import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import optuna


AUC_RE = re.compile(r"roc_auc_score:([0-9.]+)")
AP_RE = re.compile(r"average_precision:([0-9.]+)")


def parse_list(value, item_type):
    return [item_type(item.strip()) for item in value.split(",") if item.strip()]


def parse_metric(pattern, text):
    matches = pattern.findall(text)
    return float(matches[-1]) if matches else None


def safe_name(value):
    return str(value).replace(".", "p").replace("-", "m")


def suggest_near_int(trial, name, center, radius, step, blocked):
    low = center - radius
    high = center + radius
    choices = [value for value in range(low, high + 1, step) if value not in blocked]
    return trial.suggest_categorical(name, choices)


def suggest_params(trial, args):
    profile = trial.suggest_categorical("profile", ["lean", "balanced", "structure"])
    threshold = trial.suggest_categorical("threshold", ["wide", "tight"])
    beta_mode = trial.suggest_categorical("beta_mode", ["fixed", "dynamic"])

    if profile == "lean":
        embedding_choices = [40, 48, 56]
        hidden_choices = [10, 12, 14]
        epoch_choices = [60, 80]
    elif profile == "balanced":
        embedding_choices = [72, 80, 96]
        hidden_choices = [20, 24, 28]
        epoch_choices = [80, 120]
    else:
        embedding_choices = [96, 112, 128]
        hidden_choices = [24, 32, 40]
        epoch_choices = [70, 90]

    params = {
        "num_anchors": suggest_near_int(
            trial,
            "num_anchors",
            center=100,
            radius=args.anchor_radius,
            step=args.anchor_step,
            blocked={100},
        ),
        "embedding_channels": trial.suggest_categorical(
            f"{profile}_embedding_channels",
            [value for value in embedding_choices if value != 64],
        ),
        "hidden_channels": trial.suggest_categorical(
            f"{profile}_hidden_channels",
            [value for value in hidden_choices if value != 16],
        ),
        "num_layers": trial.suggest_categorical("num_layers", [3]),
        "epochs": trial.suggest_categorical(
            f"{profile}_epochs",
            [value for value in epoch_choices if value != 100],
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            args.learning_rate_min,
            args.learning_rate_max,
            log=True,
        ),
        "beta_1": trial.suggest_float("beta_1", 0.65, 1.45),
        "beta_2": trial.suggest_float("beta_2", 5e-4, 2.5e-3, log=True),
        "q": trial.suggest_categorical(
            f"{threshold}_q",
            [82, 86, 88] if threshold == "wide" else [92, 94, 96],
        ),
        "sp_epochs": trial.suggest_categorical("sp_epochs", [8, 12, 16]),
        "sp_learning_rate": trial.suggest_float("sp_learning_rate", 3e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True),
        "grad_clip": trial.suggest_categorical("grad_clip", [0.5, 1.5, 2.5]),
        "structure_chunk_size": trial.suggest_categorical(
            "structure_chunk_size",
            [128, 256],
        ),
        "ending_rounds": 1,
        "seed": trial.suggest_categorical("seed", args.seeds),
        "dynamic_beta_2": beta_mode == "dynamic",
        "min_beta_2": 5e-5,
        "max_beta_2": trial.suggest_categorical("max_beta_2", [3e-3, 6e-3, 1e-2]),
        "profile": profile,
        "threshold": threshold,
        "beta_mode": beta_mode,
    }

    # Keep the initial search close to the user-provided learning rate but never
    # exactly equal to it.
    if abs(params["learning_rate"] - 5e-3) < 1e-6:
        params["learning_rate"] = 4.8e-3

    return params


def make_run_id(trial_number, params):
    keys = [
        "profile",
        "num_anchors",
        "embedding_channels",
        "hidden_channels",
        "epochs",
        "learning_rate",
        "beta_1",
        "beta_2",
        "q",
        "beta_mode",
    ]
    parts = [f"trial{trial_number:04d}"]
    parts.extend(f"{key}-{safe_name(params[key])}" for key in keys)
    return "_".join(parts)


def run_trial(args, dataset, trial, params):
    gpu_id = args.gpus[trial.number % len(args.gpus)]
    run_id = make_run_id(trial.number, params)
    trial_dir = Path(args.output_dir) / dataset / run_id
    results_dir = trial_dir / "results"
    trial_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.python,
        args.main_script,
        "--data_flag",
        "real_world",
        "--data_dir",
        args.data_dir,
        "--results_dir",
        str(results_dir),
        "--real_world_name",
        dataset,
        "--random_seed",
        str(params["seed"]),
        "--num_anchors",
        str(params["num_anchors"]),
        "--embedding_channels",
        str(params["embedding_channels"]),
        "--hidden_channels",
        str(params["hidden_channels"]),
        "--num_layers",
        str(params["num_layers"]),
        "--epochs",
        str(params["epochs"]),
        "--sp_epochs",
        str(params["sp_epochs"]),
        "--learning_rate",
        str(params["learning_rate"]),
        "--sp_learning_rate",
        str(params["sp_learning_rate"]),
        "--weight_decay",
        str(params["weight_decay"]),
        "--grad_clip",
        str(params["grad_clip"]),
        "--beta_1",
        str(params["beta_1"]),
        "--beta_2",
        str(params["beta_2"]),
        "--min_beta_2",
        str(params["min_beta_2"]),
        "--max_beta_2",
        str(params["max_beta_2"]),
        "--q",
        str(params["q"]),
        "--structure_chunk_size",
        str(params["structure_chunk_size"]),
        "--ending_rounds",
        str(params["ending_rounds"]),
    ]
    if params["dynamic_beta_2"]:
        command.append("--dynamic_beta_2")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start = time.time()
    completed = subprocess.run(
        command,
        cwd=args.repo_dir,
        env=env,
        text=True,
        capture_output=True,
    )
    elapsed_seconds = time.time() - start

    stdout = completed.stdout
    result = {
        "dataset": dataset,
        "trial_number": trial.number,
        "gpu": gpu_id,
        "returncode": completed.returncode,
        "auc": parse_metric(AUC_RE, stdout),
        "average_precision": parse_metric(AP_RE, stdout),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "trial_dir": str(trial_dir),
        **params,
    }

    (trial_dir / "command.json").write_text(json.dumps(command, indent=2))
    (trial_dir / "stdout.txt").write_text(stdout)
    (trial_dir / "stderr.txt").write_text(completed.stderr)
    (trial_dir / "trial.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    append_trial_csv(Path(args.output_dir) / "trials.csv", result)
    return result


def append_trial_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "trial_number",
        "auc",
        "average_precision",
        "returncode",
        "elapsed_seconds",
        "gpu",
        "profile",
        "threshold",
        "beta_mode",
        "num_anchors",
        "embedding_channels",
        "hidden_channels",
        "num_layers",
        "epochs",
        "sp_epochs",
        "learning_rate",
        "sp_learning_rate",
        "weight_decay",
        "grad_clip",
        "beta_1",
        "beta_2",
        "dynamic_beta_2",
        "max_beta_2",
        "q",
        "structure_chunk_size",
        "ending_rounds",
        "seed",
        "trial_dir",
    ]
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name) for name in fieldnames})


def save_best(study, args, dataset):
    params = study.best_trial.user_attrs["all_params"]
    best = {
        "dataset": dataset,
        "auc": study.best_value,
        "trial_number": study.best_trial.number,
        "trial_dir": study.best_trial.user_attrs.get("trial_dir"),
        **params,
    }
    dataset_dir = Path(args.output_dir) / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "best.json").write_text(json.dumps(best, indent=2, sort_keys=True))
    return best


def optimize_dataset(args, dataset):
    storage = f"sqlite:///{Path(args.output_dir) / 'optuna.sqlite3'}"
    study = optuna.create_study(
        study_name=f"{args.study_name}_{dataset}",
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=args.search_seed,
            n_startup_trials=min(4, args.n_trials),
            multivariate=True,
            group=True,
        ),
        load_if_exists=True,
    )

    def objective(trial):
        params = suggest_params(trial, args)
        trial.set_user_attr("all_params", params)
        result = run_trial(args, dataset, trial, params)
        trial.set_user_attr("trial_dir", result["trial_dir"])

        if result["returncode"] != 0 or result["auc"] is None:
            raise optuna.TrialPruned(
                f"main.py failed with returncode={result['returncode']}"
            )

        if args.target_auc is not None and result["auc"] >= args.target_auc:
            trial.study.stop()
        return result["auc"]

    print(
        f"Optimizing {dataset}: n_trials={args.n_trials}, "
        f"parallel_jobs={args.parallel_jobs}, gpus={args.gpus}"
    )
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.parallel_jobs)
    best = save_best(study, args, dataset)
    print(f"Best {dataset}: {json.dumps(best, sort_keys=True)}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", type=str, default=str(Path(__file__).parent))
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--main_script", type=str, default="main.py")
    parser.add_argument("--data_dir", type=str, default="data/real_world")
    parser.add_argument("--output_dir", type=str, default="optuna_results_russia_novel_v2")
    parser.add_argument("--study_name", type=str, default="asgae_novel_v2")
    parser.add_argument("--datasets", type=str, default="russia")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--parallel_jobs", type=int, default=1)
    parser.add_argument("--n_trials", type=int, default=12)
    parser.add_argument("--target_auc", type=float, default=0.8)
    parser.add_argument("--search_seed", type=int, default=20260505)
    parser.add_argument("--seeds", type=str, default="12345")
    parser.add_argument("--anchor_radius", type=int, default=36)
    parser.add_argument("--anchor_step", type=int, default=12)
    parser.add_argument("--learning_rate_min", type=float, default=1.5e-3)
    parser.add_argument("--learning_rate_max", type=float, default=9e-3)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    args.repo_dir = str(repo_dir)
    if args.python is None:
        repo_python = repo_dir / "env" / "bin" / "python"
        args.python = str(repo_python) if repo_python.exists() else sys.executable

    args.datasets = parse_list(args.datasets, str)
    args.gpus = parse_list(args.gpus, int)
    args.seeds = parse_list(args.seeds, int)

    if not args.gpus:
        raise ValueError("At least one GPU id is required")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    best_by_dataset = {}
    for dataset in args.datasets:
        best_by_dataset[dataset] = optimize_dataset(args, dataset)

    summary_path = Path(args.output_dir) / "best_by_dataset.json"
    summary_path.write_text(json.dumps(best_by_dataset, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
