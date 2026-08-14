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


def parse_list(value, item_type):
    return [item_type(item.strip()) for item in value.split(",") if item.strip()]


def parse_auc(text):
    matches = AUC_RE.findall(text)
    return float(matches[-1]) if matches else None


def safe_name(value):
    return str(value).replace(".", "p").replace("-", "m")


def suggest(trial, name, values):
    if len(values) == 1:
        return values[0]
    return trial.suggest_categorical(name, values)


def trial_params(trial, args):
    return {
        "num_anchors": suggest(trial, "num_anchors", args.num_anchor_values),
        "embedding_channels": suggest(trial, "embedding_channels", args.embedding_channels_values),
        "hidden_channels": suggest(trial, "hidden_channels", args.hidden_channels_values),
        "num_layers": suggest(trial, "num_layers", args.num_layer_values),
        "epochs": suggest(trial, "epochs", args.epoch_values),
        "learning_rate": suggest(trial, "learning_rate", args.learning_rates),
        "beta_1": suggest(trial, "beta_1", args.beta_1_values),
        "beta_2": suggest(trial, "beta_2", args.beta_2_values),
        "q": suggest(trial, "q", args.q_values),
        "ending_rounds": suggest(trial, "ending_rounds", args.ending_round_values),
        "seed": suggest(trial, "seed", args.seeds),
    }


def make_run_id(trial_number, params):
    parts = [f"trial{trial_number:04d}"]
    for key in (
        "num_anchors",
        "embedding_channels",
        "hidden_channels",
        "num_layers",
        "epochs",
        "learning_rate",
        "beta_1",
        "beta_2",
        "q",
        "ending_rounds",
        "seed",
    ):
        parts.append(f"{key}-{safe_name(params[key])}")
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
        "--learning_rate",
        str(params["learning_rate"]),
        "--beta_1",
        str(params["beta_1"]),
        "--beta_2",
        str(params["beta_2"]),
        "--q",
        str(params["q"]),
        "--ending_rounds",
        str(params["ending_rounds"]),
    ]

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
    auc = parse_auc(completed.stdout)

    (trial_dir / "command.json").write_text(json.dumps(command, indent=2))
    (trial_dir / "stdout.txt").write_text(completed.stdout)
    (trial_dir / "stderr.txt").write_text(completed.stderr)

    result = {
        "dataset": dataset,
        "trial_number": trial.number,
        "gpu": gpu_id,
        "returncode": completed.returncode,
        "auc": auc,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "trial_dir": str(trial_dir),
        **params,
    }
    (trial_dir / "trial.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    append_trial_csv(Path(args.output_dir) / "trials.csv", result)
    return result


def append_trial_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "trial_number",
        "auc",
        "returncode",
        "elapsed_seconds",
        "gpu",
        "num_anchors",
        "embedding_channels",
        "hidden_channels",
        "num_layers",
        "epochs",
        "learning_rate",
        "beta_1",
        "beta_2",
        "q",
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
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=5
    )
    study = optuna.create_study(
        study_name=f"{args.study_name}_{dataset}",
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.search_seed,n_startup_trials=10, multivariate=True),
        pruner=pruner,
        load_if_exists=True,
    )

    def objective(trial):
        params = trial_params(trial, args)
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
    parser.add_argument("--output_dir", type=str, default="optuna_results_russia")
    parser.add_argument("--study_name", type=str, default="asgae_simple")
    parser.add_argument("--datasets", type=str, default="russia")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--parallel_jobs", type=int, default=None)
    parser.add_argument("--n_trials", type=int, default=24)
    parser.add_argument("--target_auc", type=float, default=0.8)
    parser.add_argument("--search_seed", type=int, default=12345)
    parser.add_argument("--seeds", type=str, default="12345")

    parser.add_argument("--num_anchor_values", type=str, default="75,125,150")
    parser.add_argument("--embedding_channels_values", type=str, default="32,96,128")
    parser.add_argument("--hidden_channels_values", type=str, default="8,24,32")
    parser.add_argument("--num_layer_values", type=str, default="3,4")
    parser.add_argument("--epoch_values", type=str, default="80,120,150")
    parser.add_argument("--learning_rates", type=str, default="0.002,0.003,0.007,0.01")
    parser.add_argument("--beta_1_values", type=str, default="0.5,0.75,1.25,1.5")
    parser.add_argument(
        "--beta_2_values", type=str, default="0.0005,0.00075,0.0015,0.002"
    )
    parser.add_argument("--q_values", type=str, default="85,88,92,95")
    parser.add_argument("--ending_round_values", type=str, default="1,2")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    args.repo_dir = str(repo_dir)
    if args.python is None:
        repo_python = repo_dir / "env" / "bin" / "python"
        args.python = str(repo_python) if repo_python.exists() else sys.executable

    args.datasets = parse_list(args.datasets, str)
    args.gpus = parse_list(args.gpus, int)
    args.seeds = parse_list(args.seeds, int)
    args.num_anchor_values = parse_list(args.num_anchor_values, int)
    args.embedding_channels_values = parse_list(args.embedding_channels_values, int)
    args.hidden_channels_values = parse_list(args.hidden_channels_values, int)
    args.num_layer_values = parse_list(args.num_layer_values, int)
    args.epoch_values = parse_list(args.epoch_values, int)
    args.learning_rates = parse_list(args.learning_rates, float)
    args.beta_1_values = parse_list(args.beta_1_values, float)
    args.beta_2_values = parse_list(args.beta_2_values, float)
    args.q_values = parse_list(args.q_values, int)
    args.ending_round_values = parse_list(args.ending_round_values, int)

    if not args.gpus:
        raise ValueError("At least one GPU id is required")
    if args.parallel_jobs is None:
        args.parallel_jobs = len(args.gpus)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    best_by_dataset = {}
    for dataset in args.datasets:
        best_by_dataset[dataset] = optimize_dataset(args, dataset)

    summary_path = Path(args.output_dir) / "best_by_dataset.json"
    summary_path.write_text(json.dumps(best_by_dataset, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
