import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import optuna


AUC_RE = re.compile(r"roc_auc_score:([0-9.]+)")


def parse_list(value, item_type):
    return [item_type(item.strip()) for item in value.split(",") if item.strip()]


def parse_auc(stdout):
    matches = AUC_RE.findall(stdout)
    if not matches:
        return None
    return float(matches[-1])


def choose_param(trial, name, values):
    if len(values) == 1:
        return values[0]
    return trial.suggest_categorical(name, values)

def choose_float_param(trial, name, values, low, high, log=False, mode="categorical"):
    if mode == "float":
        return trial.suggest_float(name, low, high, log=log)
    return choose_param(trial, name, values)


def safe_name(value):
    return str(value).replace(".", "p").replace("/", "_")


def run_main(args, dataset, trial, params, gpu_id):
    run_id = (
        f"trial{trial.number:04d}_"
        f"lr{safe_name(params['learning_rate'])}_"
        f"b1{safe_name(params['beta_1'])}_"
        f"b2{safe_name(params['beta_2'])}_"
        f"q{params['q']}_"
        f"emb{params['embedding_channels']}_"
        f"hid{params['hidden_channels']}_"
        f"anc{params['num_anchors']}_"
        f"ep{params['epochs']}_"
        f"sp{params['sp_epochs']}_"
        f"it{params['ending_rounds']}_"
        f"seed{params['seed']}"
    )
    trial_dir = Path(args.output_dir) / dataset / run_id
    trial_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.python,
        args.main_script,
        "--data_flag",
        "real_world",
        "--data_dir",
        args.data_dir,
        "--results_dir",
        str(trial_dir / "results"),
        "--model_path",
        str(trial_dir / "model.pt"),
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
        str(args.num_layers),
        "--epochs",
        str(params["epochs"]),
        "--sp_epochs",
        str(params["sp_epochs"]),
        "--learning_rate",
        str(params["learning_rate"]),
        "--beta_1",
        str(params["beta_1"]),
        "--beta_2",
        str(params["beta_2"]),
        "--q",
        str(params["q"]),
        "--structure_chunk_size",
        str(args.structure_chunk_size),
        "--ending_rounds",
        str(params["ending_rounds"]),
        "--early_stopping_patience",
        str(params["early_stopping_patience"]),
        "--early_stopping_min_delta",
        str(args.early_stopping_min_delta),
    ]

    if args.normalize_features:
        command.append("--normalize_features")
    else:
        command.append("--no-normalize_features")

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

    (trial_dir / "stdout.txt").write_text(completed.stdout)
    (trial_dir / "stderr.txt").write_text(completed.stderr)

    auc = parse_auc(completed.stdout)
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
    return result


def save_best(study, args, dataset):
    all_params = study.best_trial.user_attrs.get("all_params", {})
    best = {
        "dataset": dataset,
        "auc": study.best_value,
        **study.best_params,
        **all_params,
    }
    path = Path(args.output_dir) / dataset / "best.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(best, indent=2, sort_keys=True))
    return best


def optimize_dataset(args, dataset):
    storage_path = Path(args.output_dir) / "optuna.sqlite3"
    storage = f"sqlite:///{storage_path}"
    study_name = f"{args.study_prefix}_{dataset}"

    sampler = optuna.samplers.TPESampler(
        seed=args.search_seed,
        multivariate=True,
        group=True,
        n_startup_trials=min(args.startup_trials, args.n_trials),
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    mlflow.set_tracking_uri(f"file://{Path(args.output_dir).resolve() / 'mlruns'}")
    mlflow.set_experiment(study_name)

    def objective(trial):
        params = {
            "learning_rate": choose_param(trial, "learning_rate", args.learning_rates),
            "beta_1": choose_float_param(
                trial,
                "beta_1",
                args.beta_1_values,
                args.beta_1_min,
                args.beta_1_max,
                log=args.beta_log,
                mode=args.beta_sampling,
            ),
            "beta_2": choose_float_param(
                trial,
                "beta_2",
                args.beta_2_values,
                args.beta_2_min,
                args.beta_2_max,
                log=args.beta_log,
                mode=args.beta_sampling,
            ),
            "q": choose_param(trial, "q", args.q_values),
            "embedding_channels": choose_param(
                trial,
                "embedding_channels",
                args.embedding_channels_values,
            ),
            "hidden_channels": choose_param(
                trial,
                "hidden_channels",
                args.hidden_channels_values,
            ),
            "num_anchors": choose_param(trial, "num_anchors", args.num_anchor_values),
            "epochs": choose_param(trial, "epochs", args.epoch_values),
            "sp_epochs": choose_param(trial, "sp_epochs", args.sp_epoch_values),
            "ending_rounds": choose_param(trial, "ending_rounds", args.iterator_values),
            "early_stopping_patience": choose_param(
                trial,
                "early_stopping_patience",
                args.early_stopping_patience_values,
            ),
            "seed": choose_param(trial, "seed", args.seeds),
        }
        gpu_id = args.gpus[trial.number % len(args.gpus)]

        with mlflow.start_run(run_name=f"{dataset}_trial_{trial.number:04d}", nested=False):
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("gpu", gpu_id)
            mlflow.log_params(params)
            trial.set_user_attr("all_params", params)

            result = run_main(args, dataset, trial, params, gpu_id)
            mlflow.log_metric("elapsed_seconds", result["elapsed_seconds"])
            mlflow.log_metric("returncode", result["returncode"])
            mlflow.log_param("trial_dir", result["trial_dir"])

            if result["returncode"] != 0 or result["auc"] is None:
                mlflow.log_metric("auc", 0.0)
                raise optuna.TrialPruned("main.py failed or did not report AUC")

            mlflow.log_metric("auc", result["auc"])
            trial.set_user_attr("trial_dir", result["trial_dir"])
            trial.set_user_attr("gpu", gpu_id)

            if args.target_auc is not None and result["auc"] >= args.target_auc:
                trial.study.stop()

            return result["auc"]

    print(
        f"Optimizing {dataset}: n_trials={args.n_trials}, "
        f"parallel_jobs={args.parallel_jobs}, gpus={args.gpus}"
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.parallel_jobs,
        show_progress_bar=False,
    )
    best = save_best(study, args, dataset)
    print(f"Best {dataset}: {json.dumps(best, sort_keys=True)}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--main_script", type=str, default="main.py")
    parser.add_argument("--data_dir", type=str, default="data/real_world")
    parser.add_argument("--output_dir", type=str, default="optuna_results")
    parser.add_argument("--study_prefix", type=str, default="asgae_auc")
    parser.add_argument("--datasets", type=str, default="russia,venezuela,china")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--parallel_jobs", type=int, default=None)
    parser.add_argument("--n_trials", type=int, default=80)
    parser.add_argument("--startup_trials", type=int, default=16)
    parser.add_argument("--target_auc", type=float, default=0.8)
    parser.add_argument("--search_seed", type=int, default=12345)
    parser.add_argument("--learning_rates", type=str, default="1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3")
    parser.add_argument("--beta_1_values", type=str, default="0.1,0.25,0.5,1.0,2.0,5.0")
    parser.add_argument("--beta_2_values", type=str, default="0.01,0.05,0.1,0.5,1.0,2.0,5.0")
    parser.add_argument("--beta_sampling", choices=["float", "categorical"], default="float")
    parser.add_argument("--beta_1_min", type=float, default=0.01)
    parser.add_argument("--beta_1_max", type=float, default=10.0)
    parser.add_argument("--beta_2_min", type=float, default=0.001)
    parser.add_argument("--beta_2_max", type=float, default=10.0)
    parser.add_argument("--beta_log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--q_values", type=str, default="70,75,80,85,90,95")
    parser.add_argument("--embedding_channels_values", type=str, default="16,32,64,128,256")
    parser.add_argument("--hidden_channels_values", type=str, default="8,16,32,64,128")
    parser.add_argument("--num_anchor_values", type=str, default="0,25,50,100,150,200,300")
    parser.add_argument("--epoch_values", type=str, default="50,100,150,200")
    parser.add_argument("--sp_epoch_values", type=str, default="10,20,40,60")
    parser.add_argument("--iterator_values", type=str, default="1,2,3")
    parser.add_argument("--early_stopping_patience_values", type=str, default="0,10,20")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--normalize_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seeds", type=str, default="12345")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--structure_chunk_size", type=int, default=256)
    args = parser.parse_args()

    if args.python is None:
        repo_python = Path(args.repo_dir) / "env" / "bin" / "python"
        args.python = str(repo_python) if repo_python.exists() else sys.executable

    args.datasets = parse_list(args.datasets, str)
    args.gpus = parse_list(args.gpus, int)
    args.learning_rates = parse_list(args.learning_rates, float)
    args.beta_1_values = parse_list(args.beta_1_values, float)
    args.beta_2_values = parse_list(args.beta_2_values, float)
    args.q_values = parse_list(args.q_values, int)
    args.embedding_channels_values = parse_list(args.embedding_channels_values, int)
    args.hidden_channels_values = parse_list(args.hidden_channels_values, int)
    args.num_anchor_values = parse_list(args.num_anchor_values, int)
    args.epoch_values = parse_list(args.epoch_values, int)
    args.sp_epoch_values = parse_list(args.sp_epoch_values, int)
    args.iterator_values = parse_list(args.iterator_values, int)
    args.early_stopping_patience_values = parse_list(args.early_stopping_patience_values, int)
    args.seeds = parse_list(args.seeds, int)

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
