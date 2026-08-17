import argparse
import csv
import json
import math
from functools import reduce
from operator import mul
from pathlib import Path

import optuna
import torch

from run_pygod_gadnr import run_gadnr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


PAPER_SPACE = {
    "lambda_x": [0.1, 0.5, 0.8, 0.9],
    "lambda_x_prime": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
    "lambda_d": [0.1, 0.5, 0.8],
    "lambda_d_prime": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
    "lambda_n": [0.001, 0.5, 0.8],
    "lambda_n_prime": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
    "hid_dim": [8, 16, 32, 64, 128],
    "backbone": ["GCN", "GraphSAGE", "GIN"],
}


FIXED = {
    "dataset": "russia",
    "graph_key": "graph",
    "epochs": 100,
    "num_layers": 1,
    "deg_dec_layers": 4,
    "fea_dec_layers": 3,
    "sample_size": 2,
    "sample_time": 3,
    "neigh_loss": "KL",
    "real_loss": False,
    "lr": 0.01,
    "dropout": 0.0,
    "weight_decay": 0.0003,
    "batch_size": 0,
    "num_neigh": -1,
    "verbose": 0,
}


def result_dir(dataset, objective):
    suffix = "macro_f1" if objective == "macro_f1" else "auc"
    return ROOT / "benchmark_results" / dataset / f"optuna_gadnr_{suffix}"


def storage_url(dataset, objective):
    return f"sqlite:///{result_dir(dataset, objective) / 'study.db'}"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return value if math.isfinite(value) else ""


def row_from_trial(trial):
    row = {
        "trial_number": trial.number,
        "state": trial.state.name,
        "objective_value": finite(trial.value),
        "auc": finite(trial.user_attrs.get("auc")),
        "macro_f1": finite(trial.user_attrs.get("macro_f1")),
    }
    row.update(trial.params)
    for key in [
        "runtime_seconds",
        "model_size_mb",
        "num_parameters",
        "num_nodes",
        "num_anomalies",
        "feature_dim",
        "edge_count",
        "seed",
        "gpu",
        "epochs",
        "batch_size",
        "num_neigh",
    ]:
        row[key] = trial.user_attrs.get(key, "")
    return row


def best_complete_trial(complete, metric):
    return max(
        complete,
        key=lambda trial: float(trial.user_attrs.get(metric, float("-inf"))),
    ) if complete else None


def search_space_from_args(args):
    space = {key: list(values) for key, values in PAPER_SPACE.items()}
    if args.hid_dims:
        space["hid_dim"] = args.hid_dims
    if args.backbones:
        space["backbone"] = args.backbones
    return space


def search_space_size(search_space):
    return reduce(mul, (len(values) for values in search_space.values()), 1)


def export_study(study, dataset, objective, search_space=None):
    search_space = search_space or PAPER_SPACE
    out_dir = result_dir(dataset, objective)
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    failed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL]
    rows = [row_from_trial(trial) for trial in study.trials]
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with open(out_dir / "trials.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    best_auc = best_complete_trial(complete, "auc")
    best_macro_f1 = best_complete_trial(complete, "macro_f1")
    summary = {
        "dataset": dataset,
        "objective": objective,
        "sampler": "TPESampler",
        "direction": f"maximize_{objective}",
        "num_trials": len(study.trials),
        "num_complete_trials": len(complete),
        "num_failed_trials": len(failed),
        "search_space_size": search_space_size(search_space),
        "search_space": search_space,
        "fixed_parameters": FIXED,
        "best_by_objective": row_from_trial(study.best_trial) if complete else None,
        "best_by_auc": row_from_trial(best_auc) if best_auc else None,
        "best_by_macro_f1": row_from_trial(best_macro_f1) if best_macro_f1 else None,
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "paper_search_space.json", {
        "source": "GAD-NR paper grid supplied in the benchmark request.",
            "search_space": search_space,
        "parameter_mapping": {
            "lambda_x": "PyGOD lambda_loss2, self-attribute reconstruction loss",
            "lambda_d": "PyGOD lambda_loss3, degree reconstruction loss",
            "lambda_n": "PyGOD lambda_loss1, neighborhood reconstruction loss",
            "lambda_x_prime": "PyGOD feature_loss_weight for weighted decision score",
            "lambda_d_prime": "PyGOD degree_loss_weight for weighted decision score",
            "lambda_n_prime": "PyGOD h_loss_weight for weighted decision score",
        },
    })


def suggest_config(trial, search_space):
    return {
        key: trial.suggest_categorical(key, values)
        for key, values in search_space.items()
    }


def make_objective(args):
    search_space = search_space_from_args(args)

    def objective(trial):
        config = suggest_config(trial, search_space)
        run_args = {
            **FIXED,
            **config,
            "dataset": args.dataset,
            "seed": args.seed + trial.number,
            "gpu": args.gpu,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_neigh": args.num_neigh,
            "verbose": args.verbose,
        }
        payload = run_gadnr(run_args)
        trial.set_user_attr("auc", payload["auc"])
        trial.set_user_attr("macro_f1", payload["macro_f1"])
        trial.set_user_attr("runtime_seconds", payload["runtime_seconds"])
        trial.set_user_attr("model_size_mb", payload.get("model_size_mb"))
        trial.set_user_attr("num_parameters", payload.get("num_parameters"))
        trial.set_user_attr("num_nodes", payload["num_nodes"])
        trial.set_user_attr("num_anomalies", payload["num_anomalies"])
        trial.set_user_attr("feature_dim", payload["feature_dim"])
        trial.set_user_attr("edge_count", payload["edge_count"])
        trial.set_user_attr("seed", run_args["seed"])
        trial.set_user_attr("gpu", args.gpu)
        trial.set_user_attr("epochs", args.epochs)
        trial.set_user_attr("batch_size", args.batch_size)
        trial.set_user_attr("num_neigh", args.num_neigh)
        if args.empty_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return float(payload[args.objective])

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="russia")
    parser.add_argument("--objective", choices=["auc", "macro_f1"], default="auc")
    parser.add_argument("--study-name")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12121995)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-neigh", type=int, default=-1)
    parser.add_argument("--hid-dims", nargs="+", type=int)
    parser.add_argument("--backbones", nargs="+", choices=["GCN", "GraphSAGE", "GIN"])
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--max-total-trials", type=int, default=400)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--empty-cuda-cache", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.study_name is None:
        args.study_name = f"gadnr_{args.dataset}_{args.objective}"

    out_dir = result_dir(args.dataset, args.objective)
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
        group=True,
        constant_liar=True,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url(args.dataset, args.objective),
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    if args.export_only:
        export_study(study, args.dataset, args.objective, search_space_from_args(args))
        return

    callback = optuna.study.MaxTrialsCallback(
        args.max_total_trials,
        states=(optuna.trial.TrialState.COMPLETE,),
    )
    study.optimize(
        make_objective(args),
        n_trials=args.n_trials,
        callbacks=[callback, lambda study, trial: export_study(study, args.dataset, args.objective, search_space_from_args(args))],
        gc_after_trial=True,
        catch=(RuntimeError, ValueError),
    )
    export_study(study, args.dataset, args.objective, search_space_from_args(args))
    print(json.dumps({
        "objective": args.objective,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": len(study.trials),
        "output_dir": str(out_dir),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
