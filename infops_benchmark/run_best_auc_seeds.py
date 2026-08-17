import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "benchmark_results"
BEST_TABLE = RESULTS / "all_models_best_auc_by_dataset.csv"
OUT = RESULTS / "repeated_seed_best_auc"


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def value(row, key, default=None, cast=str):
    raw = row.get(key, "")
    if raw in ("", None):
        return default
    try:
        if cast is int:
            return int(float(raw))
        if cast is float:
            return float(raw)
        return cast(raw)
    except (TypeError, ValueError):
        return default


def finite_float(row, key):
    try:
        val = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def infer_contrastive_test_rounds(row):
    found = value(row, "test_rounds", None, int)
    if found is not None:
        return found
    if row.get("setting_source"):
        return 64
    return 16 if row["dataset"] == "russia" else 8


def result_json_path(dataset, model):
    return RESULTS / dataset / f"{model}.json"


def command_for(row, seed, device):
    model = row["model"]
    dataset = row["dataset"]
    if model.startswith("pygod_"):
        pygod_model = model.removeprefix("pygod_")
        gpu = device.split(":")[-1] if device.startswith("cuda:") else "-1"
        return [
            sys.executable,
            "run_pygod_russia.py",
            "--dataset",
            dataset,
            "--models",
            pygod_model,
            "--epochs",
            str(value(row, "epochs", 100, int)),
            "--hid-dim",
            str(value(row, "hid_dim", 64, int)),
            "--lr",
            str(value(row, "lr", 0.004, float)),
            "--batch-size",
            str(value(row, "batch_size", 1024, int)),
            "--num-neigh",
            str(value(row, "num_neigh", 10, int)),
            "--gpu",
            gpu,
            "--seed",
            str(seed),
        ]
    if model == "gadnr":
        gpu = device.split(":")[-1] if device.startswith("cuda:") else "-1"
        return [
            sys.executable,
            "run_pygod_gadnr.py",
            "--dataset",
            dataset,
            "--result-name",
            "gadnr",
            "--epochs",
            str(value(row, "epochs", 100, int)),
            "--hid-dim",
            str(value(row, "hid_dim", 16, int)),
            "--backbone",
            str(value(row, "backbone", "GCN", str)),
            "--lambda-x",
            str(value(row, "lambda_x", 0.8, float)),
            "--lambda-d",
            str(value(row, "lambda_d", 0.5, float)),
            "--lambda-n",
            str(value(row, "lambda_n", 0.001, float)),
            "--lambda-x-prime",
            str(value(row, "lambda_x_prime", 1.0, float)),
            "--lambda-d-prime",
            str(value(row, "lambda_d_prime", 1.0, float)),
            "--lambda-n-prime",
            str(value(row, "lambda_n_prime", 1.0, float)),
            "--batch-size",
            str(value(row, "batch_size", 0, int)),
            "--num-neigh",
            str(value(row, "num_neigh", -1, int)),
            "--gpu",
            gpu,
            "--seed",
            str(seed),
        ]
    if model in ("anemone", "gradate"):
        cmd = [
            sys.executable,
            "run_contrastive_russia.py",
            "--dataset",
            dataset,
            "--models",
            model.upper(),
            "--epochs",
            str(value(row, "epochs", 50, int)),
            "--test-rounds",
            str(infer_contrastive_test_rounds(row)),
            "--embedding-dim",
            str(value(row, "embedding_dim", 64, int)),
            "--lr",
            str(value(row, "lr", 0.001, float)),
            "--alpha",
            str(value(row, "alpha", 0.5, float)),
            "--beta",
            str(value(row, "beta", 0.1, float)),
            "--negsamp-patch",
            str(value(row, "negsamp_patch", 1, int)),
            "--negsamp-context",
            str(value(row, "negsamp_context", 1, int)),
            "--seed",
            str(seed),
            "--device",
            device,
        ]
        return cmd
    if model.startswith("iohunter_"):
        gnn_type = model.removeprefix("iohunter_")
        return [
            sys.executable,
            "run_iohunter_gnn_russia.py",
            "--dataset",
            dataset,
            "--models",
            gnn_type,
            "--epochs",
            str(value(row, "epochs", 300, int)),
            "--hidden-dim",
            str(value(row, "hidden_dim", 64, int)),
            "--lr",
            str(value(row, "lr", 0.001, float)),
            "--seed",
            str(seed),
            "--device",
            device,
        ]
    raise ValueError(f"Unsupported model: {model}")


def task_groups():
    rows = read_csv(BEST_TABLE)
    groups = []
    for row in rows:
        if not row.get("dataset") or not row.get("model"):
            continue
        groups.append(row)
    groups.sort(key=lambda r: (r["dataset"], r["model"]))
    return groups


def run_one(row, seed, device, force=False):
    dataset = row["dataset"]
    model = row["model"]
    out_path = OUT / dataset / model / f"seed_{seed}.json"
    if out_path.exists() and not force:
        print(f"skip existing {dataset}/{model}/seed_{seed}", flush=True)
        return

    cmd = command_for(row, seed, device)
    start = time.perf_counter()
    print(f"\n=== {dataset} {model} seed={seed} ===", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=HERE, check=True)
        payload = read_json(result_json_path(dataset, model))
        payload["model"] = model
        payload["dataset"] = dataset
        payload["seed"] = seed
        payload["status"] = "ok"
        payload["repeated_seed_runtime_seconds"] = time.perf_counter() - start
    except Exception as exc:
        payload = {
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "status": "failed",
            "error": repr(exc),
            "repeated_seed_runtime_seconds": time.perf_counter() - start,
        }
    payload["best_config_auc"] = finite_float(row, "auc")
    payload["best_config_macro_f1"] = finite_float(row, "macro_f1")
    payload["best_config"] = dict(row)
    payload["command"] = cmd
    write_json(out_path, payload)
    print(f"wrote {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--skip-iohunter", action="store_true")
    parser.add_argument("--include-datasets", nargs="+")
    parser.add_argument("--include-models", nargs="+")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    groups = task_groups()
    if args.skip_iohunter:
        groups = [row for row in groups if not row["model"].startswith("iohunter_")]
    if args.include_datasets:
        datasets = set(args.include_datasets)
        groups = [row for row in groups if row["dataset"] in datasets]
    if args.include_models:
        models = set(args.include_models)
        groups = [row for row in groups if row["model"] in models]
    selected = [row for idx, row in enumerate(groups) if idx % args.num_workers == args.worker_index]
    print(
        f"worker {args.worker_index}/{args.num_workers} on {args.device}: "
        f"{len(selected)} model-dataset groups, {len(selected) * len(args.seeds)} runs",
        flush=True,
    )
    for row in selected:
        for seed in args.seeds:
            run_one(row, seed, args.device, args.force)
    marker = OUT / f"worker_{args.worker_index}.complete"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("complete\n")
    print(f"wrote {marker}", flush=True)


if __name__ == "__main__":
    main()
