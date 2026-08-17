import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import load_dataset, normalized_scores


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BEST_TABLE = ROOT / "benchmark_results" / "all_models_best_auc_by_dataset.csv"
OUT = ROOT / "benchmark_results" / "seeded_model_scores"


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


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


def checkpoint_path(out_root, dataset, model, seed):
    return Path(out_root) / f"seed_{seed}" / dataset / model / "model.pt"


def run_dir(out_root, dataset, model, seed):
    return Path(out_root) / f"seed_{seed}" / dataset / model


def command_for(row, seed, device, out_root):
    model = row["model"]
    dataset = row["dataset"]
    save_dir = str(Path(out_root) / f"seed_{seed}")

    if model.startswith("pygod_"):
        pygod_model = model.removeprefix("pygod_")
        gpu = device.split(":")[-1] if device.startswith("cuda:") else "-1"
        return [
            sys.executable,
            "run_pygod_russia.py",
            "--dataset", dataset,
            "--models", pygod_model,
            "--epochs", str(value(row, "epochs", 100, int)),
            "--hid-dim", str(value(row, "hid_dim", 64, int)),
            "--lr", str(value(row, "lr", 0.004, float)),
            "--batch-size", str(value(row, "batch_size", 1024, int)),
            "--num-neigh", str(value(row, "num_neigh", 10, int)),
            "--gpu", gpu,
            "--seed", str(seed),
            "--save-dir", save_dir,
        ]

    if model in ("anemone", "gradate"):
        return [
            sys.executable,
            "run_contrastive_russia.py",
            "--dataset", dataset,
            "--models", model.upper(),
            "--epochs", str(value(row, "epochs", 50, int)),
            "--test-rounds", str(value(row, "test_rounds", 32, int)),
            "--embedding-dim", str(value(row, "embedding_dim", 64, int)),
            "--lr", str(value(row, "lr", 0.001, float)),
            "--alpha", str(value(row, "alpha", 0.5, float)),
            "--beta", str(value(row, "beta", 0.1, float)),
            "--negsamp-patch", str(value(row, "negsamp_patch", 1, int)),
            "--negsamp-context", str(value(row, "negsamp_context", 1, int)),
            "--seed", str(seed),
            "--device", device,
            "--save-dir", save_dir,
        ]

    if model == "gadnr":
        gpu = device.split(":")[-1] if device.startswith("cuda:") else "-1"
        return [
            sys.executable,
            "run_pygod_gadnr.py",
            "--dataset", dataset,
            "--result-name", "gadnr_seeded_score",
            "--epochs", str(value(row, "epochs", 100, int)),
            "--hid-dim", str(value(row, "hid_dim", 16, int)),
            "--backbone", str(value(row, "backbone", "GCN", str)),
            "--lambda-x", str(value(row, "lambda_x", 0.8, float)),
            "--lambda-d", str(value(row, "lambda_d", 0.5, float)),
            "--lambda-n", str(value(row, "lambda_n", 0.001, float)),
            "--lambda-x-prime", str(value(row, "lambda_x_prime", 1.0, float)),
            "--lambda-d-prime", str(value(row, "lambda_d_prime", 1.0, float)),
            "--lambda-n-prime", str(value(row, "lambda_n_prime", 1.0, float)),
            "--batch-size", str(value(row, "batch_size", 0, int)),
            "--num-neigh", str(value(row, "num_neigh", -1, int)),
            "--gpu", gpu,
            "--seed", str(seed),
            "--save-dir", save_dir,
        ]

    raise ValueError(f"Unsupported model: {model}")


def tensor_to_numpy(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def save_node_scores(ckpt_path, dataset):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    scores = ckpt.get("decision_score_")
    if scores is None:
        raise ValueError(f"{ckpt_path} does not contain decision_score_")
    raw = tensor_to_numpy(scores).astype(np.float64)
    _payload, _graph, labels, _sbert = load_dataset(dataset, "graph")
    if raw.shape[0] != labels.shape[0]:
        raise ValueError(f"{ckpt_path}: {raw.shape[0]} scores for {labels.shape[0]} labels")

    probs = normalized_scores(raw)
    root = ckpt_path.parent
    np.save(root / "raw_scores.npy", raw)
    np.save(root / "probabilities.npy", probs)
    with open(root / "node_scores.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["node_id", "label", "raw_score", "probability"])
        writer.writeheader()
        for idx, (label, raw_score, prob) in enumerate(zip(labels, raw, probs)):
            writer.writerow({
                "node_id": idx,
                "label": int(label),
                "raw_score": float(raw_score),
                "probability": float(prob),
            })
    return ckpt


def task_rows(include_datasets=None, include_models=None):
    rows = [row for row in read_csv(BEST_TABLE) if row.get("dataset") and row.get("model")]
    if include_datasets:
        keep = set(include_datasets)
        rows = [row for row in rows if row["dataset"] in keep]
    if include_models:
        keep = set(include_models)
        rows = [row for row in rows if row["model"] in keep]
    rows.sort(key=lambda row: (row["dataset"], row["model"]))
    return rows


def run_one(row, seed, device, out_root, force=False):
    dataset = row["dataset"]
    model = row["model"]
    root = run_dir(out_root, dataset, model, seed)
    status_path = root / "run.status.json"
    ckpt_path = checkpoint_path(out_root, dataset, model, seed)
    scores_path = root / "node_scores.csv"

    if status_path.exists() and ckpt_path.exists() and scores_path.exists() and not force:
        try:
            status = json.load(open(status_path))
            if status.get("status") == "ok":
                print(f"skip existing {dataset}/{model}/seed_{seed}", flush=True)
                return
        except json.JSONDecodeError:
            pass

    cmd = command_for(row, seed, device, out_root)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "train.log"
    start = time.perf_counter()
    print(f"\n=== {dataset} {model} seed={seed} on {device} ===", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    payload = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "device": device,
        "command": cmd,
        "best_config": dict(row),
    }

    try:
        with open(log_path, "w") as log:
            subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, check=True)
        ckpt = save_node_scores(ckpt_path, dataset)
        metrics = ckpt.get("metrics") or {}
        payload.update({
            "status": "ok",
            "checkpoint": str(ckpt_path),
            "node_scores_csv": str(scores_path),
            "raw_scores_npy": str(root / "raw_scores.npy"),
            "probabilities_npy": str(root / "probabilities.npy"),
            "auc": metrics.get("auc"),
            "macro_f1": metrics.get("macro_f1"),
            "model_size_mb": metrics.get("model_size_mb"),
            "runtime_seconds": metrics.get("runtime_seconds"),
            "best_config_auc": finite_float(row, "auc"),
            "best_config_macro_f1": finite_float(row, "macro_f1"),
        })
    except Exception as exc:
        payload.update({
            "status": "failed",
            "error": repr(exc),
            "checkpoint": str(ckpt_path),
            "node_scores_csv": str(scores_path),
            "log": str(log_path),
        })
    payload["wall_seconds"] = time.perf_counter() - start
    write_json(status_path, payload)
    print(f"wrote {status_path} status={payload['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--include-datasets", nargs="+")
    parser.add_argument("--include-models", nargs="+")
    parser.add_argument("--out-root", default=str(OUT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    groups = task_rows(args.include_datasets, args.include_models)
    selected = [row for idx, row in enumerate(groups) if idx % args.num_workers == args.worker_index]
    print(
        f"worker {args.worker_index}/{args.num_workers} on {args.device}: "
        f"{len(selected)} model-dataset groups, {len(selected) * len(args.seeds)} runs",
        flush=True,
    )
    for row in selected:
        for seed in args.seeds:
            run_one(row, seed, args.device, args.out_root, args.force)
    marker = Path(args.out_root) / f"worker_{args.worker_index}.complete"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("complete\n")
    print(f"wrote {marker}", flush=True)


if __name__ == "__main__":
    main()
