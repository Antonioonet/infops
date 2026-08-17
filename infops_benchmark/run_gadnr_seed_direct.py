import argparse
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

from run_pygod_gadnr import run_gadnr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BEST_TABLE = ROOT / "benchmark_results" / "all_models_best_auc_by_dataset.csv"
OUT = ROOT / "benchmark_results" / "repeated_seed_best_auc"


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


def best_row(dataset):
    with open(BEST_TABLE, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("dataset") == dataset and row.get("model") == "gadnr":
                return row
    raise ValueError(f"No GADNR row for {dataset}")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()

    row = best_row(args.dataset)
    gadnr_args = SimpleNamespace(
        dataset=args.dataset,
        graph_key="graph",
        seed=args.seed,
        gpu=args.gpu,
        epochs=value(row, "epochs", 100, int),
        hid_dim=value(row, "hid_dim", 16, int),
        num_layers=1,
        deg_dec_layers=4,
        fea_dec_layers=3,
        backbone=value(row, "backbone", "GCN", str),
        sample_size=2,
        sample_time=3,
        neigh_loss="KL",
        lambda_x=value(row, "lambda_x", 0.8, float),
        lambda_d=value(row, "lambda_d", 0.5, float),
        lambda_n=value(row, "lambda_n", 0.001, float),
        lambda_x_prime=value(row, "lambda_x_prime", 1.0, float),
        lambda_d_prime=value(row, "lambda_d_prime", 1.0, float),
        lambda_n_prime=value(row, "lambda_n_prime", 1.0, float),
        real_loss=False,
        lr=0.01,
        dropout=0.0,
        weight_decay=0.0003,
        batch_size=value(row, "batch_size", 0, int),
        num_neigh=value(row, "num_neigh", -1, int),
        verbose=1,
        save_dir=None,
    )

    start = time.perf_counter()
    try:
        payload = run_gadnr(gadnr_args)
        payload["model"] = "gadnr"
        payload["dataset"] = args.dataset
        payload["seed"] = args.seed
        payload["status"] = "ok"
    except Exception as exc:
        payload = {
            "dataset": args.dataset,
            "model": "gadnr",
            "seed": args.seed,
            "status": "failed",
            "error": repr(exc),
        }
    payload["repeated_seed_runtime_seconds"] = time.perf_counter() - start
    payload["best_config_auc"] = finite_float(row, "auc")
    payload["best_config_macro_f1"] = finite_float(row, "macro_f1")
    payload["best_config"] = dict(row)
    payload["command"] = ["run_gadnr_seed_direct.py", "--dataset", args.dataset, "--seed", args.seed, "--gpu", args.gpu]

    out_path = OUT / args.dataset / "gadnr" / f"seed_{args.seed}.json"
    write_json(out_path, payload)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
