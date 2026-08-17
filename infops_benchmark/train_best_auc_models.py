import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CSV = ROOT / "benchmark_results" / "all_models_best_auc_by_dataset.csv"
DEFAULT_OUT = ROOT / "benchmark_results" / "saved_models_best_auc"


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def as_int(row, key, default=None):
    value = row.get(key, "")
    if value in ("", None):
        return default
    return int(float(value))


def as_float(row, key, default=None):
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def add_if_present(cmd, flag, row, key, cast=str):
    value = row.get(key, "")
    if value not in ("", None):
        cmd.extend([flag, str(cast(value))])


def checkpoint_path(out_dir, row):
    return Path(out_dir) / row["dataset"] / row["model"] / "model.pt"


def command_for(row, device, gpu, out_dir, seed):
    model = row["model"]
    dataset = row["dataset"]
    if model in ("anemone", "gradate"):
        cmd = [
            sys.executable,
            "run_contrastive_russia.py",
            "--dataset", dataset,
            "--models", model.upper(),
            "--epochs", str(as_int(row, "epochs", 50)),
            "--embedding-dim", str(as_int(row, "embedding_dim", 64)),
            "--lr", str(as_float(row, "lr", 0.001)),
            "--alpha", str(as_float(row, "alpha", 0.5)),
            "--beta", str(as_float(row, "beta", 0.1)),
            "--test-rounds", str(as_int(row, "test_rounds", 16)),
            "--negsamp-patch", str(as_int(row, "negsamp_patch", 1)),
            "--negsamp-context", str(as_int(row, "negsamp_context", 1)),
            "--seed", str(seed),
            "--device", device,
            "--save-dir", str(out_dir),
        ]
        return cmd

    if model.startswith("pygod_"):
        pygod_model = model.removeprefix("pygod_")
        cmd = [
            sys.executable,
            "run_pygod_russia.py",
            "--dataset", dataset,
            "--models", pygod_model,
            "--epochs", str(as_int(row, "epochs", 100)),
            "--hid-dim", str(as_int(row, "hid_dim", 64)),
            "--lr", str(as_float(row, "lr", 0.004)),
            "--gpu", str(gpu),
            "--seed", str(seed),
            "--save-dir", str(out_dir),
        ]
        add_if_present(cmd, "--batch-size", row, "batch_size", lambda v: int(float(v)))
        add_if_present(cmd, "--num-neigh", row, "num_neigh", lambda v: int(float(v)))
        return cmd

    if model == "gadnr":
        cmd = [
            sys.executable,
            "run_pygod_gadnr.py",
            "--dataset", dataset,
            "--result-name", "gadnr_saved_model_train",
            "--epochs", str(as_int(row, "epochs", 100)),
            "--hid-dim", str(as_int(row, "hid_dim", 16)),
            "--backbone", row.get("backbone") or "GCN",
            "--lambda-x", str(as_float(row, "lambda_x", 0.8)),
            "--lambda-d", str(as_float(row, "lambda_d", 0.5)),
            "--lambda-n", str(as_float(row, "lambda_n", 0.001)),
            "--lambda-x-prime", str(as_float(row, "lambda_x_prime", 1.0)),
            "--lambda-d-prime", str(as_float(row, "lambda_d_prime", 1.0)),
            "--lambda-n-prime", str(as_float(row, "lambda_n_prime", 1.0)),
            "--batch-size", str(as_int(row, "batch_size", 0)),
            "--num-neigh", str(as_int(row, "num_neigh", -1)),
            "--gpu", str(gpu),
            "--seed", str(seed),
            "--verbose", "0",
            "--save-dir", str(out_dir),
        ]
        return cmd

    raise ValueError(f"Unsupported model: {model}")


def write_status(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12121995)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    if args.datasets:
        wanted = set(args.datasets)
        rows = [row for row in rows if row["dataset"] in wanted]
    if args.models:
        wanted = set(args.models)
        rows = [row for row in rows if row["model"] in wanted]
    rows = sorted(rows, key=lambda row: (row["dataset"], row["model"]))
    selected = [row for idx, row in enumerate(rows) if idx % args.num_workers == args.worker_index]
    print(f"worker {args.worker_index}/{args.num_workers}: {len(selected)} rows on {args.device}", flush=True)

    out_dir = Path(args.out_dir)
    for local_idx, row in enumerate(selected):
        ckpt = checkpoint_path(out_dir, row)
        status_path = ckpt.with_suffix(".status.json")
        if ckpt.exists() and not args.force:
            print(f"skip existing {ckpt}", flush=True)
            continue

        run_seed = args.seed + args.worker_index * 1000 + local_idx
        cmd = command_for(row, args.device, args.gpu, out_dir, run_seed)
        payload = {
            "dataset": row["dataset"],
            "model": row["model"],
            "status": "running",
            "source_row": row,
            "command": cmd,
            "checkpoint": str(ckpt),
            "seed": run_seed,
            "started_at": time.time(),
        }
        write_status(status_path, payload)
        print("\n$", " ".join(cmd), flush=True)
        start = time.perf_counter()
        try:
            subprocess.run(cmd, cwd=HERE, check=True)
            payload["status"] = "ok"
            payload["runtime_seconds"] = time.perf_counter() - start
            payload["finished_at"] = time.time()
        except Exception as exc:
            payload["status"] = "failed"
            payload["error"] = repr(exc)
            payload["runtime_seconds"] = time.perf_counter() - start
            payload["finished_at"] = time.time()
            write_status(status_path, payload)
            print(f"failed {row['dataset']}/{row['model']}: {exc!r}", flush=True)
            continue
        write_status(status_path, payload)


if __name__ == "__main__":
    main()
