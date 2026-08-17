import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def result_root(dataset_name):
    return HERE.parent / "benchmark_results" / dataset_name


def out_root(dataset_name):
    return result_root(dataset_name) / "author_recommended_non_iohunter"


def run_cmd(cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def metric_row(result, extra):
    row = {
        "model": result["model"],
        "dataset": result["dataset"],
        "auc": result["auc"],
        "macro_f1": result["macro_f1"],
        "threshold": result["threshold"],
        "runtime_seconds": result.get("runtime_seconds"),
        "model_size_mb": result.get("model_size_mb"),
        "num_parameters": result.get("num_parameters"),
    }
    row.update(extra)
    return row


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, rows):
    best_auc = {}
    best_f1 = {}
    for row in rows:
        model = row["model"]
        if model not in best_auc or row["auc"] > best_auc[model]["auc"]:
            best_auc[model] = row
        if model not in best_f1 or row["macro_f1"] > best_f1[model]["macro_f1"]:
            best_f1[model] = row
    summary = {
        "best_by_auc": best_auc,
        "best_by_macro_f1": best_f1,
        "num_trials": len(rows),
        "notes": [
            "PyGOD uses DOMINANT/CoLA default hid_dim=64, lr=0.004, epoch=100.",
            "PyGOD keeps mini-batch neighbor sampling for large InfoOps graphs: batch_size=1024, num_neigh=10.",
            "ANEMONE uses author README alpha choices 0.6 and 0.8 with code defaults for lr, embedding, subgraph, and negative sampling.",
            "GRADATE uses code defaults alpha=0.1, beta=0.1, negsamp_patch=6, negsamp_context=1, lr=0.001, embedding_dim=64.",
            "Contrastive test_rounds is configurable for runtime; the original repos default to 256.",
        ],
    }
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pygod-batch-size", type=int, default=1024)
    parser.add_argument("--pygod-num-neigh", type=int, default=10)
    parser.add_argument("--test-rounds", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = out_root(args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    complete = out_dir / ".complete"
    if complete.exists() and not args.force:
        print(f"{args.dataset} author-recommended run already complete at {out_dir}")
        return

    gpu = args.device.split(":")[-1] if args.device.startswith("cuda:") else ("0" if args.device == "cuda" else "-1")
    rows = []

    run_cmd([
        sys.executable,
        "run_pygod_russia.py",
        "--dataset",
        args.dataset,
        "--models",
        "dominant",
        "cola",
        "--epochs",
        "100",
        "--hid-dim",
        "64",
        "--lr",
        "0.004",
        "--batch-size",
        str(args.pygod_batch_size),
        "--num-neigh",
        str(args.pygod_num_neigh),
        "--gpu",
        gpu,
    ])
    for name in ("pygod_dominant", "pygod_cola"):
        result = load_json(result_root(args.dataset) / f"{name}.json")
        rows.append(metric_row(result, {
            "setting_source": "pygod_default_core",
            "epochs": 100,
            "hid_dim": 64,
            "lr": 0.004,
            "batch_size": args.pygod_batch_size,
            "num_neigh": args.pygod_num_neigh,
        }))

    for alpha in (0.6, 0.8):
        run_cmd([
            sys.executable,
            "run_contrastive_russia.py",
            "--dataset",
            args.dataset,
            "--models",
            "ANEMONE",
            "--epochs",
            "100",
            "--test-rounds",
            str(args.test_rounds),
            "--embedding-dim",
            "64",
            "--lr",
            "0.001",
            "--alpha",
            str(alpha),
            "--negsamp-patch",
            "1",
            "--negsamp-context",
            "1",
            "--device",
            args.device,
        ])
        result = load_json(result_root(args.dataset) / "anemone.json")
        rows.append(metric_row(result, {
            "setting_source": "anemone_readme",
            "epochs": 100,
            "embedding_dim": 64,
            "lr": 0.001,
            "alpha": alpha,
            "test_rounds": args.test_rounds,
            "negsamp_patch": 1,
            "negsamp_context": 1,
        }))

    run_cmd([
        sys.executable,
        "run_contrastive_russia.py",
        "--dataset",
        args.dataset,
        "--models",
        "GRADATE",
        "--epochs",
        "400",
        "--test-rounds",
        str(args.test_rounds),
        "--embedding-dim",
        "64",
        "--lr",
        "0.001",
        "--alpha",
        "0.1",
        "--beta",
        "0.1",
        "--negsamp-patch",
        "6",
        "--negsamp-context",
        "1",
        "--device",
        args.device,
    ])
    result = load_json(result_root(args.dataset) / "gradate.json")
    rows.append(metric_row(result, {
        "setting_source": "gradate_code_default",
        "epochs": 400,
        "embedding_dim": 64,
        "lr": 0.001,
        "alpha": 0.1,
        "beta": 0.1,
        "test_rounds": args.test_rounds,
        "negsamp_patch": 6,
        "negsamp_context": 1,
    }))

    write_csv(out_dir / "trials.csv", rows)
    write_summary(out_dir / "summary.json", rows)
    complete.write_text("complete\n")
    print(f"Wrote {out_dir / 'trials.csv'}")
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {complete}")


if __name__ == "__main__":
    main()
