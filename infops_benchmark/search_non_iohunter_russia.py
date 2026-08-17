import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def result_root(dataset_name):
    return HERE.parent / "benchmark_results" / dataset_name


def search_root(dataset_name):
    return result_root(dataset_name) / "hyperparam_search_non_iohunter"


def run_cmd(cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)


def load_result(path):
    with open(path) as fh:
        return json.load(fh)


def write_trial_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_pygod_search(args):
    rows = []
    gpu = args.device.split(":")[-1] if args.device.startswith("cuda:") else ("0" if args.device == "cuda" else "-1")
    for model, epochs, hid_dim, lr in itertools.product(
        ["dominant", "cola"],
        args.pygod_epochs,
        args.pygod_hid_dims,
        args.pygod_lrs,
    ):
        cmd = [
            sys.executable,
            "run_pygod_russia.py",
            "--dataset",
            args.dataset,
            "--models",
            model,
            "--epochs",
            str(epochs),
            "--hid-dim",
            str(hid_dim),
            "--lr",
            str(lr),
            "--batch-size",
            str(args.pygod_batch_size),
            "--num-neigh",
            str(args.pygod_num_neigh),
            "--gpu",
            gpu,
        ]
        run_cmd(cmd)
        result_name = f"pygod_{model}"
        result = load_result(result_root(args.dataset) / f"{result_name}.json")
        row = {
            "model": result_name,
            "epochs": epochs,
            "hid_dim": hid_dim,
            "lr": lr,
            "batch_size": result.get("batch_size"),
            "num_neigh": result.get("num_neigh"),
            "auc": result["auc"],
            "macro_f1": result["macro_f1"],
            "threshold": result["threshold"],
            "runtime_seconds": result.get("runtime_seconds"),
            "model_size_mb": result.get("model_size_mb"),
            "num_parameters": result.get("num_parameters"),
        }
        rows.append(row)
    return rows


def run_contrastive_search(args):
    rows = []
    for model, epochs, embedding_dim, lr, alpha in itertools.product(
        ["ANEMONE", "GRADATE"],
        args.contrastive_epochs,
        args.contrastive_embedding_dims,
        args.contrastive_lrs,
        args.contrastive_alphas,
    ):
        cmd = [
            sys.executable,
            "run_contrastive_russia.py",
            "--dataset",
            args.dataset,
            "--models",
            model,
            "--epochs",
            str(epochs),
            "--test-rounds",
            str(args.contrastive_test_rounds),
            "--embedding-dim",
            str(embedding_dim),
            "--lr",
            str(lr),
            "--alpha",
            str(alpha),
            "--device",
            args.device,
        ]
        run_cmd(cmd)
        result_name = model.lower()
        result = load_result(result_root(args.dataset) / f"{result_name}.json")
        row = {
            "model": result_name,
            "epochs": epochs,
            "embedding_dim": embedding_dim,
            "lr": lr,
            "alpha": alpha,
            "auc": result["auc"],
            "macro_f1": result["macro_f1"],
            "threshold": result["threshold"],
            "best_loss": result.get("best_loss"),
            "runtime_seconds": result.get("runtime_seconds"),
            "model_size_mb": result.get("model_size_mb"),
            "num_parameters": result.get("num_parameters"),
        }
        rows.append(row)
    return rows


def best_rows(rows):
    best_auc = {}
    best_f1 = {}
    for row in rows:
        model = row["model"]
        if model not in best_auc or row["auc"] > best_auc[model]["auc"]:
            best_auc[model] = row
        if model not in best_f1 or row["macro_f1"] > best_f1[model]["macro_f1"]:
            best_f1[model] = row
    return best_auc, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default="russia")
    parser.add_argument("--pygod-epochs", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--pygod-hid-dims", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--pygod-lrs", nargs="+", type=float, default=[0.001, 0.004])
    parser.add_argument("--pygod-batch-size", type=int, default=1024)
    parser.add_argument("--pygod-num-neigh", type=int, default=10)
    parser.add_argument("--contrastive-epochs", nargs="+", type=int, default=[30, 50])
    parser.add_argument("--contrastive-embedding-dims", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--contrastive-lrs", nargs="+", type=float, default=[0.001])
    parser.add_argument("--contrastive-alphas", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--contrastive-test-rounds", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = search_root(args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = out_dir / ".complete"
    if complete_marker.exists() and not args.force:
        print(f"{args.dataset} search already complete at {out_dir}; use --force to rerun.")
        return
    rows = []
    rows.extend(run_pygod_search(args))
    rows.extend(run_contrastive_search(args))

    write_trial_csv(out_dir / "trials.csv", rows)
    best_auc, best_f1 = best_rows(rows)
    summary = {
        "best_by_auc": best_auc,
        "best_by_macro_f1": best_f1,
        "num_trials": len(rows),
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {out_dir / 'trials.csv'}")
    print(f"Wrote {out_dir / 'summary.json'}")
    complete_marker.write_text("complete\n")
    print(f"Wrote {complete_marker}")


if __name__ == "__main__":
    main()
