import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def summary_paths(dataset):
    return sorted((ROOT / "benchmark_results" / dataset).glob("optuna_gadnr_auc*/summary.json"))


def read_summaries(dataset):
    summaries = []
    for path in summary_paths(dataset):
        with open(path) as fh:
            summary = json.load(fh)
        summary["_summary_path"] = str(path)
        summaries.append(summary)
    return summaries


def flatten_best(dataset, summary):
    best = summary.get("best_by_auc") or summary.get("best_by_objective") or {}
    row = {
        "dataset": dataset,
        "num_trials": summary.get("num_trials"),
        "num_complete_trials": summary.get("num_complete_trials"),
        "num_failed_trials": summary.get("num_failed_trials"),
        "summary_path": summary.get("_summary_path", ""),
    }
    row.update(best)
    return row


def best_summary(summaries):
    best = None
    best_auc = float("-inf")
    for summary in summaries:
        trial = summary.get("best_by_auc") or summary.get("best_by_objective") or {}
        auc = trial.get("auc")
        if auc is None:
            continue
        auc = float(auc)
        if auc > best_auc:
            best_auc = auc
            best = summary
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["russia", "cuba", "iran", "china", "UAE", "venezuela"],
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "benchmark_results" / "gadnr_best_auc_by_dataset.csv"),
    )
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        summary = best_summary(read_summaries(dataset))
        if summary is not None:
            rows.append(flatten_best(dataset, summary))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_path.write_text("")
    print(f"Wrote {out_path} with {len(rows)} rows")


if __name__ == "__main__":
    main()
