import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "benchmark_results" / "repeated_seed_best_auc"
OUT = ROOT / "benchmark_results" / "repeated_seed_best_auc_tables"


BASE_COLUMNS = [
    "dataset",
    "model",
    "seed",
    "status",
    "auc",
    "macro_f1",
    "threshold",
    "num_nodes",
    "num_anomalies",
    "runtime_seconds",
    "repeated_seed_runtime_seconds",
    "model_size_mb",
    "num_parameters",
    "epochs",
    "hid_dim",
    "hidden_dim",
    "embedding_dim",
    "lr",
    "alpha",
    "beta",
    "batch_size",
    "num_neigh",
    "test_rounds",
    "negsamp_patch",
    "negsamp_context",
    "best_config_auc",
    "best_config_macro_f1",
    "error",
    "result_json",
]


def flatten(value, prefix=""):
    flat = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten(child, child_key))
    elif isinstance(value, list):
        flat[prefix] = json.dumps(value, sort_keys=True)
    else:
        flat[prefix] = value
    return flat


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = {key for row in rows for key in row}
    columns = [key for key in BASE_COLUMNS if key in keys]
    columns.extend(sorted(keys.difference(columns)))
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def write_compact_csv(path, rows):
    compact = []
    for row in rows:
        compact.append({
            "model": row.get("model", ""),
            "seed": row.get("seed", ""),
            "dataset": row.get("dataset", ""),
            "auc": row.get("auc", ""),
            "macro_f1": row.get("macro_f1", ""),
            "model_size": row.get("model_size_mb", ""),
            "time": row.get("runtime_seconds", ""),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "seed", "dataset", "auc", "macro_f1", "model_size", "time"],
        )
        writer.writeheader()
        writer.writerows(compact)
    print(f"Wrote {path} ({len(compact)} rows)")


def as_float(value):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def std(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def main():
    rows = []
    for path in sorted(IN.glob("*/*/seed_*.json")):
        with open(path) as fh:
            payload = json.load(fh)
        row = flatten(payload)
        row["result_json"] = str(path.relative_to(ROOT / "benchmark_results"))
        rows.append({key: ("" if value is None else value) for key, value in row.items()})

    write_csv(OUT / "all_seed_runs.csv", rows)
    write_compact_csv(OUT / "all_seed_runs_compact.csv", rows)

    by_pair = defaultdict(list)
    for row in rows:
        by_pair[(row.get("dataset"), row.get("model"))].append(row)

    summary = []
    for (dataset, model), group in sorted(by_pair.items()):
        ok = [row for row in group if row.get("status") == "ok"]
        aucs = [as_float(row.get("auc")) for row in ok]
        f1s = [as_float(row.get("macro_f1")) for row in ok]
        runtimes = [as_float(row.get("runtime_seconds")) for row in ok]
        summary.append({
            "dataset": dataset,
            "model": model,
            "num_runs": len(group),
            "num_ok": len(ok),
            "num_failed": len(group) - len(ok),
            "auc_mean": mean(aucs),
            "auc_std": std(aucs),
            "macro_f1_mean": mean(f1s),
            "macro_f1_std": std(f1s),
            "runtime_seconds_mean": mean(runtimes),
            "best_config_auc": ok[0].get("best_config_auc", "") if ok else group[0].get("best_config_auc", ""),
            "best_config_macro_f1": ok[0].get("best_config_macro_f1", "") if ok else group[0].get("best_config_macro_f1", ""),
        })
    write_csv(OUT / "seed_summary_by_dataset_model.csv", summary)


if __name__ == "__main__":
    main()
