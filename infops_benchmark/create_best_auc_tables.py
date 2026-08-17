import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
OUT = RESULTS / "best_auc_tables"
DATASETS = ["russia", "venezuela", "iran", "china", "cuba", "UAE"]


COLUMNS = [
    "dataset",
    "model",
    "auc",
    "macro_f1",
    "threshold",
    "num_nodes",
    "num_anomalies",
    "runtime_seconds",
    "model_size_mb",
    "num_parameters",
    "num_trainable_parameters",
    "epochs",
    "hid_dim",
    "embedding_dim",
    "lr",
    "alpha",
    "beta",
    "batch_size",
    "num_neigh",
    "test_rounds",
    "negsamp_patch",
    "negsamp_context",
    "setting_source",
]


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


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


def as_float(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def clean_row(row):
    return {key: ("" if value is None else value) for key, value in row.items()}


def trial_rows(dataset, folder):
    path = RESULTS / dataset / folder / "trials.csv"
    if not path.exists():
        return []
    rows = []
    for row in read_csv(path):
        row["dataset"] = dataset
        rows.append(clean_row(row))
    return rows


def iohunter_rows(dataset):
    rows = []
    for path in sorted((RESULTS / dataset).glob("iohunter_*.json")):
        if path.name == "iohunter_gnn_summary.json":
            continue
        row = flatten(read_json(path))
        row["dataset"] = dataset
        rows.append(clean_row(row))
    return rows


def all_rows_for_dataset(dataset):
    rows = []
    rows.extend(trial_rows(dataset, "hyperparam_search_non_iohunter"))
    rows.extend(trial_rows(dataset, "author_recommended_non_iohunter"))
    rows.extend(iohunter_rows(dataset))
    return rows


def better(candidate, incumbent):
    if incumbent is None:
        return True
    auc_candidate = as_float(candidate, "auc")
    auc_incumbent = as_float(incumbent, "auc")
    if auc_candidate != auc_incumbent:
        return auc_candidate > auc_incumbent
    return as_float(candidate, "macro_f1") > as_float(incumbent, "macro_f1")


def best_by_model(rows):
    best = {}
    for row in rows:
        model = row.get("model")
        if not model:
            continue
        key = (row.get("dataset"), model)
        if better(row, best.get(key)):
            best[key] = row
    return [best[key] for key in sorted(best)]


def ordered_columns(rows):
    present = {key for row in rows for key in row}
    cols = [key for key in COLUMNS if key in present]
    cols.extend(sorted(present.difference(cols)))
    return cols


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ordered_columns(rows)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def main():
    all_best = []
    for dataset in DATASETS:
        rows = best_by_model(all_rows_for_dataset(dataset))
        write_csv(OUT / f"{dataset}_best_auc.csv", rows)
        all_best.extend(rows)
    write_csv(OUT / "all_best_auc.csv", all_best)


if __name__ == "__main__":
    main()
