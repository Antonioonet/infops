import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from common import load_dataset


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BEST_TABLE = ROOT / "benchmark_results" / "all_models_best_auc_by_dataset.csv"
SCORES_ROOT = ROOT / "benchmark_results" / "seeded_model_scores"
OUT = ROOT / "benchmark_results" / "seeded_model_score_thresholds"


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def safe_auc(labels, probs):
    labels = np.asarray(labels, dtype=np.int64)
    if np.unique(labels).shape[0] < 2:
        return ""
    return float(roc_auc_score(labels, probs))


def split_masks(dataset):
    payload, _graph, labels, _sbert = load_dataset(dataset, "graph")
    split = payload["splits"][0]
    return labels, np.asarray(split["train"], dtype=bool), np.asarray(split["val"], dtype=bool)


def probability_path(scores_root, dataset, model, seed):
    return Path(scores_root) / f"seed_{seed}" / dataset / model / "probabilities.npy"


def status_path(scores_root, dataset, model, seed):
    return Path(scores_root) / f"seed_{seed}" / dataset / model / "run.status.json"


def load_probabilities(scores_root, dataset, model, seeds):
    probs = {}
    statuses = {}
    for seed in seeds:
        status_file = status_path(scores_root, dataset, model, seed)
        if not status_file.exists():
            raise FileNotFoundError(status_file)
        status = json.load(open(status_file))
        statuses[seed] = status
        if status.get("status") != "ok":
            raise RuntimeError(f"{dataset}/{model}/seed_{seed} status={status.get('status')}: {status.get('error')}")
        path = probability_path(scores_root, dataset, model, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        probs[seed] = np.load(path).astype(np.float64)
    return probs, statuses


def candidate_thresholds(prob_by_seed, mask):
    values = np.concatenate([prob[mask] for prob in prob_by_seed.values()])
    unique = np.unique(values)
    return np.concatenate(
        (
            [np.nextafter(float(unique.min()), -np.inf)],
            unique.astype(np.float64),
            [np.nextafter(float(unique.max()), np.inf)],
        )
    )


def f1_curve_for_thresholds(labels, probs, thresholds):
    order = np.argsort(probs)
    sorted_probs = probs[order]
    sorted_labels = labels[order].astype(np.int64)
    cum_pos = np.concatenate(([0], np.cumsum(sorted_labels)))
    pos = int(sorted_labels.sum())
    n = sorted_labels.shape[0]
    neg = n - pos

    right = np.searchsorted(sorted_probs, thresholds, side="right")
    pred_pos = n - right
    tp = pos - cum_pos[right]
    fp = pred_pos - tp
    fn = pos - tp
    tn = neg - fp

    pred_neg = n - pred_pos
    f1_pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(2 * tp + fp + fn) > 0)
    f1_neg = np.divide(2 * tn, 2 * tn + fn + fp, out=np.zeros_like(tn, dtype=np.float64), where=(2 * tn + fn + fp) > 0)
    macro_f1 = 0.5 * (f1_pos + f1_neg)
    return macro_f1, f1_pos, pred_pos, pred_neg


def metrics_at(labels, probs, mask, threshold):
    y = labels[mask].astype(np.int64)
    p = probs[mask]
    pred = (p > threshold).astype(np.int64)
    return {
        "auc": safe_auc(y, p),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "binary_f1": float(f1_score(y, pred, average="binary", zero_division=0)),
        "num_nodes": int(y.shape[0]),
        "num_anomalies": int(y.sum()),
        "num_predicted_anomalies": int(pred.sum()),
    }


def best_threshold(labels, prob_by_seed, train_mask, seeds):
    thresholds = candidate_thresholds(prob_by_seed, train_mask)
    macro_curves = []
    binary_curves = []
    pred_curves = []
    y_train = labels[train_mask]
    for seed in seeds:
        macro, binary, pred_pos, _pred_neg = f1_curve_for_thresholds(
            y_train,
            prob_by_seed[seed][train_mask],
            thresholds,
        )
        macro_curves.append(macro)
        binary_curves.append(binary)
        pred_curves.append(pred_pos)

    mean_macro = np.mean(np.vstack(macro_curves), axis=0)
    mean_binary = np.mean(np.vstack(binary_curves), axis=0)
    mean_pred = np.mean(np.vstack(pred_curves), axis=0)
    best_value = float(mean_macro.max())
    best_indices = np.flatnonzero(mean_macro == best_value)
    if best_indices.shape[0] > 1:
        idx = int(best_indices[np.argmax(mean_binary[best_indices])])
    else:
        idx = int(best_indices[0])
    return {
        "threshold": float(thresholds[idx]),
        "train_mean_macro_f1": best_value,
        "train_mean_binary_f1": float(mean_binary[idx]),
        "train_mean_predicted_anomalies": float(mean_pred[idx]),
        "num_threshold_candidates": int(thresholds.shape[0]),
    }


def model_dataset_rows(include_datasets=None, include_models=None):
    rows = [row for row in read_csv(BEST_TABLE) if row.get("dataset") and row.get("model")]
    if include_datasets:
        keep = set(include_datasets)
        rows = [row for row in rows if row["dataset"] in keep]
    if include_models:
        keep = set(include_models)
        rows = [row for row in rows if row["model"] in keep]
    return sorted(rows, key=lambda row: (row["dataset"], row["model"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-root", default=str(SCORES_ROOT))
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--include-datasets", nargs="+")
    parser.add_argument("--include-models", nargs="+")
    args = parser.parse_args()

    scores_root = Path(args.scores_root)
    out_dir = Path(args.out_dir)
    detail_rows = []
    threshold_rows = []

    for row in model_dataset_rows(args.include_datasets, args.include_models):
        dataset = row["dataset"]
        model = row["model"]
        print(f"Calibrating {dataset}/{model}", flush=True)
        labels, train_mask, val_mask = split_masks(dataset)
        prob_by_seed, statuses = load_probabilities(scores_root, dataset, model, args.seeds)
        selected = best_threshold(labels, prob_by_seed, train_mask, args.seeds)
        threshold = selected["threshold"]

        threshold_rows.append({
            "dataset": dataset,
            "model": model,
            "threshold": threshold,
            "train_mean_macro_f1": selected["train_mean_macro_f1"],
            "train_mean_binary_f1": selected["train_mean_binary_f1"],
            "train_mean_predicted_anomalies": selected["train_mean_predicted_anomalies"],
            "num_threshold_candidates": selected["num_threshold_candidates"],
            "threshold_train_nodes": int(train_mask.sum()),
            "threshold_validation_nodes": int(val_mask.sum()),
            "threshold_train_anomalies": int(labels[train_mask].sum()),
            "threshold_validation_anomalies": int(labels[val_mask].sum()),
            "split_id": 0,
        })

        for seed in args.seeds:
            train = metrics_at(labels, prob_by_seed[seed], train_mask, threshold)
            val = metrics_at(labels, prob_by_seed[seed], val_mask, threshold)
            status = statuses[seed]
            detail_rows.append({
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "threshold": threshold,
                "train_macro_f1": train["macro_f1"],
                "train_binary_f1": train["binary_f1"],
                "train_auc": train["auc"],
                "validation_macro_f1": val["macro_f1"],
                "validation_binary_f1": val["binary_f1"],
                "validation_auc": val["auc"],
                "validation_predicted_anomalies": val["num_predicted_anomalies"],
                "validation_nodes": val["num_nodes"],
                "validation_anomalies": val["num_anomalies"],
                "model_size_mb": status.get("model_size_mb", ""),
                "runtime_seconds": status.get("runtime_seconds", ""),
                "checkpoint": status.get("checkpoint", ""),
                "node_scores_csv": status.get("node_scores_csv", ""),
            })

    write_csv(out_dir / "thresholds_by_dataset_model.csv", threshold_rows)
    write_csv(out_dir / "validation_scores_by_dataset_model_seed.csv", detail_rows)


if __name__ == "__main__":
    main()
