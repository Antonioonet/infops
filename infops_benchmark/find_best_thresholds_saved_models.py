import argparse
import csv
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ROOT = HERE.parent

from common import load_dataset, normalized_scores, node_features, sparse_adj_from_graph
from run_contrastive_russia import (
    augment_random_edge,
    dense_norm_adj,
    load_model_class,
    score_batches,
)


CONTRASTIVE_MODELS = {"anemone": "ANEMONE", "gradate": "GRADATE"}


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return value if math.isfinite(value) else ""


def load_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def checkpoint_path(save_root, dataset, model):
    return Path(save_root) / dataset / model / "model.pt"


def tensor_to_numpy(values):
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def namespace_from_checkpoint(ckpt, device):
    hparams = dict(ckpt.get("hyperparameters") or {})
    defaults = {
        "batch_size": 128,
        "subgraph_size": 4,
        "embedding_dim": 64,
        "negsamp_patch": 1,
        "negsamp_context": 1,
        "alpha": 0.5,
        "beta": 0.1,
        "test_rounds": 32,
        "seed": 12121995,
        "device": device,
    }
    defaults.update(hparams)
    defaults["device"] = device
    return SimpleNamespace(**defaults)


def contrastive_scores(dataset, model_name, ckpt, ckpt_path, device):
    cached = ckpt_path.with_name("decision_scores.npy")
    if cached.exists():
        return np.load(cached)

    args = namespace_from_checkpoint(ckpt, device)
    _, graph, _labels, sbert = load_dataset(dataset, "graph")
    x = node_features(graph, sbert).numpy().astype(np.float32)
    adj_sp = sparse_adj_from_graph(graph)
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    adj = dense_norm_adj(adj_sp).unsqueeze(0).to(dev)
    features = torch.from_numpy(x).unsqueeze(0).to(dev)

    model_class = CONTRASTIVE_MODELS[model_name]
    model_cls = load_model_class(model_class)
    model = model_cls(
        x.shape[1],
        int(args.embedding_dim),
        "prelu",
        int(args.negsamp_patch),
        int(args.negsamp_context),
        "avg",
    ).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    is_gradate = model_name == "gradate"
    adj_hat = None
    if is_gradate:
        adj_hat = dense_norm_adj(augment_random_edge(adj_sp, seed=int(args.seed))).unsqueeze(0).to(dev)

    scores = score_batches(
        model,
        adj,
        features,
        graph,
        args,
        dev,
        is_gradate,
        adj_hat,
        int(args.seed) + 10000,
    )
    scores = np.asarray(scores, dtype=np.float64)
    np.save(cached, scores)

    del model, adj, features, adj_hat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def load_scores(dataset, model_name, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if model_name in CONTRASTIVE_MODELS:
        return contrastive_scores(dataset, model_name, ckpt, ckpt_path, device)
    scores = ckpt.get("decision_score_")
    if scores is None:
        cached = ckpt_path.with_name("decision_scores.npy")
        if cached.exists():
            return np.load(cached)
        raise ValueError(f"{ckpt_path} has no decision_score_")
    return tensor_to_numpy(scores)


def safe_auc(labels, probs):
    labels = np.asarray(labels, dtype=np.int64)
    if np.unique(labels).shape[0] < 2:
        return ""
    return float(roc_auc_score(labels, probs))


def metrics_at(labels, probs, mask, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    y = labels[mask]
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


def best_threshold(labels, probs, mask, average):
    labels = np.asarray(labels, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    y = labels[mask]
    p = probs[mask]
    unique = np.unique(p)
    if unique.size == 0:
        return 0.5, 0.0
    thresholds = np.concatenate(
        (
            [np.nextafter(float(unique.min()), -np.inf)],
            unique.astype(np.float64),
            [np.nextafter(float(unique.max()), np.inf)],
        )
    )
    best_t = 0.5
    best_f1 = -1.0
    best_predicted = None
    for threshold in thresholds:
        pred = (p > threshold).astype(np.int64)
        score = float(f1_score(y, pred, average=average, zero_division=0))
        predicted = int(pred.sum())
        if score > best_f1 or (score == best_f1 and (best_predicted is None or predicted < best_predicted)):
            best_t = float(threshold)
            best_f1 = score
            best_predicted = predicted
    return best_t, best_f1


def dataset_splits(dataset):
    payload, _graph, labels, _sbert = load_dataset(dataset, "graph")
    splits = payload.get("splits")
    if not splits:
        raise ValueError(f"{dataset} has no predefined train/val/test splits")
    return labels, splits


def summarize(rows, metric_columns):
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["model"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (dataset, model), items in sorted(grouped.items()):
        out = {
            "dataset": dataset,
            "model": model,
            "num_splits": len(items),
        }
        for col in metric_columns:
            values = [finite_float(item.get(col)) for item in items]
            values = [value for value in values if value != ""]
            out[f"{col}_mean"] = float(np.mean(values)) if values else ""
            out[f"{col}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else "")
        summary.append(out)
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def root_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-csv", default="benchmark_results/all_models_best_auc_by_dataset.csv")
    parser.add_argument("--save-root", default="benchmark_results/saved_models_best_auc")
    parser.add_argument("--out-dir", default="benchmark_results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--datasets", nargs="*")
    args = parser.parse_args()

    best_csv = root_path(args.best_csv)
    save_root = root_path(args.save_root)
    out_dir = root_path(args.out_dir)

    best_rows = load_rows(best_csv)
    if args.models:
        keep_models = set(args.models)
        best_rows = [row for row in best_rows if row["model"] in keep_models]
    if args.datasets:
        keep_datasets = set(args.datasets)
        best_rows = [row for row in best_rows if row["dataset"] in keep_datasets]

    detailed = []
    started = time.perf_counter()
    for row in best_rows:
        dataset = row["dataset"]
        model = row["model"]
        ckpt = checkpoint_path(save_root, dataset, model)
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        print(f"Scoring {dataset}/{model}", flush=True)
        labels, splits = dataset_splits(dataset)
        raw_scores = load_scores(dataset, model, ckpt, args.device)
        if raw_scores.shape[0] != labels.shape[0]:
            raise ValueError(f"{dataset}/{model}: {raw_scores.shape[0]} scores for {labels.shape[0]} labels")
        probs = normalized_scores(raw_scores)
        full_auc = safe_auc(labels, probs)

        for split_id, split in sorted(splits.items(), key=lambda item: int(item[0])):
            train_mask = np.asarray(split["train"], dtype=bool)
            val_mask = np.asarray(split["val"], dtype=bool)
            test_mask = np.asarray(split["test"], dtype=bool)

            macro_t, val_best_macro = best_threshold(labels, probs, val_mask, "macro")
            binary_t, val_best_binary = best_threshold(labels, probs, val_mask, "binary")

            train_macro = metrics_at(labels, probs, train_mask, macro_t)
            val_macro = metrics_at(labels, probs, val_mask, macro_t)
            test_macro = metrics_at(labels, probs, test_mask, macro_t)
            train_binary = metrics_at(labels, probs, train_mask, binary_t)
            val_binary = metrics_at(labels, probs, val_mask, binary_t)
            test_binary = metrics_at(labels, probs, test_mask, binary_t)
            test_at_05 = metrics_at(labels, probs, test_mask, 0.5)

            detailed.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "split": int(split_id),
                    "full_auc": full_auc,
                    "macro_threshold_from_val": macro_t,
                    "val_macro_f1_at_macro_threshold": val_best_macro,
                    "train_macro_f1_at_macro_threshold": train_macro["macro_f1"],
                    "test_macro_f1_at_macro_threshold": test_macro["macro_f1"],
                    "test_binary_f1_at_macro_threshold": test_macro["binary_f1"],
                    "test_auc": test_macro["auc"],
                    "test_predicted_anomalies_at_macro_threshold": test_macro["num_predicted_anomalies"],
                    "binary_threshold_from_val": binary_t,
                    "val_binary_f1_at_binary_threshold": val_best_binary,
                    "train_binary_f1_at_binary_threshold": train_binary["binary_f1"],
                    "test_binary_f1_at_binary_threshold": test_binary["binary_f1"],
                    "test_macro_f1_at_binary_threshold": test_binary["macro_f1"],
                    "test_predicted_anomalies_at_binary_threshold": test_binary["num_predicted_anomalies"],
                    "test_macro_f1_at_0_5": test_at_05["macro_f1"],
                    "test_binary_f1_at_0_5": test_at_05["binary_f1"],
                    "train_nodes": train_macro["num_nodes"],
                    "val_nodes": val_macro["num_nodes"],
                    "test_nodes": test_macro["num_nodes"],
                    "train_anomalies": train_macro["num_anomalies"],
                    "val_anomalies": val_macro["num_anomalies"],
                    "test_anomalies": test_macro["num_anomalies"],
                    "score_min": float(raw_scores.min()),
                    "score_max": float(raw_scores.max()),
                    "checkpoint_path": str(ckpt),
                }
            )

    detailed_path = out_dir / "best_threshold_f1_by_dataset_model_split.csv"
    write_csv(detailed_path, detailed)
    summary_metrics = [
        "full_auc",
        "macro_threshold_from_val",
        "val_macro_f1_at_macro_threshold",
        "train_macro_f1_at_macro_threshold",
        "test_macro_f1_at_macro_threshold",
        "test_binary_f1_at_macro_threshold",
        "test_auc",
        "binary_threshold_from_val",
        "val_binary_f1_at_binary_threshold",
        "train_binary_f1_at_binary_threshold",
        "test_binary_f1_at_binary_threshold",
        "test_macro_f1_at_binary_threshold",
        "test_macro_f1_at_0_5",
        "test_binary_f1_at_0_5",
    ]
    summary = summarize(detailed, summary_metrics)
    summary_path = out_dir / "best_threshold_f1_by_dataset_model.csv"
    write_csv(summary_path, summary)
    print(f"Wrote {detailed_path} ({len(detailed)} rows)", flush=True)
    print(f"Wrote {summary_path} ({len(summary)} rows)", flush=True)
    print(f"Elapsed seconds: {time.perf_counter() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
