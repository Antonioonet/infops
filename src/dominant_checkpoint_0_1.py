import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from experiment_utils import train_validation_test_masks


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
CHECKPOINT_ROOT = (
    ROOT / "artifacts/results/benchmark/saved_models_best_auc"
)
DEFAULT_LOG_FILE = (
    ROOT / "artifacts/logs/dominant_checkpoint_0_1_2026-08-27.log"
)
DEFAULT_OUTPUT_CSV = (
    ROOT / "artifacts/results/low_data_exps/"
    "dominant_checkpoint_0_1_percent.csv"
)


def load_labels(dataset):
    """Load labels in the same node order used to create the checkpoint."""
    with open(DATA_ROOT / dataset / "0.7_datasets.pkl", "rb") as file:
        graph_data = pickle.load(file)
    return np.asarray(graph_data["labels"], dtype=np.int64)


def checkpoint_scores(dataset):
    """Load fixed anomaly scores from one already-trained DOMINANT checkpoint."""
    checkpoint_path = (
        CHECKPOINT_ROOT / dataset / "pygod_dominant" / "model.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint["dataset"] != dataset:
        raise ValueError(
            f"Checkpoint dataset mismatch: expected {dataset}, "
            f"found {checkpoint['dataset']}"
        )
    return (
        torch.as_tensor(checkpoint["decision_score_"])
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )


def best_training_threshold(scores, labels, train_mask):
    unique_scores = np.unique(scores[train_mask])
    thresholds = np.concatenate([
        [np.nextafter(unique_scores.min(), -np.inf)],
        unique_scores,
        [np.nextafter(unique_scores.max(), np.inf)],
    ])

    best_threshold = 0.5
    best_macro_f1 = -1
    best_predicted = None
    for threshold in thresholds:
        prediction = scores[train_mask] > threshold
        macro_f1 = f1_score(
            labels[train_mask],
            prediction,
            average="macro",
            zero_division=0,
        )
        predicted = int(prediction.sum())
        if macro_f1 > best_macro_f1 or (
            macro_f1 == best_macro_f1 and predicted < best_predicted
        ):
            best_threshold = float(threshold)
            best_macro_f1 = float(macro_f1)
            best_predicted = predicted
    return best_threshold, best_macro_f1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed DOMINANT checkpoint scores with 0.1% threshold "
            "calibration and five seeded evaluation splits."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--train-fraction", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.499)
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main(args):
    if not np.isclose(
        args.train_fraction + args.validation_fraction + args.test_fraction,
        1.0,
    ):
        raise ValueError("train, validation, and test fractions must sum to 1")

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or sorted(
        path.name for path in DATA_ROOT.iterdir() if path.is_dir()
    )
    fields = [
        "dataset",
        "seed",
        "split",
        "threshold",
        "train_macro_f1",
        "validation_macro_f1",
        "validation_auc",
        "test_macro_f1",
        "test_auc",
        "training_seconds",
        "status",
    ]

    with args.log_file.open("w") as log_file, args.output_csv.open(
        "w", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()

        def log(message):
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        log(
            "score_source=checkpoint.decision_score_; "
            f"train={args.train_fraction}; "
            f"validation={args.validation_fraction}; "
            f"test={args.test_fraction}; seeds={args.seeds}"
        )
        for dataset in datasets:
            labels = load_labels(dataset)
            raw_scores = checkpoint_scores(dataset)
            if len(raw_scores) != len(labels):
                raise ValueError(
                    f"{dataset}: checkpoint has {len(raw_scores)} scores but "
                    f"the dataset has {len(labels)} labels"
                )

            score_min = float(raw_scores.min())
            score_max = float(raw_scores.max())
            scores = (raw_scores - score_min) / (
                score_max - score_min + 1e-12
            )
            for seed in args.seeds:
                train_mask, validation_mask, test_mask = (
                    train_validation_test_masks(
                        labels,
                        args.train_fraction,
                        args.validation_fraction,
                        args.test_fraction,
                        seed,
                    )
                )
                threshold, train_macro_f1 = best_training_threshold(
                    scores,
                    labels,
                    train_mask,
                )
                validation_prediction = scores[validation_mask] > threshold
                test_prediction = scores[test_mask] > threshold
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "split": 0,
                    "threshold": threshold,
                    "train_macro_f1": train_macro_f1,
                    "validation_macro_f1": f1_score(
                        labels[validation_mask],
                        validation_prediction,
                        average="macro",
                        zero_division=0,
                    ),
                    "validation_auc": roc_auc_score(
                        labels[validation_mask],
                        scores[validation_mask],
                    ),
                    "test_macro_f1": f1_score(
                        labels[test_mask],
                        test_prediction,
                        average="macro",
                        zero_division=0,
                    ),
                    "test_auc": roc_auc_score(
                        labels[test_mask],
                        scores[test_mask],
                    ),
                    "training_seconds": 0.0,
                    "status": "ok",
                }
                writer.writerow(row)
                csv_file.flush()
                log(
                    f"dataset={dataset} seed={seed} "
                    f"train_nodes={train_mask.sum()} "
                    f"validation_nodes={validation_mask.sum()} "
                    f"test_nodes={test_mask.sum()} "
                    f"threshold={threshold:.6f} "
                    f"validation_macro_f1={row['validation_macro_f1']:.4f} "
                    f"validation_auc={row['validation_auc']:.4f} "
                    f"test_macro_f1={row['test_macro_f1']:.4f} "
                    f"test_auc={row['test_auc']:.4f}"
                )


if __name__ == "__main__":
    main(parse_args())
