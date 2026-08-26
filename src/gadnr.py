import argparse
import csv
import gc
import math
import pickle
import random
import time
from datetime import date
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from pygod.detector import GADNR
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import GCN, GIN, GraphSAGE

from experiment_utils import train_validation_test_masks


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
HP_FILE = ROOT / "artifacts/results/benchmark/all_models_best_auc_by_dataset.csv"
OUTPUT_ROOT = ROOT / "artifacts/results" / f"gadnr_refactoring_{date.today().isoformat()}"
SEEDS = [1, 2, 3, 4, 5]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
GPU = int(DEVICE.split(":")[1]) if DEVICE.startswith("cuda") else -1


def pyg_backbone(backbone):
    def build_backbone(**kwargs):
        kwargs.pop("tot_nodes", None)
        return backbone(**kwargs)

    return build_backbone


BACKBONES = {
    "GCN": pyg_backbone(GCN),
    "GraphSAGE": pyg_backbone(GraphSAGE),
    "GIN": pyg_backbone(GIN),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def load_data(dataset):
    dataset_root = DATA_ROOT / dataset

    with open(dataset_root / "0.7_datasets.pkl", "rb") as file:
        graph_data = pickle.load(file)

    embeddings = torch.load(
        dataset_root / "sbert_nodeattributes_mostPop5.pt",
        map_location="cpu",
    ).float()

    graph = nx.convert_node_labels_to_integers(
        graph_data["graph"],
        ordering="sorted",
    )

    degrees = np.asarray(
        [graph.degree(node) for node in range(graph.number_of_nodes())]
    )
    degree_bins = np.asarray(
        [0 if degree <= 1 else math.ceil(math.log2(degree)) for degree in degrees]
    )
    degree_features = torch.nn.functional.one_hot(
        torch.tensor(degree_bins),
        num_classes=int(degree_bins.max()) + 1,
    ).float()

    features = torch.cat([embeddings, degree_features], dim=1)

    edges = []
    for source, target in graph.edges():
        edges.append((source, target))
        if source != target:
            edges.append((target, source))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    labels = torch.tensor(graph_data["labels"], dtype=torch.long)

    data = Data(x=features, edge_index=edge_index, y=labels)
    return graph_data, data


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


def main(args=None):
    if args is None:
        args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_root = OUTPUT_ROOT / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)

    hp_df = pd.read_csv(HP_FILE)
    hp_df = hp_df[hp_df["model"] == "gadnr"]

    datasets = sorted(path.name for path in DATA_ROOT.iterdir() if path.is_dir())

    columns = [
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

    with open(OUTPUT_ROOT / "results.csv", "w", newline="") as csv_file, open(
        OUTPUT_ROOT / "run.log", "w"
    ) as log_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()

        def log(message):
            print(message)
            log_file.write(message + "\n")
            log_file.flush()

        log(f"Device: {DEVICE}")

        for dataset in datasets:
            log(f"Starting {dataset}")

            graph_data, data = load_data(dataset)
            labels = data.y.numpy()
            hp = hp_df[hp_df["dataset"] == dataset].iloc[0]
            if DEVICE.startswith("cuda"):
                data = data.to(DEVICE)

            best_seed_score = -1
            best_checkpoint = None

            for seed in SEEDS:
                set_seed(seed)

                model = GADNR(
                    hid_dim=int(hp["hid_dim"]),
                    num_layers=1,
                    deg_dec_layers=4,
                    fea_dec_layers=3,
                    backbone=BACKBONES[hp["backbone"]],
                    sample_size=2,
                    sample_time=3,
                    neigh_loss="KL",
                    lambda_loss1=float(hp["lambda_n"]),
                    lambda_loss2=float(hp["lambda_x"]),
                    lambda_loss3=float(hp["lambda_d"]),
                    real_loss=False,
                    epoch=int(hp["epochs"]),
                    lr=0.01,
                    dropout=0.0,
                    weight_decay=0.0003,
                    batch_size=int(hp["batch_size"]),
                    num_neigh=int(hp["num_neigh"]),
                    gpu=GPU,
                )

                start = time.perf_counter()
                model.fit(
                    data,
                    label=data.y,
                    h_loss_weight=float(hp["lambda_n_prime"]),
                    degree_loss_weight=float(hp["lambda_d_prime"]),
                    feature_loss_weight=float(hp["lambda_x_prime"]),
                )
                training_seconds = time.perf_counter() - start

                raw_scores = model.decision_score_
                if torch.is_tensor(raw_scores):
                    raw_scores = raw_scores.detach().cpu().numpy()
                raw_scores = np.asarray(raw_scores)
                score_min = float(raw_scores.min())
                score_max = float(raw_scores.max())
                scores = (raw_scores - score_min) / (score_max - score_min + 1e-12)

                validation_scores = []
                thresholds = []

                for split_id in range(1):
                    train_mask, val_mask, test_mask = train_validation_test_masks(
                        labels, args.train_fraction, args.validation_fraction,
                        args.test_fraction, seed,
                    )

                    threshold, train_macro_f1 = best_training_threshold(
                        scores,
                        labels,
                        train_mask,
                    )

                    val_prediction = scores[val_mask] > threshold
                    validation_macro_f1 = f1_score(
                        labels[val_mask],
                        val_prediction,
                        average="macro",
                        zero_division=0,
                    )
                    validation_auc = roc_auc_score(
                        labels[val_mask],
                        scores[val_mask],
                    )
                    test_prediction = scores[test_mask] > threshold
                    test_macro_f1 = f1_score(labels[test_mask], test_prediction, average="macro", zero_division=0)
                    test_auc = roc_auc_score(labels[test_mask], scores[test_mask])

                    writer.writerow({
                        "dataset": dataset,
                        "seed": seed,
                        "split": split_id,
                        "threshold": threshold,
                        "train_macro_f1": train_macro_f1,
                        "validation_macro_f1": validation_macro_f1,
                        "validation_auc": validation_auc,
                        "test_macro_f1": test_macro_f1,
                        "test_auc": test_auc,
                        "training_seconds": training_seconds,
                        "status": "ok",
                    })
                    csv_file.flush()

                    validation_scores.append(validation_macro_f1)
                    thresholds.append(threshold)

                    log(
                        f"{dataset} seed={seed} split={split_id} "
                        f"threshold={threshold:.4f} "
                        f"validation_macro_f1={validation_macro_f1:.4f}"
                    )

                seed_score = float(np.mean(validation_scores))

                if seed_score > best_seed_score:
                    best_seed_score = seed_score
                    best_checkpoint = {
                        "dataset": dataset,
                        "seed": seed,
                        "device": DEVICE,
                        "state_dict": {
                            name: value.detach().cpu()
                            for name, value in model.model.state_dict().items()
                        },
                        "hyperparameters": {
                            "hid_dim": int(hp["hid_dim"]),
                            "num_layers": 1,
                            "deg_dec_layers": 4,
                            "fea_dec_layers": 3,
                            "backbone": hp["backbone"],
                            "sample_size": 2,
                            "sample_time": 3,
                            "neigh_loss": "KL",
                            "lambda_n": float(hp["lambda_n"]),
                            "lambda_x": float(hp["lambda_x"]),
                            "lambda_d": float(hp["lambda_d"]),
                            "lambda_n_prime": float(hp["lambda_n_prime"]),
                            "lambda_x_prime": float(hp["lambda_x_prime"]),
                            "lambda_d_prime": float(hp["lambda_d_prime"]),
                            "real_loss": False,
                            "epoch": int(hp["epochs"]),
                            "lr": 0.01,
                            "dropout": 0.0,
                            "weight_decay": 0.0003,
                            "batch_size": int(hp["batch_size"]),
                            "num_neigh": int(hp["num_neigh"]),
                        },
                        "feature_dim": int(data.x.shape[1]),
                        "thresholds_by_split": thresholds,
                        "default_threshold": float(np.median(thresholds)),
                        "score_min": score_min,
                        "score_max": score_max,
                        "validation_macro_f1_mean": seed_score,
                    }

                # The checkpoint above contains CPU tensors only. Releasing
                # the model between seeds prevents cached CUDA allocations
                # from accumulating over a long benchmark run.
                del model
                if DEVICE.startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()

            checkpoint_dir = checkpoint_root / dataset
            checkpoint_dir.mkdir(exist_ok=True)
            if args.save_first_model:
                torch.save(best_checkpoint, checkpoint_dir / "gadnr.pt")

            log(
                f"Completed {dataset}; best_seed={best_checkpoint['seed']} "
                f"validation_macro_f1_mean={best_seed_score:.4f}"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--save-first-model", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
