import argparse
import gc
import math
import os
import pickle
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from pygod.detector import DOMINANT
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.data import Data

from experiment_utils import train_validation_test_masks


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
HP_FILE = ROOT / "artifacts/results/benchmark/all_models_best_auc_by_dataset.csv"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
GPU = int(DEVICE.split(":")[1]) if DEVICE.startswith("cuda") else -1
# `batch_size=0` and `num_neigh=-1` train on the whole graph and all
# neighbours at once. That makes DOMINANT's adjacency reconstruction grow
# quadratically with the graph size and easily exhausts GPU memory.
BATCH_SIZE = int(os.environ.get("DOMINANT_BATCH_SIZE", "512"))
NUM_NEIGHBORS = [
    int(value)
    for value in os.environ.get("DOMINANT_NUM_NEIGHBORS", "10,5,5,5").split(",")
]

if BATCH_SIZE <= 0:
    raise ValueError("DOMINANT_BATCH_SIZE must be greater than zero")
if len(NUM_NEIGHBORS) != 4 or any(value <= 0 for value in NUM_NEIGHBORS):
    raise ValueError(
        "DOMINANT_NUM_NEIGHBORS must contain four positive integers, e.g. 10,5,5,5"
    )


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


def parse_args():
    parser = argparse.ArgumentParser(description="DOMINANT node anomaly experiment")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--save-first-model", action="store_true")
    parser.add_argument("--log-file", type=Path, default=ROOT / "artifacts/logs/dominant.log")
    return parser.parse_args()


def main(args):
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    hp_df = pd.read_csv(HP_FILE)
    hp_df = hp_df[hp_df["model"] == "pygod_dominant"]
    datasets = args.datasets or sorted(path.name for path in DATA_ROOT.iterdir() if path.is_dir())
    with args.log_file.open("w") as log_file:
        def log(message):
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        log(f"device={DEVICE}; train={args.train_fraction}; validation={args.validation_fraction}; test={args.test_fraction}")
        for dataset in datasets:
            _, data = load_data(dataset)
            labels = data.y.numpy()
            hp = hp_df[hp_df["dataset"] == dataset].iloc[0]
            for seed in args.seeds:
                set_seed(seed)
                model = DOMINANT(
                    hid_dim=int(hp["hid_dim"]),
                    num_layers=4,
                    epoch=int(hp["epochs"]),
                    lr=float(hp["lr"]),
                    batch_size=BATCH_SIZE,
                    num_neigh=NUM_NEIGHBORS,
                    gpu=GPU,
                )

                start = time.perf_counter()
                model.fit(data)
                training_seconds = time.perf_counter() - start

                raw_scores = model.decision_score_.detach().cpu().numpy()
                score_min = float(raw_scores.min())
                score_max = float(raw_scores.max())
                scores = (raw_scores - score_min) / (score_max - score_min + 1e-12)
                train_mask, validation_mask, test_mask = train_validation_test_masks(
                    labels, args.train_fraction, args.validation_fraction, args.test_fraction, seed
                )
                threshold, train_macro_f1 = best_training_threshold(scores, labels, train_mask)
                validation_prediction = scores[validation_mask] > threshold
                test_prediction = scores[test_mask] > threshold
                log(
                    f"dataset={dataset} seed={seed} train_nodes={train_mask.sum()} "
                    f"validation_nodes={validation_mask.sum()} test_nodes={test_mask.sum()} "
                    f"threshold={threshold:.6f} train_macro_f1={train_macro_f1:.4f} "
                    f"validation_macro_f1={f1_score(labels[validation_mask], validation_prediction, average='macro', zero_division=0):.4f} "
                    f"validation_auc={roc_auc_score(labels[validation_mask], scores[validation_mask]):.4f} "
                    f"test_macro_f1={f1_score(labels[test_mask], test_prediction, average='macro', zero_division=0):.4f} "
                    f"test_auc={roc_auc_score(labels[test_mask], scores[test_mask]):.4f} "
                    f"training_seconds={training_seconds:.2f}"
                )
                if args.save_first_model and seed == args.seeds[0]:
                    model_path = ROOT / "artifacts/checkpoints/dominant" / dataset / "dominant.pt"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "dataset": dataset,
                        "seed": seed,
                        "state_dict": {
                            name: value.detach().cpu()
                            for name, value in model.model.state_dict().items()
                        },
                        "hyperparameters": {
                            "hid_dim": int(hp["hid_dim"]),
                            "num_layers": 4,
                            "epoch": int(hp["epochs"]),
                            "lr": float(hp["lr"]),
                            "batch_size": BATCH_SIZE,
                            "num_neigh": NUM_NEIGHBORS,
                        },
                        "feature_dim": int(data.x.shape[1]),
                        "threshold": threshold,
                    }, model_path)
                del model
                if DEVICE.startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()

if __name__ == "__main__":
    main(parse_args())
