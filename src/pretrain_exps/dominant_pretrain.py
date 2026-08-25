import argparse
import gc
import math
import pickle
import random
import time
from datetime import date
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from pygod.detector import DOMINANT
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT / "data"
DATASETS = ("UAE", "china", "cuba", "iran", "russia", "venezuela")
CALIBRATION_FRACTION = 0.001
VALIDATION_FRACTION = 0.499
DEGREE_FEATURE_DIM = 15
HID_DIM = 64
NUM_LAYERS = 4
LEARNING_RATE = 0.004
BATCH_SIZE = 1024
NUM_NEIGHBORS = 10
PRETRAIN_EPOCHS = 100
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
GPU = int(DEVICE.split(":")[1]) if DEVICE.startswith("cuda") else -1


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
    degree_bins = torch.tensor([
        0 if graph.degree(node) <= 1 else math.ceil(math.log2(graph.degree(node)))
        for node in range(graph.number_of_nodes())
    ])
    degree_bins = degree_bins.clamp(max=DEGREE_FEATURE_DIM - 1)
    degree_features = torch.nn.functional.one_hot(
        degree_bins,
        num_classes=DEGREE_FEATURE_DIM,
    ).float()
    features = torch.cat([embeddings, degree_features], dim=1)

    edges = []
    for source, target in graph.edges():
        edges.append((source, target))
        if source != target:
            edges.append((target, source))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    labels = torch.tensor(graph_data["labels"], dtype=torch.long)
    return Data(x=features, edge_index=edge_index, y=labels)


def best_training_threshold(scores, labels, calibration_mask):
    calibration_scores = scores[calibration_mask]
    thresholds = np.concatenate([
        [np.nextafter(calibration_scores.min(), -np.inf)],
        np.unique(calibration_scores),
        [np.nextafter(calibration_scores.max(), np.inf)],
    ])
    best_threshold = 0.5
    best_macro_f1 = -1.0
    best_predicted = None
    for threshold in thresholds:
        prediction = calibration_scores > threshold
        macro_f1 = f1_score(
            labels[calibration_mask],
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


def target_split(labels, seed):
    indices = np.arange(len(labels))
    calibration_size = max(
        len(np.unique(labels)),
        math.ceil(CALIBRATION_FRACTION * len(labels)),
    )
    calibration, remaining = train_test_split(
        indices,
        train_size=calibration_size,
        random_state=seed,
        stratify=labels,
    )
    validation_fraction_of_remaining = VALIDATION_FRACTION / (
        1 - CALIBRATION_FRACTION
    )
    validation, unused = train_test_split(
        remaining,
        train_size=validation_fraction_of_remaining,
        random_state=seed + 10000,
        stratify=labels[remaining],
    )
    calibration_mask = np.zeros(len(labels), dtype=bool)
    validation_mask = np.zeros(len(labels), dtype=bool)
    calibration_mask[calibration] = True
    validation_mask[validation] = True
    return calibration_mask, validation_mask, len(unused)


def initialize_model():
    return DOMINANT(
        hid_dim=HID_DIM,
        num_layers=NUM_LAYERS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_neigh=NUM_NEIGHBORS,
        gpu=GPU,
        verbose=0,
    )


def pretrain_on_graph(model, optimizer, data, epochs):
    model.process_graph(data)
    if model.model is None:
        model.num_nodes, model.in_dim = data.x.shape
        model.model = model.init_model(**model.kwargs)
    elif data.x.shape[1] != model.in_dim:
        raise ValueError("All pretraining graphs must have the same feature dimension")

    loader = NeighborLoader(
        data,
        model.num_neigh,
        batch_size=model.batch_size,
        shuffle=True,
    )
    model.model.train()
    for _ in range(epochs):
        for sampled_data in loader:
            loss, _ = model.forward_model(sampled_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def pretrain_model(source_data, seed, pretrain_epochs):
    model = initialize_model()
    optimizer = None
    for epoch_id in range(pretrain_epochs):
        source_order = list(source_data)
        random.Random(seed + epoch_id).shuffle(source_order)
        for dataset, data in source_order:
            if optimizer is None:
                pretrain_on_graph(model, None, data, 0)
                optimizer = torch.optim.Adam(
                    model.model.parameters(),
                    lr=model.lr,
                    weight_decay=model.weight_decay,
                )
            pretrain_on_graph(model, optimizer, data, 1)
    return model


def normalized_scores(model, data):
    raw_scores = model.decision_function(data).detach().cpu().numpy()
    score_min = float(raw_scores.min())
    score_max = float(raw_scores.max())
    return (raw_scores - score_min) / (score_max - score_min + 1e-12)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single shared DOMINANT pretraining experiment."
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--training-mode",
        choices=("shared", "per-dataset"),
        default="shared",
        help="Train once on all graphs or train a fresh model on each target graph.",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=PRETRAIN_EPOCHS,
        help="Epochs over all six graphs during the one shared pretraining run.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=ROOT / "artifacts/logs" / f"dominant_pretrain_{date.today().isoformat()}.log",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    all_data = {dataset: load_data(dataset) for dataset in DATASETS}
    log_file = args.log_file.open("w")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    log(
        f"device={DEVICE}; calibration_fraction={CALIBRATION_FRACTION}; "
        f"validation_fraction={VALIDATION_FRACTION}; "
        f"degree_feature_dim={DEGREE_FEATURE_DIM}; seed={args.seed}; "
        f"pretrain_epochs={args.pretrain_epochs}; training_mode={args.training_mode}"
    )
    try:
        set_seed(args.seed)
        shared_model = None
        shared_training_seconds = None
        if args.training_mode == "shared":
            start = time.perf_counter()
            shared_model = pretrain_model(
                list(all_data.items()),
                args.seed,
                args.pretrain_epochs,
            )
            shared_training_seconds = time.perf_counter() - start
            log(f"shared_training_seconds={shared_training_seconds:.2f}")

        for target_dataset in DATASETS:
            if args.training_mode == "shared":
                model = shared_model
                training_seconds = shared_training_seconds
            else:
                set_seed(args.seed)
                start = time.perf_counter()
                model = pretrain_model(
                    [(target_dataset, all_data[target_dataset])],
                    args.seed,
                    args.pretrain_epochs,
                )
                training_seconds = time.perf_counter() - start
            labels = all_data[target_dataset].y.numpy()
            start = time.perf_counter()
            scores = normalized_scores(model, all_data[target_dataset])
            scoring_seconds = time.perf_counter() - start
            calibration_mask, validation_mask, unused_nodes = target_split(
                labels,
                args.seed,
            )
            threshold, calibration_macro_f1 = best_training_threshold(
                scores,
                labels,
                calibration_mask,
            )
            prediction = scores[validation_mask] > threshold
            validation_macro_f1 = f1_score(
                labels[validation_mask], prediction, average="macro", zero_division=0
            )
            validation_auc = roc_auc_score(labels[validation_mask], scores[validation_mask])
            validation_precision = precision_score(
                labels[validation_mask], prediction, zero_division=0
            )
            validation_recall = recall_score(
                labels[validation_mask], prediction, zero_division=0
            )
            log(
                f"target={target_dataset} calibration_nodes={calibration_mask.sum()} "
                f"validation_nodes={validation_mask.sum()} unused_nodes={unused_nodes} "
                f"threshold={threshold:.6f} calibration_macro_f1={calibration_macro_f1:.4f} "
                f"validation_macro_f1={validation_macro_f1:.4f} "
                f"validation_auc={validation_auc:.4f} "
                f"validation_precision={validation_precision:.4f} "
                f"validation_recall={validation_recall:.4f} "
                f"training_seconds={training_seconds:.2f} "
                f"scoring_seconds={scoring_seconds:.2f}"
            )
            if args.training_mode == "per-dataset":
                del model
                if DEVICE.startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()
        if shared_model is not None:
            del shared_model
            if DEVICE.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
