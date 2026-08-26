import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pygod.detector import CoLA, GADNR
from sklearn.metrics import f1_score, roc_auc_score

import cola
import contrastive
import gadnr
from dominant import best_training_threshold
from experiment_utils import train_validation_test_masks


ROOT = Path(__file__).resolve().parent.parent
HP_FILE = ROOT / "artifacts/results/benchmark/all_models_best_auc_by_dataset.csv"
DATASETS = ("UAE", "china", "cuba", "iran", "russia", "venezuela")
MODELS = ("cola", "gadnr", "anemone", "gradate")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def normalized(scores):
    scores = np.asarray(scores)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)


def train_cola(data, hp):
    model = CoLA(
        hid_dim=int(hp["hid_dim"]), num_layers=4, epoch=int(hp["epochs"]),
        lr=float(hp["lr"]), batch_size=cola.BATCH_SIZE,
        num_neigh=cola.NUM_NEIGHBORS, gpu=cola.GPU,
    )
    model.fit(data)
    return model, model.decision_score_.detach().cpu().numpy()


def train_gadnr(data, hp):
    model = GADNR(
        hid_dim=int(hp["hid_dim"]), num_layers=1, deg_dec_layers=4,
        fea_dec_layers=3, backbone=gadnr.BACKBONES[hp["backbone"]],
        sample_size=2, sample_time=3, neigh_loss="KL",
        lambda_loss1=float(hp["lambda_n"]), lambda_loss2=float(hp["lambda_x"]),
        lambda_loss3=float(hp["lambda_d"]), real_loss=False,
        epoch=int(hp["epochs"]), lr=0.01, dropout=0.0, weight_decay=0.0003,
        batch_size=int(hp["batch_size"]), num_neigh=int(hp["num_neigh"]),
        gpu=gadnr.GPU,
    )
    model.fit(
        data, h_loss_weight=float(hp["lambda_n_prime"]),
        degree_loss_weight=float(hp["lambda_d_prime"]),
        feature_loss_weight=float(hp["lambda_x_prime"]),
    )
    scores = model.decision_score_
    return model, scores.detach().cpu().numpy() if torch.is_tensor(scores) else scores


def train_contrastive(name, graph, features, hp, seed):
    model, state, scores, _, _ = contrastive.train_model(name, graph, features, hp, seed)
    return model, scores


def parse_args():
    parser = argparse.ArgumentParser(description="0.1% split experiments for non-DOMINANT models")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--train-fraction", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.499)
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--save-first-model", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "artifacts/results/other_models_0_1_percent.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    hp_df = pd.read_csv(HP_FILE)
    fields = ["model", "dataset", "seed", "train_nodes", "validation_nodes", "test_nodes", "threshold", "train_macro_f1", "validation_macro_f1", "validation_auc", "test_macro_f1", "test_auc", "training_seconds"]
    with args.output_csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for name in args.models:
            hp_name = "pygod_cola" if name == "cola" else name
            for dataset in args.datasets:
                hp = hp_df[(hp_df["model"] == hp_name) & (hp_df["dataset"] == dataset)].iloc[0]
                if name in ("cola", "gadnr"):
                    _, data = (cola.load_data(dataset) if name == "cola" else gadnr.load_data(dataset))
                    labels = data.y.detach().cpu().numpy()
                    if name == "gadnr" and DEVICE.startswith("cuda"):
                        data = data.to(DEVICE)
                else:
                    _, graph, features, labels = contrastive.load_data(dataset)
                    hp = contrastive.hyperparameters(name, hp)
                for seed in args.seeds:
                    (cola.set_seed if name == "cola" else gadnr.set_seed if name == "gadnr" else contrastive.set_seed)(seed)
                    start = time.perf_counter()
                    if name == "cola":
                        model, raw_scores = train_cola(data, hp)
                    elif name == "gadnr":
                        model, raw_scores = train_gadnr(data, hp)
                    else:
                        model, raw_scores = train_contrastive(name, graph, features, hp, seed)
                    training_seconds = time.perf_counter() - start
                    scores = normalized(raw_scores)
                    train_mask, validation_mask, test_mask = train_validation_test_masks(labels, args.train_fraction, args.validation_fraction, args.test_fraction, seed)
                    threshold, train_f1 = best_training_threshold(scores, labels, train_mask)
                    validation_pred = scores[validation_mask] > threshold
                    test_pred = scores[test_mask] > threshold
                    row = {"model": name, "dataset": dataset, "seed": seed, "train_nodes": train_mask.sum(), "validation_nodes": validation_mask.sum(), "test_nodes": test_mask.sum(), "threshold": threshold, "train_macro_f1": train_f1, "validation_macro_f1": f1_score(labels[validation_mask], validation_pred, average="macro", zero_division=0), "validation_auc": roc_auc_score(labels[validation_mask], scores[validation_mask]), "test_macro_f1": f1_score(labels[test_mask], test_pred, average="macro", zero_division=0), "test_auc": roc_auc_score(labels[test_mask], scores[test_mask]), "training_seconds": training_seconds}
                    writer.writerow(row)
                    output.flush()
                    print(row, flush=True)
                    del model
                    if DEVICE.startswith("cuda"):
                        torch.cuda.empty_cache()
                    gc.collect()


if __name__ == "__main__":
    main()
