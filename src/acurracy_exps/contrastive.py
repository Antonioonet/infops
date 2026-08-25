import csv
import gc
import importlib.util
import math
import pickle
import random
import time
from datetime import date
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT / "data"
HP_FILE = ROOT / "artifacts/results/benchmark/all_models_best_auc_by_dataset.csv"
SEEDS = [1, 2, 3, 4, 5]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
SUBGRAPH_SIZE = 4
WEIGHT_DECAY = 0.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def value(row, key, default, cast):
    return default if pd.isna(row[key]) else cast(row[key])


def hyperparameters(model_name, row):
    return {
        "epochs": value(row, "epochs", 400 if model_name == "gradate" else 100, int),
        "embedding_dim": value(row, "embedding_dim", 64, int),
        "lr": value(row, "lr", 0.001, float),
        "alpha": value(row, "alpha", 0.1 if model_name == "gradate" else 1.0, float),
        "beta": value(row, "beta", 0.1, float),
        "test_rounds": value(row, "test_rounds", 64, int),
        "negsamp_patch": value(row, "negsamp_patch", 6 if model_name == "gradate" else 1, int),
        "negsamp_context": value(row, "negsamp_context", 1, int),
    }


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
    labels = np.asarray(graph_data["labels"], dtype=np.int64)
    return graph_data, graph, features, labels


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.asarray(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat).transpose().dot(d_mat).tocoo()


def dense_adj(adj):
    normalized = normalize_adj(adj) + sp.eye(adj.shape[0])
    return torch.from_numpy(np.asarray(normalized.todense(), dtype=np.float32))


def augment_adj(adj, seed):
    rng = np.random.default_rng(seed)
    upper = sp.triu(adj, k=1).tocoo()
    keep = rng.random(upper.nnz) > 0.2
    kept = sp.coo_matrix(
        (upper.data[keep], (upper.row[keep], upper.col[keep])),
        shape=adj.shape,
    )
    return (kept + kept.T).tocsr()


def sample_subgraphs(dgl_graph, seed):
    dgl.seed(seed)
    nodes = torch.arange(dgl_graph.num_nodes())
    traces, _ = dgl.sampling.random_walk(
        dgl_graph,
        nodes,
        length=SUBGRAPH_SIZE * 3,
        restart_prob=0.1,
    )
    subgraphs = []
    for node, trace in enumerate(traces):
        trace = trace[trace >= 0]
        picked = torch.unique(trace, sorted=False).tolist()
        retry = 0
        while len(picked) < SUBGRAPH_SIZE - 1:
            dgl.seed(seed + retry + 1)
            retry_trace, _ = dgl.sampling.random_walk(
                dgl_graph,
                [node],
                length=SUBGRAPH_SIZE * 5,
                restart_prob=0.1,
            )
            retry_trace = retry_trace[0]
            retry_trace = retry_trace[retry_trace >= 0]
            picked = torch.unique(retry_trace, sorted=False).tolist()
            retry += 1
            if len(picked) <= 2 and retry > 10:
                picked = picked * (SUBGRAPH_SIZE - 1)
        subgraphs.append(picked[:SUBGRAPH_SIZE - 1] + [node])
    return subgraphs


def load_model(model_name):
    model_file = ROOT / "vendor" / model_name.upper() / "model.py"
    spec = importlib.util.spec_from_file_location(f"{model_name}_model", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def model_batch(indices, subgraphs, adj, features, adj_hat=None):
    size = len(indices)
    feature_dim = features.shape[2]
    zero_row = torch.zeros((size, 1, SUBGRAPH_SIZE), device=DEVICE)
    zero_col = torch.zeros((size, SUBGRAPH_SIZE + 1, 1), device=DEVICE)
    zero_col[:, -1, :] = 1.0
    zero_feature = torch.zeros((size, 1, feature_dim), device=DEVICE)
    batch_adj = []
    batch_adj_hat = []
    batch_features = []
    for node in indices:
        nodes = subgraphs[node]
        batch_adj.append(adj[:, nodes, :][:, :, nodes])
        batch_features.append(features[:, nodes, :])
        if adj_hat is not None:
            batch_adj_hat.append(adj_hat[:, nodes, :][:, :, nodes])
    batch_adj = torch.cat((torch.cat(batch_adj), zero_row), dim=1)
    batch_adj = torch.cat((batch_adj, zero_col), dim=2)
    batch_features = torch.cat(batch_features)
    batch_features = torch.cat(
        (batch_features[:, :-1, :], zero_feature, batch_features[:, -1:, :]),
        dim=1,
    )
    if adj_hat is None:
        return batch_features, batch_adj, None
    batch_adj_hat = torch.cat((torch.cat(batch_adj_hat), zero_row), dim=1)
    batch_adj_hat = torch.cat((batch_adj_hat, zero_col), dim=2)
    return batch_features, batch_adj, batch_adj_hat


def contrastive_loss(model_name, model, features, adj, adj_hat, size, hp):
    patch_labels = torch.cat(
        (torch.ones(size), torch.zeros(size * int(hp["negsamp_patch"])))
    ).unsqueeze(1).to(DEVICE)
    context_labels = torch.cat(
        (torch.ones(size), torch.zeros(size * int(hp["negsamp_context"])))
    ).unsqueeze(1).to(DEVICE)
    patch_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([int(hp["negsamp_patch"])], device=DEVICE)
    )
    context_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([int(hp["negsamp_context"])], device=DEVICE)
    )
    alpha = float(hp["alpha"])
    if model_name == "anemone":
        logits_context, logits_patch = model(features, adj)
        return alpha * context_loss(logits_context, context_labels) + (
            1 - alpha
        ) * patch_loss(logits_patch, patch_labels)

    beta = float(hp["beta"])
    logits_context, logits_patch, subgraph, _ = model(features, adj)
    logits_context_hat, logits_patch_hat, subgraph_hat, _ = model(
        features, adj_hat
    )
    subgraph = F.normalize(subgraph, dim=1, p=2)
    subgraph_hat = F.normalize(subgraph_hat, dim=1, p=2)
    sim_one = torch.exp(torch.matmul(subgraph, subgraph_hat.t()))
    sim_two = torch.exp(torch.matmul(subgraph, subgraph.t()))
    sim_three = torch.exp(torch.matmul(subgraph_hat, subgraph_hat.t()))
    negatives = np.arange(size - 1)
    negatives = np.insert(negatives, 0, size - 1)
    denominator = torch.diagonal(
        sim_one[:, negatives] + sim_two[:, negatives] + sim_three[:, negatives]
    )
    numerator = torch.diagonal(sim_one)
    nce_loss = torch.mean(-torch.log(numerator / denominator))
    context = alpha * context_loss(logits_context, context_labels) + (
        1 - alpha
    ) * context_loss(logits_context_hat, context_labels)
    patch = alpha * patch_loss(logits_patch, patch_labels) + (
        1 - alpha
    ) * patch_loss(logits_patch_hat, patch_labels)
    return beta * context + (1 - beta) * patch + 0.1 * nce_loss


def train_model(model_name, graph, node_features, hp, seed):
    set_seed(seed)
    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=range(graph.number_of_nodes()),
        format="csr",
        dtype=np.float32,
    )
    dgl_graph = dgl.from_scipy(sp.csr_matrix(adjacency))
    adj = dense_adj(adjacency).unsqueeze(0).to(DEVICE)
    features = node_features.unsqueeze(0).to(DEVICE)
    adj_hat = None
    if model_name == "gradate":
        adj_hat = dense_adj(augment_adj(adjacency, seed)).unsqueeze(0).to(DEVICE)
    model_class = load_model(model_name)
    model = model_class(
        node_features.shape[1],
        int(hp["embedding_dim"]),
        "prelu",
        int(hp["negsamp_patch"]),
        int(hp["negsamp_context"]),
        "avg",
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(hp["lr"]),
        weight_decay=WEIGHT_DECAY,
    )
    best_loss = float("inf")
    best_state = None
    start = time.perf_counter()
    for epoch in range(int(hp["epochs"])):
        model.train()
        indices = list(range(graph.number_of_nodes()))
        np.random.default_rng(seed + epoch).shuffle(indices)
        subgraphs = sample_subgraphs(dgl_graph, seed + epoch)
        total_loss = 0.0
        for batch_start in range(0, len(indices), BATCH_SIZE):
            batch = indices[batch_start:batch_start + BATCH_SIZE]
            batch_features, batch_adj, batch_adj_hat = model_batch(
                batch,
                subgraphs,
                adj,
                features,
                adj_hat,
            )
            optimizer.zero_grad()
            loss = contrastive_loss(
                model_name,
                model,
                batch_features,
                batch_adj,
                batch_adj_hat,
                len(batch),
                hp,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch)
        mean_loss = total_loss / graph.number_of_nodes()
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    training_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    scores = score_model(
        model_name,
        model,
        graph,
        dgl_graph,
        features,
        adj,
        adj_hat,
        hp,
        seed + 10000,
    )
    return model, best_state, scores, best_loss, training_seconds


def score_model(
    model_name,
    model,
    graph,
    dgl_graph,
    features,
    adj,
    adj_hat,
    hp,
    seed,
):
    rounds = int(hp["test_rounds"])
    scores = np.zeros((rounds, graph.number_of_nodes()), dtype=np.float64)
    for round_id in range(rounds):
        indices = list(range(graph.number_of_nodes()))
        np.random.default_rng(seed + round_id).shuffle(indices)
        subgraphs = sample_subgraphs(dgl_graph, seed + round_id)
        for batch_start in range(0, len(indices), BATCH_SIZE):
            batch = indices[batch_start:batch_start + BATCH_SIZE]
            batch_features, batch_adj, batch_adj_hat = model_batch(
                batch,
                subgraphs,
                adj,
                features,
                adj_hat,
            )
            size = len(batch)
            with torch.no_grad():
                if model_name == "anemone":
                    context, patch = model(batch_features, batch_adj)
                    context = torch.sigmoid(torch.squeeze(context))
                    patch = torch.sigmoid(torch.squeeze(patch))
                    context_score = -(context[:size] - context[size:].view(
                        size, int(hp["negsamp_context"])
                    ).mean(1))
                    patch_score = -(patch[:size] - patch[size:].view(
                        size, int(hp["negsamp_patch"])
                    ).mean(1))
                    score = float(hp["alpha"]) * context_score + (
                        1 - float(hp["alpha"])
                    ) * patch_score
                else:
                    context, patch, _, _ = model(batch_features, batch_adj)
                    context_hat, patch_hat, _, _ = model(
                        batch_features,
                        batch_adj_hat,
                    )
                    context, patch, context_hat, patch_hat = [
                        torch.sigmoid(torch.squeeze(value))
                        for value in (context, patch, context_hat, patch_hat)
                    ]
                    context_score = -(context[:size] - context[size:].view(
                        size, int(hp["negsamp_context"])
                    ).mean(1))
                    context_hat_score = -(
                        context_hat[:size] - context_hat[size:].view(
                            size, int(hp["negsamp_context"])
                        ).mean(1)
                    )
                    patch_score = -(patch[:size] - patch[size:].view(
                        size, int(hp["negsamp_patch"])
                    ).mean(1))
                    patch_hat_score = -(patch_hat[:size] - patch_hat[size:].view(
                        size, int(hp["negsamp_patch"])
                    ).mean(1))
                    alpha = float(hp["alpha"])
                    beta = float(hp["beta"])
                    score = beta * (
                        alpha * context_score + (1 - alpha) * context_hat_score
                    ) + (1 - beta) * (
                        alpha * patch_score + (1 - alpha) * patch_hat_score
                    )
            scores[round_id, batch] = score.detach().cpu().numpy()
    return scores.mean(axis=0) + scores.std(axis=0)


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


def run_experiment(model_name):
    output_root = ROOT / "artifacts/results" / f"{model_name}_refactoring_{date.today().isoformat()}"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    hp_df = pd.read_csv(HP_FILE)
    hp_df = hp_df[hp_df["model"] == model_name]
    datasets = sorted(path.name for path in DATA_ROOT.iterdir() if path.is_dir())
    columns = [
        "dataset",
        "seed",
        "split",
        "threshold",
        "train_macro_f1",
        "validation_macro_f1",
        "validation_auc",
        "training_seconds",
        "status",
    ]
    with open(output_root / "results.csv", "w", newline="") as csv_file, open(
        output_root / "run.log", "w"
    ) as log_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        def log(message):
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()
        log(f"Device: {DEVICE}; batch_size={BATCH_SIZE}; subgraph_size={SUBGRAPH_SIZE}")
        for dataset in datasets:
            log(f"Starting {dataset}")
            graph_data, graph, features, labels = load_data(dataset)
            hp = hyperparameters(
                model_name,
                hp_df[hp_df["dataset"] == dataset].iloc[0],
            )
            best_seed_score = -1
            best_checkpoint = None
            for seed in SEEDS:
                model, state_dict, raw_scores, best_loss, training_seconds = train_model(
                    model_name,
                    graph,
                    features,
                    hp,
                    seed,
                )
                score_min = float(raw_scores.min())
                score_max = float(raw_scores.max())
                scores = (raw_scores - score_min) / (score_max - score_min + 1e-12)
                validation_scores = []
                thresholds = []
                for split_id in range(5):
                    split = graph_data["splits"][split_id]
                    train_mask = np.asarray(split["train"], dtype=bool)
                    val_mask = np.asarray(split["val"], dtype=bool)
                    threshold, train_macro_f1 = best_training_threshold(
                        scores,
                        labels,
                        train_mask,
                    )
                    validation_prediction = scores[val_mask] > threshold
                    validation_macro_f1 = f1_score(
                        labels[val_mask],
                        validation_prediction,
                        average="macro",
                        zero_division=0,
                    )
                    validation_auc = roc_auc_score(labels[val_mask], scores[val_mask])
                    writer.writerow({
                        "dataset": dataset,
                        "seed": seed,
                        "split": split_id,
                        "threshold": threshold,
                        "train_macro_f1": train_macro_f1,
                        "validation_macro_f1": validation_macro_f1,
                        "validation_auc": validation_auc,
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
                        "state_dict": state_dict,
                        "hyperparameters": {
                            "epochs": int(hp["epochs"]),
                            "embedding_dim": int(hp["embedding_dim"]),
                            "lr": float(hp["lr"]),
                            "alpha": float(hp["alpha"]),
                            "beta": float(hp["beta"]),
                            "batch_size": BATCH_SIZE,
                            "subgraph_size": SUBGRAPH_SIZE,
                            "test_rounds": int(hp["test_rounds"]),
                            "negsamp_patch": int(hp["negsamp_patch"]),
                            "negsamp_context": int(hp["negsamp_context"]),
                        },
                        "feature_dim": int(features.shape[1]),
                        "thresholds_by_split": thresholds,
                        "default_threshold": float(np.median(thresholds)),
                        "score_min": score_min,
                        "score_max": score_max,
                        "best_loss": best_loss,
                        "validation_macro_f1_mean": seed_score,
                    }
                del model
                if DEVICE.startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()
            checkpoint_dir = checkpoint_root / dataset
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save(best_checkpoint, checkpoint_dir / f"{model_name}.pt")
            log(
                f"Completed {dataset}; best_seed={best_checkpoint['seed']} "
                f"validation_macro_f1_mean={best_seed_score:.4f}"
            )
