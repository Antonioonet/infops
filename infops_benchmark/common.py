import json
import math
import pickle
import random
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]


def data_root(dataset_name: str) -> Path:
    return ROOT / "data" / "processed" / dataset_name


def out_root(dataset_name: str) -> Path:
    return ROOT / "benchmark_results" / dataset_name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_name: str = "russia", graph_key: str = "graph"):
    root = data_root(dataset_name)
    with open(root / "0.7_datasets.pkl", "rb") as fh:
        dataset = pickle.load(fh)
    graph = nx.convert_node_labels_to_integers(dataset[graph_key], ordering="sorted")
    labels = np.asarray(dataset["labels"], dtype=np.int64)
    sbert = torch.load(root / "sbert_nodeattributes_mostPop5.pt", map_location="cpu")
    if sbert.shape[0] != graph.number_of_nodes():
        raise ValueError(f"feature/node mismatch: {sbert.shape[0]} features for {graph.number_of_nodes()} nodes")
    return dataset, graph, labels, sbert.float()


def load_russia(graph_key: str = "graph"):
    return load_dataset("russia", graph_key)


def degree_log2_onehot(graph: nx.Graph) -> torch.Tensor:
    degrees = np.asarray([graph.degree(i) for i in range(graph.number_of_nodes())], dtype=np.int64)
    bins = np.asarray([0 if d <= 1 else int(math.ceil(math.log2(d))) for d in degrees], dtype=np.int64)
    onehot = np.zeros((graph.number_of_nodes(), int(bins.max()) + 1), dtype=np.float32)
    onehot[np.arange(graph.number_of_nodes()), bins] = 1.0
    return torch.from_numpy(onehot)


def node_features(graph: nx.Graph, sbert: torch.Tensor) -> torch.Tensor:
    return torch.cat([sbert.float(), degree_log2_onehot(graph)], dim=1)


def edge_index_from_graph(graph: nx.Graph) -> torch.Tensor:
    edges = []
    for u, v in graph.edges():
        edges.append((int(u), int(v)))
        if u != v:
            edges.append((int(v), int(u)))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def sparse_adj_from_graph(graph: nx.Graph) -> sp.csr_matrix:
    return nx.to_scipy_sparse_array(graph, nodelist=range(graph.number_of_nodes()), format="csr", dtype=np.float32)


def pyg_data(dataset_name: str = "russia", graph_key: str = "graph") -> Data:
    _, graph, labels, sbert = load_dataset(dataset_name, graph_key)
    return Data(
        x=node_features(graph, sbert),
        edge_index=edge_index_from_graph(graph),
        y=torch.from_numpy(labels).long(),
    )


def normalized_scores(scores) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).any():
        return np.zeros_like(scores, dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    scores = np.nan_to_num(scores, nan=float(np.median(finite)), posinf=float(np.max(finite)), neginf=float(np.min(finite)))
    lo = float(scores.min())
    hi = float(scores.max())
    if hi <= lo:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - lo) / (hi - lo)


def evaluate_scores(labels, scores, threshold: float = 0.5, mask=None) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probs = normalized_scores(scores)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        labels_eval = labels[mask]
        probs_eval = probs[mask]
    else:
        labels_eval = labels
        probs_eval = probs
    pred = (probs_eval > threshold).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels_eval, probs_eval)),
        "macro_f1": float(f1_score(labels_eval, pred, average="macro", zero_division=0)),
        "threshold": threshold,
        "num_nodes": int(labels_eval.shape[0]),
        "num_anomalies": int(labels_eval.sum()),
    }


def torch_model_size(module: nn.Module) -> dict:
    total_params = sum(param.numel() for param in module.parameters())
    trainable_params = sum(param.numel() for param in module.parameters() if param.requires_grad)
    param_bytes = sum(param.numel() * param.element_size() for param in module.parameters())
    buffer_bytes = sum(buf.numel() * buf.element_size() for buf in module.buffers())
    return {
        "num_parameters": int(total_params),
        "num_trainable_parameters": int(trainable_params),
        "parameter_size_mb": float(param_bytes / (1024**2)),
        "buffer_size_mb": float(buffer_bytes / (1024**2)),
        "model_size_mb": float((param_bytes + buffer_bytes) / (1024**2)),
    }


def model_size_from_object(obj) -> dict:
    if isinstance(obj, nn.Module):
        return torch_model_size(obj)
    for attr in ("model", "model_", "net", "net_", "detector", "module"):
        candidate = getattr(obj, attr, None)
        if isinstance(candidate, nn.Module):
            return torch_model_size(candidate)
    return {
        "num_parameters": None,
        "num_trainable_parameters": None,
        "parameter_size_mb": None,
        "buffer_size_mb": None,
        "model_size_mb": None,
    }


def write_result(name: str, payload: dict, dataset_name: str = "russia") -> None:
    root = out_root(dataset_name)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {path}")
