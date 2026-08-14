import math
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]


def degree_log2_onehot(graph):
    degrees = np.asarray([graph.degree(i) for i in range(graph.number_of_nodes())], dtype=np.int64)
    bins = np.asarray([0 if d <= 1 else int(math.ceil(math.log2(d))) for d in degrees], dtype=np.int64)
    onehot = np.zeros((graph.number_of_nodes(), int(bins.max()) + 1), dtype=np.float32)
    onehot[np.arange(graph.number_of_nodes()), bins] = 1.0
    return torch.from_numpy(onehot)


def edge_index_from_graph(graph):
    edges = []
    for u, v in graph.edges():
        edges.append((int(u), int(v)))
        if u != v:
            edges.append((int(v), int(u)))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_dataset(dataset_name):
    src = ROOT / "data" / "processed" / dataset_name
    with open(src / "0.7_datasets.pkl", "rb") as fh:
        payload = pickle.load(fh)

    graph = nx.convert_node_labels_to_integers(payload["graph"], ordering="sorted")
    labels = np.asarray(payload["labels"], dtype=np.int64)
    sbert = torch.load(src / "sbert_nodeattributes_mostPop5.pt", map_location="cpu").float()
    if sbert.shape[0] != graph.number_of_nodes():
        raise ValueError(f"{dataset_name}: {sbert.shape[0]} features for {graph.number_of_nodes()} nodes")

    x = torch.cat([sbert, degree_log2_onehot(graph)], dim=1).float()
    return Data(
        x=x,
        edge_index=edge_index_from_graph(graph),
        anomaly_flag=torch.from_numpy(labels.astype(bool)),
    )


def main():
    out_dir = Path(__file__).resolve().parent / "data" / "real_world"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name in ("iran", "cuba", "UAE"):
        data = build_dataset(dataset_name)
        out_path = out_dir / f"{dataset_name}.pkl"
        with open(out_path, "wb") as fh:
            pickle.dump(data, fh)
        print(
            f"wrote {out_path}: nodes={data.num_nodes} "
            f"features={data.num_features} edges={data.edge_index.size(1)} "
            f"anomalies={int(data.anomaly_flag.sum())}"
        )


if __name__ == "__main__":
    main()
