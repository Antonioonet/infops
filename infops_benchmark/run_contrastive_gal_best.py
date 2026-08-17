import argparse
import csv
import importlib.util
import json
import math
import pickle
import random
import time
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "benchmark_results"
BEST_TABLE = RESULTS / "best_auc_tables" / "all_best_auc.csv"
OUT = RESULTS / "gal_contrastive_best"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_name, graph_key="graph"):
    root = ROOT / "data" / "processed" / dataset_name
    with open(root / "0.7_datasets.pkl", "rb") as fh:
        dataset = pickle.load(fh)
    graph = nx.convert_node_labels_to_integers(dataset[graph_key], ordering="sorted")
    labels = np.asarray(dataset["labels"], dtype=np.int64)
    sbert = torch.load(root / "sbert_nodeattributes_mostPop5.pt", map_location="cpu")
    if sbert.shape[0] != graph.number_of_nodes():
        raise ValueError(f"feature/node mismatch: {sbert.shape[0]} features for {graph.number_of_nodes()} nodes")
    return dataset, graph, labels, sbert.float()


def degree_log2_onehot(graph):
    degrees = np.asarray([graph.degree(i) for i in range(graph.number_of_nodes())], dtype=np.int64)
    bins = np.asarray([0 if d <= 1 else int(math.ceil(math.log2(d))) for d in degrees], dtype=np.int64)
    onehot = np.zeros((graph.number_of_nodes(), int(bins.max()) + 1), dtype=np.float32)
    onehot[np.arange(graph.number_of_nodes()), bins] = 1.0
    return torch.from_numpy(onehot)


def node_features(graph, sbert):
    return torch.cat([sbert.float(), degree_log2_onehot(graph)], dim=1)


def sparse_adj_from_graph(graph):
    return nx.to_scipy_sparse_array(graph, nodelist=range(graph.number_of_nodes()), format="csr", dtype=np.float32)


def normalized_scores(scores):
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


def evaluate_scores(labels, scores, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int64)
    probs = normalized_scores(scores)
    pred = (probs > threshold).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "threshold": threshold,
        "num_nodes": int(labels.shape[0]),
        "num_anomalies": int(labels.sum()),
    }


def model_size_from_object(module):
    param_bytes = sum(param.numel() * param.element_size() for param in module.parameters())
    buffer_bytes = sum(buf.numel() * buf.element_size() for buf in module.buffers())
    total_params = sum(param.numel() for param in module.parameters())
    trainable_params = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return {
        "num_parameters": int(total_params),
        "num_trainable_parameters": int(trainable_params),
        "parameter_size_mb": float(param_bytes / (1024**2)),
        "buffer_size_mb": float(buffer_bytes / (1024**2)),
        "model_size_mb": float((param_bytes + buffer_bytes) / (1024**2)),
    }


def load_model_class(model_dir):
    spec = importlib.util.spec_from_file_location(f"{model_dir}_model", ROOT / model_dir / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def load_official_gal_loss_class():
    official_path = ROOT / "Graph-Anomaly-Loss" / "src" / "models.py"
    if not official_path.exists():
        raise FileNotFoundError(f"Missing official GAL repository at {official_path}")
    spec = importlib.util.spec_from_file_location("official_gal_models", official_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UnsupervisedLoss


class OfficialGALAdapter:
    """Feeds InfoOps data into zhao-tong/Graph-Anomaly-Loss UnsupervisedLoss."""

    def __init__(self, graph, labels, device, c_margin):
        self.ds = "infops"
        labels = np.asarray(labels, dtype=np.int64)
        n_nodes = graph.number_of_nodes()
        adj_lists = {i: set(int(j) for j in graph.neighbors(i)) for i in range(n_nodes)}
        counts = np.bincount(labels, minlength=int(labels.max()) + 1)
        u2size = np.asarray([counts[label] for label in labels], dtype=np.float32)
        dc = SimpleNamespace(args=SimpleNamespace(cluster_aloss=False))
        setattr(dc, f"{self.ds}_adj_lists", adj_lists)
        setattr(dc, f"{self.ds}_best_adj_lists", adj_lists)
        setattr(dc, f"{self.ds}_simis", sp.eye(n_nodes, format="csr", dtype=np.float32))
        setattr(dc, f"{self.ds}_train", np.arange(n_nodes))
        setattr(dc, f"{self.ds}_trainable", set(range(n_nodes)))
        setattr(dc, f"{self.ds}_labels_a", labels)
        setattr(dc, f"{self.ds}_u2size_of_cluster", u2size)
        official_loss = load_official_gal_loss_class()
        self.loss = official_loss(dc, self.ds, device, biased=False, a_loss="labels", C=c_margin)

    def extend_nodes(self, nodes, num_pairs):
        self.loss.positive_pairs_aloss = []
        self.loss.negative_pairs_aloss = []
        self.loss.node_positive_pairs_aloss = {}
        self.loss.node_negative_pairs_aloss = {}
        self.loss.get_label_aloss_pos_neg_nodes(np.asarray(nodes, dtype=np.int64), num_pairs)
        unique = set(int(n) for n in nodes)
        for pair in self.loss.positive_pairs_aloss + self.loss.negative_pairs_aloss:
            unique.update(int(n) for n in pair)
        self.loss.unique_nodes_batch = list(unique)
        return self.loss.unique_nodes_batch

    def anomaly_loss(self, embeddings, unique_nodes):
        maxmin, mean = self.loss.get_loss_anomaly(embeddings, np.asarray(unique_nodes, dtype=np.int64))
        return maxmin, mean


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.asarray(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat).transpose().dot(d_mat).tocoo()


def dense_norm_adj(adj):
    return torch.from_numpy((normalize_adj(adj) + sp.eye(adj.shape[0])).todense()).float()


def augment_random_edge(adj, drop_rate=0.2, seed=0):
    rng = np.random.default_rng(seed)
    coo = sp.triu(adj, k=1).tocoo()
    keep = rng.random(coo.nnz) > drop_rate
    kept = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=adj.shape)
    return (kept + kept.T).tocsr()


def sample_subgraphs(graph: nx.Graph, subgraph_size: int, seed: int):
    rng = np.random.default_rng(seed)
    reduced_size = subgraph_size - 1
    subgraphs = []
    for node in range(graph.number_of_nodes()):
        picked = []
        current = node
        for _ in range(subgraph_size * 5):
            neigh = list(graph.neighbors(current))
            if not neigh:
                current = node
                continue
            current = int(rng.choice(neigh))
            if current != node and current not in picked:
                picked.append(current)
            if len(picked) >= reduced_size:
                break
        if not picked:
            picked = [node]
        while len(picked) < reduced_size:
            picked.append(int(rng.choice(picked)))
        subgraphs.append(picked[:reduced_size] + [node])
    return subgraphs


def value(row, key, default, cast):
    raw = row.get(key, "")
    if raw in ("", None):
        return default
    return cast(float(raw)) if cast is int else cast(raw)


def best_config(dataset, model):
    with open(BEST_TABLE, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["dataset"] == dataset and row["model"] == model:
                return row
    raise ValueError(f"missing best config for {dataset}/{model}")


def anchor_embeddings(model, bf, ba):
    h_context = model.gcn_context(bf, ba)
    h_patch = model.gcn_patch(bf, ba)
    return 0.5 * (h_context[:, -1, :] + h_patch[:, -1, :])


def make_model_batch(node_ids, subgraphs, adj, features, feature_dim, subgraph_size, device, adj_hat=None):
    cur = len(node_ids)
    zero_row = torch.zeros((cur, 1, subgraph_size), device=device)
    zero_col = torch.zeros((cur, subgraph_size + 1, 1), device=device)
    zero_col[:, -1, :] = 1.0
    zero_feat = torch.zeros((cur, 1, feature_dim), device=device)
    ba, bf, ba_hat = [], [], []
    for node_id in node_ids:
        sg = subgraphs[int(node_id)]
        ba.append(adj[:, sg, :][:, :, sg])
        bf.append(features[:, sg, :])
        if adj_hat is not None:
            ba_hat.append(adj_hat[:, sg, :][:, :, sg])
    ba = torch.cat((torch.cat(ba), zero_row), dim=1)
    ba = torch.cat((ba, zero_col), dim=2)
    bf = torch.cat(bf)
    bf = torch.cat((bf[:, :-1, :], zero_feat, bf[:, -1:, :]), dim=1)
    if adj_hat is None:
        return bf, ba, None
    ba_hat = torch.cat((torch.cat(ba_hat), zero_row), dim=1)
    ba_hat = torch.cat((ba_hat, zero_col), dim=2)
    return bf, ba, ba_hat


def unpack_model_outputs(outputs):
    return outputs if isinstance(outputs, tuple) else tuple(outputs)


def run_batches(model, adj, features, graph, args, device, gradate=False, adj_hat=None, seed=0):
    nb_nodes = features.shape[1]
    ft_size = features.shape[2]
    batch_num = nb_nodes // args.batch_size + 1
    scores = np.zeros((args.test_rounds, nb_nodes), dtype=np.float64)
    for round_id in range(args.test_rounds):
        all_idx = list(range(nb_nodes))
        np.random.default_rng(seed + round_id).shuffle(all_idx)
        subgraphs = sample_subgraphs(graph, args.subgraph_size, seed + round_id)
        for batch_idx in range(batch_num):
            idx = all_idx[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size]
            if not idx:
                continue
            cur = len(idx)
            zero_row = torch.zeros((cur, 1, args.subgraph_size), device=device)
            zero_col = torch.zeros((cur, args.subgraph_size + 1, 1), device=device)
            zero_col[:, -1, :] = 1.0
            zero_feat = torch.zeros((cur, 1, ft_size), device=device)
            ba, bf, ba_hat = [], [], []
            for i in idx:
                sg = subgraphs[i]
                ba.append(adj[:, sg, :][:, :, sg])
                bf.append(features[:, sg, :])
                if gradate:
                    ba_hat.append(adj_hat[:, sg, :][:, :, sg])
            ba = torch.cat((torch.cat(ba), zero_row), dim=1)
            ba = torch.cat((ba, zero_col), dim=2)
            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], zero_feat, bf[:, -1:, :]), dim=1)
            with torch.no_grad():
                if gradate:
                    ba_hat = torch.cat((torch.cat(ba_hat), zero_row), dim=1)
                    ba_hat = torch.cat((ba_hat, zero_col), dim=2)
                    l1, l2, _, _ = unpack_model_outputs(model(bf, ba))
                    l1h, l2h, _, _ = unpack_model_outputs(model(bf, ba_hat))
                    l1, l2, l1h, l2h = [torch.sigmoid(torch.squeeze(t)) for t in (l1, l2, l1h, l2h)]
                    s1 = -(l1[:cur] - l1[cur:].view(cur, args.negsamp_context).mean(1))
                    s1h = -(l1h[:cur] - l1h[cur:].view(cur, args.negsamp_context).mean(1))
                    s2 = -(l2[:cur] - l2[cur:].view(cur, args.negsamp_patch).mean(1))
                    s2h = -(l2h[:cur] - l2h[cur:].view(cur, args.negsamp_patch).mean(1))
                    score = args.beta * (args.alpha * s1 + (1 - args.alpha) * s1h) + (1 - args.beta) * (args.alpha * s2 + (1 - args.alpha) * s2h)
                else:
                    l1, l2 = unpack_model_outputs(model(bf, ba))[:2]
                    l1 = torch.sigmoid(torch.squeeze(l1))
                    l2 = torch.sigmoid(torch.squeeze(l2))
                    s1 = -(l1[:cur] - l1[cur:].view(cur, args.negsamp_context).mean(1))
                    s2 = -(l2[:cur] - l2[cur:].view(cur, args.negsamp_patch).mean(1))
                    score = args.alpha * s1 + (1 - args.alpha) * s2
            scores[round_id, idx] = score.detach().cpu().numpy()
    return scores.mean(axis=0) + scores.std(axis=0)


def run_one(args):
    row = best_config(args.dataset, args.model)
    args.epochs = value(row, "epochs", args.epochs, int)
    args.embedding_dim = value(row, "embedding_dim", args.embedding_dim, int)
    args.lr = value(row, "lr", args.lr, float)
    args.alpha = value(row, "alpha", args.alpha, float)
    args.beta = value(row, "beta", args.beta, float)
    args.test_rounds = value(row, "test_rounds", args.test_rounds, int)
    args.negsamp_patch = value(row, "negsamp_patch", args.negsamp_patch, int)
    args.negsamp_context = value(row, "negsamp_context", args.negsamp_context, int)
    if args.limit_epochs is not None:
        args.epochs = min(args.epochs, args.limit_epochs)
    if args.limit_test_rounds is not None:
        args.test_rounds = min(args.test_rounds, args.limit_test_rounds)

    out_path = OUT / args.dataset / args.model / f"seed_{args.seed}_gal_w{args.gal_weight:g}_m{args.gal_margin_scale:g}.json"
    if out_path.exists() and not args.force:
        print(f"Skipping existing result: {out_path}", flush=True)
        return

    set_seed(args.seed)
    start = time.perf_counter()
    _, graph, labels_np, sbert = load_dataset(args.dataset, "graph")
    x = node_features(graph, sbert).numpy().astype(np.float32)
    adj_sp = sparse_adj_from_graph(graph)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    adj = dense_norm_adj(adj_sp).unsqueeze(0).to(device)
    features = torch.from_numpy(x).unsqueeze(0).to(device)
    official_gal = OfficialGALAdapter(graph, labels_np, device, args.gal_margin_scale)

    gradate = args.model == "gradate"
    model_cls = load_model_class("GRADATE" if gradate else "ANEMONE")
    model = model_cls(x.shape[1], args.embedding_dim, "prelu", args.negsamp_patch, args.negsamp_context, "avg").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce_patch = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.negsamp_patch], device=device))
    bce_context = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.negsamp_context], device=device))
    adj_hat = dense_norm_adj(augment_random_edge(adj_sp, seed=args.seed)).unsqueeze(0).to(device) if gradate else None
    best_state = None
    best_loss = float("inf")
    best_base = 0.0
    best_gal = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = total_base = total_gal = 0.0
        seen = 0
        all_idx = list(range(graph.number_of_nodes()))
        np.random.default_rng(args.seed + epoch).shuffle(all_idx)
        subgraphs = sample_subgraphs(graph, args.subgraph_size, args.seed + epoch)
        for batch_start in range(0, len(all_idx), args.batch_size):
            idx = all_idx[batch_start:batch_start + args.batch_size]
            cur = len(idx)
            zero_row = torch.zeros((cur, 1, args.subgraph_size), device=device)
            zero_col = torch.zeros((cur, args.subgraph_size + 1, 1), device=device)
            zero_col[:, -1, :] = 1.0
            zero_feat = torch.zeros((cur, 1, x.shape[1]), device=device)
            ba, bf, ba_hat = [], [], []
            for i in idx:
                sg = subgraphs[i]
                ba.append(adj[:, sg, :][:, :, sg])
                bf.append(features[:, sg, :])
                if gradate:
                    ba_hat.append(adj_hat[:, sg, :][:, :, sg])
            ba = torch.cat((torch.cat(ba), zero_row), dim=1)
            ba = torch.cat((ba, zero_col), dim=2)
            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], zero_feat, bf[:, -1:, :]), dim=1)
            lbl_patch = torch.cat((torch.ones(cur), torch.zeros(cur * args.negsamp_patch))).unsqueeze(1).to(device)
            lbl_context = torch.cat((torch.ones(cur), torch.zeros(cur * args.negsamp_context))).unsqueeze(1).to(device)
            opt.zero_grad(set_to_none=True)
            if gradate:
                ba_hat = torch.cat((torch.cat(ba_hat), zero_row), dim=1)
                ba_hat = torch.cat((ba_hat, zero_col), dim=2)
                l1, l2, sub, _ = unpack_model_outputs(model(bf, ba))
                l1h, l2h, subh, _ = unpack_model_outputs(model(bf, ba_hat))
                sub = F.normalize(sub, p=2, dim=1)
                subh = F.normalize(subh, p=2, dim=1)
                nce = F.cross_entropy((sub @ subh.t()).clamp(min=-20.0, max=20.0), torch.arange(cur, device=device))
                base_loss = args.beta * (args.alpha * bce_context(l1, lbl_context) + (1 - args.alpha) * bce_context(l1h, lbl_context))
                base_loss = base_loss + (1 - args.beta) * (args.alpha * bce_patch(l2, lbl_patch) + (1 - args.alpha) * bce_patch(l2h, lbl_patch)) + 0.1 * nce
            else:
                l1, l2 = unpack_model_outputs(model(bf, ba))[:2]
                base_loss = args.alpha * bce_context(l1, lbl_context) + (1 - args.alpha) * bce_patch(l2, lbl_patch)
            gal_nodes = official_gal.extend_nodes(idx, args.gal_pairs)
            gal_bf, gal_ba, _ = make_model_batch(
                gal_nodes,
                subgraphs,
                adj,
                features,
                x.shape[1],
                args.subgraph_size,
                device,
            )
            gal, gal_mean = official_gal.anomaly_loss(anchor_embeddings(model, gal_bf, gal_ba), gal_nodes)
            loss = base_loss + args.gal_weight * gal
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu()) * cur
            total_base += float(base_loss.detach().cpu()) * cur
            total_gal += float(gal.detach().cpu()) * cur
            seen += cur
        mean_loss = total_loss / max(seen, 1)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_base = total_base / max(seen, 1)
            best_gal = total_gal / max(seen, 1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{args.dataset} {args.model} epoch {epoch} loss={mean_loss:.6f} base={total_base/max(seen,1):.6f} gal={total_gal/max(seen,1):.6f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    scores = run_batches(model, adj, features, graph, args, device, gradate, adj_hat, args.seed + 10000)
    metrics = evaluate_scores(labels_np, scores, threshold=0.5)
    metrics.update({
        "model": f"{args.model}_gal",
        "base_model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "feature_dim": int(x.shape[1]),
        "epochs": args.epochs,
        "embedding_dim": args.embedding_dim,
        "lr": args.lr,
        "alpha": args.alpha,
        "beta": args.beta,
        "test_rounds": args.test_rounds,
        "negsamp_patch": args.negsamp_patch,
        "negsamp_context": args.negsamp_context,
        "gal_weight": args.gal_weight,
        "gal_margin_scale": args.gal_margin_scale,
        "gal_pairs": args.gal_pairs,
        "gal_implementation": "zhao-tong/Graph-Anomaly-Loss src.models.UnsupervisedLoss.get_loss_anomaly",
        "best_loss": best_loss,
        "best_base_loss": best_base,
        "best_gal_loss": best_gal,
        "runtime_seconds": time.perf_counter() - start,
        "best_config": row,
    })
    metrics.update(model_size_from_object(model))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, choices=["anemone", "gradate"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--test-rounds", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--subgraph-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--negsamp-patch", type=int, default=1)
    parser.add_argument("--negsamp-context", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gal-weight", type=float, default=0.1)
    parser.add_argument("--gal-margin-scale", type=float, default=20.0)
    parser.add_argument("--gal-pairs", type=int, default=50)
    parser.add_argument("--limit-epochs", type=int)
    parser.add_argument("--limit-test-rounds", type=int)
    parser.add_argument("--force", action="store_true")
    run_one(parser.parse_args())


if __name__ == "__main__":
    main()
