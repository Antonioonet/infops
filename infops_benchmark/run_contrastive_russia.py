import argparse
import importlib.util
import time
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import evaluate_scores, load_dataset, model_size_from_object, node_features, set_seed, sparse_adj_from_graph, write_result


ROOT = Path(__file__).resolve().parents[1]


def load_model_class(model_dir: str):
    spec = importlib.util.spec_from_file_location(f"{model_dir}_model", ROOT / model_dir / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


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
    kept = kept + kept.T
    return kept.tocsr()


def sample_subgraphs(graph: nx.Graph, subgraph_size: int, seed: int):
    rng = random_state = np.random.default_rng(seed)
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
            picked.append(int(random_state.choice(picked)))
        picked = picked[:reduced_size]
        picked.append(node)
        subgraphs.append(picked)
    return subgraphs


def score_batches(model, adj, features, graph, args, device, gradate=False, adj_hat=None, seed=0):
    nb_nodes = features.shape[1]
    ft_size = features.shape[2]
    batch_num = nb_nodes // args.batch_size + 1
    multi = np.zeros((args.test_rounds, nb_nodes), dtype=np.float64)

    for round_id in range(args.test_rounds):
        all_idx = list(range(nb_nodes))
        np.random.default_rng(seed + round_id).shuffle(all_idx)
        subgraphs = sample_subgraphs(graph, args.subgraph_size, seed + round_id)
        for batch_idx in range(batch_num):
            idx = all_idx[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size]
            if not idx:
                continue
            cur_batch_size = len(idx)
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, args.subgraph_size), device=device)
            added_adj_zero_col = torch.zeros((cur_batch_size, args.subgraph_size + 1, 1), device=device)
            added_adj_zero_col[:, -1, :] = 1.0
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size), device=device)
            ba, bf, ba_hat = [], [], []
            for i in idx:
                sg = subgraphs[i]
                ba.append(adj[:, sg, :][:, :, sg])
                bf.append(features[:, sg, :])
                if gradate:
                    ba_hat.append(adj_hat[:, sg, :][:, :, sg])
            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
            with torch.no_grad():
                if gradate:
                    ba_hat = torch.cat(ba_hat)
                    ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
                    ba_hat = torch.cat((ba_hat, added_adj_zero_col), dim=2)
                    l1, l2, _, _ = model(bf, ba)
                    l1h, l2h, _, _ = model(bf, ba_hat)
                    l1, l2, l1h, l2h = [torch.sigmoid(torch.squeeze(t)) for t in (l1, l2, l1h, l2h)]
                    s1 = -(l1[:cur_batch_size] - l1[cur_batch_size:].view(cur_batch_size, args.negsamp_context).mean(1))
                    s1h = -(l1h[:cur_batch_size] - l1h[cur_batch_size:].view(cur_batch_size, args.negsamp_context).mean(1))
                    s2 = -(l2[:cur_batch_size] - l2[cur_batch_size:].view(cur_batch_size, args.negsamp_patch).mean(1))
                    s2h = -(l2h[:cur_batch_size] - l2h[cur_batch_size:].view(cur_batch_size, args.negsamp_patch).mean(1))
                    score = args.beta * (args.alpha * s1 + (1 - args.alpha) * s1h) + (1 - args.beta) * (args.alpha * s2 + (1 - args.alpha) * s2h)
                else:
                    l1, l2 = model(bf, ba)
                    l1 = torch.sigmoid(torch.squeeze(l1))
                    l2 = torch.sigmoid(torch.squeeze(l2))
                    s1 = -(l1[:cur_batch_size] - l1[cur_batch_size:].view(cur_batch_size, args.negsamp_context).mean(1))
                    s2 = -(l2[:cur_batch_size] - l2[cur_batch_size:].view(cur_batch_size, args.negsamp_patch).mean(1))
                    score = args.alpha * s1 + (1 - args.alpha) * s2
            multi[round_id, idx] = score.detach().cpu().numpy()
    return multi.mean(axis=0) + multi.std(axis=0)


def run_contrastive(kind: str, args):
    run_start = time.perf_counter()
    set_seed(args.seed)
    _, graph, labels, sbert = load_dataset(args.dataset, "graph")
    x = node_features(graph, sbert).numpy().astype(np.float32)
    adj_sp = sparse_adj_from_graph(graph)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    adj = dense_norm_adj(adj_sp).unsqueeze(0).to(device)
    features = torch.from_numpy(x).unsqueeze(0).to(device)

    gradate = kind == "GRADATE"
    model_cls = load_model_class(kind)
    model = model_cls(x.shape[1], args.embedding_dim, "prelu", args.negsamp_patch, args.negsamp_context, "avg").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce_patch = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.negsamp_patch], device=device))
    bce_context = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.negsamp_context], device=device))
    adj_hat = dense_norm_adj(augment_random_edge(adj_sp, seed=args.seed)).unsqueeze(0).to(device) if gradate else None
    best_state = None
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        all_idx = list(range(graph.number_of_nodes()))
        np.random.default_rng(args.seed + epoch).shuffle(all_idx)
        subgraphs = sample_subgraphs(graph, args.subgraph_size, args.seed + epoch)
        for batch_start in range(0, len(all_idx), args.batch_size):
            idx = all_idx[batch_start:batch_start + args.batch_size]
            cur = len(idx)
            added_adj_zero_row = torch.zeros((cur, 1, args.subgraph_size), device=device)
            added_adj_zero_col = torch.zeros((cur, args.subgraph_size + 1, 1), device=device)
            added_adj_zero_col[:, -1, :] = 1.0
            added_feat_zero_row = torch.zeros((cur, 1, x.shape[1]), device=device)
            ba, bf, ba_hat = [], [], []
            for i in idx:
                sg = subgraphs[i]
                ba.append(adj[:, sg, :][:, :, sg])
                bf.append(features[:, sg, :])
                if gradate:
                    ba_hat.append(adj_hat[:, sg, :][:, :, sg])
            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
            lbl_patch = torch.cat((torch.ones(cur), torch.zeros(cur * args.negsamp_patch))).unsqueeze(1).to(device)
            lbl_context = torch.cat((torch.ones(cur), torch.zeros(cur * args.negsamp_context))).unsqueeze(1).to(device)
            opt.zero_grad(set_to_none=True)
            if gradate:
                ba_hat = torch.cat(ba_hat)
                ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
                ba_hat = torch.cat((ba_hat, added_adj_zero_col), dim=2)
                l1, l2, sub, _ = model(bf, ba)
                l1h, l2h, subh, _ = model(bf, ba_hat)
                sub = F.normalize(sub, p=2, dim=1)
                subh = F.normalize(subh, p=2, dim=1)
                logits = (sub @ subh.t()).clamp(min=-20.0, max=20.0)
                target = torch.arange(cur, device=device)
                nce = F.cross_entropy(logits, target)
                loss = args.beta * (args.alpha * bce_context(l1, lbl_context) + (1 - args.alpha) * bce_context(l1h, lbl_context))
                loss = loss + (1 - args.beta) * (args.alpha * bce_patch(l2, lbl_patch) + (1 - args.alpha) * bce_patch(l2h, lbl_patch)) + 0.1 * nce
            else:
                l1, l2 = model(bf, ba)
                loss = args.alpha * bce_context(l1, lbl_context) + (1 - args.alpha) * bce_patch(l2, lbl_patch)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu()) * cur
            seen += cur
        mean_loss = total_loss / max(seen, 1)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{kind} epoch {epoch} loss={mean_loss:.6f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    scores = score_batches(model, adj, features, graph, args, device, gradate, adj_hat, args.seed + 10000)
    runtime_seconds = time.perf_counter() - run_start
    metrics = evaluate_scores(labels, scores, threshold=0.5)
    metrics.update({
        "model": kind.lower(),
        "dataset": args.dataset,
        "feature_dim": int(x.shape[1]),
        "best_loss": best_loss,
        "epochs": args.epochs,
        "embedding_dim": args.embedding_dim,
        "lr": args.lr,
        "alpha": args.alpha,
        "beta": args.beta,
        "test_rounds": args.test_rounds,
        "runtime_seconds": runtime_seconds,
    })
    metrics.update(model_size_from_object(model))
    if args.save_dir:
        save_root = Path(args.save_dir) / args.dataset / kind.lower()
        save_root.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_root / "model.pt"
        torch.save({
            "model": kind.lower(),
            "dataset": args.dataset,
            "model_class": kind,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "best_state_dict": best_state,
            "metrics": metrics,
            "hyperparameters": vars(args),
            "decision_score_": torch.from_numpy(np.asarray(scores, dtype=np.float64)),
            "feature_dim": int(x.shape[1]),
            "num_nodes": int(graph.number_of_nodes()),
            "num_edges": int(graph.number_of_edges()),
        }, ckpt_path)
        print(f"Saved checkpoint {ckpt_path}", flush=True)
    write_result(kind.lower(), metrics, args.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["ANEMONE", "GRADATE"])
    parser.add_argument("--dataset", default="russia")
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
    parser.add_argument("--seed", type=int, default=12121995)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-dir")
    args = parser.parse_args()
    summary = {}
    for model in args.models:
        summary[model.lower()] = run_contrastive(model, args)
    write_result("contrastive_summary", summary, args.dataset)


if __name__ == "__main__":
    main()
