import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from pygod.detector import CoLA, DOMINANT
import pygod.detector.base as pygod_base

from common import evaluate_scores, model_size_from_object, pyg_data, set_seed, write_result


class FullBatchNeighborLoader:
    def __init__(self, data, num_neigh=None, batch_size=None, **kwargs):
        self.data = data

    def __iter__(self):
        data = self.data.clone()
        data.batch_size = data.x.shape[0]
        data.n_id = torch.arange(data.x.shape[0], dtype=torch.long)
        yield data

    def __len__(self):
        return 1


class PythonNeighborLoader:
    def __init__(self, data, num_neigh=None, batch_size=1024, **kwargs):
        self.data = data
        self.num_nodes = int(data.x.shape[0])
        self.batch_size = int(batch_size or self.num_nodes)
        if isinstance(num_neigh, (list, tuple)):
            num_neigh = num_neigh[0] if num_neigh else -1
        self.num_neigh = -1 if num_neigh is None else int(num_neigh)
        edge_index = data.edge_index.detach().cpu()
        self.adj = [[] for _ in range(self.num_nodes)]
        for src, dst in edge_index.t().tolist():
            self.adj[int(src)].append(int(dst))
        self._batches = None

    def __len__(self):
        return math.ceil(self.num_nodes / self.batch_size)

    def _neighbors(self, node):
        neigh = self.adj[node]
        if self.num_neigh < 0 or len(neigh) <= self.num_neigh:
            return neigh
        if self.num_neigh == 0:
            return []
        # Deterministic spread across the adjacency list, avoiding a first-k bias.
        idx = np.linspace(0, len(neigh) - 1, self.num_neigh, dtype=np.int64)
        return [neigh[int(i)] for i in idx]

    def __iter__(self):
        if self._batches is None:
            self._batches = list(self._build_batches())
        yield from self._batches

    def _build_batches(self):
        for start in range(0, self.num_nodes, self.batch_size):
            seeds = list(range(start, min(start + self.batch_size, self.num_nodes)))
            local = list(seeds)
            seen = set(local)
            for node in seeds:
                for nbr in self._neighbors(node):
                    if nbr not in seen:
                        seen.add(nbr)
                        local.append(nbr)

            local_pos = {node: idx for idx, node in enumerate(local)}
            rows, cols = [], []
            for src in local:
                src_pos = local_pos[src]
                for dst in self.adj[src]:
                    dst_pos = local_pos.get(dst)
                    if dst_pos is not None:
                        rows.append(src_pos)
                        cols.append(dst_pos)
            if rows:
                edge_index = torch.tensor([rows, cols], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            sampled = self.data.__class__(
                x=self.data.x[local],
                edge_index=edge_index,
                y=self.data.y[seeds] if hasattr(self.data, "y") else None,
            )
            sampled.batch_size = len(seeds)
            sampled.n_id = torch.tensor(local, dtype=torch.long)

            # Local seed-by-subgraph adjacency for the fallback DOMINANT
            # forward path below.
            s = torch.zeros((len(seeds), len(local)), dtype=torch.float32)
            for row_idx, node in enumerate(seeds):
                for nbr in self.adj[node]:
                    dst_pos = local_pos.get(nbr)
                    if dst_pos is not None:
                        s[row_idx, dst_pos] = 1.0
            sampled.s = s
            yield sampled


def dominant_local_forward(detector, data):
    batch_size = data.batch_size
    x = data.x.to(detector.device)
    s = data.s.to(detector.device)
    edge_index = data.edge_index.to(detector.device)
    x_, s_ = detector.model(x, edge_index)
    score = detector.model.loss_func(
        x[:batch_size],
        x_[:batch_size],
        s,
        s_[:batch_size],
        detector.weight,
    )
    loss = torch.mean(score)
    return loss, score.detach().cpu()


def cpu_state_dict(module):
    if module is None or not hasattr(module, "state_dict"):
        return None
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def save_pygod_checkpoint(save_dir, name, model, metrics, args, data):
    if not save_dir:
        return
    save_root = Path(save_dir) / args.dataset / name
    save_root.mkdir(parents=True, exist_ok=True)
    inner = getattr(model, "model", None)
    scores = getattr(model, "decision_score_", None)
    if torch.is_tensor(scores):
        scores = scores.detach().cpu()
    checkpoint = {
        "model": name,
        "dataset": args.dataset,
        "detector_class": model.__class__.__name__,
        "state_dict": cpu_state_dict(inner),
        "metrics": metrics,
        "hyperparameters": vars(args),
        "threshold_": float(getattr(model, "threshold_", float("nan"))),
        "decision_score_": scores,
        "feature_dim": int(data.x.shape[1]),
        "num_nodes": int(data.x.shape[0]),
        "num_edges": int(data.edge_index.shape[1]),
    }
    ckpt_path = save_root / "model.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint {ckpt_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hid-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-neigh", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=["dominant", "cola"], choices=["dominant", "cola"])
    parser.add_argument("--dataset", default="russia")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=12121995)
    parser.add_argument("--save-dir")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.batch_size == 0:
        pygod_base.NeighborLoader = FullBatchNeighborLoader
    else:
        pygod_base.NeighborLoader = PythonNeighborLoader
    data = pyg_data(args.dataset, "graph")
    labels = data.y.cpu().numpy()
    contamination = max(float(labels.mean()), 1.0 / labels.shape[0])

    candidates = {
        "pygod_dominant": DOMINANT(
            hid_dim=args.hid_dim,
            epoch=args.epochs,
            lr=args.lr,
            gpu=args.gpu,
            batch_size=args.batch_size,
            num_neigh=args.num_neigh,
            contamination=contamination,
            verbose=1,
        ),
        "pygod_cola": CoLA(
            hid_dim=args.hid_dim,
            epoch=args.epochs,
            lr=args.lr,
            gpu=args.gpu,
            batch_size=args.batch_size,
            num_neigh=args.num_neigh,
            contamination=contamination,
            verbose=1,
        ),
    }
    models = {f"pygod_{name}": candidates[f"pygod_{name}"] for name in args.models}
    if args.batch_size > 0:
        for name, model in models.items():
            model.process_graph = lambda data: None
            if name == "pygod_dominant":
                model.forward_model = lambda data, model=model: dominant_local_forward(model, data)

    summary = {}
    for name, model in models.items():
        print(f"\n=== {name} ===", flush=True)
        start = time.perf_counter()
        model.fit(data)
        if hasattr(model, "decision_score_"):
            scores = model.decision_score_
        else:
            scores = model.decision_function(data)
        runtime_seconds = time.perf_counter() - start
        if torch.is_tensor(scores):
            scores = scores.detach().cpu().numpy()
        metrics = evaluate_scores(labels, np.asarray(scores), threshold=0.5)
        metrics.update({
            "model": name,
            "dataset": args.dataset,
            "feature_dim": int(data.x.shape[1]),
            "epochs": args.epochs,
            "hid_dim": args.hid_dim,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "num_neigh": args.num_neigh,
            "runtime_seconds": runtime_seconds,
        })
        metrics.update(model_size_from_object(model))
        save_pygod_checkpoint(args.save_dir, name, model, metrics, args, data)
        write_result(name, metrics, args.dataset)
        summary[name] = metrics
    write_result("pygod_summary", summary, args.dataset)


if __name__ == "__main__":
    main()
