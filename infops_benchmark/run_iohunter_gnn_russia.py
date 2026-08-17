import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from common import edge_index_from_graph, load_dataset, model_size_from_object, node_features, set_seed, write_result


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "iohunter" / "src"))
from models import GNN  # noqa: E402


def eval_prob(labels, prob, mask):
    y = labels[mask]
    p = prob[mask]
    pred = (p > 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(y, p)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "num_nodes": int(y.shape[0]),
        "num_anomalies": int(y.sum()),
    }


def run_one(gnn_type, args):
    start = time.perf_counter()
    dataset, graph, labels_np, sbert = load_dataset(args.dataset, "graph")
    labels = torch.from_numpy(labels_np.astype(np.float32))
    x = node_features(graph, sbert)
    edge_index = edge_index_from_graph(graph)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    x = x.to(device)
    edge_index = edge_index.to(device)
    labels_t = labels.to(device)

    split_metrics = []
    model_size = None
    for split_id, split in dataset["splits"].items():
        split_start = time.perf_counter()
        set_seed(args.seed + int(split_id))
        train_mask = torch.from_numpy(split["train"].astype(bool)).to(device)
        val_mask_np = split["val"].astype(bool)
        test_mask_np = split["test"].astype(bool)
        val_mask = torch.from_numpy(val_mask_np).to(device)

        model = GNN(x.shape[1], args.hidden_dim, 2, dropout_p=args.dropout, gnn_type=gnn_type).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = torch.nn.BCELoss()
        best_state = None
        best_val = -1.0
        patience = 0

        for epoch in range(args.epochs):
            model.train()
            opt.zero_grad(set_to_none=True)
            prob = model(x, edge_index).flatten()
            loss = loss_fn(prob[train_mask], labels_t[train_mask])
            loss.backward()
            opt.step()

            if epoch % args.check_every == 0:
                model.eval()
                with torch.no_grad():
                    prob_np = model(x, edge_index).flatten().detach().cpu().numpy()
                val_auc = roc_auc_score(labels_np[val_mask_np], prob_np[val_mask_np])
                if val_auc > best_val:
                    best_val = val_auc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience >= args.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            prob_np = model(x, edge_index).flatten().detach().cpu().numpy()
        metrics = eval_prob(labels_np, prob_np, test_mask_np)
        metrics["split"] = int(split_id)
        metrics["runtime_seconds"] = time.perf_counter() - split_start
        split_metrics.append(metrics)
        if model_size is None:
            model_size = model_size_from_object(model)
        print(f"{gnn_type} split {split_id}: {metrics}", flush=True)

    result = {
        "model": f"iohunter_{gnn_type}",
        "dataset": args.dataset,
        "feature_dim": int(x.shape[1]),
        "splits": split_metrics,
        "auc": float(np.mean([m["auc"] for m in split_metrics])),
        "macro_f1": float(np.mean([m["macro_f1"] for m in split_metrics])),
        "threshold": 0.5,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "runtime_seconds": time.perf_counter() - start,
    }
    if model_size is not None:
        result.update(model_size)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gcn", "gat", "sage"])
    parser.add_argument("--dataset", default="russia")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--check-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12121995)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    summary = {}
    for model_name in args.models:
        result = run_one(model_name, args)
        write_result(result["model"], result, args.dataset)
        summary[result["model"]] = result
    write_result("iohunter_gnn_summary", summary, args.dataset)


if __name__ == "__main__":
    main()
