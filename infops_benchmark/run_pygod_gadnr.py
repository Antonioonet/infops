import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from pygod.detector import GADNR
from torch_geometric.nn import GCN, GIN, GraphSAGE

from common import evaluate_scores, model_size_from_object, pyg_data, set_seed, write_result


def pyg_backbone(backbone_cls):
    def build_backbone(**kwargs):
        # PyGOD 1.1.0 forwards this GADNRBase-only value to PyG's backbone.
        kwargs.pop("tot_nodes", None)
        return backbone_cls(**kwargs)

    build_backbone.__name__ = backbone_cls.__name__
    return build_backbone


BACKBONES = {
    "GCN": pyg_backbone(GCN),
    "GraphSAGE": pyg_backbone(GraphSAGE),
    "GIN": pyg_backbone(GIN),
}


def cpu_state_dict(module):
    if module is None or not hasattr(module, "state_dict"):
        return None
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def positive_contamination(labels: np.ndarray) -> float:
    return min(max(float(labels.mean()), 1.0 / labels.shape[0]), 0.5)


def tensor_to_numpy(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def run_gadnr(args):
    if isinstance(args, dict):
        args = SimpleNamespace(**args)
    set_seed(args.seed)
    data = pyg_data(args.dataset, args.graph_key)
    labels = data.y.cpu().numpy()
    if args.gpu >= 0:
        data = data.to(f"cuda:{args.gpu}")

    model = GADNR(
        hid_dim=args.hid_dim,
        num_layers=args.num_layers,
        deg_dec_layers=args.deg_dec_layers,
        fea_dec_layers=args.fea_dec_layers,
        backbone=BACKBONES[args.backbone],
        sample_size=args.sample_size,
        sample_time=args.sample_time,
        neigh_loss=args.neigh_loss,
        lambda_loss1=args.lambda_n,
        lambda_loss2=args.lambda_x,
        lambda_loss3=args.lambda_d,
        real_loss=args.real_loss,
        lr=args.lr,
        epoch=args.epochs,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        gpu=args.gpu,
        batch_size=args.batch_size,
        num_neigh=args.num_neigh,
        contamination=positive_contamination(labels),
        verbose=args.verbose,
    )

    start = time.perf_counter()
    model.fit(
        data,
        label=data.y,
        h_loss_weight=args.lambda_n_prime,
        degree_loss_weight=args.lambda_d_prime,
        feature_loss_weight=args.lambda_x_prime,
    )
    runtime_seconds = time.perf_counter() - start

    scores = tensor_to_numpy(model.decision_score_)
    metrics = evaluate_scores(labels, scores, threshold=0.5)
    payload = {
        **metrics,
        "model": "pygod_gadnr",
        "dataset": args.dataset,
        "graph_key": args.graph_key,
        "feature_dim": int(data.x.shape[1]),
        "edge_count": int(data.edge_index.shape[1]),
        "seed": args.seed,
        "runtime_seconds": runtime_seconds,
        "hyperparameters": {
            "epochs": args.epochs,
            "hid_dim": args.hid_dim,
            "num_layers": args.num_layers,
            "deg_dec_layers": args.deg_dec_layers,
            "fea_dec_layers": args.fea_dec_layers,
            "backbone": args.backbone,
            "sample_size": args.sample_size,
            "sample_time": args.sample_time,
            "neigh_loss": args.neigh_loss,
            "lambda_x": args.lambda_x,
            "lambda_d": args.lambda_d,
            "lambda_n": args.lambda_n,
            "lambda_x_prime": args.lambda_x_prime,
            "lambda_d_prime": args.lambda_d_prime,
            "lambda_n_prime": args.lambda_n_prime,
            "real_loss": args.real_loss,
            "lr": args.lr,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "num_neigh": args.num_neigh,
            "gpu": args.gpu,
        },
        "pygod_parameter_mapping": {
            "lambda_loss1": "lambda_n",
            "lambda_loss2": "lambda_x",
            "lambda_loss3": "lambda_d",
            "h_loss_weight": "lambda_n_prime",
            "degree_loss_weight": "lambda_d_prime",
            "feature_loss_weight": "lambda_x_prime",
        },
    }
    payload.update(model_size_from_object(model))
    if getattr(args, "save_dir", None):
        save_root = Path(args.save_dir) / args.dataset / "gadnr"
        save_root.mkdir(parents=True, exist_ok=True)
        scores_tensor = model.decision_score_
        if torch.is_tensor(scores_tensor):
            scores_tensor = scores_tensor.detach().cpu()
        ckpt_path = save_root / "model.pt"
        torch.save({
            "model": "gadnr",
            "dataset": args.dataset,
            "detector_class": model.__class__.__name__,
            "state_dict": cpu_state_dict(getattr(model, "model", None)),
            "metrics": payload,
            "hyperparameters": payload["hyperparameters"],
            "threshold_": float(getattr(model, "threshold_", float("nan"))),
            "decision_score_": scores_tensor,
            "feature_dim": int(data.x.shape[1]),
            "num_nodes": int(data.x.shape[0]),
            "num_edges": int(data.edge_index.shape[1]),
        }, ckpt_path)
        print(f"Saved checkpoint {ckpt_path}", flush=True)
    return payload


def write_gadnr_payload(payload, result_name):
    write_result(result_name, payload, payload["dataset"])

    flat_path = Path(__file__).resolve().parents[1] / "benchmark_results" / payload["dataset"] / f"{result_name}_flat.json"
    flat_payload = {
        key: value for key, value in payload.items()
        if not isinstance(value, (dict, list))
    }
    flat_payload.update(payload["hyperparameters"])
    flat_path.write_text(json.dumps(flat_payload, indent=2, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="russia")
    parser.add_argument("--graph-key", default="graph")
    parser.add_argument("--result-name", default="pygod_gadnr")
    parser.add_argument("--seed", type=int, default=12121995)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hid-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--deg-dec-layers", type=int, default=4)
    parser.add_argument("--fea-dec-layers", type=int, default=3)
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="GCN")
    parser.add_argument("--sample-size", type=int, default=2)
    parser.add_argument("--sample-time", type=int, default=3)
    parser.add_argument("--neigh-loss", choices=["KL", "W2"], default="KL")
    parser.add_argument("--lambda-x", type=float, default=0.8)
    parser.add_argument("--lambda-d", type=float, default=0.5)
    parser.add_argument("--lambda-n", type=float, default=0.001)
    parser.add_argument("--lambda-x-prime", type=float, default=1.0)
    parser.add_argument("--lambda-d-prime", type=float, default=1.0)
    parser.add_argument("--lambda-n-prime", type=float, default=1.0)
    parser.add_argument("--real-loss", action="store_true")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0003)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-neigh", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--save-dir")
    return parser


def main():
    args = build_parser().parse_args()
    payload = run_gadnr(args)
    write_gadnr_payload(payload, args.result_name)


if __name__ == "__main__":
    main()
