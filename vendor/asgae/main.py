import os.path as osp
import matplotlib.pyplot as plt
import torch
import torch_geometric.transforms as T
import numpy as np
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU
import argparse
import os, sys
from tqdm import tqdm
import time
import pickle as pkl
import copy
import networkx as nx
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.utils import subgraph
from torch.autograd import Variable
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter
from networkx.algorithms.components import strongly_connected_components
from torch_geometric.transforms import NormalizeFeatures
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch_geometric.utils import (
    negative_sampling,
    remove_self_loops,
    add_self_loops,
)
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import itertools
from model.encoder import *
from model.decoder import *
from model.model import *
from model.smgnn import *


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_node_features(x: torch.Tensor) -> torch.Tensor:
    """Stable normalization for dense SBERT-like node features."""
    x = x.float()
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return F.normalize(x, p=2, dim=1)


def add_anchors(data, args):
    """
    Stable version of anchor features.

    Original repo used raw shortest-path distances. For large graphs, those
    distances can be much larger than SBERT values and can dominate training.
    Here we normalize anchor distances to [0, 1].
    """
    seed = args.random_seed
    num_anchors = args.num_anchors

    if num_anchors <= 0:
        return data

    if num_anchors > data.num_nodes:
        raise ValueError(
            f"num_anchors={num_anchors} cannot be larger than num_nodes={data.num_nodes}"
        )

    rng = np.random.default_rng(seed)
    G = to_networkx(data, to_undirected=True)
    anchors = rng.choice(data.num_nodes, num_anchors, replace=False)

    # sqrt(N) is safer than N / num_anchors for large graphs.
    cutoff = max(2, int(np.sqrt(data.num_nodes)))
    default_dist = cutoff + 1

    dist_to_anchors = []

    for anchor in anchors:
        length = nx.single_source_shortest_path_length(
            G,
            int(anchor),
            cutoff=cutoff,
        )

        dist = np.full(data.num_nodes, default_dist, dtype=np.float32)

        for node, d in length.items():
            dist[node] = d

        dist_to_anchors.append(dist)

    dist_to_anchors = np.stack(dist_to_anchors, axis=1)

    # Normalize anchor distances to [0, 1].
    dist_to_anchors = dist_to_anchors / float(default_dist)

    anchor_x = torch.tensor(
        dist_to_anchors,
        dtype=torch.float,
        device=data.x.device,
    )

    data.x = torch.cat([data.x, anchor_x], dim=-1)
    return data


def load_structure_synthetic(args):
    n = args.size
    n_anomaly = int(n * args.anomaly_ratio)

    DATA_DIR = args.data_dir
    dataset = args.dataset
    anomaly_type = args.anomaly_type
    size = args.size
    FILE_NAME = os.path.join(
        DATA_DIR,
        'structure_anomaly',
        dataset,
        anomaly_type,
        f"size_{size}.pkl",
    )
    with open(FILE_NAME, 'rb') as file:
        data = pkl.load(file)

    anomaly_flag = np.array([False] * (data.num_nodes - n_anomaly) + [True] * n_anomaly)
    return data, anomaly_flag


def load_attribute_synthetic(args):
    DATA_DIR = args.data_dir
    FILE_NAME = f"{args.size}_{args.dim}_{args.anomaly_attr_ratio}_{args.diff_ratio}.pkl"
    FILE_PATH = os.path.join(DATA_DIR, 'attribute_anomaly', FILE_NAME)

    with open(FILE_PATH, 'rb') as file:
        data = pkl.load(file)

    anomaly_flag = data.anomaly_flag.numpy()
    return data, anomaly_flag


def load_material(args):
    DATA_DIR = args.data_dir
    FILE_NAME = f"{args.half_num}.pkl"
    FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

    with open(FILE_PATH, 'rb') as file:
        data = pkl.load(file)

    anomaly_flag = data.anomaly_flag.numpy()
    return data, anomaly_flag


def load_real_world(args):
    DATA_DIR = args.data_dir
    FILE_NAME = f"{args.real_world_name}.pkl"
    FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

    with open(FILE_PATH, 'rb') as file:
        data = pkl.load(file)

    anomaly_flag = data.anomaly_flag.numpy()
    return data, anomaly_flag


def recon_forward(model, data):
    model.train()
    x = data.x
    edge_index = data.edge_index
    z = model.encode(x, edge_index)
    pos_loss, neg_loss, attr_loss = model.recon_loss(z, edge_index, x=x)
    return pos_loss, neg_loss, attr_loss


@torch.no_grad()
def compute_structure_error_chunked(
    model,
    z,
    edge_index,
    num_nodes,
    device,
    chunk_size=256,
):
    """
    Computes the same idea as:

        abs((real_adj - pred_adj).mean(axis=1))

    but without materializing a dense N x N adjacency matrix.
    """
    edge_index_clean, _ = remove_self_loops(edge_index)
    edge_index_clean, _ = add_self_loops(edge_index_clean, num_nodes=num_nodes)

    real_degree = torch.bincount(
        edge_index_clean[0],
        minlength=num_nodes,
    ).float().to(device)

    real_row_mean = real_degree / float(num_nodes)
    all_cols = torch.arange(num_nodes, device=device)
    errors = []

    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        rows = torch.arange(start, end, device=device)

        row_index = rows.repeat_interleave(num_nodes)
        col_index = all_cols.repeat(end - start)
        full_edge_index = torch.stack([row_index, col_index], dim=0)

        pred_struct, _ = model.decoder(z, full_edge_index)
        pred_struct = pred_struct.view(end - start, num_nodes)
        pred_row_mean = pred_struct.mean(dim=1)

        error = torch.abs(real_row_mean[start:end] - pred_row_mean)
        errors.append(error.detach().cpu())

    return torch.cat(errors, dim=0).numpy()


@torch.no_grad()
def compute_feature_error(model, z, data):
    """Stable feature reconstruction error for anomaly scoring."""
    _, pred_attr = model.decoder(z, data.edge_index)

    # SmoothL1 is less sensitive to extreme reconstruction errors than MSE.
    feature_error = F.smooth_l1_loss(
        pred_attr,
        data.x,
        reduction='none',
    ).mean(dim=1)

    return feature_error.detach().cpu().numpy()


def tensor_to_device_long(x, device):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.long)
    return torch.tensor(x, device=device, dtype=torch.long)


def build_continuous_predicted_value(
    data,
    anomaly_scores,
    residual_data,
    anomaly_nodes_index,
    device,
):
    """Converts subgraph scores into one score per node."""
    continuous_predicted_value = torch.zeros(data.num_nodes, device=device)

    if not hasattr(residual_data, 'batch'):
        return continuous_predicted_value, False

    batch = residual_data.batch
    if batch is None or batch.numel() == 0:
        return continuous_predicted_value, False

    batch = batch.to(device)
    anomaly_nodes_index = tensor_to_device_long(anomaly_nodes_index, device)
    scores = anomaly_scores.view(-1).to(device)

    num_groups = int(batch.max().item()) + 1

    for i in range(num_groups):
        if i >= scores.numel():
            break

        mask = batch == i
        if mask.sum().item() == 0:
            continue

        continuous_predicted_value[anomaly_nodes_index[mask]] = scores[i]

    return continuous_predicted_value, True


def freeze_model_except_r(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.r.requires_grad_(True)


def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad_(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_flag', type=str, default='structure_anomaly')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--real_world_name', type=str, default='email')
    parser.add_argument('--dataset', type=str, default='random')
    parser.add_argument('--anomaly_type', type=str, default='chain')
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--anomaly_ratio', type=float, default=0.02)
    parser.add_argument('--dim', type=int, default=50)
    parser.add_argument('--anomaly_scale', type=float, default=0.3)
    parser.add_argument('--anomaly_attr_ratio', type=float, default=0.2)
    parser.add_argument('--diff_ratio', type=int, default=5)
    parser.add_argument('--half_num', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--num_anchors', type=int, default=100)
    parser.add_argument('--embedding_channels', type=int, default=64)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--sp_epochs', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=90)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--sp_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--beta_1', type=float, default=1.0)
    parser.add_argument('--beta_2', type=float, default=1e-3)
    parser.add_argument('--dynamic_beta_2', action='store_true')
    parser.add_argument('--min_beta_2', type=float, default=1e-4)
    parser.add_argument('--max_beta_2', type=float, default=1e-2)
    parser.add_argument('--q', type=int, default=85)
    parser.add_argument('--structure_chunk_size', type=int, default=256)
    parser.add_argument('--convergence', type=float, default=1e-4)
    parser.add_argument('--ending_rounds', type=int, default=1)
    args = parser.parse_args()

    set_seed(args.random_seed)

    # Load data.
    if args.data_flag == 'structure_anomaly':
        data, anomaly_flag = load_structure_synthetic(args)
        RESULT_DIR = os.path.join(
            args.results_dir,
            'structure_synthetic',
            args.dataset,
            args.anomaly_type,
        )
        RESULT_FILE_NAME = (
            f"{args.dataset}_{args.anomaly_type}_{args.size}_"
            f"{args.embedding_channels}_{args.num_anchors}.txt"
        )
    elif args.data_flag == 'attribute_anomaly':
        data, anomaly_flag = load_attribute_synthetic(args)
        RESULT_DIR = os.path.join(
            args.results_dir,
            str(args.dim),
            str(args.anomaly_attr_ratio),
            str(args.diff_ratio),
        )
        RESULT_FILE_NAME = (
            f"{args.dim}_{args.anomaly_attr_ratio}_{args.diff_ratio}_"
            f"{args.size}_{args.embedding_channels}_{args.num_anchors}.txt"
        )
    elif args.data_flag == 'material':
        data, anomaly_flag = load_material(args)
        RESULT_DIR = os.path.join(args.results_dir, str(args.half_num))
        RESULT_FILE_NAME = (
            f"{args.half_num}_{args.size}_{args.embedding_channels}_{args.num_anchors}.txt"
        )
    elif args.data_flag == 'real_world':
        data, anomaly_flag = load_real_world(args)
        RESULT_DIR = os.path.join(args.results_dir, str(args.real_world_name))
        RESULT_FILE_NAME = (
            f"{args.real_world_name}_{args.embedding_channels}_{args.num_anchors}.txt"
        )
    else:
        raise ValueError(f"Unknown data_flag: {args.data_flag}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    result_path = os.path.join(RESULT_DIR, RESULT_FILE_NAME)

    anomaly_flag = np.asarray(anomaly_flag).astype(int)

    # Normalize original node features first.
    data.x = normalize_node_features(data.x)

    # Add normalized anchor features.
    if args.num_anchors > 0:
        data = add_anchors(data, args)

    # Normalize again after concatenating anchor features.
    data.x = normalize_node_features(data.x)

    # Model parameters.
    out_channels = data.num_features
    num_features = data.num_features
    embedding_channels = args.embedding_channels
    hidden_channels = args.hidden_channels
    num_layer = args.num_layers - 2

    innerproduct_decoder = InnerProductDecoder(
        embedding_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
    )

    model = GAE(
        encoder=GCNEncoder(
            num_features,
            hidden_channels,
            embedding_channels,
            num_layers=num_layer,
            aggr='add',
        ),
        decoder=innerproduct_decoder,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # The second phase should optimize only model.r.
    optimizer_sp = torch.optim.AdamW(
        [model.r],
        lr=args.sp_learning_rate,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
    )

    scheduler_sp = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_sp,
        mode='min',
        factor=0.5,
        patience=5,
    )

    prev_loss = 1e12
    flag = True
    accumulated_rounds = 1
    continuous_predicted_value = torch.zeros(data.num_nodes, device=device)

    with open(result_path, 'a') as f:
        while flag:
            print(f'Iteration #{accumulated_rounds:2d}')

            # Phase 1: graph autoencoder training.
            last_loss = None
            last_beta_2 = args.beta_2

            for epoch in range(1, args.epochs + 1):
                optimizer.zero_grad()

                pos_loss, neg_loss, attr_loss = recon_forward(model, data)
                structure_loss = pos_loss + neg_loss
                feature_loss = attr_loss.mean()

                if args.dynamic_beta_2:
                    with torch.no_grad():
                        beta_2_epoch = structure_loss.detach() / (feature_loss.detach() + 1e-8)
                        beta_2_epoch = beta_2_epoch.clamp(args.min_beta_2, args.max_beta_2)
                else:
                    beta_2_epoch = torch.tensor(args.beta_2, device=device)

                loss = args.beta_1 * structure_loss + beta_2_epoch * feature_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    print('Bad loss detected. Stopping phase 1.')
                    print('structure_loss:', structure_loss.item())
                    print('feature_loss:', feature_loss.item())
                    print('beta_2:', float(beta_2_epoch))
                    break

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.grad_clip,
                )

                optimizer.step()
                scheduler.step(loss.item())

                last_loss = loss.item()
                last_beta_2 = float(beta_2_epoch)

                if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                    print(
                        f'Epoch:{epoch:3d} | '
                        f'loss:{loss.item():.6f} | '
                        f'struct:{structure_loss.item():.6f} | '
                        f'feat:{feature_loss.item():.6f} | '
                        f'beta2:{float(beta_2_epoch):.6g} | '
                        f'grad:{float(grad_norm):.4f} | '
                        f'lr:{optimizer.param_groups[0]["lr"]:.2e}'
                    )

            if last_loss is None:
                print('No valid training loss was produced. Exiting.')
                break

            denom = abs(last_loss) + 1e-12
            if abs(last_loss - prev_loss) / denom < args.convergence or accumulated_rounds >= args.ending_rounds:
                flag = False
            else:
                prev_loss = last_loss

            # Phase 2: supermodular scoring / radius optimization.
            freeze_model_except_r(model)

            for sp_epoch in range(1, args.sp_epochs + 1):
                optimizer_sp.zero_grad()

                with torch.no_grad():
                    z = model.encode(data.x, data.edge_index)

                    mean_error_struct = compute_structure_error_chunked(
                        model=model,
                        z=z,
                        edge_index=data.edge_index,
                        num_nodes=data.num_nodes,
                        device=device,
                        chunk_size=args.structure_chunk_size,
                    )

                    mean_error_feature = compute_feature_error(model, z, data)

                    total_error = (
                        args.beta_1 * mean_error_struct
                        + last_beta_2 * mean_error_feature
                    )

                    threshold = np.percentile(total_error, q=args.q)
                    selected = total_error > threshold

                    if selected.sum() == 0:
                        print(
                            f'No residual nodes selected at q={args.q}. '
                            'Try a smaller q, for example 80.'
                        )
                        continue

                    smgnn = SuperModularModel(threshold=threshold)
                    anomaly_scores, residual_data, anomaly_nodes_index = smgnn(data, total_error)

                    continuous_predicted_value, valid_residual = build_continuous_predicted_value(
                        data=data,
                        anomaly_scores=anomaly_scores,
                        residual_data=residual_data,
                        anomaly_nodes_index=anomaly_nodes_index,
                        device=device,
                    )

                    if not valid_residual:
                        print(
                            f'Empty residual graph at q={args.q}. '
                            'Skipping this SP epoch. Try q=80 or q=85.'
                        )
                        continue

                sp_loss = torch.clamp(
                    model.r - continuous_predicted_value,
                    min=0.0,
                ).sum() + model.r ** 2

                if torch.isnan(sp_loss) or torch.isinf(sp_loss):
                    print('Bad sp_loss detected. Skipping this SP epoch.')
                    continue

                sp_loss.backward()

                grad_norm_sp = torch.nn.utils.clip_grad_norm_(
                    [model.r],
                    max_norm=args.grad_clip,
                )

                optimizer_sp.step()
                scheduler_sp.step(sp_loss.item())

                if sp_epoch == 1 or sp_epoch == args.sp_epochs:
                    print(
                        f'sp_epoch:{sp_epoch:3d} | '
                        f'sp_loss:{sp_loss.item():.6f} | '
                        f'r:{float(model.r.detach().cpu()):.6f} | '
                        f'grad_r:{float(grad_norm_sp):.4f} | '
                        f'sp_lr:{optimizer_sp.param_groups[0]["lr"]:.2e}'
                    )

            unfreeze_model(model)

            scores_np = continuous_predicted_value.detach().cpu().numpy()

            if len(np.unique(anomaly_flag)) < 2:
                auc = float('nan')
                ap = float('nan')
                macro_f1 = float('nan')
                print('Cannot compute ROC-AUC/AP: anomaly_flag has only one class.')
            else:
                auc = roc_auc_score(anomaly_flag, scores_np)
                ap = average_precision_score(anomaly_flag, scores_np)
                score_min = float(np.min(scores_np))
                score_max = float(np.max(scores_np))
                if score_max > score_min:
                    probs_np = (scores_np - score_min) / (score_max - score_min)
                else:
                    probs_np = np.zeros_like(scores_np)
                macro_f1 = f1_score(
                    anomaly_flag,
                    (probs_np > 0.5).astype(int),
                    average='macro',
                    zero_division=0,
                )
                print(
                    f'roc_auc_score:{auc:.3f} | '
                    f'average_precision:{ap:.3f} | '
                    f'macro_f1:{macro_f1:.3f}'
                )

            f.write(
                f'round={accumulated_rounds}, '
                f'loss={last_loss:.6f}, '
                f'beta2={last_beta_2:.6g}, '
                f'auc={auc:.6f}, '
                f'ap={ap:.6f}, '
                f'macro_f1={macro_f1:.6f}\n'
            )
            f.flush()

            accumulated_rounds += 1

    torch.save(model.state_dict(), 'model.pt')
