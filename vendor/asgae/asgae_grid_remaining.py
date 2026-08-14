import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


AUC_RE = re.compile(r"roc_auc_score:([0-9.]+)")
AP_RE = re.compile(r"average_precision:([0-9.]+)")
F1_RE = re.compile(r"macro_f1:([0-9.]+)")


BASE_CONFIGS = [
    {
        "name": "asgae_venezuela_best",
        "num_anchors": 125,
        "embedding_channels": 64,
        "hidden_channels": 32,
        "num_layers": 2,
        "epochs": 250,
        "learning_rate": 0.0001,
        "beta_1": 1.5,
        "beta_2": 0.001,
        "q": 85,
        "ending_rounds": 5,
    },
    {
        "name": "asgae_venezuela_sparse_top",
        "num_anchors": 250,
        "embedding_channels": 32,
        "hidden_channels": 8,
        "num_layers": 2,
        "epochs": 150,
        "learning_rate": 0.0005,
        "beta_1": 0.5,
        "beta_2": 0.001,
        "q": 85,
        "ending_rounds": 4,
    },
    {
        "name": "asgae_venezuela_anchor200_top",
        "num_anchors": 200,
        "embedding_channels": 8,
        "hidden_channels": 24,
        "num_layers": 4,
        "epochs": 200,
        "learning_rate": 0.0005,
        "beta_1": 1.5,
        "beta_2": 0.001,
        "q": 85,
        "ending_rounds": 3,
    },
    {
        "name": "asgae_china_best",
        "num_anchors": 75,
        "embedding_channels": 16,
        "hidden_channels": 16,
        "num_layers": 4,
        "epochs": 150,
        "learning_rate": 0.001,
        "beta_1": 1.25,
        "beta_2": 0.1,
        "q": 85,
        "ending_rounds": 4,
    },
    {
        "name": "asgae_china_wide_top",
        "num_anchors": 75,
        "embedding_channels": 96,
        "hidden_channels": 8,
        "num_layers": 4,
        "epochs": 150,
        "learning_rate": 0.001,
        "beta_1": 0.5,
        "beta_2": 0.00001,
        "q": 85,
        "ending_rounds": 5,
    },
    {
        "name": "asgae_china_128_top",
        "num_anchors": 75,
        "embedding_channels": 128,
        "hidden_channels": 8,
        "num_layers": 2,
        "epochs": 150,
        "learning_rate": 0.001,
        "beta_1": 1.5,
        "beta_2": 1.0,
        "q": 85,
        "ending_rounds": 4,
    },
    {
        "name": "asgae_russia_fast_top",
        "num_anchors": 150,
        "embedding_channels": 128,
        "hidden_channels": 24,
        "num_layers": 4,
        "epochs": 50,
        "learning_rate": 0.01,
        "beta_1": 1.5,
        "beta_2": 0.0001,
        "q": 92,
        "ending_rounds": 1,
    },
    {
        "name": "asgae_russia_deep_best",
        "num_anchors": 150,
        "embedding_channels": 32,
        "hidden_channels": 24,
        "num_layers": 8,
        "epochs": 200,
        "learning_rate": 0.001,
        "beta_1": 1.25,
        "beta_2": 0.0001,
        "q": 85,
        "ending_rounds": 2,
    },
    {
        "name": "asgae_russia_deep_128",
        "num_anchors": 150,
        "embedding_channels": 128,
        "hidden_channels": 24,
        "num_layers": 8,
        "epochs": 200,
        "learning_rate": 0.001,
        "beta_1": 1.25,
        "beta_2": 0.001,
        "q": 85,
        "ending_rounds": 2,
    },
    {
        "name": "asgae_russia_manual_best",
        "num_anchors": 150,
        "embedding_channels": 128,
        "hidden_channels": 8,
        "num_layers": 4,
        "epochs": 200,
        "learning_rate": 0.005,
        "beta_1": 1.25,
        "beta_2": 0.01,
        "q": 92,
        "ending_rounds": 3,
    },
    {
        "name": "asgae_china_anchor125",
        "num_anchors": 125,
        "embedding_channels": 96,
        "hidden_channels": 8,
        "num_layers": 4,
        "epochs": 150,
        "learning_rate": 0.001,
        "beta_1": 1.5,
        "beta_2": 0.00001,
        "q": 85,
        "ending_rounds": 4,
    },
]


DATASET_EXTRA = {
    "iran": [],
    "cuba": [],
    "UAE": [],
}


def parse_metric(pattern, text):
    matches = pattern.findall(text)
    return float(matches[-1]) if matches else None


def safe_name(value):
    return str(value).replace(".", "p").replace("-", "m")


def configs_for_dataset(dataset):
    seen = set()
    configs = []
    for cfg in [*DATASET_EXTRA.get(dataset, []), *BASE_CONFIGS]:
        key = tuple((k, cfg[k]) for k in sorted(cfg) if k != "name")
        if key in seen:
            continue
        seen.add(key)
        configs.append(dict(cfg))
    return configs


def command_for(args, dataset, cfg, trial_dir):
    results_dir = trial_dir / "results"
    cmd = [
        args.python,
        "main.py",
        "--data_flag",
        "real_world",
        "--data_dir",
        "data/real_world",
        "--results_dir",
        str(results_dir),
        "--real_world_name",
        dataset,
        "--random_seed",
        str(args.seed),
        "--num_anchors",
        str(cfg["num_anchors"]),
        "--embedding_channels",
        str(cfg["embedding_channels"]),
        "--hidden_channels",
        str(cfg["hidden_channels"]),
        "--num_layers",
        str(cfg["num_layers"]),
        "--epochs",
        str(cfg["epochs"]),
        "--sp_epochs",
        str(args.sp_epochs),
        "--learning_rate",
        str(cfg["learning_rate"]),
        "--sp_learning_rate",
        str(args.sp_learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--grad_clip",
        str(args.grad_clip),
        "--beta_1",
        str(cfg["beta_1"]),
        "--beta_2",
        str(cfg["beta_2"]),
        "--q",
        str(cfg["q"]),
        "--structure_chunk_size",
        str(args.structure_chunk_size),
        "--ending_rounds",
        str(cfg["ending_rounds"]),
    ]
    return cmd


def run_task(args, task):
    dataset, cfg_idx, cfg = task
    trial_name = f"{cfg_idx:03d}_{cfg['name']}_a{cfg['num_anchors']}_e{cfg['embedding_channels']}_h{cfg['hidden_channels']}_lr{safe_name(cfg['learning_rate'])}_b2{safe_name(cfg['beta_2'])}_q{cfg['q']}"
    trial_dir = Path(args.output_dir) / dataset / trial_name
    trial_path = trial_dir / "trial.json"
    if trial_path.exists() and not args.force:
        print(f"skip existing {trial_path}", flush=True)
        return

    trial_dir.mkdir(parents=True, exist_ok=True)
    cmd = command_for(args, dataset, cfg, trial_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    start = time.perf_counter()
    completed = subprocess.run(cmd, cwd=args.repo_dir, env=env, text=True, capture_output=True)
    runtime = time.perf_counter() - start
    result = {
        "dataset": dataset,
        "config_index": cfg_idx,
        "config_name": cfg["name"],
        "gpu": args.gpu,
        "returncode": completed.returncode,
        "auc": parse_metric(AUC_RE, completed.stdout),
        "average_precision": parse_metric(AP_RE, completed.stdout),
        "macro_f1": parse_metric(F1_RE, completed.stdout),
        "runtime_seconds": runtime,
        "seed": args.seed,
        "command": cmd,
        **cfg,
    }
    (trial_dir / "stdout.txt").write_text(completed.stdout)
    (trial_dir / "stderr.txt").write_text(completed.stderr)
    (trial_dir / "command.json").write_text(json.dumps(cmd, indent=2))
    trial_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True), flush=True)


def all_tasks(datasets):
    tasks = []
    for dataset in datasets:
        for idx, cfg in enumerate(configs_for_dataset(dataset)):
            tasks.append((dataset, idx, cfg))
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--python", default=str(Path(__file__).resolve().parent / "env" / "bin" / "python"))
    parser.add_argument("--output-dir", default="../benchmark_results/asgae_asgae_only_grid")
    parser.add_argument("--datasets", nargs="+", default=["iran", "cuba", "UAE"])
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--sp-epochs", type=int, default=20)
    parser.add_argument("--sp-learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--structure-chunk-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.repo_dir = str(Path(args.repo_dir).resolve())
    selected = [task for i, task in enumerate(all_tasks(args.datasets)) if i % args.num_workers == args.worker_index]
    print(
        f"worker {args.worker_index}/{args.num_workers} gpu={args.gpu} "
        f"tasks={len(selected)} datasets={args.datasets}",
        flush=True,
    )
    for task in selected:
        run_task(args, task)

    marker = Path(args.output_dir) / f"worker_{args.worker_index}.complete"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("complete\n")
    print(f"wrote {marker}", flush=True)


if __name__ == "__main__":
    main()
