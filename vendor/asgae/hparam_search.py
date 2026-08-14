import argparse
import csv
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


AUC_RE = re.compile(r"roc_auc_score:([0-9.]+)")


@dataclass(frozen=True)
class Trial:
    dataset: str
    learning_rate: float
    beta_1: float
    beta_2: float
    q: int
    embedding_channels: int
    hidden_channels: int
    num_anchors: int
    seed: int


def parse_list(value, item_type):
    return [item_type(item.strip()) for item in value.split(",") if item.strip()]


def build_trials(args, dataset):
    combos = itertools.product(
        args.learning_rates,
        args.beta_1_values,
        args.beta_2_values,
        args.q_values,
        args.embedding_channels_values,
        args.hidden_channels_values,
        args.num_anchor_values,
        args.seeds,
    )
    trials = [
        Trial(
            dataset=dataset,
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            q=q,
            embedding_channels=embedding_channels,
            hidden_channels=hidden_channels,
            num_anchors=num_anchors,
            seed=seed,
        )
        for (
            learning_rate,
            beta_1,
            beta_2,
            q,
            embedding_channels,
            hidden_channels,
            num_anchors,
            seed,
        ) in combos
    ]

    rng = random.Random(args.search_seed)
    rng.shuffle(trials)

    if args.max_trials is not None and args.max_trials > 0:
        trials = trials[: args.max_trials]

    return trials


def trial_id(index, trial):
    return (
        f"{index:04d}_"
        f"lr{trial.learning_rate:g}_"
        f"b1{trial.beta_1:g}_"
        f"b2{trial.beta_2:g}_"
        f"q{trial.q}_"
        f"emb{trial.embedding_channels}_"
        f"hid{trial.hidden_channels}_"
        f"anc{trial.num_anchors}_"
        f"seed{trial.seed}"
    ).replace(".", "p")


def trial_key(trial):
    return (
        trial.dataset,
        trial.learning_rate,
        trial.beta_1,
        trial.beta_2,
        trial.q,
        trial.embedding_channels,
        trial.hidden_channels,
        trial.num_anchors,
        trial.seed,
    )


def make_trial(args, dataset, config):
    return Trial(
        dataset=dataset,
        learning_rate=config["learning_rate"],
        beta_1=config["beta_1"],
        beta_2=config["beta_2"],
        q=config["q"],
        embedding_channels=config["embedding_channels"],
        hidden_channels=config["hidden_channels"],
        num_anchors=config["num_anchors"],
        seed=config["seed"],
    )


def parse_auc(stdout):
    matches = AUC_RE.findall(stdout)
    if not matches:
        return None
    return float(matches[-1])


def run_trial(args, gpu_id, index, trial):
    run_id = trial_id(index, trial)
    trial_dir = Path(args.output_dir) / trial.dataset / run_id
    trial_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = trial_dir / "stdout.txt"
    stderr_path = trial_dir / "stderr.txt"
    model_path = trial_dir / "model.pt"
    results_dir = trial_dir / "results"

    command = [
        args.python,
        args.main_script,
        "--data_flag",
        "real_world",
        "--data_dir",
        args.data_dir,
        "--results_dir",
        str(results_dir),
        "--model_path",
        str(model_path),
        "--real_world_name",
        trial.dataset,
        "--random_seed",
        str(trial.seed),
        "--num_anchors",
        str(trial.num_anchors),
        "--embedding_channels",
        str(trial.embedding_channels),
        "--hidden_channels",
        str(trial.hidden_channels),
        "--num_layers",
        str(args.num_layers),
        "--epochs",
        str(args.epochs),
        "--learning_rate",
        str(trial.learning_rate),
        "--beta_1",
        str(trial.beta_1),
        "--beta_2",
        str(trial.beta_2),
        "--q",
        str(trial.q),
        "--structure_chunk_size",
        str(args.structure_chunk_size),
        "--ending_rounds",
        str(args.ending_rounds),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start = time.time()
    completed = subprocess.run(
        command,
        cwd=args.repo_dir,
        env=env,
        text=True,
        capture_output=True,
    )
    elapsed_seconds = time.time() - start

    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)

    auc = parse_auc(completed.stdout)
    row = {
        **asdict(trial),
        "run_id": run_id,
        "gpu": gpu_id,
        "returncode": completed.returncode,
        "auc": auc,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "trial_dir": str(trial_dir),
    }

    (trial_dir / "trial.json").write_text(json.dumps(row, indent=2, sort_keys=True))
    return row


def append_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    fieldnames = [
        "dataset",
        "run_id",
        "gpu",
        "returncode",
        "auc",
        "elapsed_seconds",
        "learning_rate",
        "beta_1",
        "beta_2",
        "q",
        "embedding_channels",
        "hidden_channels",
        "num_anchors",
        "seed",
        "trial_dir",
    ]

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_best_summary(output_dir, dataset, rows):
    valid_rows = [
        row
        for row in rows
        if row["returncode"] == 0 and row["auc"] is not None
    ]
    if not valid_rows:
        return None

    best = max(valid_rows, key=lambda row: row["auc"])
    summary_path = Path(output_dir) / dataset / "best.json"
    summary_path.write_text(json.dumps(best, indent=2, sort_keys=True))
    return best


def run_trial_batch(args, trials, start_index=1, stop_on_target=True):
    rows = []
    best = None
    stop_search = False
    next_offset = 0

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {}

        while next_offset < len(trials) and len(futures) < len(args.gpus):
            index = start_index + next_offset
            trial = trials[next_offset]
            gpu_id = args.gpus[next_offset % len(args.gpus)]
            future = executor.submit(run_trial, args, gpu_id, index, trial)
            futures[future] = next_offset
            next_offset += 1

        while futures:
            for future in as_completed(futures):
                completed_offset = futures.pop(future)
                break

            row = future.result()
            rows.append(row)
            status = "ok" if row["returncode"] == 0 else f"fail:{row['returncode']}"
            auc = "nan" if row["auc"] is None else f"{row['auc']:.6f}"
            print(f"{row['dataset']} {row['run_id']} gpu={row['gpu']} {status} auc={auc}")

            if row["returncode"] == 0 and row["auc"] is not None:
                if best is None or row["auc"] > best["auc"]:
                    best = row

                if (
                    stop_on_target
                    and args.target_auc is not None
                    and row["auc"] >= args.target_auc
                ):
                    print(
                        f"Target AUC reached for {row['dataset']}: "
                        f"{row['auc']:.6f} >= {args.target_auc:.6f}"
                    )
                    stop_search = True

            if not stop_search and next_offset < len(trials):
                index = start_index + next_offset
                trial = trials[next_offset]
                gpu_id = args.gpus[next_offset % len(args.gpus)]
                new_future = executor.submit(run_trial, args, gpu_id, index, trial)
                futures[new_future] = next_offset
                next_offset += 1

    return rows, best


def run_dataset(args, dataset):
    trials = build_trials(args, dataset)
    if not trials:
        print(f"No trials for {dataset}")
        return None

    print(f"Running {len(trials)} trials for {dataset} on GPUs {args.gpus}")
    rows, _ = run_trial_batch(args, trials, start_index=1, stop_on_target=True)

    rows.sort(key=lambda row: row["run_id"])
    append_rows(Path(args.output_dir) / "trials.csv", rows)
    best = write_best_summary(args.output_dir, dataset, rows)

    if best is None:
        print(f"No successful AUC result for {dataset}")
    else:
        print(
            f"Best {dataset}: auc={best['auc']:.6f}, "
            f"lr={best['learning_rate']}, beta_1={best['beta_1']}, "
            f"beta_2={best['beta_2']}, q={best['q']}, "
            f"embedding={best['embedding_channels']}, hidden={best['hidden_channels']}, "
            f"anchors={best['num_anchors']}"
        )

    return best


def initial_config(args):
    return {
        "learning_rate": args.initial_learning_rate,
        "beta_1": args.initial_beta_1,
        "beta_2": args.initial_beta_2,
        "q": args.initial_q,
        "embedding_channels": args.initial_embedding_channels,
        "hidden_channels": args.initial_hidden_channels,
        "num_anchors": args.initial_num_anchors,
        "seed": args.seeds[0],
    }


def row_to_config(row):
    return {
        "learning_rate": row["learning_rate"],
        "beta_1": row["beta_1"],
        "beta_2": row["beta_2"],
        "q": row["q"],
        "embedding_channels": row["embedding_channels"],
        "hidden_channels": row["hidden_channels"],
        "num_anchors": row["num_anchors"],
        "seed": row["seed"],
    }


def greedy_candidates(args, dataset, config, parameter_name, seen):
    value_lists = {
        "learning_rate": args.learning_rates,
        "beta_1": args.beta_1_values,
        "beta_2": args.beta_2_values,
        "q": args.q_values,
        "embedding_channels": args.embedding_channels_values,
        "hidden_channels": args.hidden_channels_values,
        "num_anchors": args.num_anchor_values,
    }

    trials = []
    for value in value_lists[parameter_name]:
        if value == config[parameter_name]:
            continue

        candidate_config = dict(config)
        candidate_config[parameter_name] = value
        for seed in args.seeds:
            candidate_config["seed"] = seed
            trial = make_trial(args, dataset, candidate_config)
            key = trial_key(trial)
            if key in seen:
                continue
            seen.add(key)
            trials.append(trial)

    return trials


def run_dataset_greedy(args, dataset):
    print(f"Running greedy search for {dataset} on GPUs {args.gpus}")
    rows = []
    seen = set()
    next_index = 1
    config = initial_config(args)

    baseline = make_trial(args, dataset, config)
    seen.add(trial_key(baseline))
    baseline_rows, best = run_trial_batch(
        args,
        [baseline],
        start_index=next_index,
        stop_on_target=True,
    )
    rows.extend(baseline_rows)
    next_index += len(baseline_rows)

    if best is not None:
        config = row_to_config(best)

    parameter_order = parse_list(args.greedy_order, str)

    for round_index in range(1, args.greedy_rounds + 1):
        improved = False
        print(f"Greedy round {round_index} for {dataset}")

        for parameter_name in parameter_order:
            trials = greedy_candidates(args, dataset, config, parameter_name, seen)
            if not trials:
                continue

            print(
                f"Testing {len(trials)} {parameter_name} candidates "
                f"from current best auc={best['auc'] if best else 'nan'}"
            )
            batch_rows, batch_best = run_trial_batch(
                args,
                trials,
                start_index=next_index,
                stop_on_target=True,
            )
            rows.extend(batch_rows)
            next_index += len(batch_rows)

            if batch_best is not None and (best is None or batch_best["auc"] > best["auc"]):
                best = batch_best
                config = row_to_config(best)
                improved = True
                print(
                    f"New greedy best for {dataset}: auc={best['auc']:.6f}, "
                    f"{parameter_name}={config[parameter_name]}"
                )

            if best is not None and args.target_auc is not None and best["auc"] >= args.target_auc:
                print(
                    f"Target AUC reached for {dataset}: "
                    f"{best['auc']:.6f} >= {args.target_auc:.6f}"
                )
                rows.sort(key=lambda row: row["run_id"])
                append_rows(Path(args.output_dir) / "trials.csv", rows)
                write_best_summary(args.output_dir, dataset, rows)
                return best

        if not improved:
            print(f"No greedy improvement in round {round_index} for {dataset}")
            break

    rows.sort(key=lambda row: row["run_id"])
    append_rows(Path(args.output_dir) / "trials.csv", rows)
    best = write_best_summary(args.output_dir, dataset, rows)

    if best is None:
        print(f"No successful AUC result for {dataset}")
    else:
        print(
            f"Best {dataset}: auc={best['auc']:.6f}, "
            f"lr={best['learning_rate']}, beta_1={best['beta_1']}, "
            f"beta_2={best['beta_2']}, q={best['q']}, "
            f"embedding={best['embedding_channels']}, hidden={best['hidden_channels']}, "
            f"anchors={best['num_anchors']}"
        )

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--main_script", type=str, default="main.py")
    parser.add_argument("--data_dir", type=str, default="data/real_world")
    parser.add_argument("--output_dir", type=str, default="hparam_results")
    parser.add_argument("--datasets", type=str, default="russia,venezuela,china")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--learning_rates", type=str, default="1e-5,5e-5,1e-4,5e-4")
    parser.add_argument("--beta_1_values", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--beta_2_values", type=str, default="0.1,1.0,2.0")
    parser.add_argument("--q_values", type=str, default="80,85,90,95")
    parser.add_argument("--embedding_channels_values", type=str, default="32,64,128")
    parser.add_argument("--hidden_channels_values", type=str, default="16,32,64")
    parser.add_argument("--num_anchor_values", type=str, default="0,50,100,200")
    parser.add_argument("--seeds", type=str, default="12345")
    parser.add_argument("--search_strategy", type=str, choices=["greedy", "random"], default="greedy")
    parser.add_argument("--max_trials", type=int, default=32)
    parser.add_argument("--search_seed", type=int, default=12345)
    parser.add_argument("--target_auc", type=float, default=None)
    parser.add_argument("--greedy_rounds", type=int, default=3)
    parser.add_argument(
        "--greedy_order",
        type=str,
        default="num_anchors,q,learning_rate,beta_2,beta_1,embedding_channels,hidden_channels",
    )
    parser.add_argument("--initial_learning_rate", type=float, default=1e-4)
    parser.add_argument("--initial_beta_1", type=float, default=1.0)
    parser.add_argument("--initial_beta_2", type=float, default=1.0)
    parser.add_argument("--initial_q", type=int, default=90)
    parser.add_argument("--initial_embedding_channels", type=int, default=64)
    parser.add_argument("--initial_hidden_channels", type=int, default=32)
    parser.add_argument("--initial_num_anchors", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ending_rounds", type=int, default=1)
    parser.add_argument("--num_anchors", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--structure_chunk_size", type=int, default=256)
    args = parser.parse_args()

    if args.python is None:
        repo_python = Path(args.repo_dir) / "env" / "bin" / "python"
        args.python = str(repo_python) if repo_python.exists() else sys.executable

    args.datasets = parse_list(args.datasets, str)
    args.gpus = parse_list(args.gpus, int)
    args.learning_rates = parse_list(args.learning_rates, float)
    args.beta_1_values = parse_list(args.beta_1_values, float)
    args.beta_2_values = parse_list(args.beta_2_values, float)
    args.q_values = parse_list(args.q_values, int)
    args.embedding_channels_values = parse_list(args.embedding_channels_values, int)
    args.hidden_channels_values = parse_list(args.hidden_channels_values, int)
    args.num_anchor_values = parse_list(args.num_anchor_values, int)
    args.seeds = parse_list(args.seeds, int)

    if not args.gpus:
        raise ValueError("At least one GPU id is required")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    best_by_dataset = {}
    for dataset in args.datasets:
        if args.search_strategy == "greedy":
            best_by_dataset[dataset] = run_dataset_greedy(args, dataset)
        else:
            best_by_dataset[dataset] = run_dataset(args, dataset)

    summary_path = Path(args.output_dir) / "best_by_dataset.json"
    summary_path.write_text(json.dumps(best_by_dataset, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
