import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use shorter smoke-test epochs.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default="russia")
    args = parser.parse_args()

    pygod_epochs = "20" if args.quick else "100"
    gnn_epochs = "50" if args.quick else "300"
    contrast_epochs = "10" if args.quick else "50"
    contrast_rounds = "8" if args.quick else "32"

    gpu = args.device.split(":")[-1] if args.device.startswith("cuda:") else "-1"
    run([sys.executable, "run_pygod_russia.py", "--dataset", args.dataset, "--epochs", pygod_epochs, "--gpu", gpu])
    run([sys.executable, "run_iohunter_gnn_russia.py", "--dataset", args.dataset, "--epochs", gnn_epochs, "--device", args.device])
    run([
        sys.executable,
        "run_contrastive_russia.py",
        "--dataset",
        args.dataset,
        "--epochs",
        contrast_epochs,
        "--test-rounds",
        contrast_rounds,
        "--device",
        args.device,
    ])


if __name__ == "__main__":
    main()
