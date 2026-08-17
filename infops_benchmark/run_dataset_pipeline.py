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
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-iohunter", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if not args.skip_iohunter:
        run([
            sys.executable,
            "run_iohunter_gnn_russia.py",
            "--dataset",
            args.dataset,
            "--epochs",
            "50" if args.quick else "300",
            "--device",
            args.device,
        ])

    run([
        sys.executable,
        "search_non_iohunter_russia.py",
        "--dataset",
        args.dataset,
        "--device",
        args.device,
    ])

    run([sys.executable, "summarize_results.py", "--dataset", args.dataset])


if __name__ == "__main__":
    main()
