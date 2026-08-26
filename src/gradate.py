import argparse

from contrastive import run_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--save-first-model", action="store_true")
    run_experiment("gradate", parser.parse_args())
