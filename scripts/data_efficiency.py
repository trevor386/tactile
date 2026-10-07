"""Accuracy vs. training-data fraction for several models (Phase 1 criterion).

    python scripts/data_efficiency.py --config configs/experiments/data_efficiency_mock.yaml \
        [--set fractions=[0.1,1.0] seeds=[0] base_overrides.train.epochs=5]

Writes results.csv, summary.md and learning_curves.png (if matplotlib is installed) to output_dir.
"""

import argparse

from train import parse_overrides

from somato.training.data_efficiency import run_data_efficiency


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/experiments/data_efficiency_mock.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    run_data_efficiency(args.config, parse_overrides(args.set))


if __name__ == "__main__":
    main()
