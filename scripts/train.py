"""Train and evaluate one model.

    python scripts/train.py --config configs/experiments/terrain_mock.yaml \
        [--set train.epochs=5 model=configs/models/egnn.yaml sensors=configs/sensors/capacitive.yaml]
"""

import argparse

import yaml

from somato.training.experiment import ExperimentConfig, run_experiment


def parse_overrides(items):
    """``a.b=value`` strings -> nested dict (values parsed as YAML)."""
    out = {}
    for item in items or []:
        key, value = item.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/experiments/terrain_mock.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=5")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config, parse_overrides(args.set))
    run_experiment(cfg)


if __name__ == "__main__":
    main()
