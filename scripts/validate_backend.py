"""Run the simulator integration checks on the mock backend.

    python scripts/validate_backend.py [--config configs/experiments/collect_mock.yaml]

For Isaac Lab use scripts/isaac/validate_isaac.py.
"""

import argparse
import sys

from somato.sim.setup import CollectConfig, make_mock_runner
from somato.utils.seeding import seed_everything
from somato.validation import BackendValidator, format_report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/experiments/collect_mock.yaml")
    parser.add_argument("--num_envs", type=int, default=10)
    args = parser.parse_args()
    cfg = CollectConfig.from_yaml(args.config, {"num_envs": args.num_envs})
    gen = seed_everything(cfg.seed)
    runner, desc, layout, _ = make_mock_runner(cfg, gen)
    validator = BackendValidator(runner.backend, desc, layout, cfg.rates, cfg.gait, cfg.contact)
    results = validator.run()
    print(format_report(results))
    sys.exit(1 if any(r.passed is False for r in results) else 0)


if __name__ == "__main__":
    main()
