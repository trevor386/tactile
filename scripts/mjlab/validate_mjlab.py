"""Validate the mjlab (MuJoCo-Warp) integration with the same checks as Isaac Lab.

    python scripts/mjlab/validate_mjlab.py [--num_envs 10] [--device cuda:0]

Exit code 1 on failure.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _common import make_mjlab_runner  # noqa: E402

from somato.sim.setup import CollectConfig  # noqa: E402
from somato.utils.seeding import seed_everything  # noqa: E402
from somato.validation import BackendValidator, format_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/experiments/collect_mjlab.yaml")
    parser.add_argument("--num_envs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = CollectConfig.from_yaml(args.config, {"num_envs": args.num_envs})
    gen = seed_everything(cfg.seed)
    runner, desc, layout, _ = make_mjlab_runner(cfg, args.device, gen)
    validator = BackendValidator(runner.backend, desc, layout, cfg.rates, cfg.gait, cfg.contact)
    results = validator.run()
    print(format_report(results))
    return 1 if any(r.passed is False for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
