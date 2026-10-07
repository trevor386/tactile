"""Validate the Isaac Lab integration (frames, contact reporting, friction sign, terrain materials).

    python scripts/isaac/validate_isaac.py --headless [--num_envs 10]

Run this after any change to the robot, the Isaac Lab version, or the scene setup. Exit code 1 on failure.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default="configs/experiments/collect_isaac.yaml")
parser.add_argument("--num_envs", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

sys.path.insert(0, os.path.dirname(__file__))
from _common import make_isaac_runner  # noqa: E402

from somato.sim.setup import CollectConfig  # noqa: E402
from somato.utils.seeding import seed_everything  # noqa: E402
from somato.validation import BackendValidator, format_report  # noqa: E402


def main() -> int:
    cfg = CollectConfig.from_yaml(args.config, {"num_envs": args.num_envs})
    gen = seed_everything(cfg.seed)
    runner, desc, layout, _ = make_isaac_runner(cfg, args.device, gen)
    validator = BackendValidator(runner.backend, desc, layout, cfg.rates, cfg.gait, cfg.contact)
    results = validator.run()
    print(format_report(results), flush=True)
    runner.backend.close()
    return 1 if any(r.passed is False for r in results) else 0


if __name__ == "__main__":
    code = main()
    # SimulationApp.close() terminates the process with status 0 (Isaac Sim 5.1), which would hide failures.
    sys.stdout.flush()
    os._exit(code)
