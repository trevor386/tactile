"""Collect a terrain dataset in Isaac Lab (Isaac Sim 5.x).

    python scripts/isaac/collect_isaac.py --config configs/experiments/collect_isaac.yaml --headless

Writes the same dataset format as scripts/collect.py, so training code is shared.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default="configs/experiments/collect_isaac.yaml")
parser.add_argument("--out", default=None)
parser.add_argument("--rounds", type=int, default=None)
parser.add_argument("--num_envs", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

sys.path.insert(0, os.path.dirname(__file__))
from _common import make_isaac_runner  # noqa: E402

from somato.data.collect import collect_dataset  # noqa: E402
from somato.sim.setup import CollectConfig  # noqa: E402
from somato.utils.seeding import seed_everything  # noqa: E402


def main():
    overrides = {k: v for k, v in {"out": args.out, "rounds": args.rounds, "num_envs": args.num_envs}.items()
                 if v is not None}
    cfg = CollectConfig.from_yaml(args.config, overrides)
    gen = seed_everything(cfg.seed)
    runner, desc, layout, catalog = make_isaac_runner(cfg, args.device, gen)
    print(layout.summary())
    collect_dataset(runner, desc, cfg.out, cfg.rounds, cfg.steps_per_episode, cfg.warmup_steps, catalog.names, gen,
                    info={"simulator": "isaaclab", "config": args.config})
    print(f"Wrote dataset to {cfg.out}")


if __name__ == "__main__":
    main()
    app.close()
