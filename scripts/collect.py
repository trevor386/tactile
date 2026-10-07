"""Collect a simulated dataset with the CPU/GPU mock simulator.

    python scripts/collect.py --config configs/experiments/collect_mock.yaml [--out DIR] [--rounds N]

For Isaac Sim use scripts/isaac/collect_isaac.py (same config format).
"""

import argparse


from somato.data.collect import collect_dataset
from somato.sim.setup import CollectConfig, make_mock_runner
from somato.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/experiments/collect_mock.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help="latent steps per episode")
    args = parser.parse_args()
    overrides = {k: v for k, v in {"out": args.out, "rounds": args.rounds, "num_envs": args.num_envs,
                                   "steps_per_episode": args.steps}.items() if v is not None}
    cfg = CollectConfig.from_yaml(args.config, overrides)
    gen = seed_everything(cfg.seed)
    runner, desc, layout, catalog = make_mock_runner(cfg, gen)
    print(layout.summary())
    collect_dataset(runner, desc, cfg.out, cfg.rounds, cfg.steps_per_episode, cfg.warmup_steps, catalog.names, gen,
                    info={"simulator": "mock_planar", "config": args.config})
    print(f"Wrote dataset to {cfg.out}")


if __name__ == "__main__":
    main()
