"""Run a trained model online against a live Isaac Lab simulation and report accuracy and latency.

    python scripts/isaac/online_demo.py --checkpoint runs/terrain_hierarchical/model.pt --headless

The model runs one latent step at a time (stage-1 recurrent state persists), exactly as on a robot.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--config", default="configs/experiments/collect_isaac.yaml")
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--steps", type=int, default=250, help="latent steps to run")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

sys.path.insert(0, os.path.dirname(__file__))
import torch  # noqa: E402
from _common import make_isaac_runner  # noqa: E402

from somato.geometry import LayoutInfo  # noqa: E402
from somato.runtime import OnlineEncoder  # noqa: E402
from somato.sensors import SensorSuite  # noqa: E402
from somato.sim.setup import CollectConfig  # noqa: E402
from somato.sources import SimSource  # noqa: E402
from somato.training.experiment import load_trained_model  # noqa: E402
from somato.utils.seeding import seed_everything  # noqa: E402


def main():
    cfg = CollectConfig.from_yaml(args.config, {"num_envs": args.num_envs})
    gen = seed_everything(cfg.seed)
    model, ckpt = load_trained_model(args.checkpoint, map_location=args.device)
    runner, desc, layout, catalog = make_isaac_runner(cfg, args.device, gen)
    if ckpt["terrain_names"] != catalog.names:
        print(f"warning: model classes {ckpt['terrain_names']} != simulator classes {catalog.names}")
    suite = SensorSuite.from_config(ckpt["sensors"], layout, cfg.rates.sensors)
    info = LayoutInfo.from_layout(layout, desc, ckpt["model_cfg"].get("cluster_mode", "body"))
    online = OnlineEncoder(model, layout, info, suite, args.device)
    source = SimSource(runner, terrain_class=torch.arange(args.num_envs) % len(catalog))
    source.reset()
    correct = []
    for t in range(args.steps):
        frame = source.read()
        out = online.step(frame)
        pred = out["terrain"].argmax(-1)
        truth = frame.labels["terrain"].to(pred.device)
        correct.append((pred == truth).float().mean().item())
        if (t + 1) % 25 == 0:
            names = [catalog.names[i] for i in pred.tolist()]
            print(f"t={(t + 1) / cfg.rates.latent_hz:5.2f}s acc={correct[-1]:.2f} predictions={names}")
    lat = torch.tensor(online.latency_ms[5:])
    print(f"mean accuracy over last 50 steps: {sum(correct[-50:]) / 50:.3f}")
    print(f"latency per step ({args.num_envs} envs): median {lat.median():.2f} ms, p95 {lat.quantile(0.95):.2f} ms",
          flush=True)
    runner.backend.close()


if __name__ == "__main__":
    main()
    app.close()
