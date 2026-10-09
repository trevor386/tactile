"""Calibrate the mjlab terrain compliance: soft-contact parameters -> physical sinkage, and check that locomotion
still depends on friction on soft ground.

    python scripts/mjlab/calibrate_compliance.py [--timeconsts 0.02] [--dmins 0.9 0.5 0.3] [--widths 0.001 0.01]
        [--frictions 0.1 0.3 0.6] [--impratio 1]

One env per (time constant, surface impedance dmin, impedance width, friction, repeat). Phase 1: the snake rests
straight for 1 s; sinkage = how far the loaded capsules' lowest points sit below the ground (mean over links in
contact, last 0.3 s). Phase 2: a fixed sidewinding gait (midpoint parameters) for 4 s; reports centroid speed,
sinkage while moving, the fraction of in-contact links that lose contact for a single physics step (flicker), and
the speed-friction correlation per compliance setting.
"""

import argparse
import itertools

import torch

from somato.control.gaits import GaitConfig, SerpenoidGait
from somato.robots.factory import build_robot
from somato.sim.mjlab.backend import MjlabBackend
from somato.sim.mjlab.config import MjlabSnakeConfig
from somato.sim.terrain import TerrainCatalog, TerrainClass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="configs/robots/snake_3d.yaml")
    p.add_argument("--timeconsts", type=float, nargs="+", default=[0.02])
    p.add_argument("--dampratio", type=float, default=1.0)
    p.add_argument("--dmins", type=float, nargs="+", default=[0.9])
    p.add_argument("--widths", type=float, nargs="+", default=[0.001])
    p.add_argument("--frictions", type=float, nargs="+", default=[0.1, 0.3, 0.6])
    p.add_argument("--impratio", type=float, default=1.0)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--rest_s", type=float, default=1.0)
    p.add_argument("--gait_s", type=float, default=4.0)
    args = p.parse_args()

    desc, layout = build_robot(args.robot)
    settings = list(itertools.product(args.timeconsts, args.dmins, args.widths))
    grid = [(s, mu) for s in settings for mu in args.frictions for _ in range(args.repeats)]
    E = len(grid)
    setting = torch.tensor([settings.index(g[0]) for g in grid])
    tc, dmin, width = (torch.tensor([g[0][i] for g in grid]) for i in range(3))
    mu = torch.tensor([g[1] for g in grid])
    catalog = TerrainCatalog([TerrainClass("calib")])
    backend = MjlabBackend(desc, layout, catalog, MjlabSnakeConfig(num_envs=E, impratio=args.impratio),
                           torch.Generator().manual_seed(0))
    dev = backend.device
    m = backend.sim.model
    for name, idx, val in (("pair_friction", 0, mu), ("pair_friction", 1, mu), ("pair_solref", 0, tc),
                           ("pair_solimp", 0, dmin), ("pair_solimp", 2, width)):
        getattr(m, name)[torch.arange(E, device=dev), :, idx] = val.to(dev)[:, None]
    m.pair_solref[..., 1] = args.dampratio
    radius = max(s.size.get("radius", 0.0) for b in desc.bodies for s in b.shapes)
    dt = backend.physics_dt

    def sinkage(state):
        """Per env: mean depth of in-contact capsule bottoms below the ground (m), and the contact mask."""
        z = state.body_pos[..., 2] - radius - backend.terrain.ground_height[:, None]
        touching = state.contact_normal_force > 1e-3
        depth = (-z).clamp_min(0.0) * touching
        return depth.sum(1) / touching.sum(1).clamp_min(1), touching

    zeros = torch.zeros(E, len(desc.joint_names), device=dev)
    rest = []
    for i in range(int(args.rest_s / dt)):
        backend.step(zeros)
        if i >= int((args.rest_s - 0.3) / dt):
            rest.append(sinkage(backend.get_state())[0])
    rest = torch.stack(rest).mean(0).cpu()

    gcfg = GaitConfig(kind="sidewinding", yaw_amplitude=(0.7, 0.7), pitch_amplitude=(0.22, 0.22),
                      frequency=(0.7, 0.7), yaw_beta=(0.7, 0.7), pitch_beta=(0.7, 0.7), yaw_offset=(0.0, 0.0))
    gait = SerpenoidGait(desc, gcfg, E, dev, torch.Generator().manual_seed(1))
    gait.params["phase"].zero_()
    start = backend.get_state().body_pos.mean(1)
    moving, prev2, prev1 = [], None, None
    flick, active = torch.zeros(E, device=dev), torch.zeros(E, device=dev)
    steps = int(args.gait_s / dt)
    for i in range(steps):
        backend.step(gait(torch.full((E,), i * dt, device=dev)))
        s, touch = sinkage(backend.get_state())
        if i > steps // 4:
            moving.append(s)
            if prev2 is not None:  # in contact at t-2 and t, not at t-1: a one-step dropout
                flick += (prev2 & ~prev1 & touch).float().sum(1)
                active += prev1.float().sum(1)
        prev2, prev1 = prev1, touch
    end = backend.get_state().body_pos.mean(1)
    finite = torch.isfinite(end).all(-1).cpu()
    speed = ((end - start)[:, :2].norm(dim=-1) / args.gait_s).cpu()
    moving = torch.stack(moving).mean(0).cpu()
    flicker = (flick / active.clamp_min(1)).cpu()

    print(f"impratio {args.impratio}, dampratio {args.dampratio}; non-finite envs: {int((~finite).sum())}")
    print(f"{'timeconst':>9s} {'dmin':>5s} {'width mm':>8s} {'mu':>5s} {'rest mm':>8s} {'moving mm':>9s} "
          f"{'speed m/s':>9s} {'flicker %':>9s}")
    for k, (t, d, w) in enumerate(settings):
        for f in args.frictions:
            sel = (setting == k) & (mu == f) & finite
            print(f"{t:9.3f} {d:5.2f} {1e3 * w:8.1f} {f:5.2f} {1e3 * rest[sel].mean():8.2f} "
                  f"{1e3 * moving[sel].mean():9.2f} {speed[sel].mean():9.3f} {100 * flicker[sel].mean():9.2f}")
        sel = (setting == k) & finite
        r = torch.corrcoef(torch.stack([mu[sel], speed[sel]]))[0, 1]
        print(f"{'':9s} speed-friction r = {float(r):+.2f}")


if __name__ == "__main__":
    main()
