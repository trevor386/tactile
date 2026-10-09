"""Proxy dataset for moving tactile stimuli (H-9, E-13): which way is a contact sliding over the skin?

    python scripts/proxy/make_slide_dataset.py --out datasets/proxy_slide_1200 --episodes 1200

The snake rests in a random, static curved pose; a pressure patch (Gaussian, 8-20 mm wide, 5-30 kPa) slides over its
skin for the whole episode in one of four directions, the class label:
* 0 ``toward_tail`` and 1 ``toward_head``: along the body (0.1-0.4 m/s), following the link axes across joints;
* 2 ``around_pos`` and 3 ``around_neg``: around the body (3-12 rad/s, i.e. 0.08-0.33 m/s of arc).

Distractors that a direction detector must ignore:
* 0-2 stationary patches;
* the static ground load on the downward-facing taxels;
* random patch widths and amplitudes.

Proprioception and the IMU are static (the joint and IMU stimuli carry no information), shear is zero (an FSR does
not sense it anyway), and so the class can only be read from how the pressure pattern moves across neighbouring
taxels over time. Stimuli are written in the dataset format of the simulator collections, so every model, sensor model
and training script runs on it unchanged.
"""

import argparse
import math
from dataclasses import asdict

import torch

from somato.data.episode import DatasetMeta, DatasetWriter, Episode
from somato.geometry.rotations import matrix_to_quat
from somato.robots.factory import build_robot
from somato.sensors.base import STIMULUS_CHANNELS
from somato.sim.runner import RateConfig

CLASSES = ["toward_tail", "toward_head", "around_pos", "around_neg"]
G = 9.81


def surface_point(s: torch.Tensor, theta: torch.Tensor, bpos, brot, link_len: float, radius: float) -> torch.Tensor:
    """World point on the skin at arc length ``s`` along the body (from the head) and angle ``theta`` around it."""
    k = (s / link_len).floor().clamp(0, bpos.shape[0] - 1).long()
    local = torch.stack([s - k * link_len, radius * torch.cos(theta), radius * torch.sin(theta)], -1)
    return bpos[k] + (brot[k] @ local.unsqueeze(-1)).squeeze(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="configs/robots/snake_3d.yaml")
    p.add_argument("--out", default="datasets/proxy_slide_1200")
    p.add_argument("--episodes", type=int, default=1200)
    p.add_argument("--steps", type=int, default=150, help="latent steps per episode (50 Hz)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    gen = torch.Generator().manual_seed(args.seed)
    desc, layout = build_robot(args.robot)
    rates = RateConfig(physics_hz=1000, control_hz=100, latent_hz=50, sensors={"tactile": 500, "joint": 250, "imu": 250})
    S = {g: rates.substeps(g) for g in layout.group_names}
    capsule = desc.bodies[0].shapes[0]
    link_len = float(desc.joints[0].pos[0])
    skin = 0.002
    radius = capsule.size["radius"] + skin
    body_len = link_len * len(layout.body_names)
    nj = len(desc.joint_names)
    meta = DatasetMeta(data_kind="stimulus", rates=asdict(rates),
                       groups={n: {"kind": g.kind, "count": len(g), "channels": list(STIMULUS_CHANNELS[g.kind])}
                               for n, g in layout.groups.items()},
                       terrain_names=CLASSES, info={"proxy": "slide", "generator": "scripts/proxy/make_slide_dataset.py"})
    writer = DatasetWriter(args.out, meta, desc, layout)
    U = lambda lo, hi, n=(): lo + (hi - lo) * torch.rand(n, generator=gen)  # noqa: E731
    T_tac = args.steps * S["tactile"]
    t = torch.arange(T_tac) / rates.sensors["tactile"]
    for e in range(args.episodes):
        cls = e % len(CLASSES)
        q = U(-0.35, 0.35, (nj,))
        bpos, brot = desc.forward_kinematics(torch.tensor([0.0, 0.0, capsule.size["radius"]]), torch.eye(3), q)
        pos, rot = layout.world_poses(bpos, brot, group="tactile")  # [N, 3], [N, 3, 3], static
        duration = float(t[-1])
        if cls < 2:  # along the body
            v = float(U(0.1, 0.4))
            span = v * duration
            s0 = float(U(0.03, body_len - 0.03 - span))
            s = s0 + v * t if cls == 0 else (s0 + span) - v * t
            theta = torch.full_like(t, float(U(0, 2 * math.pi)))
            speed = v
        else:  # around the body
            w = float(U(3.0, 12.0))
            s = torch.full_like(t, float(U(0.03, body_len - 0.03)))
            theta = float(U(0, 2 * math.pi)) + (w if cls == 2 else -w) * t
            speed = w * radius
        width, peak = float(U(0.008, 0.02)), float(U(5e3, 3e4))
        centre = surface_point(s, theta, bpos, brot, link_len, radius)  # [T, 3]
        d2 = (pos[None] - centre[:, None]).pow(2).sum(-1)  # [T, N]
        pressure = peak * torch.exp(-0.5 * d2 / width**2)
        for _ in range(int(torch.randint(0, 3, (1,), generator=gen))):  # stationary distractors
            c = surface_point(U(0.03, body_len - 0.03, (1,)), U(0, 2 * math.pi, (1,)), bpos, brot, link_len, radius)
            pressure = pressure + float(U(2e3, 2e4)) * torch.exp(-0.5 * (pos - c).pow(2).sum(-1) / float(U(0.008, 0.02)) ** 2)
        facing = (-rot[:, 2, 2]).clamp_min(0.0) ** 4  # static ground load on downward-facing taxels
        pressure = pressure + float(U(2e3, 1e4)) * facing
        tactile = torch.zeros(T_tac, len(pos), 3)
        tactile[..., 0] = pressure
        jstate = torch.stack([q, torch.zeros(nj), torch.zeros(nj), q], -1)  # pos, vel, torque, target
        ipos, irot = layout.world_poses(bpos, brot, group="imu")
        spec_force = (irot.transpose(-1, -2) @ torch.tensor([0.0, 0.0, G])).reshape(-1, 3)
        imu = torch.cat([spec_force, torch.zeros_like(spec_force)], -1)  # [1, 6]
        L = args.steps
        ep = Episode(
            data={"tactile": tactile.view(L, S["tactile"], -1, 3),
                  "joint": jstate.expand(L, S["joint"], nj, 4).clone(),
                  "imu": imu.expand(L, S["imu"], 1, 6).clone()},
            body_pos=bpos.expand(L, -1, -1).clone(),
            body_quat=matrix_to_quat(brot).expand(L, -1, -1).clone(),
            labels={"terrain": torch.tensor(cls), "in_contact": (facing.new_zeros(L, len(layout.body_names)) + 1),
                    "slip_speed": torch.zeros(L, len(layout.body_names)),
                    "normal_force": torch.zeros(L, len(layout.body_names))},
            params={"speed": speed, "width": width, "peak": peak},
        )
        writer.write(ep)
        if (e + 1) % 200 == 0:
            print(f"[slide] {e + 1}/{args.episodes} episodes")
    writer.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
