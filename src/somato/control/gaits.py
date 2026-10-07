"""Nominal (open-loop) snake gaits. These use no somatosensory feedback; they generate the motion
whose terrain interaction the encoder learns to interpret.

Serpenoid curve (Hirose): joint ``j`` of axis ``a`` follows
``phi_j(t) = A_a sin(omega t - beta_a j + psi_a) + offset_a``.

* lateral undulation: yaw only.
* sidewinding: yaw and pitch waves with a quarter-period phase shift (works on isotropic friction).
* rolling: ``beta = 0`` on both axes with a quarter-period shift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from somato.robots.description import RobotDescription

Range = tuple[float, float]


@dataclass
class GaitConfig:
    kind: str = "lateral_undulation"  # lateral_undulation | sidewinding | rolling
    yaw_amplitude: Range = (0.4, 0.8)  # rad
    pitch_amplitude: Range = (0.15, 0.35)
    frequency: Range = (0.5, 1.2)  # Hz
    yaw_beta: Range = (0.5, 0.9)  # rad per yaw joint
    pitch_beta: Range = (0.5, 0.9)
    yaw_offset: Range = (-0.1, 0.1)  # turning bias
    ramp_time: float = 0.5  # s, amplitude ramp-up after reset


_PARAMS = ("A_y", "A_p", "w", "b_y", "b_p", "off", "phase")


class SerpenoidGait:
    """Batched serpenoid gait with per-environment randomized parameters."""

    def __init__(self, desc: RobotDescription, cfg: GaitConfig, num_envs: int, device="cpu",
                 generator: torch.Generator | None = None):
        self.cfg, self.num_envs, self.device, self.generator = cfg, num_envs, torch.device(device), generator
        axes = [desc.joint(n).axis for n in desc.joint_names]
        self.is_yaw = torch.tensor([abs(a[2]) > 0.5 for a in axes], device=self.device)
        # Index of each joint among joints of the same axis (wave phase advances per same-axis joint).
        yaw_i = torch.cumsum(self.is_yaw.long(), 0) - 1
        pitch_i = torch.cumsum((~self.is_yaw).long(), 0) - 1
        self.axis_index = torch.where(self.is_yaw, yaw_i, pitch_i).float()
        self.params = {k: torch.zeros(num_envs, device=self.device) for k in _PARAMS}
        self.reset()

    def _uniform(self, rng: Range, n: int) -> torch.Tensor:
        u = torch.rand(n, generator=self.generator)
        return (rng[0] + (rng[1] - rng[0]) * u).to(self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids.to(self.device)
        n, c = len(ids), self.cfg
        p = self.params
        p["A_y"][ids] = self._uniform(c.yaw_amplitude, n)
        p["A_p"][ids] = self._uniform(c.pitch_amplitude, n) if c.kind != "lateral_undulation" else 0.0
        p["w"][ids] = 2 * math.pi * self._uniform(c.frequency, n)
        p["b_y"][ids] = self._uniform(c.yaw_beta, n) if c.kind != "rolling" else 0.0
        p["b_p"][ids] = self._uniform(c.pitch_beta, n) if c.kind != "rolling" else 0.0
        p["off"][ids] = self._uniform(c.yaw_offset, n)
        p["phase"][ids] = 2 * math.pi * self._uniform((0.0, 1.0), n)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Joint targets ``[E, Nj]`` at per-environment time since reset ``t [E]``."""
        p = {k: v[:, None] for k, v in self.params.items()}
        t = t.to(self.device)[:, None]
        ramp = (t / max(self.cfg.ramp_time, 1e-6)).clamp(0.0, 1.0)
        phase = p["w"] * t + p["phase"]
        yaw = p["A_y"] * torch.sin(phase - p["b_y"] * self.axis_index) + p["off"]
        pitch = p["A_p"] * torch.sin(phase - p["b_p"] * self.axis_index + math.pi / 2)
        return ramp * torch.where(self.is_yaw, yaw, pitch)
