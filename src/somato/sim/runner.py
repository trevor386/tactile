"""Multi-rate stepping of a backend: physics -> controller -> stimuli at each sensor group's rate.

Rates are organized around the *latent step* (the rate at which stage 1 emits latents and stages 2-3
run). Every sensor group samples ``S_g = rate_g / latent_hz`` times per latent step; all rates must
divide the physics rate. One call to :meth:`SimRunner.step_latent` returns one latent step of data.

With ``anti_alias`` (default), each sensor sample is the mean of the stimuli over the physics steps of its
sample period (integrate-and-dump), like a sensor that low-pass filters before sampling. Point sampling
instead aliases physics-rate solver chatter (e.g. resting-contact jitter in PhysX) into the sensor stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from somato.sim.backend import SimBackend
from somato.sim.stimulus import StimulusPipeline


@dataclass
class RateConfig:
    physics_hz: float = 1000.0
    control_hz: float = 100.0
    latent_hz: float = 50.0
    sensors: dict[str, float] = field(default_factory=lambda: {"tactile": 1000.0, "joint": 500.0, "imu": 500.0})
    anti_alias: bool = True  # average stimuli over each sample period instead of point sampling

    def _ratio(self, a: float, b: float, what: str) -> int:
        r = a / b
        if abs(r - round(r)) > 1e-6 or round(r) < 1:
            raise ValueError(f"{what}: {a} Hz is not an integer multiple of {b} Hz")
        return int(round(r))

    def physics_per_latent(self) -> int:
        return self._ratio(self.physics_hz, self.latent_hz, "physics/latent")

    def physics_per_control(self) -> int:
        return self._ratio(self.physics_hz, self.control_hz, "physics/control")

    def substeps(self, group: str) -> int:
        """High-rate samples per latent step for ``group``."""
        return self._ratio(self.sensors[group], self.latent_hz, f"{group}/latent")

    def physics_per_sample(self, group: str) -> int:
        return self._ratio(self.physics_hz, self.sensors[group], f"physics/{group}")


@dataclass
class LatentFrame:
    """One latent step of simulated data for all environments."""

    stimuli: dict[str, torch.Tensor]  # group -> [E, S_g, N_g, C]
    body_pos: torch.Tensor  # [E, Nb, 3] at the end of the step (layout body order)
    body_quat: torch.Tensor  # [E, Nb, 4]
    labels: dict[str, torch.Tensor]  # per-body labels averaged over the step, e.g. slip_speed [E, Nb]


class SimRunner:
    """Steps ``backend`` with ``controller(time [E]) -> targets [E, Nj]`` and the stimulus pipeline."""

    def __init__(self, backend: SimBackend, pipeline: StimulusPipeline, controller, rates: RateConfig):
        self.backend, self.pipeline, self.controller, self.rates = backend, pipeline, controller, rates
        pipeline.bind(backend.body_names, backend.joint_names)
        missing = [g for g in pipeline.layout.group_names if g not in rates.sensors]
        if missing:
            raise ValueError(f"No sample rate configured for sensor groups {missing}")
        self._n_physics = rates.physics_per_latent()
        self._ctrl_every = rates.physics_per_control()
        self._sample_every = {g: rates.physics_per_sample(g) for g in pipeline.layout.group_names}
        self._targets: torch.Tensor | None = None
        self._step_count = 0
        self._layout_body_map = torch.tensor(
            [backend.body_names.index(b) for b in pipeline.layout.body_names], device=backend.device
        )

    @property
    def time(self) -> torch.Tensor:
        return self._step_count * self.backend.physics_dt * torch.ones(self.backend.num_envs, device=self.backend.device)

    def reset(self, terrain_class: torch.Tensor | None = None) -> None:
        """Reset all environments together (keeps environments time-aligned)."""
        self.backend.reset(None, terrain_class)
        if hasattr(self.controller, "reset"):
            self.controller.reset()
        self.pipeline.reset()
        self._targets = None
        self._step_count = 0

    def step_latent(self) -> LatentFrame:
        dt = self.backend.physics_dt
        stim: dict[str, list[torch.Tensor]] = {g: [] for g in self.pipeline.layout.group_names}
        # Running sums over the current sample period (sample periods divide the latent step).
        period_sum: dict[str, torch.Tensor | int] = {g: 0 for g in stim}
        label_acc: dict[str, torch.Tensor] = {}
        n_label = 0
        state = None
        for _ in range(self._n_physics):
            if self._targets is None or self._step_count % self._ctrl_every == 0:
                self._targets = self.controller(self.time)
            self.backend.step(self._targets)
            self._step_count += 1
            state = self.backend.get_state()
            need = [g for g, every in self._sample_every.items() if self._step_count % every == 0]
            # The stimulus pipeline must run every physics step for IMU finite differences to be correct.
            stimuli, labels = self.pipeline(state, self.backend.terrain, dt)
            if self.rates.anti_alias:
                for g in stim:
                    period_sum[g] = period_sum[g] + stimuli[g]
            for g in need:
                if self.rates.anti_alias:
                    stim[g].append(period_sum[g] / self._sample_every[g])
                    period_sum[g] = 0
                else:
                    stim[g].append(stimuli[g])
            for k, v in labels.items():
                label_acc[k] = label_acc.get(k, 0) + v
            n_label += 1
        bpos = state.body_pos[:, self._layout_body_map]
        bquat = state.body_quat[:, self._layout_body_map]
        return LatentFrame(
            stimuli={g: torch.stack(v, dim=1) for g, v in stim.items()},
            body_pos=bpos,
            body_quat=bquat,
            labels={k: v / n_label for k, v in label_acc.items()},
        )
