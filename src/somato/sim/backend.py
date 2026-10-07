"""Simulator-agnostic backend interface.

A backend owns the physics of ``num_envs`` parallel environments and exposes the *raw physical
state* each physics step. Everything sensor-related (taxel stimuli, IMU specific force, sensor
technology models) is computed downstream from this state by :class:`somato.sim.stimulus.StimulusPipeline`,
so adding a simulator means implementing this interface only.

Conventions: world frame z-up, gravity -z, quaternions wxyz, body velocities are those of the body
*frame origin* (not the COM), forces in Newtons.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from somato.sim.terrain import TerrainBatch


@dataclass
class RawSimState:
    body_pos: torch.Tensor  # [E, Nb, 3]
    body_quat: torch.Tensor  # [E, Nb, 4] (wxyz)
    body_lin_vel: torch.Tensor  # [E, Nb, 3] velocity of the body frame origin (world)
    body_ang_vel: torch.Tensor  # [E, Nb, 3] (world)
    joint_pos: torch.Tensor  # [E, Nj]
    joint_vel: torch.Tensor  # [E, Nj]
    joint_torque: torch.Tensor  # [E, Nj] actuator effort
    joint_target: torch.Tensor  # [E, Nj] position command
    # Contact with the ground (any may be None if the backend cannot provide it):
    contact_normal_force: torch.Tensor | None = None  # [E, Nb] magnitude along the ground normal
    contact_friction_force: torch.Tensor | None = None  # [E, Nb, 3] tangential force on the body (world)
    contact_point: torch.Tensor | None = None  # [E, Nb, 3] mean contact location (world)
    # Native IMU readings, if the simulator has an IMU sensor: name -> {"acc": [E, 3], "gyro": [E, 3]}
    # in the IMU frame (acc is specific force, i.e. reads +g when at rest).
    imu: dict[str, dict[str, torch.Tensor]] | None = None


class SimBackend(ABC):
    """Physics backend for ``num_envs`` environments of one robot."""

    @property
    @abstractmethod
    def num_envs(self) -> int: ...

    @property
    @abstractmethod
    def physics_dt(self) -> float: ...

    @property
    @abstractmethod
    def body_names(self) -> list[str]: ...

    @property
    @abstractmethod
    def joint_names(self) -> list[str]: ...

    @property
    @abstractmethod
    def terrain(self) -> TerrainBatch:
        """Current per-environment terrain parameters (labels and contact-model inputs)."""

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | None = None, terrain_class: torch.Tensor | None = None) -> None:
        """Reset environments (all if ``env_ids`` is None), assigning terrain classes (random if None)."""

    @abstractmethod
    def step(self, joint_targets: torch.Tensor) -> None:
        """Apply joint position targets ``[E, Nj]`` and advance one physics step."""

    @abstractmethod
    def get_state(self) -> RawSimState: ...

    def close(self) -> None:
        pass
