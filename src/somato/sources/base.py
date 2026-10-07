"""Sources of somatosensory data, one latent step at a time.

A source yields :class:`SourceFrame` objects with the same structure regardless of where the data
come from -- a simulator, a recorded episode, or real hardware -- so online inference
(:class:`somato.runtime.OnlineEncoder`) and logging code are written once.

On hardware, sensor poses come from forward kinematics of the joint encoders
(:meth:`HardwareSource.body_poses_from_joints`). Because stage 2 uses relative geometry in the
sensors' own frames, the root pose can be left at the identity unless a model uses world-frame
features (``rel_pos_world``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from somato.data.episode import Episode
from somato.geometry.layout import SensorLayout
from somato.geometry.rotations import matrix_to_quat
from somato.robots.description import RobotDescription
from somato.sim.runner import SimRunner


@dataclass
class SourceFrame:
    data: dict[str, torch.Tensor]  # group -> [E, S_g, N_g, C]
    body_pos: torch.Tensor  # [E, Nb, 3] (layout body order)
    body_quat: torch.Tensor  # [E, Nb, 4]
    kind: str = "stimulus"  # "stimulus" (apply sensor models) or "readings" (already raw readings)
    labels: dict[str, torch.Tensor] | None = None


class SomatoSource(ABC):
    layout: SensorLayout

    def reset(self) -> None:
        pass

    @abstractmethod
    def read(self) -> SourceFrame | None:
        """Next latent step, or None when the source is exhausted."""


class SimSource(SomatoSource):
    """Live simulation (mock or Isaac Lab) through a :class:`SimRunner`."""

    def __init__(self, runner: SimRunner, terrain_class: torch.Tensor | None = None):
        self.runner, self.layout, self.terrain_class = runner, runner.pipeline.layout, terrain_class

    def reset(self) -> None:
        self.runner.reset(self.terrain_class)

    def read(self) -> SourceFrame:
        f = self.runner.step_latent()
        labels = dict(f.labels)
        labels["terrain"] = self.runner.backend.terrain.class_id
        return SourceFrame(f.stimuli, f.body_pos, f.body_quat, "stimulus", labels)


class ReplaySource(SomatoSource):
    """Replays stored episodes (batched as environments)."""

    def __init__(self, episodes: list[Episode], layout: SensorLayout, kind: str = "stimulus"):
        self.episodes, self.layout, self.kind = episodes, layout, kind
        self.t = 0

    def reset(self) -> None:
        self.t = 0

    def read(self) -> SourceFrame | None:
        if self.t >= min(ep.num_steps for ep in self.episodes):
            return None
        t = self.t
        self.t += 1
        data = {g: torch.stack([ep.data[g][t].float() for ep in self.episodes]) for g in self.layout.group_names}
        labels = {"terrain": torch.stack([ep.labels["terrain"] for ep in self.episodes])}
        return SourceFrame(data, torch.stack([ep.body_pos[t] for ep in self.episodes]),
                           torch.stack([ep.body_quat[t] for ep in self.episodes]), self.kind, labels)


class HardwareSource(SomatoSource):
    """Template for real robots. Subclass and implement :meth:`read_group` (and optionally
    :meth:`read_joint_positions`) for your drivers, e.g. a serial FSR array, a capacitive skin over
    I2C/SPI, servo buses, or ROS topics.

    Readings must be raw sensor outputs in the same channel order as the sensor model used in
    simulation for that group (e.g. ``fsr_v`` for FSRs), so models trained on simulated data with a
    matching sensor model can be applied directly.
    """

    def __init__(self, desc: RobotDescription, layout: SensorLayout, joint_group: str = "joint"):
        self.desc, self.layout, self.joint_group = desc, layout, joint_group

    @abstractmethod
    def read_group(self, group: str) -> torch.Tensor:
        """Block until one latent step of samples is available: ``[S_g, N_g, C]`` readings."""

    def read_joint_positions(self, joint_readings: torch.Tensor) -> torch.Tensor:
        """Joint angles ``[num_joints]`` (description order) from the joint group's readings ``[S, N, C]``.

        Default: the last sample's first channel, mapped by joint name.
        """
        g = self.layout.groups[self.joint_group]
        q = torch.zeros(len(self.desc.joint_names))
        for i, name in enumerate(g.joint_names):
            q[self.desc.joint_names.index(name)] = joint_readings[-1, i, 0]
        return q

    def body_poses_from_joints(self, q: torch.Tensor, root_rot: torch.Tensor | None = None):
        root_rot = torch.eye(3) if root_rot is None else root_rot
        pos, rot = self.desc.forward_kinematics(torch.zeros(3), root_rot, q)
        order = [self.desc.body_names.index(b) for b in self.layout.body_names]
        return pos[order], rot[order]

    def read(self) -> SourceFrame:
        data = {g: self.read_group(g)[None] for g in self.layout.group_names}
        q = self.read_joint_positions(data[self.joint_group][0])
        pos, rot = self.body_poses_from_joints(q)
        return SourceFrame(data, pos[None], matrix_to_quat(rot)[None], "readings")
