"""Model input containers."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from somato.geometry.layout import LayoutInfo


@dataclass
class GroupSpec:
    """What the model needs to know about a sensor group's readings."""

    kind: str
    channels: int  # reading channels produced by the group's sensor model
    substeps: int  # high-rate samples per latent step (sensor rate / latent rate)


@dataclass
class SomatoBatch:
    """A batch of windows (or a single streaming step when ``L == 1``).

    Attributes:
        readings: per group ``[B, L, S_g, N_g, C_g]`` raw sensor readings.
        pos / rot: ``[B, L, N, 3]`` / ``[B, L, N, 3, 3]`` sensor world poses at each latent step
            (all groups concatenated in layout order).
        info: layout tensors (group slices, areas, clusters, rest pose).
        node_mask: ``[B, N]`` True for working sensors (e.g. to simulate dead taxels).
        labels: task labels, passed through untouched.
    """

    readings: dict[str, torch.Tensor]
    pos: torch.Tensor
    rot: torch.Tensor
    info: LayoutInfo
    node_mask: torch.Tensor | None = None
    labels: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.pos.shape[0]

    @property
    def num_steps(self) -> int:
        return self.pos.shape[1]

    def to(self, device) -> SomatoBatch:
        return SomatoBatch(
            readings={k: v.to(device) for k, v in self.readings.items()},
            pos=self.pos.to(device),
            rot=self.rot.to(device),
            info=self.info.to(device),
            node_mask=None if self.node_mask is None else self.node_mask.to(device),
            labels={k: v.to(device) for k, v in self.labels.items()},
        )

    def step(self, t: int) -> SomatoBatch:
        """The single latent step ``t`` (keeps the time dimension, ``L = 1``)."""
        return SomatoBatch(
            readings={k: v[:, t : t + 1] for k, v in self.readings.items()},
            pos=self.pos[:, t : t + 1],
            rot=self.rot[:, t : t + 1],
            info=self.info,
            node_mask=self.node_mask,
            labels=self.labels,
        )
