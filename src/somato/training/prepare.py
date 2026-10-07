"""Turn collated dataset samples into model batches (sensor models, poses, augmentation)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from somato.geometry.layout import LayoutInfo, SensorLayout
from somato.geometry.rotations import quat_to_matrix, rot_z
from somato.models.batch import SomatoBatch
from somato.sensors.suite import SensorSuite


@dataclass
class AugmentConfig:
    random_yaw: bool = True  # rotate the whole scene about gravity (keeps IMU readings consistent)
    random_translation: float = 1.0  # m, uniform xy shift
    sensor_dropout: float = 0.0  # fraction of sensors marked dead (node_mask) per sample
    dropout_groups: list[str] = field(default_factory=lambda: ["tactile"])


class BatchPreparer:
    """Applies sensor models to stimuli, computes sensor poses, and augments (train only).

    If ``suite`` is None the dataset is assumed to already contain raw readings.
    """

    def __init__(self, layout: SensorLayout, info: LayoutInfo, suite: SensorSuite | None, device,
                 augment: AugmentConfig | None = None):
        self.layout, self.suite, self.device = layout, suite, torch.device(device)
        self.info = info.to(self.device)
        self.augment = augment or AugmentConfig()

    def readings(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """``[B, L, S, N, C]`` stimuli -> readings with the same leading shape."""
        if self.suite is None:
            return data
        flat = {g: x.flatten(1, 2) for g, x in data.items()}  # [B, L*S, N, C]
        out, _ = self.suite(flat)
        return {g: out[g].view(*data[g].shape[:3], *out[g].shape[2:]) for g in data}

    def __call__(self, sample: dict[str, torch.Tensor], train: bool = False) -> SomatoBatch:
        sample = {k: v.to(self.device, non_blocking=True) for k, v in sample.items()}
        data = {k[5:]: v for k, v in sample.items() if k.startswith("data/")}
        readings = self.readings(data)
        bpos, brot = sample["body_pos"], quat_to_matrix(sample["body_quat"])
        B = bpos.shape[0]
        aug = self.augment
        if train and (aug.random_yaw or aug.random_translation > 0):
            yaw = 2 * math.pi * torch.rand(B, device=self.device) if aug.random_yaw else torch.zeros(B, device=self.device)
            R = rot_z(yaw)[:, None, None]  # [B, 1, 1, 3, 3]
            t = torch.zeros(B, 3, device=self.device)
            t[:, :2] = (torch.rand(B, 2, device=self.device) * 2 - 1) * aug.random_translation
            bpos = (R @ bpos.unsqueeze(-1)).squeeze(-1) + t[:, None, None]
            brot = R @ brot
        pos, rot = self.layout.world_poses(bpos, brot)
        node_mask = None
        if train and aug.sensor_dropout > 0:
            node_mask = torch.ones(B, self.info.num_nodes, dtype=torch.bool, device=self.device)
            for g in aug.dropout_groups:
                if g in self.info.slices:
                    sl = self.info.slices[g]
                    node_mask[:, sl] = torch.rand(B, sl.stop - sl.start, device=self.device) >= aug.sensor_dropout
        labels = {k[6:]: v for k, v in sample.items() if k.startswith("label/")}
        labels.update({k: v for k, v in sample.items() if k.startswith("param/")})
        return SomatoBatch(readings, pos, rot, self.info, node_mask, labels)
