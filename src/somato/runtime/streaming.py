"""Online (streaming) inference: one latent step at a time with persistent recurrent state."""

from __future__ import annotations

import time

import torch
from torch import nn

from somato.geometry.layout import LayoutInfo, SensorLayout
from somato.geometry.rotations import quat_to_matrix
from somato.models.batch import SomatoBatch
from somato.sensors.suite import SensorSuite
from somato.sources.base import SourceFrame


class OnlineEncoder:
    """Wraps a trained model for deployment.

    Sensor-model state (hysteresis, drift) and model state (stage-1 hidden states, head memory) persist
    across calls, so ``step`` on consecutive latent steps reproduces offline window processing.
    """

    def __init__(self, model: nn.Module, layout: SensorLayout, info: LayoutInfo, suite: SensorSuite | None = None,
                 device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.layout, self.info, self.suite = layout, info.to(self.device), suite
        self.state: dict | None = None
        self.sensor_state = None
        self.latency_ms: list[float] = []

    def reset(self) -> None:
        self.state, self.sensor_state = None, None

    @torch.no_grad()
    def step(self, frame: SourceFrame) -> dict[str, torch.Tensor]:
        t0 = time.perf_counter()
        data = {g: v.to(self.device).float() for g, v in frame.data.items()}
        if frame.kind == "stimulus":
            if self.suite is None:
                raise ValueError("Stimulus frames need a sensor suite to produce readings")
            data, self.sensor_state = self.suite(data, self.sensor_state)
        bpos = frame.body_pos.to(self.device).float()
        brot = quat_to_matrix(frame.body_quat.to(self.device).float())
        pos, rot = self.layout.world_poses(bpos, brot)
        batch = SomatoBatch({g: v[:, None] for g, v in data.items()}, pos[:, None], rot[:, None], self.info)
        outputs, self.state = self.model(batch, self.state)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.latency_ms.append(1e3 * (time.perf_counter() - t0))
        return {k: v[:, -1] for k, v in outputs.items()}
