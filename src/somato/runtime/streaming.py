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

    ``window`` (latent steps) switches to sliding-window inference: the last ``window`` steps of readings
    and poses are kept and the model is re-run on them from a fresh state at every step, which reproduces
    the training regime of models trained on independent windows (whose recurrent state is not trained
    beyond the window length). It costs about ``window`` times the stage-1 compute per step. Sensor-model
    state still persists.
    """

    def __init__(self, model: nn.Module, layout: SensorLayout, info: LayoutInfo, suite: SensorSuite | None = None,
                 device: str | torch.device = "cpu", window: int | None = None):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.layout, self.info, self.suite = layout, info.to(self.device), suite
        self.window = window
        self.state: dict | None = None
        self.sensor_state = None
        self.history: list[SomatoBatch] = []
        self.latency_ms: list[float] = []
        self.last_outputs: dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self.state, self.sensor_state, self.history, self.last_outputs = None, None, [], {}

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
        if self.window:
            self.history = (self.history + [batch])[-self.window:]
            h = self.history
            batch = SomatoBatch({g: torch.cat([b.readings[g] for b in h], 1) for g in batch.readings},
                                torch.cat([b.pos for b in h], 1), torch.cat([b.rot for b in h], 1), self.info)
            outputs, _ = self.model(batch, output_steps=1)
        else:
            outputs, self.state = self.model(batch, self.state)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.latency_ms.append(1e3 * (time.perf_counter() - t0))
        if outputs:  # models with brain_stride > 1 predict only at brain steps; hold the last prediction between them
            self.last_outputs = {k: v[:, -1] for k, v in outputs.items()}
        return self.last_outputs
