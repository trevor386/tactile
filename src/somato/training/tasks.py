"""Training tasks: map head outputs + labels to losses and per-sample metrics."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from somato.models.batch import SomatoBatch
from somato.utils.registry import Registry

TASKS: Registry = Registry("task")


class Task:
    def __init__(self, name: str, head: str | None = None, weight: float = 1.0):
        self.name, self.head, self.weight = name, head or name, weight

    def loss(self, outputs: dict[str, torch.Tensor], batch: SomatoBatch) -> torch.Tensor:
        raise NotImplementedError

    def metrics(self, outputs: dict[str, torch.Tensor], batch: SomatoBatch) -> dict[str, torch.Tensor]:
        """Per-sample metric values ``[B]`` (averaged over the dataset by the trainer)."""
        raise NotImplementedError


@TASKS.register("classification")
class ClassificationTask(Task):
    """Window-level classification (e.g. terrain class); the loss is applied at every output step."""

    def __init__(self, name: str, label: str = "terrain", head: str | None = None, weight: float = 1.0,
                 label_smoothing: float = 0.0):
        super().__init__(name, head, weight)
        self.label, self.label_smoothing = label, label_smoothing

    def loss(self, outputs, batch):
        logits = outputs[self.head]  # [B, k, C]
        y = batch.labels[self.label].long()
        return F.cross_entropy(logits.flatten(0, 1), y.repeat_interleave(logits.shape[1]),
                               label_smoothing=self.label_smoothing)

    def predict(self, outputs) -> torch.Tensor:
        return outputs[self.head][:, -1].argmax(-1)

    def metrics(self, outputs, batch):
        logits = outputs[self.head]
        y = batch.labels[self.label].long()
        return {
            "acc_last": (logits[:, -1].argmax(-1) == y).float(),
            "acc_mean": (logits.mean(1).argmax(-1) == y).float(),
            "nll_last": F.cross_entropy(logits[:, -1], y, reduction="none"),
        }


@TASKS.register("per_body_regression")
class PerBodyRegressionTask(Task):
    """Per-body, per-step regression of a label such as ``slip_speed [B, L, Nb]`` (Huber loss)."""

    def __init__(self, name: str, label: str = "slip_speed", head: str | None = None, weight: float = 1.0,
                 delta: float = 0.1):
        super().__init__(name, head, weight)
        self.label, self.delta = label, delta

    def _pair(self, outputs, batch):
        pred = outputs[self.head][..., 0]  # [B, k, Nb]
        target = batch.labels[self.label][:, -pred.shape[1]:]
        if target.shape != pred.shape:
            raise ValueError(f"Per-body prediction {tuple(pred.shape)} does not match label {tuple(target.shape)}; "
                             "use cluster_mode 'body' with sensors on every body")
        return pred, target

    def loss(self, outputs, batch):
        pred, target = self._pair(outputs, batch)
        return F.huber_loss(pred, target, delta=self.delta)

    def metrics(self, outputs, batch):
        pred, target = self._pair(outputs, batch)
        return {"mae": (pred - target).abs().mean((1, 2))}


def build_tasks(cfg: dict[str, dict[str, Any]]) -> list[Task]:
    tasks = []
    for name, spec in cfg.items():
        spec = dict(spec)
        tasks.append(TASKS.build(spec.pop("type"), name, **spec))
    return tasks
