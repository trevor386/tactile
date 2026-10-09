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
        labels = batch.labels[self.label]
        if "steps" in outputs:  # the latent steps the model predicted at (brain_stride > 1: not the last k)
            target = labels[:, outputs["steps"][0].long()]
        else:
            target = labels[:, -pred.shape[1]:]
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


@TASKS.register("property_regression")
class PropertyRegressionTask(Task):
    """Window-level regression of physical terrain properties (episode-level labels ``param/<target>``).

    The properties (friction, measured sinkage, ...) are what transfers to terrains without a class label (e.g. other
    worlds). Targets are standardized with statistics fitted on the training episodes (``fit``; stored in ``stats``
    so evaluation restores them), after a log transform for ``log_targets`` (quantities spanning orders of
    magnitude). Huber loss on every output step; metrics at the last step: per-target MAE in the original units
    (``mae_<target>``) and the mean absolute error in standardized units (``zmae``).
    """

    def __init__(self, name: str, targets: list[str], head: str | None = None, weight: float = 1.0,
                 log_targets: list[str] | None = None, delta: float = 1.0, eps: float = 0.01,
                 stats: dict[str, list[float]] | None = None):
        super().__init__(name, head, weight)
        self.targets, self.log_targets = list(targets), set(log_targets or [])
        unknown = self.log_targets - set(self.targets)
        if unknown:
            raise ValueError(f"log_targets {sorted(unknown)} are not targets")
        self.delta, self.eps = delta, eps
        self.stats = stats or {"mean": [0.0] * len(self.targets), "std": [1.0] * len(self.targets)}

    @property
    def out_dim(self) -> int:
        return len(self.targets)

    def _forward_transform(self, y: torch.Tensor) -> torch.Tensor:
        cols = [torch.log(y[..., i] + self.eps) if t in self.log_targets else y[..., i]
                for i, t in enumerate(self.targets)]
        return torch.stack(cols, -1)

    def _inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        cols = [torch.exp(z[..., i]) - self.eps if t in self.log_targets else z[..., i]
                for i, t in enumerate(self.targets)]
        return torch.stack(cols, -1)

    def fit(self, values: torch.Tensor) -> None:
        """Standardization statistics from raw training targets ``[E, n_targets]``."""
        z = self._forward_transform(values.double())
        self.stats = {"mean": z.mean(0).tolist(), "std": z.std(0).clamp_min(1e-6).tolist()}

    def raw_targets(self, batch: SomatoBatch) -> torch.Tensor:
        return torch.stack([batch.labels[f"param/{t}"].float() for t in self.targets], -1)  # [B, n]

    def _standardize(self, y: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.stats["mean"], device=y.device, dtype=y.dtype)
        std = torch.tensor(self.stats["std"], device=y.device, dtype=y.dtype)
        return (self._forward_transform(y) - mean) / std

    def _raw_prediction(self, z: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.stats["mean"], device=z.device, dtype=z.dtype)
        std = torch.tensor(self.stats["std"], device=z.device, dtype=z.dtype)
        return self._inverse_transform(z * std + mean)

    def loss(self, outputs, batch):
        pred = outputs[self.head].float()  # [B, k, n] standardized
        target = self._standardize(self.raw_targets(batch))[:, None].expand_as(pred)
        return F.huber_loss(pred, target, delta=self.delta)

    def metrics(self, outputs, batch):
        z = outputs[self.head][:, -1].float()
        y = self.raw_targets(batch)
        err = (self._raw_prediction(z) - y).abs()
        out = {f"mae_{t}": err[:, i] for i, t in enumerate(self.targets)}
        out["zmae"] = (z - self._standardize(y)).abs().mean(-1)
        return out


def build_tasks(cfg: dict[str, dict[str, Any]]) -> list[Task]:
    tasks = []
    for name, spec in cfg.items():
        spec = dict(spec)
        tasks.append(TASKS.build(spec.pop("type"), name, **spec))
    return tasks
