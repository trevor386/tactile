"""A compact, dependency-free training loop."""

from __future__ import annotations

import contextlib
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from somato.training.prepare import BatchPreparer
from somato.training.tasks import Task


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_epochs: float = 1.0
    output_steps: int = 4  # stages 2-3 (and the loss) run on the last k latent steps of each window
    # Truncated BPTT: when > 0, each sample is processed in consecutive chunks of this many latent steps with the
    # recurrent state carried (detached) from chunk to chunk; the loss and an optimizer step follow every chunk,
    # and evaluation averages the metrics over all chunk ends. Training samples longer than the chunk then teach
    # the model to predict from the long state histories of streaming deployment. 0 = independent windows.
    tbptt_chunk: int = 0
    # Lower bound on optimizer steps: epochs are raised to reach it, so small training sets (learning curves) still
    # get enough updates to converge (combine with ``patience`` for early stopping). 0 = exactly ``epochs``.
    min_steps: int = 0
    eval_every: int = 0  # validate every N optimizer steps instead of every epoch (patience then counts evaluations)
    amp: str = "none"  # "bf16": autocast forward/loss in bfloat16 on CUDA (neighbor search stays float32)
    device: str = "auto"
    num_workers: int = 0
    seed: int = 0
    patience: int = 0  # early stopping on the selection metric (0 = off)
    select_metric: str = "terrain/acc_last"
    normalizer_batches: int = 8
    log_every: int = 0  # steps between progress prints (0 = once per epoch)
    max_steps_per_epoch: int = 0  # 0 = full epoch


def detach_state(state: Any) -> Any:
    """Detach every tensor of a (nested) recurrent state so gradients stop at chunk boundaries."""
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, dict):
        return {k: detach_state(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(detach_state(v) for v in state)
    return state


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class Trainer:
    def __init__(self, model: nn.Module, tasks: list[Task], preparer: BatchPreparer, cfg: TrainConfig,
                 log_path: str | Path | None = None):
        self.model, self.tasks, self.preparer, self.cfg = model, tasks, preparer, cfg
        self.device = preparer.device
        self.model.to(self.device)
        self.log_path = Path(log_path) if log_path else None
        self.history: list[dict[str, Any]] = []

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        gen = torch.Generator().manual_seed(self.cfg.seed)
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle, num_workers=self.cfg.num_workers,
                          drop_last=False, generator=gen, pin_memory=self.device.type == "cuda")

    @torch.no_grad()
    def fit_normalizers(self, ds: Dataset) -> None:
        if not hasattr(self.model, "fit_normalizers"):
            return
        acc: dict[str, list[torch.Tensor]] = {}
        for i, sample in enumerate(self._loader(ds, shuffle=True)):
            if i >= self.cfg.normalizer_batches:
                break
            batch = self.preparer(sample, train=False)
            for g, x in batch.readings.items():
                acc.setdefault(g, []).append(x.reshape(-1, x.shape[-1]))
        self.model.fit_normalizers({g: torch.cat(v) for g, v in acc.items()})

    def _loss(self, outputs, batch) -> tuple[torch.Tensor, dict[str, float]]:
        total, parts = 0.0, {}
        for t in self.tasks:
            loss = t.loss(outputs, batch)
            total = total + t.weight * loss
            parts[f"{t.name}/loss"] = float(loss.detach())
        return total, parts

    def _autocast(self):
        if self.cfg.amp == "bf16":
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        if self.cfg.amp != "none":
            raise ValueError(f"Unknown amp mode {self.cfg.amp!r} (none | bf16)")
        return contextlib.nullcontext()

    def _chunks(self, batch):
        """Consecutive ``tbptt_chunk``-step pieces of ``batch`` (the whole batch when chunking is off)."""
        c, L = self.cfg.tbptt_chunk, batch.num_steps
        if c <= 0 or L <= c:
            return [batch]
        return [batch.steps(s, min(s + c, L)) for s in range(0, L, c)]

    def fit(self, train_ds: Dataset, val_ds: Dataset | None = None) -> dict[str, Any]:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        self.fit_normalizers(train_ds)
        loader = self._loader(train_ds, shuffle=True)
        steps_per_epoch = len(loader) if not cfg.max_steps_per_epoch else min(len(loader), cfg.max_steps_per_epoch)
        if cfg.tbptt_chunk > 0:  # one optimizer step per chunk
            steps_per_epoch *= max(1, math.ceil(getattr(train_ds, "window", cfg.tbptt_chunk) / cfg.tbptt_chunk))
        epochs = max(cfg.epochs, math.ceil(cfg.min_steps / max(1, steps_per_epoch)))
        total_steps = max(1, epochs * steps_per_epoch)
        warmup = max(1, int(cfg.warmup_epochs * steps_per_epoch))
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
        )
        best = {"score": -math.inf, "state": None, "epoch": -1, "step": 0, "bad": 0}
        acc = {"t0": time.time(), "running": {}, "n": 0}
        step = 0

        def checkpoint(epoch: int) -> bool:
            """Validate, keep the best weights, log; returns True when early stopping triggers."""
            n = max(1, acc["n"])
            record = {"epoch": epoch, "step": step, "time": time.time() - acc["t0"], "lr": sched.get_last_lr()[0],
                      **{f"train/{k}": v / n for k, v in acc["running"].items()}}
            if val_ds is not None and len(val_ds) > 0:
                record.update({f"val/{k}": v for k, v in self.evaluate(val_ds).items()})
                score = record.get(f"val/{cfg.select_metric}", -record.get("val/loss", 0.0))
                if score > best["score"]:
                    best.update(score=score, epoch=epoch, step=step, bad=0, state=copy.deepcopy(self.model.state_dict()))
                else:
                    best["bad"] += 1
            self.history.append(record)
            self._log(record)
            acc.update(t0=time.time(), running={}, n=0)
            self.model.train()
            return bool(cfg.patience and best["bad"] >= cfg.patience)

        stop = False
        for epoch in range(epochs):
            self.model.train()
            for i, sample in enumerate(loader):
                if cfg.max_steps_per_epoch and i >= cfg.max_steps_per_epoch:
                    break
                state = None
                for chunk in self._chunks(self.preparer(sample, train=True)):
                    with self._autocast():
                        outputs, state = self.model(chunk, state, output_steps=cfg.output_steps)
                        loss, parts = self._loss(outputs, chunk)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    # pre-clip gradient norm (logged; clipping only when grad_clip > 0)
                    gnorm = nn.utils.clip_grad_norm_(self.model.parameters(),
                                                     cfg.grad_clip if cfg.grad_clip > 0 else float("inf"))
                    opt.step()
                    sched.step()
                    state = detach_state(state)
                    step += 1
                    acc["n"] += 1
                    for k, v in {**parts, "grad_norm": float(gnorm)}.items():
                        acc["running"][k] = acc["running"].get(k, 0.0) + v
                    if cfg.log_every and step % cfg.log_every == 0:
                        print(f"  step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in parts.items()))
                if cfg.eval_every and step % cfg.eval_every == 0 and checkpoint(epoch):
                    stop = True
                    break
            if stop:
                break
            if not cfg.eval_every and checkpoint(epoch):
                break
        if cfg.eval_every and acc["n"] and not stop:
            checkpoint(epochs - 1)  # steps since the last evaluation
        if best["state"] is not None:
            self.model.load_state_dict(best["state"])
        return {"best_epoch": best["epoch"], "best_step": best["step"], "best_score": best["score"],
                "history": self.history}

    @torch.no_grad()
    def evaluate(self, ds: Dataset) -> dict[str, float]:
        self.model.eval()
        sums: dict[str, float] = {}
        count, loss_sum = 0, 0.0
        for sample in self._loader(ds, shuffle=False):
            state = None
            for chunk in self._chunks(self.preparer(sample, train=False)):
                with self._autocast():
                    outputs, state = self.model(chunk, state, output_steps=self.cfg.output_steps)
                    loss, _ = self._loss(outputs, chunk)
                outputs = {k: v.float() for k, v in outputs.items()}
                B = chunk.batch_size
                loss_sum += float(loss) * B
                count += B
                for t in self.tasks:
                    for k, v in t.metrics(outputs, chunk).items():
                        sums[f"{t.name}/{k}"] = sums.get(f"{t.name}/{k}", 0.0) + float(v.sum())
        out = {k: v / max(count, 1) for k, v in sums.items()}
        out["loss"] = loss_sum / max(count, 1)
        return out

    @torch.no_grad()
    def confusion_matrix(self, ds: Dataset, task: Task, num_classes: int) -> torch.Tensor:
        self.model.eval()
        cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for sample in self._loader(ds, shuffle=False):
            state = None
            for chunk in self._chunks(self.preparer(sample, train=False)):
                with self._autocast():
                    outputs, state = self.model(chunk, state, output_steps=self.cfg.output_steps)
                pred = task.predict(outputs).cpu()
                y = chunk.labels[task.label].long().cpu()
                cm.index_put_((y, pred), torch.ones_like(y), accumulate=True)
        return cm

    def _log(self, record: dict[str, Any]) -> None:
        keys = [k for k in record if k.startswith("val/") and ("acc" in k or "mae" in k or k == "val/loss")]
        msg = f"epoch {record['epoch']:3d} | train " + ", ".join(
            f"{k[6:]}={v:.4f}" for k, v in record.items() if k.startswith("train/"))
        if keys:
            msg += " | " + ", ".join(f"{k}={record[k]:.4f}" for k in keys)
        print(msg)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")


def save_checkpoint(path: str | Path, model: nn.Module, extras: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **extras}, path)


def train_config_dict(cfg: TrainConfig) -> dict[str, Any]:
    return asdict(cfg)
