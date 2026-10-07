"""Unstructured baselines for the data-efficiency comparison.

The Phase 1 completion criterion is that the hierarchical encoder matches or beats an *unstructured*
baseline of comparable size with less training data. Two kinds of baselines are provided:

* :class:`FlatRecurrentModel` -- every reading of every sensor is flattened into one vector per
  latent step and fed to an MLP + GRU (the "flatten all taxels" approach of most prior work). It
  is tied to one sensor layout.
* Config-only variants of the hierarchical model (see ``configs/models``): ``spatial.type: none``
  (no interaction), or ``full_attention`` with ``edge_features: none`` and ``sensor_id_embedding``
  (a transformer over sensors without geometry).

:func:`match_parameter_count` scales a model family's width to a parameter budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn

from somato.geometry.layout import LayoutInfo
from somato.models.batch import GroupSpec, SomatoBatch
from somato.models.common import InputNormalizer, mlp
from somato.models.heads import HeadConfig
from somato.utils.config import from_dict


@dataclass
class FlatBaselineConfig:
    architecture: str = "flat_recurrent"
    hidden: int = 128
    mlp_layers: int = 2
    rnn_layers: int = 1
    chunk_features: str = "stats"  # "stats": mean/std/last per sensor channel; "flatten": every sample
    heads: dict[str, Any] = field(default_factory=lambda: {"terrain": HeadConfig()})
    dropout: float = 0.0

    def __post_init__(self):
        self.heads = {k: v if isinstance(v, HeadConfig) else from_dict(HeadConfig, v) for k, v in self.heads.items()}


class FlatRecurrentModel(nn.Module):
    def __init__(self, groups: dict[str, GroupSpec], info: LayoutInfo, cfg: FlatBaselineConfig):
        super().__init__()
        self.cfg, self.groups = cfg, groups
        self.sizes = {g: info.slices[g].stop - info.slices[g].start for g in info.group_names}
        per = 3 if cfg.chunk_features == "stats" else None
        in_dim = sum(
            self.sizes[g] * s.channels * (per if per else s.substeps) for g, s in groups.items() if g in self.sizes
        )
        self.normalizers = nn.ModuleDict({g: InputNormalizer(s.channels) for g, s in groups.items()})
        self.encoder = mlp(in_dim, cfg.hidden, cfg.hidden, layers=cfg.mlp_layers, dropout=cfg.dropout)
        self.rnn = nn.GRU(cfg.hidden, cfg.hidden, num_layers=cfg.rnn_layers, batch_first=True)
        for name, h in cfg.heads.items():
            if h.type != "global":
                raise ValueError(f"FlatRecurrentModel only supports global heads (head '{name}' is '{h.type}')")
        self.heads = nn.ModuleDict({n: mlp(cfg.hidden, cfg.hidden, h.out_dim, 2, dropout=cfg.dropout)
                                    for n, h in cfg.heads.items()})

    @torch.no_grad()
    def fit_normalizers(self, readings: dict[str, torch.Tensor]) -> None:
        for g, x in readings.items():
            if g in self.normalizers:
                self.normalizers[g].fit(x)

    def forward(self, batch: SomatoBatch, state: dict | None = None, output_steps: int | None = None,
                return_features: bool = False):
        feats = []
        for g in batch.info.group_names:
            x = self.normalizers[g](batch.readings[g])  # [B, L, S, N, C]
            if x.shape[3] != self.sizes[g]:
                raise ValueError(f"FlatRecurrentModel was built for {self.sizes[g]} '{g}' sensors, got {x.shape[3]}")
            if self.cfg.chunk_features == "stats":
                x = torch.stack([x.mean(2), x.std(2, unbiased=False), x[:, :, -1]], dim=2)
            feats.append(x.flatten(2))
        z = self.encoder(torch.cat(feats, dim=-1))
        h0 = None if state is None else state.get("h")
        out, h = self.rnn(z, h0)
        L = out.shape[1]
        k = L if output_steps is None else min(output_steps, L)
        out = out[:, L - k :]
        outputs = {n: head(out) for n, head in self.heads.items()}
        if return_features:
            outputs["features"] = out
        return outputs, {"h": h}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_parameter_count(make: Callable[[int], nn.Module], target: int, low: int = 4, high: int = 4096) -> tuple[int, nn.Module]:
    """Binary-search the integer width ``w`` so that ``count_parameters(make(w))`` is closest to ``target``."""
    best = None
    while low <= high:
        mid = (low + high) // 2
        model = make(mid)
        n = count_parameters(model)
        if best is None or abs(n - target) < abs(best[2] - target):
            best = (mid, model, n)
        if n < target:
            low = mid + 1
        else:
            high = mid - 1
    return best[0], best[1]
