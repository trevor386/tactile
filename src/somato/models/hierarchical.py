"""The hierarchical somatosensory encoder.

    readings[g] [B,L,S,N_g,C] --(stage 1: shared per-sensor temporal encoder per group)--> [B,L,N_g,d_g]
        --(project + group embedding, concat groups)--> [B,L,N,D]
        --(stage 2: geometric interaction at each step, using sensor poses)--> [B,L,N,D]
        --(stage 3: task heads)--> {task: [B,L,...]}

Nothing in the weights depends on the number of sensors or their placement (unless the
``sensor_id_embedding`` baseline option is enabled), so a trained model can be evaluated on a robot
with a different sensor layout as long as its sensor groups have the same names and channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from somato.models.batch import GroupSpec, SomatoBatch
from somato.models.common import InputNormalizer
from somato.models.heads import HeadConfig, build_head
from somato.models.spatial import SpatialConfig, SpatialStack
from somato.models.temporal import TEMPORAL_ENCODERS
from somato.utils.config import from_dict


@dataclass
class TemporalConfig:
    type: str = "conv_gru"
    latent_dim: int = 64
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    architecture: str = "hierarchical"
    dim: int = 64
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    # Per-group overrides of the temporal config, e.g. {"imu": {"type": "gru"}}.
    temporal_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    heads: dict[str, Any] = field(default_factory=lambda: {"terrain": HeadConfig()})
    cluster_mode: str = "body"  # how LayoutInfo clusters are built for cluster-attention heads
    num_clusters: int | None = None
    sensor_id_embedding: int = 0  # >0: absolute per-sensor embedding (unstructured baselines only)

    def __post_init__(self):
        self.heads = {k: v if isinstance(v, HeadConfig) else from_dict(HeadConfig, v) for k, v in self.heads.items()}


class HierarchicalSomatoModel(nn.Module):
    def __init__(self, groups: dict[str, GroupSpec], cfg: ModelConfig):
        super().__init__()
        self.cfg, self.groups = cfg, groups
        D = cfg.dim
        self.normalizers = nn.ModuleDict({g: InputNormalizer(s.channels) for g, s in groups.items()})
        self.temporal = nn.ModuleDict()
        self.proj = nn.ModuleDict()
        for g, spec in groups.items():
            tcfg = from_dict(TemporalConfig, {**_as_dict(cfg.temporal), **cfg.temporal_overrides.get(g, {})})
            self.temporal[g] = TEMPORAL_ENCODERS.build(
                tcfg.type, in_channels=spec.channels, substeps=spec.substeps, latent_dim=tcfg.latent_dim, **tcfg.params
            )
            self.proj[g] = nn.Linear(tcfg.latent_dim, D)
        self.group_embed = nn.ParameterDict({g: nn.Parameter(torch.zeros(D)) for g in groups})
        self.sensor_id = nn.Embedding(cfg.sensor_id_embedding, D) if cfg.sensor_id_embedding > 0 else None
        self.spatial = SpatialStack(D, cfg.spatial)
        self.heads = nn.ModuleDict({name: build_head(D, hcfg) for name, hcfg in cfg.heads.items()})

    # ------------------------------------------------------------------ utilities
    @torch.no_grad()
    def fit_normalizers(self, readings: dict[str, torch.Tensor]) -> None:
        """Fit per-group input standardization from example readings ``[..., C]``."""
        for g, x in readings.items():
            if g in self.normalizers:
                self.normalizers[g].fit(x)

    def init_state(self, batch_size: int, num_nodes: dict[str, int], device) -> dict:
        return {
            "temporal": {g: self.temporal[g].init_state(batch_size, n, device) for g, n in num_nodes.items()},
            "heads": {n: h.init_state(batch_size, device) for n, h in self.heads.items()},
        }

    # ------------------------------------------------------------------ forward
    def encode_temporal(self, batch: SomatoBatch, state: dict | None = None) -> tuple[torch.Tensor, dict]:
        """Stage 1 for all groups -> ``[B, L, N, D]`` (groups concatenated in layout order)."""
        tstate = (state or {}).get("temporal", {})
        zs, new_state = [], {}
        for g in batch.info.group_names:
            if g not in self.temporal:
                raise KeyError(f"Model has no encoder for sensor group '{g}' (has {list(self.temporal)})")
            x = self.normalizers[g](batch.readings[g])
            z, new_state[g] = self.temporal[g](x, tstate.get(g))
            zs.append(self.proj[g](z) + self.group_embed[g])
        h = torch.cat(zs, dim=2)
        if self.sensor_id is not None:
            h = h + self.sensor_id.weight[: h.shape[2]]
        return h, new_state

    def forward(self, batch: SomatoBatch, state: dict | None = None, output_steps: int | None = None,
                return_features: bool = False) -> tuple[dict[str, torch.Tensor], dict]:
        """Run all stages.

        Args:
            batch: inputs with ``L`` latent steps.
            state: recurrent state from a previous call (``None`` starts fresh).
            output_steps: run stages 2-3 only on the last ``k`` steps (stage 1 always runs on all
                ``L`` steps to build up its memory). ``None`` = all steps.
            return_features: also return the stage-2 node features under ``"features"``.

        Returns:
            ``(outputs, state)`` where ``outputs[task]`` is ``[B, k, ...]``.
        """
        h, tstate = self.encode_temporal(batch, state)
        B, L, N, D = h.shape
        k = L if output_steps is None else min(output_steps, L)
        h, pos, rot = h[:, L - k :], batch.pos[:, L - k :], batch.rot[:, L - k :]
        mask = None
        if batch.node_mask is not None:
            mask = batch.node_mask[:, None].expand(B, k, N).reshape(B * k, N)
        hs = self.spatial(h.reshape(B * k, N, D), pos.reshape(B * k, N, 3), rot.reshape(B * k, N, 3, 3),
                          batch.info, mask).view(B, k, N, D)
        outputs, hstate = {}, {}
        prev_heads = (state or {}).get("heads", {})
        for name, head in self.heads.items():
            outputs[name], hstate[name] = head(hs, pos, batch.info, batch.node_mask, prev_heads.get(name))
        if return_features:
            outputs["features"] = hs
        return outputs, {"temporal": tstate, "heads": hstate}


def _as_dict(cfg) -> dict[str, Any]:
    return {"type": cfg.type, "latent_dim": cfg.latent_dim, "params": dict(cfg.params)}
