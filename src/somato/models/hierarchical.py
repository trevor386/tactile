"""The hierarchical somatosensory encoder.

    readings[g] [B,L,S,N_g,C] --(stage 1, "receptors": shared per-sensor temporal encoder per group)--> [B,L,N_g,d_g]
        --(project per group)--> [B,L,N_g,D]
        --(stage 2, "spinal cord": spatial interaction within each group, from current sensor poses)--> [B,L,N_g,D]
        --(stage 3, "brain": task heads; the brain head fuses groups across the body)--> {task: [B,L,...]}

Sensor groups (modalities) stay segregated until stage 3 (``fusion: segregated``): each group has its own stage-1
encoder and its own stage-2 operator, and only the heads see several groups. ``fusion: mixed`` instead runs one
stage-2 graph over all groups (version 0; kept as an ablation). Stages 2-3 run every ``brain_stride`` latent steps.

Nothing in the weights depends on the number of sensors or their placement (unless the
``sensor_id_embedding`` baseline option is enabled), so a trained model can be evaluated on a robot
with a different sensor layout as long as its sensor groups have the same names and channels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn

from somato.models.batch import GroupSpec, SomatoBatch
from somato.models.common import InputNormalizer
from somato.models.heads import HeadConfig, build_head
from somato.models.spatial import SpatialConfig, SpatialStack
from somato.models.temporal import TEMPORAL_ENCODERS
from somato.utils.config import deep_merge, from_dict


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
    # "segregated": one stage-2 operator per sensor group, built from `spatial` merged with
    # `spatial_overrides[group]`, so modalities meet only in stage 3; a `graph.k_per_group` entry is then that group's
    # k. "mixed": one stage-2 graph over all groups (version 0).
    fusion: str = "segregated"
    spatial_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    heads: dict[str, Any] = field(default_factory=lambda: {"terrain": HeadConfig()})
    # Stages 2-3 run at every `brain_stride`-th latent step (counted from the start of the stream, so a window of
    # L steps ends on one); stage 1 runs at every step.
    brain_stride: int = 1
    cluster_mode: str = "body"  # how LayoutInfo clusters are built for cluster-attention heads
    num_clusters: int | None = None
    sensor_id_embedding: int = 0  # >0: absolute per-sensor embedding (unstructured baselines only)

    def __post_init__(self):
        self.heads = {k: v if isinstance(v, HeadConfig) else from_dict(HeadConfig, v) for k, v in self.heads.items()}
        if self.fusion not in ("segregated", "mixed"):
            raise ValueError(f"Unknown fusion {self.fusion!r} (segregated | mixed)")
        if self.brain_stride < 1:
            raise ValueError("brain_stride must be >= 1")

    def group_spatial(self, group: str) -> SpatialConfig:
        """Stage-2 config of one group in segregated mode (an override that changes the operator type does not
        inherit the default operator's ``params``, which are type-specific; the base's ``k_per_group`` entry for the
        group becomes its ``k`` before the override, so an override's own ``graph.k`` wins)."""
        base, over = asdict(self.spatial), self.spatial_overrides.get(group, {})
        if base["graph"].get("k_per_group"):
            base["graph"]["k"] = base["graph"]["k_per_group"].get(group, base["graph"]["k"])
            base["graph"]["k_per_group"] = None
        merged = deep_merge(base, over)
        if over.get("type", base["type"]) != base["type"]:
            merged["params"] = dict(over.get("params", {}))
        cfg = from_dict(SpatialConfig, merged)
        if cfg.graph.k_per_group:  # given in the override itself
            cfg.graph.k = cfg.graph.k_per_group.get(group, cfg.graph.k)
            cfg.graph.k_per_group = None
        return cfg


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
        if cfg.fusion == "mixed":
            self.spatial = SpatialStack(D, cfg.spatial)
        else:
            self.spatial = nn.ModuleDict({g: SpatialStack(D, cfg.group_spatial(g)) for g in groups})
        self.heads = nn.ModuleDict({name: build_head(D, hcfg) for name, hcfg in cfg.heads.items()})
        # Recurrent heads need every brain step of a window, not only the output steps.
        self._recurrent_heads = any(getattr(h, "recurrent", False) for h in self.heads.values())

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

    def encode_spatial(self, h: torch.Tensor, pos: torch.Tensor, rot: torch.Tensor, info, mask) -> torch.Tensor:
        """Stage 2 on ``[B', N, D]`` features of one step each: per group (segregated) or over all groups (mixed)."""
        if self.cfg.fusion == "mixed":
            return self.spatial(h, pos, rot, info, mask)
        outs = []
        for g in info.group_names:
            sl = info.slices[g]
            outs.append(self.spatial[g](h[:, sl], pos[:, sl], rot[:, sl], info.subset(g),
                                        None if mask is None else mask[:, sl]))
        return torch.cat(outs, dim=1)

    def forward(self, batch: SomatoBatch, state: dict | None = None, output_steps: int | None = None,
                return_features: bool = False) -> tuple[dict[str, torch.Tensor], dict]:
        """Run all stages.

        Args:
            batch: inputs with ``L`` latent steps.
            state: recurrent state from a previous call (``None`` starts fresh).
            output_steps: predict at the last ``k`` brain steps only (stage 1 always runs on all ``L`` steps to
                build up its memory, recurrent heads on all brain steps). ``None`` = every brain step.
            return_features: also return the stage-2 node features under ``"features"``.

        Returns:
            ``(outputs, state)`` where ``outputs[task]`` is ``[B, k, ...]`` and ``outputs["steps"]`` ``[B, k]`` the
            latent steps (indices into this call's ``L``) the predictions belong to, for per-step labels; empty when
            no brain step falls in this call (streaming or truncated BPTT with ``brain_stride > 1``).
        """
        h, tstate = self.encode_temporal(batch, state)
        B, L, N, D = h.shape
        t0 = int((state or {}).get("t", 0))
        steps = [t for t in range(L) if (t0 + t + 1) % self.cfg.brain_stride == 0]
        n_out = len(steps) if output_steps is None else min(output_steps, len(steps))
        run = steps if self._recurrent_heads else steps[len(steps) - n_out:]
        prev_heads = (state or {}).get("heads", {})
        new_state = {"temporal": tstate, "heads": dict(prev_heads), "t": t0 + L}
        if not run:
            return {}, new_state
        k = len(run)
        idx = torch.tensor(run, device=h.device)
        h, pos, rot = h[:, idx], batch.pos[:, idx], batch.rot[:, idx]
        mask = None
        if batch.node_mask is not None:
            mask = batch.node_mask[:, None].expand(B, k, N).reshape(B * k, N)
        hs = self.encode_spatial(h.reshape(B * k, N, D), pos.reshape(B * k, N, 3), rot.reshape(B * k, N, 3, 3),
                                 batch.info, mask).view(B, k, N, D)
        outputs = {}
        for name, head in self.heads.items():
            out, new_state["heads"][name] = head(hs, pos, batch.info, batch.node_mask, prev_heads.get(name))
            outputs[name] = out[:, k - n_out:]
        if return_features:
            outputs["features"] = hs[:, k - n_out:]
        outputs["steps"] = idx[k - n_out:].expand(B, -1)
        return outputs, new_state


def _as_dict(cfg) -> dict[str, Any]:
    return {"type": cfg.type, "latent_dim": cfg.latent_dim, "params": dict(cfg.params)}
