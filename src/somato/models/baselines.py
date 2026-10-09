"""Unstructured baselines for the data-efficiency comparison.

The Phase 1 completion criterion is that the hierarchical encoder matches or beats an *unstructured*
baseline of comparable size with less training data. The baselines get the same information as the structured
model, including the kinematics (sensor geometry), so a difference measures the inductive bias, not an input:

* :class:`FlatRecurrentModel`: every reading of every sensor is flattened into one vector per latent step, plus
  the pose of every body in the robot frame, and fed to an MLP + GRU (the "flatten all taxels" approach of most
  prior work). It is tied to one sensor layout.
* :class:`TransformerBaseline`: one token per (sensor, time patch) from its raw samples, with the sensor's
  current position and normal in the robot frame as positional encoding, and global self-attention over all
  sensors, modalities and time: no locality, no hierarchy, no modality segregation.
* Config-only variants of the hierarchical model (see ``configs/models``), e.g. ``spatial.type: none``.

The *robot frame* is the frame of the IMU (the first ``imu``-kind sensor; node 0 if there is none), which
hardware knows from its own kinematics, so neither baseline sees the world pose.

:func:`match_parameter_count` scales a model family's width to a parameter budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn

from somato.geometry.layout import LayoutInfo
from somato.models.batch import GroupSpec, SomatoBatch
from somato.models.common import FourierEncoding, InputNormalizer, mlp
from somato.models.heads import HeadConfig
from somato.utils.config import from_dict


def reference_node(info: LayoutInfo) -> int:
    """Index of the node whose frame is the robot frame: the first IMU sensor, else node 0."""
    for g, kind in zip(info.group_names, info.group_kinds):
        if kind == "imu":
            return info.slices[g].start
    return 0


def in_robot_frame(pos: torch.Tensor, rot: torch.Tensor, ref: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sensor poses ``[..., N, 3]``, ``[..., N, 3, 3]`` expressed in the frame of node ``ref``."""
    r_t = rot[..., ref, :, :].transpose(-1, -2).unsqueeze(-3)  # [..., 1, 3, 3]
    p = (r_t @ (pos - pos[..., ref : ref + 1, :]).unsqueeze(-1)).squeeze(-1)
    return p, r_t @ rot


@dataclass
class FlatBaselineConfig:
    architecture: str = "flat_recurrent"
    hidden: int = 128
    mlp_layers: int = 2
    rnn_layers: int = 1
    chunk_features: str = "stats"  # "stats": mean/std/last per sensor channel; "flatten": every sample
    # Append the pose of every sensor-carrying body in the robot frame (position / `kinematics_scale` and the first
    # two rotation columns) at each latent step: the same geometric information the structured model gets.
    kinematics: bool = True
    kinematics_scale: float = 0.5  # m
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
        # One anchor sensor per body: its pose is a fixed transform of the body pose (information-equivalent).
        bodies = torch.unique(info.body_index)
        anchors = torch.stack([(info.body_index == b).nonzero()[0, 0] for b in bodies])
        self.register_buffer("anchors", anchors, persistent=False)
        self.ref = reference_node(info)
        if cfg.kinematics:
            in_dim += 9 * len(anchors)
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
            if batch.node_mask is not None:  # dead sensors read their mean (0 after normalization)
                sl = batch.info.slices[g]
                x = x * batch.node_mask[:, None, None, sl, None].to(x.dtype)
            if self.cfg.chunk_features == "stats":
                x = torch.stack([x.mean(2), x.std(2, unbiased=False), x[:, :, -1]], dim=2)
            feats.append(x.flatten(2))
        if self.cfg.kinematics:
            p, r = in_robot_frame(batch.pos, batch.rot, self.ref)  # [B, L, N, 3], [B, L, N, 3, 3]
            a = self.anchors
            feats.append(torch.cat([p[:, :, a] / self.cfg.kinematics_scale, r[:, :, a, :, :2].flatten(-2)], -1)
                         .flatten(2).to(feats[0].dtype))
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


@dataclass
class TransformerBaselineConfig:
    architecture: str = "transformer"
    dim: int = 64
    layers: int = 4
    attn_heads: int = 4
    ffn_mult: int = 2
    patch_steps: int = 5  # latent steps per token (time patch); windows are padded at the start to a multiple
    pos_bands: int = 4  # Fourier bands of the kinematic encoding (position / `kinematics_scale`, normal)
    kinematics_scale: float = 0.5  # m
    sensor_id: bool = False  # also a learned per-sensor embedding (absolute identity, layout-specific)
    max_sensors: int = 1024
    max_patches: int = 64
    heads: dict[str, Any] = field(default_factory=lambda: {"terrain": HeadConfig()})
    dropout: float = 0.0

    def __post_init__(self):
        self.heads = {k: v if isinstance(v, HeadConfig) else from_dict(HeadConfig, v) for k, v in self.heads.items()}
        if self.dim % self.attn_heads:
            raise ValueError("dim must be divisible by attn_heads")


class TransformerBaseline(nn.Module):
    """The least structured baseline: global self-attention over (sensor, time patch) tokens of every modality.

    Each token embeds one sensor's raw samples over ``patch_steps`` latent steps (a linear map per sensor group, the
    only weight sharing), plus a Fourier encoding of the sensor's position and normal in the robot frame at the end
    of the patch (kinematics), a group embedding, a time-patch embedding and optionally a sensor-ID embedding. A
    class token reads out one prediction per window, ``[B, 1, out]``. Dead sensors (``node_mask``) read their mean
    (0 after normalization), as in the flat baseline; masking them out of the attention instead would need a dense
    ``[B, heads, T*N, T*N]`` mask and rule out flash attention. Window-level only (streaming via
    ``OnlineEncoder(window=...)``).
    """

    def __init__(self, groups: dict[str, GroupSpec], info: LayoutInfo, cfg: TransformerBaselineConfig):
        super().__init__()
        self.cfg, self.groups = cfg, groups
        D, P = cfg.dim, cfg.patch_steps
        self.normalizers = nn.ModuleDict({g: InputNormalizer(s.channels) for g, s in groups.items()})
        self.patch = nn.ModuleDict({g: nn.Linear(P * s.substeps * s.channels, D) for g, s in groups.items()})
        self.group_embed = nn.Embedding(len(groups), D)
        self.time_embed = nn.Embedding(cfg.max_patches, D)
        self.kin_enc = FourierEncoding(6, cfg.pos_bands)
        self.kin = nn.Linear(self.kin_enc.out_dim, D)
        self.sensor_embed = nn.Embedding(cfg.max_sensors, D) if cfg.sensor_id else None
        self.cls = nn.Parameter(torch.randn(D) * 0.02)
        layer = nn.TransformerEncoderLayer(D, cfg.attn_heads, cfg.ffn_mult * D, cfg.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, norm=nn.LayerNorm(D), enable_nested_tensor=False)
        for name, h in cfg.heads.items():
            if h.type != "global":
                raise ValueError(f"TransformerBaseline only supports global heads (head '{name}' is '{h.type}')")
        self.heads = nn.ModuleDict({n: mlp(D, h.hidden, h.out_dim, 2, dropout=cfg.dropout)
                                    for n, h in cfg.heads.items()})
        self.ref = reference_node(info)

    @torch.no_grad()
    def fit_normalizers(self, readings: dict[str, torch.Tensor]) -> None:
        for g, x in readings.items():
            if g in self.normalizers:
                self.normalizers[g].fit(x)

    def forward(self, batch: SomatoBatch, state: dict | None = None, output_steps: int | None = None,
                return_features: bool = False):
        cfg, P = self.cfg, self.cfg.patch_steps
        B, L = batch.pos.shape[:2]
        T = math.ceil(L / P)
        pad = T * P - L
        tokens = []
        for gi, g in enumerate(batch.info.group_names):
            x = self.normalizers[g](batch.readings[g])  # [B, L, S, N, C]
            sl = batch.info.slices[g]
            if batch.node_mask is not None:  # dead sensors read their mean (0 after normalization)
                x = x * batch.node_mask[:, None, None, sl, None].to(x.dtype)
            if pad:  # repeat the first step so every patch is full
                x = torch.cat([x[:, :1].expand(B, pad, *x.shape[2:]), x], 1)
            _, _, S, N, C = x.shape
            x = x.view(B, T, P, S, N, C).permute(0, 1, 4, 2, 3, 5).reshape(B, T, N, P * S * C)
            tok = self.patch[g](x) + self.group_embed.weight[gi]
            if self.sensor_embed is not None:
                tok = tok + self.sensor_embed.weight[sl]
            tokens.append(tok)
        tok = torch.cat(tokens, 2)  # [B, T, N, D]
        # Kinematics at the last latent step of each patch, in the robot frame.
        last = torch.arange(P - 1 - pad, L, P, device=tok.device).clamp_min(0)
        p, r = in_robot_frame(batch.pos[:, last], batch.rot[:, last], self.ref)
        tok = tok + self.kin(self.kin_enc(torch.cat([p / cfg.kinematics_scale, r[..., :, 2]], -1)).to(tok.dtype))
        tok = tok + self.time_embed.weight[:T, None]
        N = tok.shape[2]
        seq = torch.cat([self.cls.expand(B, 1, -1).to(tok.dtype), tok.reshape(B, T * N, -1)], 1)
        out = self.encoder(seq)[:, :1]  # class token, [B, 1, D]
        outputs = {n: head(out) for n, head in self.heads.items()}
        if return_features:
            outputs["features"] = out
        return outputs, {}


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
