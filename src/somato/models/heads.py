"""Stage 3: task-specific heads.

Heads consume the per-sensor features of stage 2 at each output step, ``h [B, L, N, D]``, and return
``[B, L, ...]`` predictions, one per step, so the same head runs offline (windows) and online (one
step at a time). Pooling options, from least to most structured:

* ``mean`` / ``max`` / ``meanmax`` over all sensors,
* ``attention``: learned-query attention pooling (global attention over sensors),
* ``cluster_attention``: sensors -> fixed geometric clusters (e.g. robot links) -> transformer
  layers between clusters with a relative-distance bias -> pooled. This is the "global attention"
  stage of the proposal, where distant regions can influence each other.

The ``brain`` head (stage 3 of the segregated model) is where sensor groups first meet: per-(group, body) region
tokens, attention biased by their *current* distances, and a memory over brain steps (see :class:`BrainHead`).

Heads may optionally carry their own recurrent state across steps (``recurrent: true``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from somato.geometry.layout import LayoutInfo
from somato.models.common import BiasedSelfAttention, FeedForward, masked_max, masked_mean, masked_softmax, mlp
from somato.models.graph import pairwise_sq_distance
from somato.utils.registry import Registry

HEADS: Registry[nn.Module] = Registry("head")


@dataclass
class HeadConfig:
    type: str = "global"  # "global" | "per_body"
    out_dim: int = 5
    pool: str = "cluster_attention"  # mean | max | meanmax | attention | cluster_attention
    hidden: int = 128
    heads: int = 4
    cluster_layers: int = 2
    cluster_length_scale: float = 0.2  # m, scale of the inter-cluster distance bias
    recurrent: bool = False
    dropout: float = 0.0
    max_groups: int = 8  # brain head: size of the modality-embedding table (sensor groups in layout order)
    node_pool: bool = False  # brain head: also feed the mean and max over all sensors (the static pool head's input)


def cluster_reduce(h: torch.Tensor, cluster_id: torch.Tensor, num_clusters: int, mask: torch.Tensor | None):
    """Mean and max of node features per cluster. ``h [B, N, D]`` -> ``([B, C, D], [B, C, D], count [B, C])``."""
    B, N, D = h.shape
    m = torch.ones(B, N, dtype=torch.bool, device=h.device) if mask is None else mask
    w = m.to(h.dtype)
    idx = cluster_id.to(h.device).view(1, N, 1).expand(B, N, D)
    total = h.new_zeros(B, num_clusters, D).scatter_add_(1, idx, h * w.unsqueeze(-1))
    count = h.new_zeros(B, num_clusters).scatter_add_(1, idx[..., 0], w)
    mean = total / count.clamp_min(1).unsqueeze(-1)
    hm = h.masked_fill(~m.unsqueeze(-1), float("-inf"))
    mx = h.new_full((B, num_clusters, D), float("-inf")).scatter_reduce_(1, idx, hm, "amax", include_self=True)
    mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
    return mean, mx, count


class AttentionPool(nn.Module):
    """Multi-head attention pooling with learned queries."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, heads)
        self.value = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = self.norm(h)
        w = masked_softmax(self.score(x), None if mask is None else mask.unsqueeze(-1), dim=1)  # [B, N, H]
        v = self.value(x).view(*x.shape[:2], self.heads, -1)  # [B, N, H, D/H]
        return (w.unsqueeze(-1) * v).sum(1).flatten(1)


class ClusterAttentionPool(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, length_scale: float, dropout: float = 0.0):
        super().__init__()
        self.heads, self.length_scale = heads, length_scale
        self.token = nn.Linear(2 * dim, dim)
        self.dist_bias = mlp(1, 32, heads, layers=2)
        self.blocks = nn.ModuleList(BiasedSelfAttention(dim, heads, dropout) for _ in range(layers))
        self.ffns = nn.ModuleList(FeedForward(dim, 2, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, pos, info: LayoutInfo, mask):
        C = info.num_clusters
        mean, mx, count = cluster_reduce(h, info.cluster_id, C, mask)
        tokens = self.token(torch.cat([mean, mx], -1))
        cpos, _, _ = cluster_reduce(pos, info.cluster_id, C, mask)  # cluster centroids [B, C, 3]
        valid = count > 0
        d = torch.cdist(cpos, cpos, compute_mode="donot_use_mm_for_euclid_dist") / self.length_scale
        bias = self.dist_bias(d.unsqueeze(-1)).permute(0, 3, 1, 2)  # [B, H, C, C]
        for attn, ffn in zip(self.blocks, self.ffns):
            tokens = ffn(attn(tokens, bias, valid))
        tokens = self.norm(tokens)
        return torch.cat([masked_mean(tokens, valid, 1), masked_max(tokens, valid, 1)], -1)


@HEADS.register("global")
class GlobalHead(nn.Module):
    """One prediction per step for the whole robot (classification logits or regression)."""

    def __init__(self, dim: int, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.pool == "attention":
            self.pool, pooled = AttentionPool(dim, cfg.heads), dim
        elif cfg.pool == "cluster_attention":
            self.pool = ClusterAttentionPool(dim, cfg.heads, cfg.cluster_layers, cfg.cluster_length_scale, cfg.dropout)
            pooled = 2 * dim
        elif cfg.pool in ("mean", "max"):
            self.pool, pooled = None, dim
        elif cfg.pool == "meanmax":
            self.pool, pooled = None, 2 * dim
        else:
            raise ValueError(f"Unknown pool '{cfg.pool}'")
        self.recurrent = cfg.recurrent
        self.rnn = nn.GRU(pooled, pooled, batch_first=True) if cfg.recurrent else None
        self.out = mlp(pooled, cfg.hidden, cfg.out_dim, layers=2, dropout=cfg.dropout)

    def init_state(self, batch: int, device) -> dict:
        if self.rnn is None:
            return {}
        return {"h": torch.zeros(1, batch, self.rnn.hidden_size, device=device)}

    def _pool(self, h, pos, info, mask):
        if self.cfg.pool == "mean":
            return masked_mean(h, mask, 1)
        if self.cfg.pool == "max":
            return masked_max(h, mask, 1)
        if self.cfg.pool == "meanmax":
            return torch.cat([masked_mean(h, mask, 1), masked_max(h, mask, 1)], -1)
        if self.cfg.pool == "attention":
            return self.pool(h, mask)
        return self.pool(h, pos, info, mask)

    def forward(self, h: torch.Tensor, pos: torch.Tensor, info: LayoutInfo, node_mask: torch.Tensor | None,
                state: dict | None = None) -> tuple[torch.Tensor, dict]:
        B, L, N, D = h.shape
        mask = None if node_mask is None else node_mask[:, None].expand(B, L, N).reshape(B * L, N)
        pooled = self._pool(h.reshape(B * L, N, D), pos.reshape(B * L, N, 3), info, mask).view(B, L, -1)
        new_state = {}
        if self.rnn is not None:
            state = state or self.init_state(B, h.device)
            pooled, hn = self.rnn(pooled, state["h"])
            new_state = {"h": hn}
        return self.out(pooled), new_state


def region_tokens(info: LayoutInfo) -> tuple[torch.Tensor, torch.Tensor]:
    """Region of every node and group of every region: one region per (sensor group, body) pair that has sensors.

    Returns ``(region_id [N], region_group [R])``; regions are ordered by group (layout order), then body.
    """
    if "regions" not in info.cache:
        nb = int(info.body_index.max()) + 1
        key = info.group_id.long() * nb + info.body_index.long()
        uniq, region = torch.unique(key, sorted=True, return_inverse=True)
        info.cache["regions"] = (region, torch.div(uniq, nb, rounding_mode="floor"))
    return info.cache["regions"]


@HEADS.register("brain")
class BrainHead(nn.Module):
    """Stage 3 ("brain"): fuses the sensor groups across the whole body, with dynamic geometry and memory.

    1. **Region tokens.** Stage-2 features are pooled (masked mean and max) per (group, body) region, e.g. one
       tactile token per link, one token per joint sensor and one IMU token, and get a learned embedding of their
       group. Modalities first meet here.
    2. **Dynamic geometry.** A token sits at the current centroid of its sensors; each attention head adds a learned
       function of the current pairwise token distance to its logits. The attention pattern therefore follows the
       body as it moves; there is no fixed map and no per-region identity, so one set of weights runs on robots with
       other link counts.
    3. **Global attention** layers over the tokens, then masked mean and max over the tokens, concatenated with the
       mean and max of the input tokens. That short path trains from the first steps (attention-only pooling
       started on a long loss plateau).
    4. **Memory.** With ``recurrent: true`` (intended for this head), a GRU over brain steps; then an MLP to the
       output.

    Config: ``hidden`` (fused width), ``heads``, ``cluster_layers`` (attention layers), ``cluster_length_scale``
    (distance unit of the bias), ``max_groups``.
    """

    def __init__(self, dim: int, cfg: HeadConfig):
        super().__init__()
        self.cfg, self.recurrent = cfg, cfg.recurrent
        self.token = nn.Linear(2 * dim, dim)
        self.group_embed = nn.Embedding(cfg.max_groups, dim)
        nn.init.zeros_(self.group_embed.weight)
        self.dist_bias = mlp(1, 32, cfg.heads, layers=2)
        self.blocks = nn.ModuleList(BiasedSelfAttention(dim, cfg.heads, cfg.dropout) for _ in range(cfg.cluster_layers))
        self.ffns = nn.ModuleList(FeedForward(dim, 2, cfg.dropout) for _ in range(cfg.cluster_layers))
        self.norm = nn.LayerNorm(dim)
        self.fuse = nn.Sequential(nn.Linear((6 if cfg.node_pool else 4) * dim, cfg.hidden), nn.GELU())
        self.rnn = nn.GRU(cfg.hidden, cfg.hidden, batch_first=True) if cfg.recurrent else None
        self.out = mlp(cfg.hidden, cfg.hidden, cfg.out_dim, layers=2, dropout=cfg.dropout)

    def init_state(self, batch: int, device) -> dict:
        if self.rnn is None:
            return {}
        return {"h": torch.zeros(1, batch, self.rnn.hidden_size, device=device)}

    def forward(self, h: torch.Tensor, pos: torch.Tensor, info: LayoutInfo, node_mask: torch.Tensor | None,
                state: dict | None = None) -> tuple[torch.Tensor, dict]:
        B, L, N, D = h.shape
        region, region_group = region_tokens(info)
        R = region_group.shape[0]
        mask = None if node_mask is None else node_mask[:, None].expand(B, L, N).reshape(B * L, N)
        mean, mx, count = cluster_reduce(h.reshape(B * L, N, D), region, R, mask)
        tokens = self.token(torch.cat([mean, mx], -1)) + self.group_embed(region_group.to(h.device))
        cpos, _, _ = cluster_reduce(pos.reshape(B * L, N, 3), region, R, mask)  # current region centroids
        valid = count > 0
        d = pairwise_sq_distance(cpos).clamp_min(0).sqrt() / self.cfg.cluster_length_scale
        bias = self.dist_bias(d.unsqueeze(-1).to(tokens.dtype)).permute(0, 3, 1, 2)  # [BL, H, R, R]
        x = tokens
        for attn, ffn in zip(self.blocks, self.ffns):
            x = ffn(attn(x, bias, valid))
        x = self.norm(x)
        pooled = [masked_mean(x, valid, 1), masked_max(x, valid, 1),
                  masked_mean(tokens, valid, 1), masked_max(tokens, valid, 1)]
        if self.cfg.node_pool:
            hn = h.reshape(B * L, N, D)
            pooled += [masked_mean(hn, mask, 1), masked_max(hn, mask, 1)]
        pooled = torch.cat(pooled, -1)
        z = self.fuse(pooled).view(B, L, -1)
        new_state = {}
        if self.rnn is not None:
            state = state or self.init_state(B, h.device)
            z, hn = self.rnn(z, state["h"])
            new_state = {"h": hn}
        return self.out(z), new_state


@HEADS.register("per_body")
class PerBodyHead(nn.Module):
    """One prediction per sensor-carrying body and step, e.g. per-link slip. Output ``[B, L, num_bodies, out]``.

    Bodies are indexed by ``LayoutInfo.cluster_id`` with ``cluster_mode: body``.
    """

    def __init__(self, dim: int, cfg: HeadConfig):
        super().__init__()
        self.out = mlp(2 * dim, cfg.hidden, cfg.out_dim, layers=2, dropout=cfg.dropout)

    def init_state(self, batch: int, device) -> dict:
        return {}

    def forward(self, h, pos, info, node_mask, state=None):
        B, L, N, D = h.shape
        mask = None if node_mask is None else node_mask[:, None].expand(B, L, N).reshape(B * L, N)
        mean, mx, _ = cluster_reduce(h.reshape(B * L, N, D), info.cluster_id, info.num_clusters, mask)
        return self.out(torch.cat([mean, mx], -1)).view(B, L, info.num_clusters, -1), {}


def build_head(dim: int, cfg: HeadConfig) -> nn.Module:
    return HEADS.build(cfg.type, dim, cfg)
