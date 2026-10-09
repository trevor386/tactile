"""Stage 2: geometry-informed interaction between sensors.

The default operator is a **continuous kernel convolution**, the 3D, irregular-grid analogue of a CNN
kernel. A small network maps the relative geometry of a neighbor to kernel weights that are shared by
every sensor ("similar relative geometry => similar interaction"):

    y_i = sum_{j in N(i)} W(e_ij) h_j,   W(e) = sum_b phi_b(e) W_b

where ``phi(e)`` is an MLP of the edge features and ``W_b`` are ``num_basis`` learned matrices (a
low-rank factorization that keeps memory at ``O(E * num_basis)`` instead of ``O(E * D^2)``).

Alternatives (same interface, for ablations): geometry-biased local attention, EGNN (the
Jiang et al. 2025 e-skin baseline), full global attention, and no interaction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from somato.geometry.layout import LayoutInfo
from somato.models.common import BiasedSelfAttention, FeedForward, FourierEncoding, gather_nodes, masked_softmax, mlp
from somato.models.graph import EDGE_FEATURE_DIMS, NeighborGraph, edge_features, knn_graph, pairwise_sq_distance
from somato.utils.registry import Registry

SPATIAL_LAYERS: Registry[nn.Module] = Registry("spatial layer")


@dataclass
class SpatialContext:
    """Everything a spatial layer may use. ``B`` here is batch x time steps."""

    pos: torch.Tensor  # [B, N, 3] sensor positions (world)
    rot: torch.Tensor  # [B, N, 3, 3] sensor frames (world)
    area: torch.Tensor  # [N] sensor areas
    length_scale: float
    graph: NeighborGraph | None = None
    edges: torch.Tensor | None = None  # [B, N, K, F]
    node_mask: torch.Tensor | None = None  # [B, N]


# ---------------------------------------------------------------------------------------- layers


@SPATIAL_LAYERS.register("kernel3d")
@SPATIAL_LAYERS.register("continuous_conv")
class ContinuousConvBlock(nn.Module):
    """Continuous kernel convolution + feed-forward, pre-norm residual (registered as ``kernel3d``).

    The 3D-kernel approach to stage 2: a CNN kernel made continuous in space. Its *support* is the set of
    neighbours, best given as a physical radius (``graph.radius``, with ``graph.k`` only a cap), and its weights are
    a learned function of where the neighbour sits in sensor i's frame, evaluated at the current poses: when the
    body bends, sensors on adjacent links move relative to each other, so both the support and the kernel weights
    follow the motion. The alternative, graph approach is message passing on a neighbour graph whose messages also
    depend on the features (``geo_attention``, ``egnn``).

    Args:
        aggregation: ``mean`` (normalize by neighbor count), ``sum``, or ``area`` -- weight each
            neighbor by its taxel area, a quadrature approximation of the continuous convolution
            integral that makes the operator insensitive to sensor density.
        envelope: multiply the kernel by a learnable Gaussian window in distance (smooth locality prior).
    """

    def __init__(self, dim: int, edge_dim: int, num_basis: int = 16, kernel_hidden: int = 64,
                 kernel_layers: int = 3, fourier_bands: int = 0, aggregation: str = "mean", envelope: bool = True,
                 ffn_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.aggregation, self.envelope = aggregation, envelope
        self.enc = FourierEncoding(max(edge_dim, 1), fourier_bands)
        self.kernel = mlp(self.enc.out_dim, kernel_hidden, num_basis, layers=kernel_layers)
        self.weight = nn.Parameter(torch.randn(num_basis, dim, dim) / math.sqrt(num_basis * dim))  # [nb, D_in, D_out]
        self.bias = nn.Parameter(torch.zeros(dim))
        self.log_width = nn.Parameter(torch.zeros(()))  # envelope width in units of length_scale
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.ffn = FeedForward(dim, ffn_mult, dropout)

    def forward(self, h: torch.Tensor, ctx: SpatialContext) -> torch.Tensor:
        g, e = ctx.graph, ctx.edges
        if e.shape[-1] == 0:
            e = e.new_ones(e.shape[:-1] + (1,))  # constant kernel == plain neighborhood mean
        phi = self.kernel(self.enc(e)) * g.mask.unsqueeze(-1)  # [B, N, K, nb]
        if self.envelope:
            pj = gather_nodes(ctx.pos, g.index)
            d2 = (pj - ctx.pos[:, :, None]).pow(2).sum(-1) / ctx.length_scale**2
            phi = phi * torch.exp(-0.5 * d2 / self.log_width.exp().pow(2)).unsqueeze(-1)
        if self.aggregation == "area":
            a = ctx.area[g.index]  # [B, N, K]
            phi = phi * (a / ctx.area.mean()).unsqueeze(-1)
        if self.aggregation in ("mean", "area"):
            phi = phi / g.mask.sum(-1, keepdim=True).clamp_min(1).unsqueeze(-1)
        x = gather_nodes(self.norm(h), g.index)  # [B, N, K, D]
        agg = torch.matmul(phi.transpose(-1, -2), x)  # [B, N, nb, D]
        y = agg.flatten(-2) @ self.weight.flatten(0, 1) + self.bias  # sum_b W_b^T agg_b
        h = h + self.drop(y)
        return self.ffn(h)


@SPATIAL_LAYERS.register("geo_attention")
class GeoAttentionBlock(nn.Module):
    """Local multi-head attention over the neighbor graph with geometric biases.

    ``logit_ij = q_i.k_j / sqrt(d) + b(e_ij) - lambda_h d_ij^2``: physical proximity is a prior
    (``lambda`` learnable, initialized local) that content can override.
    """

    def __init__(self, dim: int, edge_dim: int, heads: int = 4, bias_hidden: int = 32, value_edges: bool = True,
                 distance_prior: bool = True, ffn_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dh, self.distance_prior = heads, dim // heads, distance_prior
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.edge_bias = mlp(edge_dim, bias_hidden, heads) if edge_dim > 0 else None
        self.edge_value = nn.Linear(edge_dim, dim) if (value_edges and edge_dim > 0) else None
        self.raw_lambda = nn.Parameter(torch.full((heads,), 0.5413))  # softplus -> 1.0
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.ffn = FeedForward(dim, ffn_mult, dropout)

    def forward(self, h, ctx):
        g = ctx.graph
        B, N, K = g.index.shape
        q, k, v = self.qkv(self.norm(h)).chunk(3, dim=-1)
        kj = gather_nodes(k, g.index).view(B, N, K, self.heads, self.dh)
        vj = gather_nodes(v, g.index)
        if self.edge_value is not None:
            vj = vj + self.edge_value(ctx.edges)
        vj = vj.view(B, N, K, self.heads, self.dh)
        logits = (q.view(B, N, 1, self.heads, self.dh) * kj).sum(-1) / math.sqrt(self.dh)  # [B, N, K, H]
        if self.edge_bias is not None:
            logits = logits + self.edge_bias(ctx.edges)
        if self.distance_prior:
            pj = gather_nodes(ctx.pos, g.index)
            d2 = (pj - ctx.pos[:, :, None]).pow(2).sum(-1, keepdim=True) / ctx.length_scale**2
            logits = logits - nn.functional.softplus(self.raw_lambda) * d2
        attn = masked_softmax(logits, g.mask.unsqueeze(-1), dim=2)
        y = (attn.unsqueeze(-1) * vj).sum(2).reshape(B, N, -1)
        h = h + self.drop(self.out(y))
        return self.ffn(h)


@SPATIAL_LAYERS.register("egnn")
class EGNNBlock(nn.Module):
    """E(n)-equivariant graph convolution (Satorras et al. 2021), as used by Jiang et al. (2025).

    Messages use only invariant quantities (squared distances + optional edge features). With
    ``update_coords`` the layer also moves node coordinates equivariantly for subsequent layers.
    """

    def __init__(self, dim: int, edge_dim: int, hidden: int = 64, update_coords: bool = False, dropout: float = 0.0):
        super().__init__()
        self.update_coords = update_coords
        self.norm = nn.LayerNorm(dim)
        self.phi_e = mlp(2 * dim + 1 + edge_dim, hidden, hidden, layers=2)
        self.phi_h = mlp(dim + hidden, hidden, dim, layers=2, dropout=dropout)
        self.phi_x = mlp(hidden, hidden, 1, layers=2) if update_coords else None

    def forward(self, h, ctx):
        g = ctx.graph
        x = self.norm(h)
        xi = x.unsqueeze(2).expand(-1, -1, g.index.shape[-1], -1)
        xj = gather_nodes(x, g.index)
        rel = gather_nodes(ctx.pos, g.index) - ctx.pos[:, :, None]
        d2 = rel.pow(2).sum(-1, keepdim=True) / ctx.length_scale**2
        m = self.phi_e(torch.cat([xi, xj, d2, ctx.edges], dim=-1)) * g.mask.unsqueeze(-1)
        deg = g.mask.sum(-1, keepdim=True).clamp_min(1)
        h = h + self.phi_h(torch.cat([x, m.sum(2) / deg], dim=-1))
        if self.update_coords:
            ctx.pos = ctx.pos - (rel * self.phi_x(m)).sum(2) / deg
        return h


@SPATIAL_LAYERS.register("full_attention")
class FullAttentionBlock(nn.Module):
    """Global self-attention over all sensors (O(N^2)); optional distance-bias prior.

    With ``distance_prior=False`` and ``edge_features: none`` this is the unstructured transformer
    baseline (add ``sensor_id_embedding`` in the model config for absolute sensor identity).
    """

    def __init__(self, dim: int, edge_dim: int, heads: int = 4, distance_prior: bool = False, ffn_mult: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.heads, self.distance_prior = heads, distance_prior
        self.attn = BiasedSelfAttention(dim, heads, dropout)
        self.raw_lambda = nn.Parameter(torch.full((heads,), 0.5413))
        self.ffn = FeedForward(dim, ffn_mult, dropout)

    def forward(self, h, ctx):
        bias = None
        if self.distance_prior:
            d2 = pairwise_sq_distance(ctx.pos) / ctx.length_scale**2  # [B, N, N]
            bias = -nn.functional.softplus(self.raw_lambda)[None, :, None, None] * d2[:, None]
        return self.ffn(self.attn(h, bias, ctx.node_mask))


@SPATIAL_LAYERS.register("none")
class NoInteractionBlock(nn.Module):
    """Ablation: per-sensor feed-forward only (sensors never exchange information in stage 2)."""

    def __init__(self, dim: int, edge_dim: int, ffn_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.ffn = FeedForward(dim, ffn_mult, dropout)

    def forward(self, h, ctx):
        return self.ffn(h)


# ---------------------------------------------------------------------------------------- stack


@dataclass
class GraphConfig:
    k: int = 16
    k_per_group: dict[str, int] | None = None  # e.g. {tactile: 12, joint: 2, imu: 1}
    radius: float | None = None
    dynamic: bool = True  # rebuild neighbors from current poses (False: from the rest pose, once)
    include_self: bool = True


@dataclass
class SpatialConfig:
    type: str = "continuous_conv"
    num_layers: int = 2
    edge_features: str = "rel_pos_local"
    length_scale: float = 0.05  # meters; typical neighbor spacing
    graph: GraphConfig = field(default_factory=GraphConfig)
    params: dict[str, Any] = field(default_factory=dict)  # layer-specific keyword arguments


class SpatialStack(nn.Module):
    def __init__(self, dim: int, cfg: SpatialConfig):
        super().__init__()
        self.cfg = cfg
        edge_dim = EDGE_FEATURE_DIMS[cfg.edge_features]
        self.layers = nn.ModuleList(
            SPATIAL_LAYERS.build(cfg.type, dim=dim, edge_dim=edge_dim, **cfg.params) for _ in range(cfg.num_layers)
        )
        self.norm = nn.LayerNorm(dim)
        self._static_cache: tuple[int, NeighborGraph] | None = None

    @property
    def needs_graph(self) -> bool:
        return self.cfg.type not in ("full_attention", "none")

    def build_graph(self, pos: torch.Tensor, info: LayoutInfo, node_mask: torch.Tensor | None) -> NeighborGraph:
        gc = self.cfg.graph
        k_per_group = None
        if gc.k_per_group:
            k_per_group = {info.group_names.index(n): k for n, k in gc.k_per_group.items() if n in info.group_names}
        if gc.dynamic:
            return knn_graph(pos, gc.k, node_mask, info.group_id, k_per_group, gc.radius, gc.include_self)
        key = id(info)
        if self._static_cache is None or self._static_cache[0] != key:
            rest = info.rest_pos.to(pos)[None]
            graph = knn_graph(rest, gc.k, None, info.group_id, k_per_group, gc.radius, gc.include_self)
            self._static_cache = (key, graph)
        graph = self._static_cache[1].expand(pos.shape[0])
        if node_mask is not None:
            graph = NeighborGraph(graph.index, graph.mask & node_mask[:, :, None] & node_mask.gather(
                1, graph.index.reshape(pos.shape[0], -1)).view_as(graph.mask))
        return graph

    def forward(self, h: torch.Tensor, pos: torch.Tensor, rot: torch.Tensor, info: LayoutInfo,
                node_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``h [B, N, D]``, ``pos [B, N, 3]``, ``rot [B, N, 3, 3]`` -> ``[B, N, D]``."""
        ctx = SpatialContext(pos=pos, rot=rot, area=info.area.to(h), length_scale=self.cfg.length_scale,
                             node_mask=node_mask)
        if self.needs_graph:
            ctx.graph = self.build_graph(pos, info, node_mask)
            ctx.edges = edge_features(self.cfg.edge_features, pos, rot, ctx.graph, self.cfg.length_scale)
        for layer in self.layers:
            h = layer(h, ctx)
        return self.norm(h)
