"""Neighborhood graphs and geometric edge features.

Graphs are stored densely as a ``[B, N, K]`` neighbor index plus a validity mask, which keeps every
operation a gather + reduction (fast on GPU, no scatter dependencies such as torch_geometric).

Edge features encode the geometry of sensor ``j`` *relative to sensor ``i``*. Choosing the kind of
feature sets the inductive bias and is the main ablation axis of stage 2:

=================  ====  ==========================================================================
kind               dim   meaning / invariance
=================  ====  ==========================================================================
``none``           0     no geometry (neighbors still selected geometrically)
``distance``       1     ``|p_j - p_i|``: invariant to rotations, reflections, translations
``rel_pos_world``  3     ``p_j - p_i`` in the world frame: translation-invariant only
``rel_pos_local``  3     ``R_i^T (p_j - p_i)``: SE(3)-invariant (uses sensor i's frame)
``rel_pose_local`` 9     ``rel_pos_local`` + 6D rotation of ``R_i^T R_j``: SE(3)-invariant, full pose
``ppf``            4     point-pair feature ``(d, n_i.d, n_j.d, n_i.n_j)``: SE(3)-invariant and
                         independent of the arbitrary in-plane x/y axes of taxel frames
=================  ====  ==========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from somato.geometry.rotations import rotation_6d
from somato.models.common import gather_nodes

EDGE_FEATURE_DIMS = {"none": 0, "distance": 1, "rel_pos_world": 3, "rel_pos_local": 3, "rel_pose_local": 9, "ppf": 4}


@dataclass
class NeighborGraph:
    index: torch.Tensor  # [B, N, K] long
    mask: torch.Tensor  # [B, N, K] bool

    @property
    def degree(self) -> torch.Tensor:
        return self.mask.sum(-1)

    def expand(self, batch: int) -> NeighborGraph:
        return NeighborGraph(self.index.expand(batch, -1, -1), self.mask.expand(batch, -1, -1))


def pairwise_sq_distance(x: torch.Tensor) -> torch.Tensor:
    """Exact squared Euclidean distances ``[..., N, N]`` between the points ``x [..., N, C]``.

    Accumulated coordinate by coordinate: no ``|a|^2 + |b|^2 - 2ab`` matmul shortcut (which loses ~1e-4 m on
    translated coordinates), and ~100x faster on GPU than ``torch.cdist``'s exact (non-matmul) kernel.
    """
    d2 = None
    for c in range(x.shape[-1]):
        diff = x[..., :, None, c] - x[..., None, :, c]
        d2 = diff * diff if d2 is None else d2 + diff * diff
    return d2


def _topk_smallest(dist: torch.Tensor, k: int, quantum: float, extra: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """``k`` smallest distances, robust to exact ties.

    Regular layouts (taxel rings, grids) have many exactly equal distances, and which of several tied
    candidates ``topk`` keeps would depend on float round-off, breaking invariance to rigid motions.
    Distances are quantized to ``quantum`` with ties broken by node index, and up to ``extra``
    additional candidates that tie with the k-th neighbor (within ``2 * quantum``) are kept as well, so
    a tie group straddling the k-th position is included as a whole. Non-kept slots get ``inf``.
    """
    N = dist.shape[-1]
    kk = min(k + extra, N)
    finite = torch.isfinite(dist)
    q = torch.round(torch.where(finite, dist, torch.zeros_like(dist)) / quantum).long().clamp_max(2**40)
    q = torch.where(finite, q, torch.full_like(q, 2**41))
    key = q * N + torch.arange(N, device=dist.device)
    _, idx = key.topk(kk, dim=-1, largest=False)  # sorted ascending
    d = dist.gather(-1, idx)
    if kk > k:
        kth = d[..., k - 1 : k]
        keep = d <= kth + 2 * quantum
        keep[..., :k] = True
        d = torch.where(keep, d, torch.full_like(d, float("inf")))
    return d, idx


def knn_graph(
    pos: torch.Tensor, k: int, node_mask: torch.Tensor | None = None, group_id: torch.Tensor | None = None,
    k_per_group: dict[int, int] | None = None, radius: float | None = None, include_self: bool = True,
    tie_quantum: float = 1e-4, tie_extra: int | None = None,
) -> NeighborGraph:
    """k-nearest-neighbor graph over ``pos [B, N, 3]``.

    Args:
        k: neighbors per node (used when ``k_per_group`` is None).
        node_mask: ``[B, N]`` True for present sensors; absent sensors are never neighbors.
        group_id: ``[N]`` group index per node (required with ``k_per_group``).
        k_per_group: ``{group_id: k_g}`` -- each node takes its ``k_g`` nearest neighbors *from each
            group*. This guarantees e.g. that every taxel also sees its nearest joint and IMU, which a
            plain KNN over thousands of dense taxels would never select.
        radius: optional ball-query cutoff (neighbors farther than this are masked).
        include_self: keep the self-edge (distance 0).
        tie_quantum: distance resolution (m) used for deterministic tie-breaking.
        tie_extra: extra neighbor slots for ties with the k-th neighbor (default ``max(1, k // 4)``).
    """
    B, N, _ = pos.shape
    # Exact pairwise distances: the matmul shortcut loses ~1e-4 m precision on translated coordinates.
    centered = pos - pos.mean(dim=1, keepdim=True)
    dist = pairwise_sq_distance(centered).sqrt()  # [B, N, N]
    inf = torch.tensor(float("inf"), device=pos.device, dtype=pos.dtype)
    if node_mask is not None:
        dist = torch.where(node_mask[:, None, :], dist, inf)
    if not include_self:
        dist = dist + torch.diag_embed(torch.full((N,), float("inf"), device=pos.device, dtype=pos.dtype))
    if k_per_group is None:
        kk = min(k, N)
        d, idx = _topk_smallest(dist, kk, tie_quantum, max(1, kk // 4) if tie_extra is None else tie_extra)
    else:
        if group_id is None:
            raise ValueError("k_per_group requires group_id")
        ds, idxs = [], []
        for g, kg in k_per_group.items():
            cols = (group_id.to(pos.device) == g).nonzero().squeeze(1)
            if kg <= 0 or cols.numel() == 0:
                continue
            kg = min(kg, cols.numel())
            extra = max(1, kg // 4) if tie_extra is None else tie_extra
            d_g, i_local = _topk_smallest(dist.index_select(-1, cols), kg, tie_quantum, extra)
            ds.append(d_g)
            idxs.append(cols[i_local])
        d, idx = torch.cat(ds, -1), torch.cat(idxs, -1)
    mask = torch.isfinite(d)
    if radius is not None:
        mask = mask & (d <= radius)
    if node_mask is not None:
        mask = mask & node_mask[:, :, None]
    idx = torch.where(mask, idx, torch.zeros_like(idx))
    return NeighborGraph(idx, mask)


def edge_features(kind: str, pos: torch.Tensor, rot: torch.Tensor, graph: NeighborGraph,
                  length_scale: float = 1.0) -> torch.Tensor:
    """Edge features ``[B, N, K, F]`` (zeros on masked edges). Lengths are divided by ``length_scale``."""
    if kind not in EDGE_FEATURE_DIMS:
        raise ValueError(f"Unknown edge feature kind '{kind}'. Options: {list(EDGE_FEATURE_DIMS)}")
    B, N, K = graph.index.shape
    if kind == "none":
        return pos.new_zeros(B, N, K, 0)
    pj = gather_nodes(pos, graph.index)  # [B, N, K, 3]
    rel = (pj - pos[:, :, None, :]) / length_scale  # world frame
    if kind == "rel_pos_world":
        feat = rel
    elif kind == "distance":
        feat = rel.norm(dim=-1, keepdim=True)
    else:
        Ri = rot[:, :, None]  # [B, N, 1, 3, 3]
        rel_local = (Ri.transpose(-1, -2) @ rel.unsqueeze(-1)).squeeze(-1)
        if kind == "rel_pos_local":
            feat = rel_local
        elif kind == "rel_pose_local":
            Rj = gather_nodes(rot, graph.index)
            feat = torch.cat([rel_local, rotation_6d(Ri.transpose(-1, -2) @ Rj)], dim=-1)
        else:  # ppf
            ni = rot[..., :, 2][:, :, None, :]  # sensor normals (z axes)
            nj = gather_nodes(rot[..., :, 2], graph.index)
            d = rel.norm(dim=-1, keepdim=True)
            u = rel / d.clamp_min(1e-9)
            feat = torch.cat([d, (ni * u).sum(-1, keepdim=True), (nj * u).sum(-1, keepdim=True),
                              (ni * nj).sum(-1, keepdim=True)], dim=-1)
    return feat * graph.mask.unsqueeze(-1).to(feat.dtype)
