"""Shared neural-network building blocks."""

from __future__ import annotations

import math

import torch
from torch import nn


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2, act: type[nn.Module] = nn.GELU,
        dropout: float = 0.0) -> nn.Sequential:
    """``layers`` linear layers with activations in between (no activation on the output)."""
    dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
    mods: list[nn.Module] = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(act())
            if dropout > 0:
                mods.append(nn.Dropout(dropout))
    return nn.Sequential(*mods)


class FeedForward(nn.Module):
    """Pre-norm residual feed-forward block: ``x + FFN(LN(x))``."""

    def __init__(self, dim: int, mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


def gather_nodes(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather neighbor features. ``x [B, N, *F]``, ``index [B, N, K]`` -> ``[B, N, K, *F]``."""
    B, N, K = index.shape
    flat = index.reshape(B, N * K)
    feat_shape = x.shape[2:]
    idx = flat.view(B, N * K, *([1] * len(feat_shape))).expand(B, N * K, *feat_shape)
    return torch.gather(x, 1, idx).view(B, N, K, *feat_shape)


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    if mask is None:
        return x.mean(dim)
    m = mask.to(x.dtype)
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    return (x * m).sum(dim) / m.sum(dim).clamp_min(1.0)


def masked_max(x: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    if mask is None:
        return x.amax(dim)
    m = mask
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    out = x.masked_fill(~m, float("-inf")).amax(dim)
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    """Softmax that returns zeros (not NaN) for fully-masked rows."""
    if mask is None:
        return logits.softmax(dim)
    logits = logits.masked_fill(~mask, float("-inf"))
    w = logits.softmax(dim)
    return torch.nan_to_num(w, nan=0.0)


class BiasedSelfAttention(nn.Module):
    """Dense multi-head self-attention with an additive per-head bias ``[B, H, N, N]``.

    Used for global (cluster-level or all-node) attention where the bias carries relative geometry.
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads, self.dh = heads, dim // heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None, mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        """``x [B, N, D]``, ``bias [B, H, N, N]``, ``mask [B, N]`` (True = valid key)."""
        B, N, D = x.shape
        q, k, v = self.qkv(self.norm(x)).view(B, N, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        if bias is not None:
            logits = logits + bias
        key_mask = None if mask is None else mask[:, None, None, :].expand(B, self.heads, N, N)
        attn = self.drop(masked_softmax(logits, key_mask, dim=-1))
        y = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return x + self.out(y)


class FourierEncoding(nn.Module):
    """``x -> [x, sin(2^k pi x), cos(2^k pi x)]`` for ``k < bands`` (no-op when ``bands == 0``)."""

    def __init__(self, in_dim: int, bands: int = 0):
        super().__init__()
        self.bands = bands
        self.out_dim = in_dim * (1 + 2 * bands)
        self.register_buffer("freqs", (2.0 ** torch.arange(bands)) * math.pi, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bands == 0:
            return x
        xf = x.unsqueeze(-1) * self.freqs
        return torch.cat([x, xf.sin().flatten(-2), xf.cos().flatten(-2)], dim=-1)


class InputNormalizer(nn.Module):
    """Per-channel standardization with statistics fitted from data (``fit``) and saved in the state dict."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(channels))
        self.register_buffer("std", torch.ones(channels))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, x.shape[-1]).double()
        self.mean.copy_(flat.mean(0).to(self.mean))
        self.std.copy_(flat.std(0).clamp_min(self.eps).to(self.std))
        self.fitted.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
