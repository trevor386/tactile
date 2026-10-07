"""Stage 1: per-sensor temporal encoders.

Every sensor of a group shares one encoder. The encoder sees only that sensor's own reading history
(no position, no neighbors), consumes ``S`` high-rate samples per low-rate *latent step* and keeps a
recurrent hidden state, so it can model dynamic events (vibration, slip onset) and sensor artifacts
(hysteresis, creep) while emitting a latent at the lower rate.

All encoders implement::

    z, state = encoder(x, state)     # x [B, L, S, N, C] -> z [B, L, N, D]

and running ``L`` steps at once gives exactly the same result as ``L`` calls with one step each
(streaming), which is how the model is deployed online.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from somato.utils.registry import Registry

TEMPORAL_ENCODERS: Registry[nn.Module] = Registry("temporal encoder")

State = dict[str, torch.Tensor]


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels: int, substeps: int, latent_dim: int):
        super().__init__()
        self.in_channels, self.substeps, self.latent_dim = in_channels, substeps, latent_dim

    def init_state(self, batch: int, nodes: int, device: torch.device | str = "cpu") -> State:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, state: State | None = None) -> tuple[torch.Tensor, State]:
        raise NotImplementedError

    def _check(self, x: torch.Tensor) -> tuple[int, int, int, int, int]:
        B, L, S, N, C = x.shape
        if S != self.substeps or C != self.in_channels:
            raise ValueError(f"Expected [B, L, {self.substeps}, N, {self.in_channels}], got {list(x.shape)}")
        return B, L, S, N, C


def _per_node_sequence(x: torch.Tensor) -> torch.Tensor:
    """``[B, L, S, N, C] -> [B*N, L, S, C]``."""
    B, L, S, N, C = x.shape
    return x.permute(0, 3, 1, 2, 4).reshape(B * N, L, S, C)


class _RecurrentCore(nn.Module):
    """Low-rate recurrent core shared by the chunked encoders (GRU or LSTM)."""

    def __init__(self, in_dim: int, hidden: int, layers: int, cell: str):
        super().__init__()
        self.hidden, self.layers, self.cell = hidden, layers, cell
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn(in_dim, hidden, num_layers=layers, batch_first=True)

    def init_state(self, batch_nodes: int, device) -> State:
        h = torch.zeros(self.layers, batch_nodes, self.hidden, device=device)
        return {"h": h, "c": h.clone()} if self.cell == "lstm" else {"h": h}

    def forward(self, seq: torch.Tensor, state: State) -> tuple[torch.Tensor, State]:
        if self.cell == "lstm":
            out, (h, c) = self.rnn(seq, (state["h"], state["c"]))
            return out, {"h": h, "c": c}
        out, h = self.rnn(seq, state["h"])
        return out, {"h": h}


@TEMPORAL_ENCODERS.register("conv_gru")
class ConvRecurrentEncoder(TemporalEncoder):
    """Conv front-end within each chunk of ``S`` samples, recurrent core across chunks.

    The 1D convolutions act as a learned filter bank for high-frequency content (vibration, impact
    transients) inside a latent step; chunk statistics (mean/max/last of the conv features plus the
    raw mean and last sample, which carry the DC load level) feed a GRU/LSTM running at the latent rate.
    """

    def __init__(self, in_channels: int, substeps: int, latent_dim: int, hidden: int = 64, conv_channels: int = 32,
                 conv_layers: int = 2, kernel_size: int = 3, rnn_layers: int = 1, cell: str = "gru"):
        super().__init__(in_channels, substeps, latent_dim)
        convs: list[nn.Module] = []
        c_in = in_channels
        for _ in range(conv_layers):
            convs += [nn.Conv1d(c_in, conv_channels, kernel_size, padding=kernel_size // 2), nn.GELU()]
            c_in = conv_channels
        self.conv = nn.Sequential(*convs)
        feat_dim = 3 * conv_channels + 2 * in_channels
        self.chunk_proj = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU())
        self.core = _RecurrentCore(hidden, hidden, rnn_layers, cell)
        self.out = nn.Linear(hidden, latent_dim)

    def init_state(self, batch, nodes, device="cpu"):
        return self.core.init_state(batch * nodes, device)

    def forward(self, x, state=None):
        B, L, S, N, C = self._check(x)
        if state is None:
            state = self.init_state(B, N, x.device)
        seq = _per_node_sequence(x)  # [BN, L, S, C]
        chunks = seq.reshape(B * N * L, S, C)
        f = self.conv(chunks.transpose(1, 2))  # [BNL, F, S]
        feats = torch.cat([f.mean(-1), f.amax(-1), f[..., -1], chunks.mean(1), chunks[:, -1]], dim=-1)
        feats = self.chunk_proj(feats).view(B * N, L, -1)
        out, state = self.core(feats, state)
        z = self.out(out).view(B, N, L, -1).transpose(1, 2)
        return z, state


@TEMPORAL_ENCODERS.register("gru")
class SampleRecurrentEncoder(TemporalEncoder):
    """A GRU/LSTM stepped at the full sensor rate; its hidden state is read out every ``S`` samples."""

    def __init__(self, in_channels: int, substeps: int, latent_dim: int, hidden: int = 64, rnn_layers: int = 1,
                 cell: str = "gru"):
        super().__init__(in_channels, substeps, latent_dim)
        self.inp = nn.Linear(in_channels, hidden)
        self.core = _RecurrentCore(hidden, hidden, rnn_layers, cell)
        self.out = nn.Linear(hidden, latent_dim)

    def init_state(self, batch, nodes, device="cpu"):
        return self.core.init_state(batch * nodes, device)

    def forward(self, x, state=None):
        B, L, S, N, C = self._check(x)
        if state is None:
            state = self.init_state(B, N, x.device)
        seq = _per_node_sequence(x).reshape(B * N, L * S, C)
        out, state = self.core(self.inp(seq), state)
        z = self.out(out[:, S - 1 :: S]).view(B, N, L, -1).transpose(1, 2)
        return z, state


@TEMPORAL_ENCODERS.register("receptor")
class ReceptorEncoder(TemporalEncoder):
    """Mechanoreceptor-inspired multi-timescale filter bank + low-rate recurrent core.

    Each input channel passes through ``num_filters`` learnable first-order low-pass filters
    (time constants log-spaced from ``tau_min`` to ``tau_max`` samples, implemented as causal FIR
    kernels of length ``kernel_size``). Per chunk the encoder extracts

    * slowly-adapting (SA-like) features: the low-pass outputs (mean and last value), and
    * rapidly-adapting (RA/Pacinian-like) features: the energy of the high-pass residual
      ``x - lowpass`` (band-limited vibration power),

    and feeds them to a GRU at the latent rate. The FIR history (last ``kernel_size - 1`` samples) is
    part of the state, so streaming is exact.
    """

    def __init__(self, in_channels: int, substeps: int, latent_dim: int, hidden: int = 64, num_filters: int = 4,
                 kernel_size: int = 64, tau_min: float = 1.0, tau_max: float = 32.0, rnn_layers: int = 1,
                 cell: str = "gru"):
        super().__init__(in_channels, substeps, latent_dim)
        self.num_filters, self.kernel_size = num_filters, kernel_size
        self.log_tau = nn.Parameter(torch.linspace(math.log(tau_min), math.log(tau_max), num_filters))
        feat_dim = 3 * in_channels * num_filters + 2 * in_channels
        self.chunk_proj = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU())
        self.core = _RecurrentCore(hidden, hidden, rnn_layers, cell)
        self.out = nn.Linear(hidden, latent_dim)

    def _kernels(self) -> torch.Tensor:
        """FIR weights ``[C*K, 1, M]`` ordered for ``conv1d`` (time-reversed impulse responses)."""
        tau = self.log_tau.exp().clamp_min(0.25)
        n = torch.arange(self.kernel_size, device=tau.device, dtype=tau.dtype)
        h = torch.exp(-n[None, :] / tau[:, None])  # [K, M]
        h = h / h.sum(-1, keepdim=True)
        w = h.flip(-1).repeat(self.in_channels, 1)  # [C*K, M]
        return w.unsqueeze(1)

    def init_state(self, batch, nodes, device="cpu"):
        st = self.core.init_state(batch * nodes, device)
        st["fir"] = torch.zeros(batch * nodes, self.in_channels, self.kernel_size - 1, device=device)
        st["fir_init"] = torch.zeros(batch * nodes, dtype=torch.bool, device=device)
        return st

    def forward(self, x, state=None):
        B, L, S, N, C = self._check(x)
        if state is None:
            state = self.init_state(B, N, x.device)
        seq = _per_node_sequence(x).reshape(B * N, L * S, C).transpose(1, 2)  # [BN, C, T]
        # On the very first call, fill the FIR history with the first sample (no zero-padding transient).
        first = seq[..., :1].expand(-1, -1, self.kernel_size - 1)
        hist = torch.where(state["fir_init"][:, None, None], state["fir"], first)
        padded = torch.cat([hist, seq], dim=-1)
        lp = F.conv1d(padded, self._kernels(), groups=C)  # [BN, C*K, T]
        hp = seq.repeat_interleave(self.num_filters, dim=1) - lp
        lp = lp.view(B * N, C * self.num_filters, L, S)
        hp_pow = hp.pow(2).view(B * N, C * self.num_filters, L, S).mean(-1)
        raw = seq.view(B * N, C, L, S)
        feats = torch.cat([lp.mean(-1), lp[..., -1], torch.log1p(hp_pow), raw.mean(-1), raw[..., -1]], dim=1)
        feats = self.chunk_proj(feats.transpose(1, 2))  # [BN, L, H]
        out, core_state = self.core(feats, state)
        new_state = dict(core_state)
        new_state["fir"] = padded[..., -(self.kernel_size - 1):]
        new_state["fir_init"] = torch.ones_like(state["fir_init"])
        z = self.out(out).view(B, N, L, -1).transpose(1, 2)
        return z, new_state


@TEMPORAL_ENCODERS.register("mlp")
class ChunkMLPEncoder(TemporalEncoder):
    """Memoryless ablation: an MLP on each chunk independently (no recurrent state)."""

    def __init__(self, in_channels: int, substeps: int, latent_dim: int, hidden: int = 64):
        super().__init__(in_channels, substeps, latent_dim)
        self.net = nn.Sequential(nn.Linear(in_channels * substeps, hidden), nn.GELU(), nn.Linear(hidden, latent_dim))

    def init_state(self, batch, nodes, device="cpu"):
        return {}

    def forward(self, x, state=None):
        B, L, S, N, C = self._check(x)
        z = self.net(x.permute(0, 1, 3, 2, 4).reshape(B, L, N, S * C))
        return z, {}
