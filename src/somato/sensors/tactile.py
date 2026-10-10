"""Tactile sensing technologies.

All models take the ideal taxel stimulus ``[..., T, N, 3]`` = (normal pressure, shear_x, shear_y) in Pa.
Parameters default to plausible values for small (~1 cm^2) taxels; tune them against bench data.
"""

from __future__ import annotations

import math

import torch

from somato.sensors.base import SENSOR_MODELS, SensorModel, quantize, randn, randn_like


@SENSOR_MODELS.register("ideal_tactile")
class IdealTactile(SensorModel):
    """Noise-free pressure (and optionally shear), reported in kPa. Useful for early experiments."""

    kind = "tactile"

    def __init__(self, dt: float, shear: bool = False, noise_kpa: float = 0.0):
        super().__init__(dt)
        self.shear = shear
        self.noise_kpa = noise_kpa
        self.output_channels = ("normal_kpa", "shear_x_kpa", "shear_y_kpa") if shear else ("normal_kpa",)

    def forward(self, stimulus, state, generator=None):
        out = stimulus[..., : self.num_outputs] * 1e-3
        if self.noise_kpa > 0:
            out = out + self.noise_kpa * randn_like(out, generator)
        return out, state


@SENSOR_MODELS.register("binary_tactile")
class BinaryTactile(SensorModel):
    """On/off contact switches with a force threshold (cheap dense skins, membrane switches)."""

    kind = "tactile"
    output_channels = ("contact",)

    def __init__(self, dt: float, area: float = 1e-4, force_threshold: float = 0.3, flip_prob: float = 0.0):
        super().__init__(dt)
        self.area, self.force_threshold, self.flip_prob = area, force_threshold, flip_prob

    def forward(self, stimulus, state, generator=None):
        out = (stimulus[..., :1] * self.area > self.force_threshold).to(stimulus.dtype)
        if self.flip_prob > 0:
            flip = torch.rand(out.shape, generator=generator, device=out.device) < self.flip_prob
            out = torch.where(flip, 1 - out, out)
        return out, state


@SENSOR_MODELS.register("fsr")
class FSRTactile(SensorModel):
    """Force-sensing resistor read through a voltage divider and an ADC.

    Model (per taxel):

    * Rate-dependent hysteresis: the internal load follows the applied force with a fast loading
      time constant and a slower unloading time constant. ``tau_unload`` may be a range ``[lo, hi]``: each taxel
      then draws its own (log-uniform), as real taxels differ.
    * Creep: a slow drift up under sustained load (fraction ``creep_frac``, time constant ``creep_tau``).
    * Power-law conductance above a turn-on threshold, ``G = (F_eff - F_th)^gamma / (F_ref^gamma R_ref)``,
      with per-taxel log-normal gain spread (manufacturing variation).
    * Divider output ``V = R_m G / (1 + R_m G)`` (normalized to Vcc = 1).
    * Rate-independent hysteresis (``hysteresis``, a fraction of full scale, or a per-taxel uniform range): a play
      operator on the output, which moves only once the input leaves a band of that width around it, so loading
      and unloading curves differ by the band whatever the speed (measured 7-17 % for FSRs, varying per sensor).
      The band narrows to zero as the output goes to zero, so the loop closes at no load (no residual offset after
      a contact ends and no dead zone for light contacts).
    * Additive noise, ADC quantization.

    Sensor-model randomization (``randomize``, training only): ``{param: [lo, hi]}`` draws the listed parameters per
    sample (window) instead of using their fixed values, so a model cannot rely on one simulated sensor's signature
    (F-23). Scalars (``gain_spread``, ``creep_frac``, ``creep_tau``, ``noise_std``) get one draw per sample. The
    per-taxel parameters (``tau_unload``, ``hysteresis``) get a per-sample sub-range (two sorted draws), within which
    each taxel then draws its own value: samples differ in both the typical value and the taxel-to-taxel spread.
    Time constants are drawn log-uniformly, everything else uniformly.

    Shear is not sensed. Output channel: ``fsr_v`` in [0, 1].
    """

    kind = "tactile"
    output_channels = ("fsr_v",)
    RANDOMIZABLE = ("tau_unload", "hysteresis", "gain_spread", "creep_frac", "creep_tau", "noise_std")
    LOG_PARAMS = ("tau_unload", "creep_tau")

    def __init__(
        self, dt: float, area: float = 1e-4, force_threshold: float = 0.15, force_ref: float = 1.0,
        exponent: float = 1.1, r_ref: float = 1.0e4, r_divider: float = 1.0e4, gain_spread: float = 0.15,
        tau_load: float = 0.002, tau_unload: float | tuple[float, float] = 0.02, creep_frac: float = 0.06,
        creep_tau: float = 1.5, hysteresis: float | tuple[float, float] = 0.0, noise_std: float = 0.003,
        adc_bits: int = 12, randomize: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__(dt)
        self.area, self.force_threshold, self.force_ref, self.exponent = area, force_threshold, force_ref, exponent
        self.r_ref, self.r_divider, self.gain_spread = r_ref, r_divider, gain_spread
        self.tau_load, self.tau_unload, self.creep_frac, self.creep_tau = tau_load, tau_unload, creep_frac, creep_tau
        self.hysteresis, self.noise_std, self.adc_bits = hysteresis, noise_std, adc_bits
        self.randomize = {k: tuple(float(x) for x in v) for k, v in (randomize or {}).items()}
        unknown = set(self.randomize) - set(self.RANDOMIZABLE)
        if unknown:
            raise ValueError(f"cannot randomize {sorted(unknown)}; randomizable: {self.RANDOMIZABLE}")

    @staticmethod
    def _per_taxel(value, shape, ref, generator, log: bool = False) -> torch.Tensor:
        """A scalar, or one uniform (log-uniform) draw per taxel from a ``[lo, hi]`` range (floats, or per-sample
        tensors ``[*batch, 1]``)."""
        if isinstance(value, (int, float)):
            return torch.full(shape, float(value), device=ref.device, dtype=ref.dtype)
        lo, hi = (torch.as_tensor(v, device=ref.device, dtype=ref.dtype) for v in value)
        if log:
            lo, hi = lo.log(), hi.log()
        x = lo + (hi - lo) * torch.rand(shape, generator=generator, device=ref.device, dtype=ref.dtype)
        return x.exp() if log else x

    def _per_sample(self, name, batch, ref, generator) -> torch.Tensor:
        """One draw ``[*batch, 1]`` per sample from the randomization range of ``name``."""
        return self._per_taxel(self.randomize[name], batch + (1,), ref, generator, log=name in self.LOG_PARAMS)

    def _taxel_range(self, name, batch, ref, generator):
        """The fixed value or range of a per-taxel parameter, or a per-sample sub-range when it is randomized."""
        if name not in self.randomize:
            return getattr(self, name)
        a, b = (self._per_sample(name, batch, ref, generator) for _ in range(2))
        return torch.minimum(a, b), torch.maximum(a, b)

    def init_state(self, stimulus, generator=None):
        batch = stimulus.shape[:-3]
        shape = batch + stimulus.shape[-2:-1]  # [*batch, N]
        # Randomized scalars live in the state ([*batch, 1]) and override the fixed values in forward.
        rnd = {k: self._per_sample(k, batch, stimulus, generator)
               for k in ("gain_spread", "creep_frac", "creep_tau", "noise_std") if k in self.randomize}
        gain = torch.exp(rnd.get("gain_spread", self.gain_spread) * randn(shape, stimulus, generator))
        # Start at the first sample's force (at steady state) to avoid a start-up transient.
        f0 = stimulus[..., 0, :, 0].clamp_min(0.0) * self.area
        tau_unload = self._per_taxel(self._taxel_range("tau_unload", batch, stimulus, generator), shape, stimulus,
                                     generator, log=True)
        state = {"load": f0.clone(), "creep": f0.clone(), "gain": gain, "a_unload": self.dt / (tau_unload + self.dt),
                 **{k: v for k, v in rnd.items() if k != "gain_spread"}}
        if self.hysteresis or "hysteresis" in self.randomize:
            hysteresis = self._taxel_range("hysteresis", batch, stimulus, generator)
            state["play_width"] = self._per_taxel(hysteresis, shape, stimulus, generator)
            creep_frac = state.get("creep_frac", self.creep_frac)
            state["play"] = self._transfer(f0 * (1 + creep_frac), gain)  # output at the steady initial load
        return state

    def _transfer(self, force: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        f = (force - self.force_threshold).clamp_min(0.0) / self.force_ref
        g = gain * f.pow(self.exponent) / self.r_ref
        return self.r_divider * g / (1.0 + self.r_divider * g)

    def forward(self, stimulus, state, generator=None):
        force = stimulus[..., 0].clamp_min(0.0) * self.area  # [*batch, T, N]
        a_load = self.dt / (self.tau_load + self.dt)
        a_creep = self.dt / (state.get("creep_tau", self.creep_tau) + self.dt)
        creep_frac = state.get("creep_frac", self.creep_frac)  # per-sample tensors when randomized
        load, creep, gain = state["load"], state["creep"], state["gain"]
        a_unload = state["a_unload"] if "a_unload" in state else self.dt / (self.tau_unload + self.dt)
        # Only the load/creep recursion is sequential (few kernels per sample); the elementwise conductance
        # transfer then runs on the whole window at once.
        eff = torch.empty_like(force)
        for t in range(force.shape[-2]):
            f = force[..., t, :]
            load = torch.lerp(load, f, torch.where(f > load, a_load, a_unload))
            creep = torch.lerp(creep, load, a_creep)
            if isinstance(creep_frac, torch.Tensor):
                torch.addcmul(load, creep, creep_frac, out=eff[..., t, :])
            else:
                torch.add(load, creep, alpha=creep_frac, out=eff[..., t, :])
        v = self._transfer(eff, gain.unsqueeze(-2))
        new_state = {**state, "load": load, "creep": creep}
        if "play" in state:  # rate-independent hysteresis: play operator of width w, closing at zero load
            w, y = state["play_width"], state["play"]
            out = torch.empty_like(v)
            for t in range(v.shape[-2]):
                vt = v[..., t, :]
                half = 0.5 * torch.minimum(vt, w)  # the band shrinks to 0 at zero output: the loop closes there
                y = torch.minimum(torch.maximum(y, vt - half), vt + half)
                out[..., t, :] = y
            v, new_state["play"] = out, y
        if "noise_std" in state:
            v = v + state["noise_std"].unsqueeze(-2) * randn_like(v, generator)
        elif self.noise_std > 0:
            v = v + self.noise_std * randn_like(v, generator)
        v = quantize(v.clamp(0.0, 1.0), 1.0 / (2**self.adc_bits - 1))
        return v.unsqueeze(-1), new_state


@SENSOR_MODELS.register("capacitive")
class CapacitiveTactile(SensorModel):
    """Parallel-plate capacitive taxel with an elastomer dielectric.

    Strain of the dielectric is ``eps = p / E`` (instantaneous) plus a viscoelastic component that
    relaxes with ``tau_visco``; the gap ``d = d0 (1 - eps)`` (limited to ``min_gap_frac``). Output
    is the relative capacitance change ``C/C0 - 1 = 1/(1 - eps) - 1`` with per-taxel gain spread, a
    slowly drifting baseline (temperature / humidity), white noise and CDC quantization.
    """

    kind = "tactile"
    output_channels = ("dc_c0",)

    def __init__(
        self, dt: float, modulus: float = 2.0e5, visco_frac: float = 0.2, tau_visco: float = 0.3,
        min_gap_frac: float = 0.25, gain_spread: float = 0.08, drift_std: float = 2e-3, noise_std: float = 1e-3,
        resolution: float = 2e-4,
    ):
        super().__init__(dt)
        self.modulus, self.visco_frac, self.tau_visco, self.min_gap_frac = modulus, visco_frac, tau_visco, min_gap_frac
        self.gain_spread, self.drift_std, self.noise_std, self.resolution = gain_spread, drift_std, noise_std, resolution

    def init_state(self, stimulus, generator=None):
        shape = stimulus.shape[:-3] + stimulus.shape[-2:-1]
        return {
            "visco": stimulus[..., 0, :, 0].clamp_min(0.0) / self.modulus,
            "baseline": 0.01 * randn(shape, stimulus, generator),
            "gain": torch.exp(self.gain_spread * randn(shape, stimulus, generator)),
        }

    def forward(self, stimulus, state, generator=None):
        strain_inst = stimulus[..., 0].clamp_min(0.0) / self.modulus  # [*batch, T, N]
        a = self.dt / (self.tau_visco + self.dt)
        visco = state["visco"]
        viscos = torch.empty_like(strain_inst)
        for t in range(strain_inst.shape[-2]):
            visco = torch.lerp(visco, strain_inst[..., t, :], a)
            viscos[..., t, :] = visco
        strain = ((1 - self.visco_frac) * strain_inst + self.visco_frac * viscos).clamp(max=1.0 - self.min_gap_frac)
        dc = state["gain"].unsqueeze(-2) * (1.0 / (1.0 - strain) - 1.0)
        # Baseline drift as a random walk over the window.
        steps = self.drift_std * math.sqrt(self.dt) * randn(strain.shape, strain, generator)
        drift = state["baseline"].unsqueeze(-2) + steps.cumsum(dim=-2)
        out = dc + drift
        if self.noise_std > 0:
            out = out + self.noise_std * randn_like(out, generator)
        out = quantize(out, self.resolution)
        new_state = {"visco": visco, "baseline": drift[..., -1, :], "gain": state["gain"]}
        return out.unsqueeze(-1), new_state
