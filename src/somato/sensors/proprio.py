"""Proprioceptive sensor models: joint/motor sensing and IMUs."""

from __future__ import annotations

import math

import torch

from somato.sensors.base import SENSOR_MODELS, SensorModel, quantize, randn, randn_like

G = 9.81


@SENSOR_MODELS.register("ideal_joint")
class IdealJoint(SensorModel):
    """Exact joint position, velocity, actuator torque and tracking error (target - pos)."""

    kind = "joint"
    output_channels = ("pos", "vel", "torque", "error")

    def forward(self, stimulus, state, generator=None):
        pos, vel, tau, target = stimulus.unbind(-1)
        return torch.stack([pos, vel, tau, target - pos], dim=-1), state


@SENSOR_MODELS.register("motor")
class MotorJoint(SensorModel):
    """A geared servo with an absolute encoder and motor-current torque sensing.

    * Position: encoder quantization (``counts_per_rev``) + small offset per joint.
    * Velocity: finite difference of the quantized position, low-pass filtered (as on most servos).
    * Torque: from motor current, so it also contains the motor's internal friction
      (Coulomb ``tau_coulomb`` and viscous ``viscous`` terms), plus gain error, noise and a
      current-loop low-pass filter. This is the "implicit" interaction signal through the joint.
    * Error: commanded target minus measured position.
    """

    kind = "joint"
    output_channels = ("pos", "vel", "torque", "error")

    def __init__(
        self, dt: float, counts_per_rev: int = 4096, offset_std: float = 0.003, vel_cutoff_hz: float = 50.0,
        tau_coulomb: float = 0.03, viscous: float = 0.01, gain_spread: float = 0.05, torque_noise: float = 0.01,
        torque_cutoff_hz: float = 100.0,
    ):
        super().__init__(dt)
        self.counts_per_rev, self.offset_std, self.vel_cutoff_hz = counts_per_rev, offset_std, vel_cutoff_hz
        self.tau_coulomb, self.viscous, self.gain_spread = tau_coulomb, viscous, gain_spread
        self.torque_noise, self.torque_cutoff_hz = torque_noise, torque_cutoff_hz

    def init_state(self, stimulus, generator=None):
        # Filters start at the first sample so windows cropped mid-episode have no start-up transient.
        shape = stimulus.shape[:-3] + stimulus.shape[-2:-1]
        offset = self.offset_std * randn(shape, stimulus, generator)
        first = stimulus[..., 0, :, :]
        return {
            "offset": offset,
            "gain": 1.0 + self.gain_spread * randn(shape, stimulus, generator),
            "prev_pos": quantize(first[..., 0] + offset, 2 * math.pi / self.counts_per_rev),
            "vel": first[..., 1].clone(),
            "torque": first[..., 2].clone(),
        }

    def forward(self, stimulus, state, generator=None):
        pos, vel, tau, target = stimulus.unbind(-1)  # each [*batch, T, N]
        step = 2 * math.pi / self.counts_per_rev
        pos_m = quantize(pos + state["offset"].unsqueeze(-2), step)
        motor_tau = tau + self.tau_coulomb * torch.tanh(vel / 0.05) + self.viscous * vel
        tau_raw = state["gain"].unsqueeze(-2) * motor_tau + self.torque_noise * randn_like(tau, generator)
        a_v = 1.0 - math.exp(-2 * math.pi * self.vel_cutoff_hz * self.dt)
        a_t = 1.0 - math.exp(-2 * math.pi * self.torque_cutoff_hz * self.dt)
        prev = torch.cat([state["prev_pos"].unsqueeze(-2), pos_m[..., :-1, :]], dim=-2)
        raw = torch.stack([(pos_m - prev) / self.dt, tau_raw], dim=-1)  # [*batch, T, N, 2]: velocity, torque
        # Both first-order low-pass recursions advance together (one kernel per sample).
        a = torch.tensor([a_v, a_t], dtype=raw.dtype, device=raw.device)
        y = torch.stack([state["vel"], state["torque"]], dim=-1)
        filt = torch.empty_like(raw)
        for t in range(pos.shape[-2]):
            y = torch.lerp(y, raw[..., t, :, :], a)
            filt[..., t, :, :] = y
        out = torch.stack([pos_m, filt[..., 0], filt[..., 1], target - pos_m], dim=-1)
        new_state = dict(state, prev_pos=pos_m[..., -1, :], vel=y[..., 0], torque=y[..., 1])
        return out, new_state


@SENSOR_MODELS.register("ideal_imu")
class IdealIMU(SensorModel):
    kind = "imu"
    output_channels = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")

    def forward(self, stimulus, state, generator=None):
        return stimulus.clone(), state


@SENSOR_MODELS.register("mems_imu")
class MEMSIMU(SensorModel):
    """Consumer MEMS IMU: white noise (noise densities), turn-on bias, bias random walk, scale
    error, range saturation and quantization. Defaults are roughly BMI088 / ICM-42688 class."""

    kind = "imu"
    output_channels = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")

    def __init__(
        self, dt: float, acc_noise_density: float = 1.5e-4 * G, gyro_noise_density: float = math.radians(0.005),
        acc_bias_std: float = 0.02 * G, gyro_bias_std: float = math.radians(0.5), acc_bias_walk: float = 1e-4,
        gyro_bias_walk: float = 1e-5, scale_std: float = 0.005, acc_range: float = 16 * G,
        gyro_range: float = math.radians(2000), acc_lsb: float = 16 * G / 2**15, gyro_lsb: float = math.radians(2000) / 2**15,
    ):
        super().__init__(dt)
        self.acc_noise_density, self.gyro_noise_density = acc_noise_density, gyro_noise_density
        self.acc_bias_std, self.gyro_bias_std = acc_bias_std, gyro_bias_std
        self.acc_bias_walk, self.gyro_bias_walk, self.scale_std = acc_bias_walk, gyro_bias_walk, scale_std
        self.acc_range, self.gyro_range, self.acc_lsb, self.gyro_lsb = acc_range, gyro_range, acc_lsb, gyro_lsb

    def _pair(self, ref, a, g):
        return torch.tensor([a, a, a, g, g, g], dtype=ref.dtype, device=ref.device)

    def init_state(self, stimulus, generator=None):
        shape = stimulus.shape[:-3] + stimulus.shape[-2:]  # [*batch, N, 6]
        bias = randn(shape, stimulus, generator) * self._pair(stimulus, self.acc_bias_std, self.gyro_bias_std)
        scale = 1.0 + self.scale_std * randn(shape, stimulus, generator)
        return {"bias": bias, "scale": scale}

    def forward(self, stimulus, state, generator=None):
        T = stimulus.shape[-3]
        walk = randn(stimulus.shape, stimulus, generator) * self._pair(
            stimulus, self.acc_bias_walk, self.gyro_bias_walk) * math.sqrt(self.dt)
        bias = state["bias"].unsqueeze(-3) + walk.cumsum(dim=-3)
        white = randn(stimulus.shape, stimulus, generator) * self._pair(
            stimulus, self.acc_noise_density, self.gyro_noise_density) / math.sqrt(self.dt)
        out = state["scale"].unsqueeze(-3) * stimulus + bias + white
        rng = self._pair(stimulus, self.acc_range, self.gyro_range)
        out = torch.maximum(torch.minimum(out, rng), -rng)
        lsb = self._pair(stimulus, self.acc_lsb, self.gyro_lsb)
        out = torch.round(out / lsb) * lsb
        return out, {"bias": bias[..., T - 1, :, :], "scale": state["scale"]}
