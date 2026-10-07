"""Sensor model interface.

A *sensor model* turns ideal physical stimuli (what a perfect sensor would measure) into the raw
readings of a specific sensing technology (FSR voltage, capacitance change, quantized encoder
counts, noisy MEMS IMU, ...). Simulators only ever produce stimuli; swapping the sensor technology is
a config change and does not require re-simulating, because datasets store stimuli and the sensor
model runs on the fly (with fresh noise each epoch).

Stimulus channels per sensor kind (all in the sensor's local frame):

* ``tactile``: ``normal`` (compressive pressure, Pa), ``shear_x``, ``shear_y`` (traction, Pa)
* ``joint``: ``pos`` (rad), ``vel`` (rad/s), ``torque`` (N m, actuator effort), ``target`` (rad, command)
* ``imu``: ``acc_x..z`` (specific force, m/s^2), ``gyro_x..z`` (rad/s)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from somato.utils.registry import Registry

STIMULUS_CHANNELS: dict[str, tuple[str, ...]] = {
    "tactile": ("normal", "shear_x", "shear_y"),
    "joint": ("pos", "vel", "torque", "target"),
    "imu": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
}

SENSOR_MODELS: Registry[SensorModel] = Registry("sensor model")

State = dict[str, torch.Tensor]


def randn_like(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)


def randn(shape, ref: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=ref.device, dtype=ref.dtype)


def quantize(x: torch.Tensor, step: float) -> torch.Tensor:
    return torch.round(x / step) * step if step > 0 else x


class SensorModel(ABC):
    """Maps stimuli ``[*batch, T, N, C_in]`` to readings ``[*batch, T, N, C_out]``.

    Models may be stateful (hysteresis, drift, filters). ``init_state`` also samples per-episode
    random parameters (biases, per-sensor gain spread), so calling it once per episode/window gives
    realistic unit-to-unit variation.
    """

    kind: str = ""
    output_channels: tuple[str, ...] = ()

    def __init__(self, dt: float):
        self.dt = float(dt)

    @property
    def num_outputs(self) -> int:
        return len(self.output_channels)

    def init_state(self, stimulus: torch.Tensor, generator: torch.Generator | None = None) -> State:
        """Initial state for a stimulus tensor shaped like the one that will be passed to ``forward``."""
        return {}

    @abstractmethod
    def forward(self, stimulus: torch.Tensor, state: State, generator: torch.Generator | None = None
                ) -> tuple[torch.Tensor, State]:
        ...

    def __call__(self, stimulus: torch.Tensor, state: State | None = None,
                 generator: torch.Generator | None = None) -> tuple[torch.Tensor, State]:
        expected = len(STIMULUS_CHANNELS[self.kind])
        if stimulus.shape[-1] != expected:
            raise ValueError(f"{type(self).__name__} expects {expected} stimulus channels, got {stimulus.shape[-1]}")
        if state is None:
            state = self.init_state(stimulus, generator)
        return self.forward(stimulus, state, generator)

    def __repr__(self) -> str:
        params = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return f"{type(self).__name__}({params})"
