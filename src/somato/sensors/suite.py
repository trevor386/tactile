"""A sensor suite assigns one sensor model to every sensor group of a layout."""

from __future__ import annotations

from typing import Any

import torch

from somato.geometry.layout import SensorLayout
from somato.sensors.base import SENSOR_MODELS, SensorModel, State


class SensorSuite:
    """Applies per-group sensor models to stimuli.

    Build from a config mapping group name -> ``{model: <registered name>, params: {...}}``::

        suite = SensorSuite.from_config(
            {"tactile": {"model": "fsr"}, "joint": {"model": "motor"}, "imu": {"model": "mems_imu"}},
            layout, rates={"tactile": 1000, "joint": 500, "imu": 500},
        )
    """

    def __init__(self, models: dict[str, SensorModel]):
        self.models = models

    @classmethod
    def from_config(cls, cfg: dict[str, Any], layout: SensorLayout, rates: dict[str, float]) -> SensorSuite:
        models = {}
        for name, group in layout.groups.items():
            if name not in cfg:
                raise KeyError(f"No sensor model configured for group '{name}'")
            spec = cfg[name]
            model = SENSOR_MODELS.build(spec["model"], dt=1.0 / rates[name], **spec.get("params", {}))
            if model.kind != group.kind:
                raise ValueError(f"Sensor model '{spec['model']}' is for '{model.kind}', group '{name}' is '{group.kind}'")
            models[name] = model
        return cls(models)

    @property
    def output_channels(self) -> dict[str, tuple[str, ...]]:
        return {n: m.output_channels for n, m in self.models.items()}

    def __call__(
        self, stimuli: dict[str, torch.Tensor], state: dict[str, State] | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, State]]:
        """``stimuli[group]``: ``[*batch, T, N, C_in]`` -> ``readings[group]``: ``[*batch, T, N, C_out]``."""
        readings, new_state = {}, {}
        for name, model in self.models.items():
            readings[name], new_state[name] = model(stimuli[name], None if state is None else state[name], generator)
        return readings, new_state

    def __repr__(self) -> str:
        return "SensorSuite(" + ", ".join(f"{n}={type(m).__name__}" for n, m in self.models.items()) + ")"
