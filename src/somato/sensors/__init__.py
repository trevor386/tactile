"""Sensor technology models. Importing this package registers all built-in models."""

from . import proprio, tactile  # noqa: F401  (registration side effects)
from .base import SENSOR_MODELS, STIMULUS_CHANNELS, SensorModel
from .suite import SensorSuite

__all__ = ["SENSOR_MODELS", "STIMULUS_CHANNELS", "SensorModel", "SensorSuite"]
