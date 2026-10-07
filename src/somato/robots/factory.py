"""Build a robot description and its sensor layout from a config (YAML) dictionary.

Example config::

    robot:
      type: snake            # or: urdf (with `path: my_robot.urdf`)
      params: {num_links: 12, joint_axes: [z]}
    sensors:                 # build_layout spec; optional for the snake (defaults used)
      tactile: {placements: [{generator: cylinder, bodies: "link_\\\\d+", n_rings: 3, n_per_ring: 8}]}
      joint: {placements: [{generator: joints}]}
      imu: {placements: [{generator: point, body: link_0}]}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from somato.geometry.layout import SensorLayout, build_layout
from somato.robots.description import RobotDescription
from somato.robots.snake import SnakeConfig, default_snake_sensor_spec, make_snake_description
from somato.utils.config import from_dict, load_yaml


def build_robot(cfg: dict[str, Any] | str | Path) -> tuple[RobotDescription, SensorLayout]:
    if isinstance(cfg, (str, Path)):
        cfg = load_yaml(cfg)
    rcfg = cfg["robot"]
    kind = rcfg.get("type", "snake")
    if kind == "snake":
        desc = make_snake_description(from_dict(SnakeConfig, rcfg.get("params", {})))
        spec = cfg.get("sensors") or default_snake_sensor_spec()
    elif kind == "urdf":
        desc = RobotDescription.from_urdf(rcfg["path"])
        if "sensors" not in cfg:
            raise ValueError("URDF robots need an explicit `sensors` layout spec")
        spec = cfg["sensors"]
    else:
        raise ValueError(f"Unknown robot type {kind}")
    return desc, build_layout(desc, spec)
