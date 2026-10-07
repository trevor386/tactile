"""Shared helpers for Isaac Lab scripts. Import only after the app has been launched."""

from __future__ import annotations

import torch

from somato.robots.factory import build_robot
from somato.sim.isaaclab.backend import IsaacLabBackend
from somato.sim.isaaclab.config import IsaacSnakeConfig
from somato.sim.setup import CollectConfig, make_runner


def make_isaac_runner(cfg: CollectConfig, device: str, generator: torch.Generator | None = None):
    """Returns ``(runner, desc, layout, catalog)`` backed by Isaac Lab."""
    desc, layout = build_robot(cfg.robot)
    catalog = cfg.catalog()
    icfg = IsaacSnakeConfig(**{**cfg.isaac, "num_envs": cfg.num_envs, "physics_dt": 1.0 / cfg.rates.physics_hz,
                               "device": device})
    backend = IsaacLabBackend(desc, layout, catalog, icfg, generator)
    return make_runner(backend, desc, layout, cfg, generator), desc, layout, catalog
