"""Shared helpers for mjlab scripts."""

from __future__ import annotations

import torch

from somato.robots.factory import build_robot
from somato.sim.mjlab.backend import MjlabBackend
from somato.sim.mjlab.config import MjlabSnakeConfig
from somato.sim.setup import CollectConfig, make_runner


def make_mjlab_runner(cfg: CollectConfig, device: str, generator: torch.Generator | None = None):
    """Returns ``(runner, desc, layout, catalog)`` backed by mjlab."""
    desc, layout = build_robot(cfg.robot)
    catalog = cfg.catalog()
    overrides = dict(cfg.mjlab)
    if "contact_solref" in overrides:
        overrides["contact_solref"] = tuple(overrides["contact_solref"])
    mcfg = MjlabSnakeConfig(**{**overrides, "num_envs": cfg.num_envs, "physics_dt": 1.0 / cfg.rates.physics_hz,
                               "device": device})
    backend = MjlabBackend(desc, layout, catalog, mcfg, generator)
    return make_runner(backend, desc, layout, cfg, generator), desc, layout, catalog
