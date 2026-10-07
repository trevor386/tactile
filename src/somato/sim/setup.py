"""Assemble backends + runners from a collection config (shared by mock and Isaac scripts)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from somato.control.gaits import GaitConfig, SerpenoidGait
from somato.geometry.layout import SensorLayout
from somato.robots.description import RobotDescription
from somato.robots.factory import build_robot
from somato.sim.backend import SimBackend
from somato.sim.contact_model import ContactModelConfig
from somato.sim.mock.planar_snake import PlanarSnakeBackend, PlanarSnakeConfig
from somato.sim.runner import RateConfig, SimRunner
from somato.sim.stimulus import StimulusPipeline
from somato.sim.terrain import TerrainCatalog, default_catalog_path
from somato.utils.config import from_dict, load_yaml


@dataclass
class CollectConfig:
    robot: Any = "configs/robots/snake_planar.yaml"  # path or inline dict (see robots/factory.py)
    terrains: str = ""  # terrain catalog YAML (default: configs/terrains/ice_forms.yaml)
    out: str = "datasets/mock_terrain"
    seed: int = 0
    num_envs: int = 32
    rounds: int = 4
    steps_per_episode: int = 150
    warmup_steps: int = 25
    device: str = "cpu"
    rates: RateConfig = field(default_factory=RateConfig)
    gait: GaitConfig = field(default_factory=GaitConfig)
    contact: ContactModelConfig = field(default_factory=ContactModelConfig)
    mock: PlanarSnakeConfig = field(default_factory=PlanarSnakeConfig)
    isaac: dict[str, Any] = field(default_factory=dict)  # IsaacSnakeConfig overrides

    @classmethod
    def from_yaml(cls, path, overrides: dict | None = None) -> CollectConfig:
        d = load_yaml(path)
        d.update(overrides or {})
        for key in ("gait", "mock"):
            for k, v in list(d.get(key, {}).items()):
                if isinstance(v, list):
                    d[key][k] = tuple(v)
        return from_dict(cls, d)

    def catalog(self) -> TerrainCatalog:
        return TerrainCatalog.from_yaml(self.terrains or default_catalog_path())


def make_runner(backend: SimBackend, desc: RobotDescription, layout: SensorLayout, cfg: CollectConfig,
                generator: torch.Generator | None = None) -> SimRunner:
    gait = SerpenoidGait(desc, cfg.gait, backend.num_envs, backend.device, generator)
    return SimRunner(backend, StimulusPipeline(layout, cfg.contact), gait, cfg.rates)


def make_mock_runner(cfg: CollectConfig, generator: torch.Generator | None = None):
    """Returns ``(runner, desc, layout, catalog)`` for the planar mock simulator."""
    desc, layout = build_robot(cfg.robot)
    catalog = cfg.catalog()
    mock_cfg = PlanarSnakeConfig(**{**cfg.mock.__dict__, "physics_dt": 1.0 / cfg.rates.physics_hz})
    backend = PlanarSnakeBackend(desc, catalog, cfg.num_envs, mock_cfg, cfg.device, generator)
    return make_runner(backend, desc, layout, cfg, generator), desc, layout, catalog
