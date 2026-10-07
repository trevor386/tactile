"""Simulation layer: backend interface, terrain, contact/stimulus models, multi-rate runner.

Isaac Lab support lives in :mod:`somato.sim.isaaclab` and is imported lazily (it needs Isaac Sim).
"""

from .backend import RawSimState, SimBackend
from .contact_model import ContactModelConfig, TaxelContactModel
from .runner import LatentFrame, RateConfig, SimRunner
from .stimulus import StimulusPipeline
from .terrain import TerrainBatch, TerrainCatalog, TerrainClass, default_catalog_path

__all__ = [
    "RawSimState", "SimBackend", "ContactModelConfig", "TaxelContactModel", "LatentFrame", "RateConfig",
    "SimRunner", "StimulusPipeline", "TerrainBatch", "TerrainCatalog", "TerrainClass", "default_catalog_path",
]
