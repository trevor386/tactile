"""Build models from config dictionaries (``architecture`` selects the model family)."""

from __future__ import annotations

from typing import Any

from torch import nn

from somato.geometry.layout import LayoutInfo
from somato.models.baselines import (
    FlatBaselineConfig, FlatRecurrentModel, TransformerBaseline, TransformerBaselineConfig,
)
from somato.models.batch import GroupSpec
from somato.models.hierarchical import HierarchicalSomatoModel, ModelConfig
from somato.utils.config import from_dict

CONFIGS = {"hierarchical": ModelConfig, "flat_recurrent": FlatBaselineConfig, "transformer": TransformerBaselineConfig}


def parse_model_config(cfg: dict[str, Any] | ModelConfig | FlatBaselineConfig | TransformerBaselineConfig):
    if not isinstance(cfg, dict):
        return cfg
    arch = cfg.get("architecture", "hierarchical")
    if arch not in CONFIGS:
        raise ValueError(f"Unknown architecture '{arch}' (options: {list(CONFIGS)})")
    return from_dict(CONFIGS[arch], cfg)


def build_model(cfg: dict[str, Any] | ModelConfig | FlatBaselineConfig | TransformerBaselineConfig,
                groups: dict[str, GroupSpec], info: LayoutInfo | None = None) -> nn.Module:
    cfg = parse_model_config(cfg)
    if isinstance(cfg, ModelConfig):
        return HierarchicalSomatoModel(groups, cfg)
    if info is None:
        raise ValueError(f"{type(cfg).__name__} needs a LayoutInfo (the baselines read the layout's kinematics)")
    if isinstance(cfg, TransformerBaselineConfig):
        return TransformerBaseline(groups, info, cfg)
    return FlatRecurrentModel(groups, info, cfg)
