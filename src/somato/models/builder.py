"""Build models from config dictionaries (``architecture`` selects the model family)."""

from __future__ import annotations

from typing import Any

from torch import nn

from somato.geometry.layout import LayoutInfo
from somato.models.baselines import FlatBaselineConfig, FlatRecurrentModel
from somato.models.batch import GroupSpec
from somato.models.hierarchical import HierarchicalSomatoModel, ModelConfig
from somato.utils.config import from_dict


def parse_model_config(cfg: dict[str, Any] | ModelConfig | FlatBaselineConfig):
    if not isinstance(cfg, dict):
        return cfg
    arch = cfg.get("architecture", "hierarchical")
    if arch == "hierarchical":
        return from_dict(ModelConfig, cfg)
    if arch == "flat_recurrent":
        return from_dict(FlatBaselineConfig, cfg)
    raise ValueError(f"Unknown architecture '{arch}'")


def build_model(cfg: dict[str, Any] | ModelConfig | FlatBaselineConfig, groups: dict[str, GroupSpec],
                info: LayoutInfo | None = None) -> nn.Module:
    cfg = parse_model_config(cfg)
    if isinstance(cfg, ModelConfig):
        return HierarchicalSomatoModel(groups, cfg)
    if info is None:
        raise ValueError("The flat baseline needs a LayoutInfo (it is tied to one sensor layout)")
    return FlatRecurrentModel(groups, info, cfg)
