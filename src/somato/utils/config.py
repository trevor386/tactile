"""Typed configuration helpers.

Configs are plain dataclasses so they are easy to construct in code and in tests. ``from_dict``
builds them from (nested) dictionaries loaded from YAML and rejects unknown keys, so typos in a
config file fail loudly instead of being silently ignored.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _resolve_dataclass_type(tp: Any) -> type | None:
    """Return the dataclass type contained in ``tp`` (handles ``Optional[X]`` / ``X | None``)."""
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        return tp
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:
        for arg in typing.get_args(tp):
            if dataclasses.is_dataclass(arg) and isinstance(arg, type):
                return arg
    return None


def from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
    """Recursively build dataclass ``cls`` from ``data``, rejecting unknown keys."""
    if data is None:
        return cls()
    if dataclasses.is_dataclass(data):
        return data  # already built
    hints = typing.get_type_hints(cls)
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise KeyError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}. Valid keys: {sorted(field_names)}")
    kwargs = {}
    for key, value in data.items():
        sub = _resolve_dataclass_type(hints.get(key))
        if sub is not None and isinstance(value, dict):
            kwargs[key] = from_dict(sub, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def to_dict(obj: Any) -> Any:
    """Convert (nested) dataclasses to plain python containers suitable for YAML."""
    if dataclasses.is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj
