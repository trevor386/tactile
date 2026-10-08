"""mjlab (MuJoCo-Warp) integration, for comparing MuJoCo's soft contacts against Isaac Sim / PhysX.

Only :class:`MjlabSnakeConfig` and :func:`is_available` are importable without mjlab. mjlab mutates global
Warp settings on import, so never import it into an Isaac Sim process::

    from somato.sim.mjlab.backend import MjlabBackend
"""

from __future__ import annotations

import importlib.util

from .config import MjlabSnakeConfig


def is_available() -> bool:
    """True if the ``mjlab`` package can be imported."""
    return importlib.util.find_spec("mjlab") is not None


__all__ = ["MjlabSnakeConfig", "is_available"]
