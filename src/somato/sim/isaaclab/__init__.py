"""Isaac Lab (Isaac Sim 5.x) integration.

Only :class:`IsaacSnakeConfig` and :func:`is_available` are importable without Isaac Sim. Import the
backend explicitly after launching the app::

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    from somato.sim.isaaclab.backend import IsaacLabBackend
"""

from __future__ import annotations

import importlib.util

from .config import IsaacSnakeConfig


def is_available() -> bool:
    """True if the ``isaaclab`` package can be imported (does not start the app)."""
    return importlib.util.find_spec("isaaclab") is not None


__all__ = ["IsaacSnakeConfig", "is_available"]
