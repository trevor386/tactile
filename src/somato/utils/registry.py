"""Name -> constructor registries so components can be swapped from config files."""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A small registry mapping string keys to classes or factory functions.

    Example::

        SENSOR_MODELS = Registry("sensor model")

        @SENSOR_MODELS.register("fsr")
        class FSRTactile(SensorModel): ...

        model = SENSOR_MODELS.build("fsr", area=1e-4)
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(obj: Callable[..., T]) -> Callable[..., T]:
            if name in self._entries:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._entries[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable[..., T]:
        if name not in self._entries:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {sorted(self._entries)}")
        return self._entries[name]

    def build(self, name: str, *args: Any, **kwargs: Any) -> T:
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries
