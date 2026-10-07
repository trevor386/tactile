"""Terrain classes and per-environment terrain parameters.

A :class:`TerrainClass` describes a surface type by parameter *ranges*; each environment samples
concrete values on reset (domain randomization), so classes overlap and classification is not
trivially solvable from a single number.

Which parameters are enacted where:

=====================  ====================  ===========================  ==================
parameter              mock planar snake     Isaac Lab                     taxel contact model
=====================  ====================  ===========================  ==================
friction               yes (Coulomb)         yes (PhysX material)          shear estimate
anisotropy             yes (scale-like)      no (PhysX is isotropic)       --
restitution            --                    yes                           --
sinkage                --                    no                            footprint width
texture amp/wavelength --                    --                            pressure vibration
undulation             load sharing          no                            --
=====================  ====================  ===========================  ==================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path

import torch

from somato.utils.config import from_dict, load_yaml

Range = tuple[float, float]


@dataclass
class TerrainClass:
    name: str
    friction: Range = (0.5, 0.5)  # Coulomb coefficient (along the body axis for anisotropic snakes)
    anisotropy: Range = (1.0, 1.0)  # lateral / axial friction ratio (snake scales)
    restitution: Range = (0.0, 0.0)
    sinkage: Range = (0.0, 0.0)  # m, extra contact depth on soft terrain (wider taxel footprint)
    texture_amp: Range = (0.0, 0.0)  # relative pressure modulation by surface micro-texture
    texture_wavelength: Range = (0.003, 0.003)  # m
    undulation_amp: Range = (0.0, 0.0)  # relative load variation from macro unevenness
    undulation_wavelength: Range = (0.2, 0.2)  # m


SAMPLED = [f.name for f in fields(TerrainClass) if f.name != "name"]


@dataclass
class TerrainBatch:
    """Concrete terrain parameters for ``E`` environments (all tensors ``[E]`` unless noted)."""

    class_id: torch.Tensor
    params: dict[str, torch.Tensor]
    ground_height: torch.Tensor
    texture_k: torch.Tensor  # [E, M, 2] wave vectors of the micro-texture field
    texture_phase: torch.Tensor  # [E, M]
    undulation_k: torch.Tensor  # [E, M, 2]
    undulation_phase: torch.Tensor  # [E, M]

    def __getattr__(self, name):
        params = self.__dict__.get("params", {})
        if name in params:
            return params[name]
        raise AttributeError(name)

    @property
    def num_envs(self) -> int:
        return self.class_id.shape[0]

    @staticmethod
    def _field(xy: torch.Tensor, k: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Unit-variance random field: ``xy [E, ..., 2]`` -> ``[E, ...]``."""
        E, M = phase.shape
        extra = xy.dim() - 2
        kk = k.view(E, *([1] * extra), M, 2)
        ph = phase.view(E, *([1] * extra), M)
        arg = (xy.unsqueeze(-2) * kk).sum(-1) + ph
        return torch.sin(arg).sum(-1) * math.sqrt(2.0 / M)

    def texture(self, xy: torch.Tensor) -> torch.Tensor:
        return self._field(xy, self.texture_k, self.texture_phase)

    def undulation(self, xy: torch.Tensor) -> torch.Tensor:
        return self._field(xy, self.undulation_k, self.undulation_phase)

    def to(self, device) -> TerrainBatch:
        return TerrainBatch(
            self.class_id.to(device), {k: v.to(device) for k, v in self.params.items()},
            *(getattr(self, n).to(device) for n in ("ground_height", "texture_k", "texture_phase", "undulation_k",
                                                    "undulation_phase")),
        )

    def update(self, env_ids: torch.Tensor, other: TerrainBatch) -> None:
        """Overwrite environments ``env_ids`` with ``other`` (which has ``len(env_ids)`` envs)."""
        self.class_id[env_ids] = other.class_id
        for k in self.params:
            self.params[k][env_ids] = other.params[k]
        for name in ("ground_height", "texture_k", "texture_phase", "undulation_k", "undulation_phase"):
            getattr(self, name)[env_ids] = getattr(other, name)


@dataclass
class TerrainCatalog:
    classes: list[TerrainClass] = field(default_factory=list)
    texture_modes: int = 8

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.classes]

    def __len__(self) -> int:
        return len(self.classes)

    @classmethod
    def from_dict(cls, d: dict) -> TerrainCatalog:
        classes = [from_dict(TerrainClass, {"name": n, **{k: tuple(v) for k, v in p.items()}})
                   for n, p in d["classes"].items()]
        return cls(classes, d.get("texture_modes", 8))

    @classmethod
    def from_yaml(cls, path: str | Path) -> TerrainCatalog:
        return cls.from_dict(load_yaml(path))

    def sample(self, class_id: torch.Tensor, generator: torch.Generator | None = None, device="cpu") -> TerrainBatch:
        class_id = class_id.long().cpu()
        E, M = class_id.shape[0], self.texture_modes

        def uniform(n=E):
            return torch.rand(n, generator=generator)

        params = {}
        for name in SAMPLED:
            lo = torch.tensor([getattr(self.classes[c], name)[0] for c in class_id.tolist()])
            hi = torch.tensor([getattr(self.classes[c], name)[1] for c in class_id.tolist()])
            params[name] = lo + (hi - lo) * uniform()

        def wave(wavelength):
            ang = 2 * math.pi * torch.rand(E, M, generator=generator)
            # Spread wavelengths +-50% around the sampled value for a broadband texture.
            lam = wavelength[:, None] * (0.5 + torch.rand(E, M, generator=generator))
            k = (2 * math.pi / lam).unsqueeze(-1) * torch.stack([ang.cos(), ang.sin()], -1)
            return k, 2 * math.pi * torch.rand(E, M, generator=generator)

        tk, tp = wave(params["texture_wavelength"])
        uk, up = wave(params["undulation_wavelength"])
        batch = TerrainBatch(class_id, params, torch.zeros(E), tk, tp, uk, up)
        return batch.to(device)


def default_catalog_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "terrains" / "ice_forms.yaml"
