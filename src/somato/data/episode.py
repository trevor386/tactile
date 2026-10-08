"""On-disk dataset format.

A dataset directory contains::

    meta.yaml        rates, sensor groups, terrain class names, data kind ("stimulus" | "readings")
    robot.urdf       the robot description used to generate the data
    layout.pt        the SensorLayout
    episodes/ep_000000.npz ...

Each episode file holds, for ``T`` latent steps:

* ``data/<group>``: ``[T, S_g, N_g, C]`` float16 -- ideal stimuli (simulation) or raw readings (hardware);
  float32 for arrays that exceed the float16 range (e.g. taxel pressure peaks above 65 kPa in Isaac Sim)
* ``body_pos`` ``[T, Nb, 3]`` / ``body_quat`` ``[T, Nb, 4]``: body poses at the end of each latent step
* ``label/<name>``: per-episode scalars (``terrain``) or per-step arrays (``slip_speed [T, Nb]``)
* ``param/<name>``: per-episode terrain parameters (for regression tasks / analysis)

Storing *stimuli* lets you change the sensor technology (FSR, capacitive, ...) at training time
without re-simulating. Files are compressed; mostly-zero taxel arrays compress very well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from somato.geometry.layout import SensorLayout
from somato.robots.description import RobotDescription
from somato.utils.config import load_yaml, save_yaml


_F16_MAX = float(np.finfo(np.float16).max)


def _compact(x: np.ndarray) -> np.ndarray:
    """float16 when every finite value fits, else float32 (a cast would turn the overflow into inf)."""
    finite = x[np.isfinite(x)]
    return x.astype(np.float16) if finite.size == 0 or np.abs(finite).max() <= _F16_MAX else x.astype(np.float32)


@dataclass
class Episode:
    data: dict[str, torch.Tensor]
    body_pos: torch.Tensor
    body_quat: torch.Tensor
    labels: dict[str, torch.Tensor] = field(default_factory=dict)
    params: dict[str, float] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return self.body_pos.shape[0]

    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {
            "body_pos": self.body_pos.float().cpu().numpy(),
            "body_quat": self.body_quat.float().cpu().numpy(),
        }
        for k, v in self.data.items():
            arrays[f"data/{k}"] = _compact(v.cpu().numpy())
        for k, v in self.labels.items():
            v = torch.as_tensor(v).cpu()
            arrays[f"label/{k}"] = v.numpy() if not v.is_floating_point() else _compact(v.numpy())
        for k, v in self.params.items():
            arrays[f"param/{k}"] = np.asarray(v, dtype=np.float32)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Episode:
        with np.load(path) as f:
            ep = cls(data={}, body_pos=torch.from_numpy(f["body_pos"]), body_quat=torch.from_numpy(f["body_quat"]))
            for key in f.files:
                if key.startswith("data/"):
                    ep.data[key[5:]] = torch.from_numpy(f[key])
                elif key.startswith("label/"):
                    ep.labels[key[6:]] = torch.from_numpy(np.array(f[key]))
                elif key.startswith("param/"):
                    ep.params[key[6:]] = float(f[key])
        return ep


@dataclass
class DatasetMeta:
    data_kind: str  # "stimulus" | "readings"
    rates: dict[str, Any]  # RateConfig fields
    groups: dict[str, dict[str, Any]]  # name -> {kind, count, channels}
    terrain_names: list[str]
    num_episodes: int = 0
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"data_kind": self.data_kind, "rates": self.rates, "groups": self.groups,
                "terrain_names": self.terrain_names, "num_episodes": self.num_episodes, "info": self.info}


class DatasetWriter:
    def __init__(self, root: str | Path, meta: DatasetMeta, desc: RobotDescription, layout: SensorLayout):
        self.root = Path(root)
        (self.root / "episodes").mkdir(parents=True, exist_ok=True)
        self.meta = meta
        desc.write_urdf(self.root / "robot.urdf")
        layout.save(self.root / "layout.pt")
        existing = sorted((self.root / "episodes").glob("ep_*.npz"))
        self.count = len(existing)

    def write(self, episode: Episode) -> Path:
        path = self.root / "episodes" / f"ep_{self.count:06d}.npz"
        episode.save(path)
        self.count += 1
        return path

    def close(self) -> None:
        self.meta.num_episodes = self.count
        save_yaml(self.meta.to_dict(), self.root / "meta.yaml")


def read_meta(root: str | Path) -> DatasetMeta:
    d = load_yaml(Path(root) / "meta.yaml")
    return DatasetMeta(**d)


def episode_paths(root: str | Path) -> list[Path]:
    return sorted((Path(root) / "episodes").glob("ep_*.npz"))


def load_robot(root: str | Path) -> tuple[RobotDescription, SensorLayout]:
    root = Path(root)
    return RobotDescription.from_urdf(root / "robot.urdf"), SensorLayout.load(root / "layout.pt")
