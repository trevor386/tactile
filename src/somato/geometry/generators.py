"""Sensor placement generators.

Each generator returns a list of :class:`Placement` (sensors on one body, expressed in that body's
frame). Generators are registered by name so a robot's sensor geometry can be described in YAML,
e.g.::

    tactile:
      placements:
        - generator: cylinder
          bodies: "link_\\d+"
          params: {n_rings: 3, n_per_ring: 8, skin: 0.002}

Taxel frames follow the e-skin convention: +z is the outward surface normal. For cylinders the
x-axis is aligned with the cylinder axis so shear channels have a consistent meaning.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import torch

from somato.geometry.rotations import frame_from_normal, rpy_to_matrix
from somato.utils.registry import Registry

if TYPE_CHECKING:
    from somato.robots.description import RobotDescription, Shape


@dataclass
class Placement:
    body: str
    local_pos: torch.Tensor  # [n, 3]
    local_rot: torch.Tensor  # [n, 3, 3]
    area: torch.Tensor  # [n]
    names: list[str]
    joint_names: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


PLACEMENT_GENERATORS: Registry[Callable[..., list[Placement]]] = Registry("placement generator")


def _match_bodies(desc: RobotDescription, pattern: str) -> list[str]:
    names = [n for n in desc.body_names if re.fullmatch(pattern, n)]
    if not names:
        raise ValueError(f"No bodies match pattern '{pattern}' in {desc.body_names}")
    return names


def _shape_transform(shape: Shape) -> tuple[torch.Tensor, torch.Tensor]:
    rot = rpy_to_matrix(torch.tensor(shape.rpy, dtype=torch.float32))
    pos = torch.tensor(shape.pos, dtype=torch.float32)
    return rot, pos


def cylinder_surface(
    radius: float, length: float, n_rings: int, n_per_ring: int, skin: float = 0.0,
    angle_offset: float = 0.0, stagger: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Taxels on a cylinder aligned with local z. Returns ``(pos [n,3], rot [n,3,3], area [n])``."""
    ring_s = -length / 2 + (torch.arange(n_rings, dtype=torch.float32) + 0.5) * (length / n_rings)
    k = torch.arange(n_per_ring, dtype=torch.float32)
    pos, normal = [], []
    for r_i, s in enumerate(ring_s):
        shift = 0.5 if (stagger and r_i % 2 == 1) else 0.0
        ang = angle_offset + (k + shift) * (2 * math.pi / n_per_ring)
        n = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1)
        pos.append(n * (radius + skin) + torch.tensor([0.0, 0.0, float(s)]))
        normal.append(n)
    pos_t, normal_t = torch.cat(pos), torch.cat(normal)
    axis = torch.tensor([0.0, 0.0, 1.0]).expand_as(normal_t)
    rot = frame_from_normal(normal_t, axis)
    area = torch.full((pos_t.shape[0],), 2 * math.pi * radius * length / (n_rings * n_per_ring))
    return pos_t, rot, area


@PLACEMENT_GENERATORS.register("cylinder")
def cylinder_generator(
    desc: RobotDescription, bodies: str = ".*", shape_index: int = 0, n_rings: int = 3, n_per_ring: int = 8,
    skin: float = 0.0, angle_offset: float = 0.0, stagger: bool = True, prefix: str = "taxel",
) -> list[Placement]:
    """Rings of taxels around a body's cylinder/capsule collision shape (cylindrical section)."""
    out = []
    for name in _match_bodies(desc, bodies):
        shape = desc.body(name).shapes[shape_index]
        if shape.kind not in ("cylinder", "capsule"):
            raise ValueError(f"Body {name} shape {shape_index} is a {shape.kind}, expected cylinder/capsule")
        p, R, a = cylinder_surface(shape.size["radius"], shape.size["length"], n_rings, n_per_ring, skin,
                                   angle_offset, stagger)
        s_rot, s_pos = _shape_transform(shape)
        out.append(Placement(name, p @ s_rot.T + s_pos, s_rot @ R, a,
                             [f"{name}/{prefix}_{i}" for i in range(p.shape[0])]))
    return out


@PLACEMENT_GENERATORS.register("sphere")
def sphere_generator(
    desc: RobotDescription, bodies: str = ".*", shape_index: int = 0, n: int = 32, skin: float = 0.0,
    lower_hemisphere_only: bool = False, prefix: str = "taxel",
) -> list[Placement]:
    """Fibonacci-sphere taxels on a spherical collision shape (e.g. a foot)."""
    out = []
    for name in _match_bodies(desc, bodies):
        shape = desc.body(name).shapes[shape_index]
        radius = shape.size["radius"]
        i = torch.arange(n, dtype=torch.float32) + 0.5
        phi = torch.acos(1 - 2 * i / n)
        theta = math.pi * (1 + 5**0.5) * i
        normal = torch.stack([torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)], -1)
        if lower_hemisphere_only:
            normal = normal[normal[:, 2] <= 0]
        s_rot, s_pos = _shape_transform(shape)
        rot = s_rot @ frame_from_normal(normal)
        pos = (normal * (radius + skin)) @ s_rot.T + s_pos
        area = torch.full((normal.shape[0],), 4 * math.pi * radius**2 / n)
        out.append(Placement(name, pos, rot, area, [f"{name}/{prefix}_{k}" for k in range(normal.shape[0])]))
    return out


@PLACEMENT_GENERATORS.register("grid_patch")
def grid_patch_generator(
    desc: RobotDescription, bodies: str = ".*", center: tuple = (0.0, 0.0, 0.0), rpy: tuple = (0.0, 0.0, 0.0),
    nx: int = 8, ny: int = 8, dx: float = 0.01, dy: float = 0.01, prefix: str = "taxel",
) -> list[Placement]:
    """A flat ``nx x ny`` taxel grid. ``center``/``rpy`` place the patch (normal = patch +z) in the body frame."""
    gx = (torch.arange(nx, dtype=torch.float32) - (nx - 1) / 2) * dx
    gy = (torch.arange(ny, dtype=torch.float32) - (ny - 1) / 2) * dy
    xx, yy = torch.meshgrid(gx, gy, indexing="ij")
    local = torch.stack([xx.flatten(), yy.flatten(), torch.zeros(nx * ny)], -1)
    R = rpy_to_matrix(torch.tensor(rpy, dtype=torch.float32))
    c = torch.tensor(center, dtype=torch.float32)
    out = []
    for name in _match_bodies(desc, bodies):
        out.append(Placement(name, local @ R.T + c, R.expand(nx * ny, 3, 3).clone(),
                             torch.full((nx * ny,), dx * dy), [f"{name}/{prefix}_{k}" for k in range(nx * ny)]))
    return out


@PLACEMENT_GENERATORS.register("explicit")
def explicit_generator(
    desc: RobotDescription, body: str, positions: list | None = None, normals: list | None = None,
    areas: list | float = 1e-4, file: str | None = None, prefix: str = "taxel",
) -> list[Placement]:
    """Arbitrary taxels given as lists or loaded from a ``.pt``/``.npz`` file with ``positions``/``normals``.

    Use this for skins designed in CAD (e.g. exported taxel centers and normals for an arm skin).
    """
    if file is not None:
        if file.endswith(".npz"):
            import numpy as np

            data = np.load(file)
            positions, normals = data["positions"].tolist(), data["normals"].tolist()
            areas = data["areas"].tolist() if "areas" in data else areas
        else:
            data = torch.load(file)
            positions, normals = data["positions"].tolist(), data["normals"].tolist()
            areas = data.get("areas", areas)
            areas = areas.tolist() if torch.is_tensor(areas) else areas
    if body not in desc.body_names:
        raise ValueError(f"Unknown body {body}")
    pos = torch.tensor(positions, dtype=torch.float32)
    rot = frame_from_normal(torch.tensor(normals, dtype=torch.float32))
    area = torch.as_tensor(areas, dtype=torch.float32).expand(pos.shape[0]).clone()
    return [Placement(body, pos, rot, area, [f"{body}/{prefix}_{k}" for k in range(pos.shape[0])])]


@PLACEMENT_GENERATORS.register("point")
def point_generator(
    desc: RobotDescription, body: str, pos: tuple = (0.0, 0.0, 0.0), rpy: tuple = (0.0, 0.0, 0.0),
    name: str | None = None,
) -> list[Placement]:
    """A single sensor (e.g. an IMU) rigidly attached to ``body``."""
    if body not in desc.body_names:
        raise ValueError(f"Unknown body {body}")
    R = rpy_to_matrix(torch.tensor(rpy, dtype=torch.float32))
    return [Placement(body, torch.tensor([pos], dtype=torch.float32), R[None], torch.ones(1),
                      [name or f"{body}/sensor"])]


@PLACEMENT_GENERATORS.register("joints")
def joints_generator(desc: RobotDescription, joints: str = ".*") -> list[Placement]:
    """One sensor per movable joint, located at the joint origin with +z along the joint axis.

    The sensor is attached to the joint's child body (whose frame coincides with the joint frame).
    """
    out = []
    for j in desc.joints:
        if not j.movable or not re.fullmatch(joints, j.name):
            continue
        rot = frame_from_normal(torch.tensor([j.axis], dtype=torch.float32))
        out.append(Placement(j.child, torch.zeros(1, 3), rot, torch.ones(1), [j.name], joint_names=[j.name]))
    if not out:
        raise ValueError(f"No movable joints match '{joints}'")
    return out
