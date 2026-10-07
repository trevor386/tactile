"""MJCF (MuJoCo XML) export of a :class:`~somato.robots.description.RobotDescription`.

The MJCF reproduces the URDF written by ``to_urdf`` (and hence what Isaac Sim imports), so the same
robot can be loaded into MuJoCo / mjlab (MuJoCo-Warp) and compared against Isaac.

Conventions
-----------
* One ``<body>`` per link, nested along the kinematic tree under ``<worldbody>``. The body frame is the
  URDF link frame (= the frame of the joint whose child it is): ``pos`` is the joint origin ``xyz`` and
  ``quat`` (wxyz) the joint origin ``rpy`` (``R = Rz Ry Rx``), both in the parent body frame.
* Joints sit at the child body origin with the axis in the child frame. ``revolute`` -> ``hinge`` with
  ``range`` and ``actuatorfrcrange = [-effort, effort]``; ``continuous`` -> unlimited ``hinge``;
  ``prismatic`` -> ``slide`` with range; ``fixed`` -> no joint (the child body is welded). MuJoCo orders
  ``qpos`` by tree traversal, not by ``desc.joint_names``: address joints by name.
* ``floating_base`` adds ``<freejoint name="root"/>`` to the root body (qpos = position + wxyz quaternion).
* ``<inertial>`` is written explicitly from ``mass``/``com``/``inertia`` (principal moments about the COM,
  body-frame axes) and ``inertiafromgeom`` is off, so geom shapes never influence mass properties.
* One geom per :class:`~somato.robots.description.Shape`, named ``{body}_geom{i}``, with ``pos``/``quat``
  from the shape's ``pos``/``rpy``. Cylinders and capsules are aligned with local z; ``length`` is the
  length of the cylindrical section, so ``size = (radius, length / 2)`` (a capsule's total length is
  ``length + 2 * radius``). Cylinder shapes become capsules by default, matching Isaac's URDF importer
  with ``replace_cylinders_with_capsules=True``.
* No ``<option>``, actuators, sensors, lights or ground plane: the simulator layer adds those.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from somato.geometry.rotations import matrix_to_quat, rpy_to_matrix
from somato.robots.description import RobotDescription, Vec3

Quat = tuple[float, float, float, float]
SiteSpec = tuple[str, str, Vec3, Quat]  # (name, body, pos, quat_wxyz) in the body frame

_FREEJOINT_NAME = "root"
_SITE_SIZE = "0.005"


def _fmt(values) -> str:
    return " ".join(f"{float(x):.9g}" for x in values)


def _rpy_to_quat(rpy: Vec3) -> list[float]:
    """URDF roll-pitch-yaw -> unit quaternion (wxyz), evaluated in float64."""
    return matrix_to_quat(rpy_to_matrix(torch.tensor(rpy, dtype=torch.float64))).tolist()


def description_to_mjcf(
    desc: RobotDescription,
    *,
    floating_base: bool = True,
    cylinders_as_capsules: bool = True,
    self_collision: bool = False,
    sites: list[SiteSpec] | None = None,
) -> str:
    """Return the MJCF (XML string) for ``desc``.

    Args:
        floating_base: add a free joint named ``root`` to the root body (else it is welded to the world).
        cylinders_as_capsules: export ``cylinder`` shapes as capsules (``capsule`` shapes always are).
        self_collision: if False, robot geoms get ``contype=0 conaffinity=1``.
        sites: optional ``(name, body, pos, quat_wxyz)`` entries, each added as a ``<site>`` in that body.
    """
    sites = list(sites or [])
    bodies = {b.name: b for b in desc.bodies}
    for name, body, _, _ in sites:
        if body not in bodies:
            raise ValueError(f"Site {name!r} refers to unknown body {body!r}")
    if floating_base and any(j.name == _FREEJOINT_NAME for j in desc.joints):
        raise ValueError(f"Joint name {_FREEJOINT_NAME!r} is reserved for the floating-base free joint")

    mujoco = ET.Element("mujoco", model=desc.name)
    ET.SubElement(mujoco, "compiler", angle="radian", autolimits="true", inertiafromgeom="false")
    worldbody = ET.SubElement(mujoco, "worldbody")

    # MuJoCo collides geoms i, j iff (contype_i & conaffinity_j) | (contype_j & conaffinity_i) is non-zero.
    # contype=0 / conaffinity=1 on the robot means robot-robot pairs never match, while a default geom such
    # as a ground plane (contype=1, conaffinity=1) still collides with the robot via the plane's contype.
    # self_collision=True keeps the defaults (contype=1, conaffinity=1) so all pairs collide, except
    # parent-child pairs, which MuJoCo always filters.
    collide = {} if self_collision else {"contype": "0", "conaffinity": "1"}

    def add_body(parent: ET.Element, name: str, pos: Vec3 | None, rpy: Vec3 | None, joint) -> ET.Element:
        b = bodies[name]
        attrs = {"name": name}
        if pos is not None:
            attrs["pos"] = _fmt(pos)
            attrs["quat"] = _fmt(_rpy_to_quat(rpy))
        el = ET.SubElement(parent, "body", attrs)

        if joint is None:
            if floating_base:
                ET.SubElement(el, "freejoint", name=_FREEJOINT_NAME)
        elif joint.type in ("revolute", "continuous", "prismatic"):
            kind = "slide" if joint.type == "prismatic" else "hinge"
            j = ET.SubElement(el, "joint", name=joint.name, type=kind, axis=_fmt(joint.axis))
            if joint.type == "continuous":
                j.set("limited", "false")
            else:
                j.set("range", _fmt((joint.lower, joint.upper)))
            if math.isfinite(joint.effort) and joint.effort > 0:
                j.set("actuatorfrcrange", _fmt((-joint.effort, joint.effort)))
        elif joint.type != "fixed":
            raise ValueError(f"Unsupported joint type {joint.type}")

        ET.SubElement(el, "inertial", pos=_fmt(b.com), mass=f"{b.mass:.9g}", diaginertia=_fmt(b.inertia))

        for i, s in enumerate(b.shapes):
            geom = {"name": f"{name}_geom{i}", "pos": _fmt(s.pos), "quat": _fmt(_rpy_to_quat(s.rpy))}
            if s.kind in ("cylinder", "capsule"):
                kind = "cylinder" if (s.kind == "cylinder" and not cylinders_as_capsules) else "capsule"
                geom.update(type=kind, size=_fmt((s.size["radius"], 0.5 * s.size["length"])))
            elif s.kind == "box":
                geom.update(type="box", size=_fmt((0.5 * s.size[k] for k in ("x", "y", "z"))))
            elif s.kind == "sphere":
                geom.update(type="sphere", size=_fmt((s.size["radius"],)))
            else:
                raise ValueError(f"Unsupported shape kind {s.kind}")
            ET.SubElement(el, "geom", {**geom, **collide})

        for site_name, site_body, site_pos, site_quat in sites:
            if site_body == name:
                ET.SubElement(el, "site", name=site_name, pos=_fmt(site_pos), quat=_fmt(site_quat),
                              size=_SITE_SIZE)
        return el

    # Joints are visited parents-first, so each parent element exists (and already holds its joint,
    # inertial and geoms) before its child bodies are appended to it.
    elements = {desc.root_body: add_body(worldbody, desc.root_body, None, None, None)}
    for j in desc._ordered_joints():
        elements[j.child] = add_body(elements[j.parent], j.child, j.pos, j.rpy, j)

    ET.indent(mujoco)
    return '<?xml version="1.0"?>\n' + ET.tostring(mujoco, encoding="unicode")


def write_mjcf(desc: RobotDescription, path: str | Path, **kwargs) -> Path:
    """Write :func:`description_to_mjcf` output to ``path`` (parent directories are created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(description_to_mjcf(desc, **kwargs))
    return path
