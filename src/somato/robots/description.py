"""Simulator-agnostic robot description (bodies, collision shapes, joints).

The description is the single source of truth for a robot's geometry. It is used to

* generate the URDF that Isaac Sim imports (``to_urdf``),
* place sensors on the robot (see :mod:`somato.geometry.layout`),
* compute body poses from joint encoders on hardware (``forward_kinematics``).

Existing robots can be loaded with ``RobotDescription.from_urdf``.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import torch

from somato.geometry.rotations import axis_angle_to_matrix, rpy_to_matrix

Vec3 = tuple[float, float, float]


@dataclass
class Shape:
    """A collision/visual primitive attached to a body.

    ``kind`` is one of ``cylinder``, ``capsule``, ``box`` or ``sphere``. Cylinders and capsules are
    aligned with their local z-axis (URDF convention); ``length`` is the length of the cylindrical
    section. ``size`` keys: cylinder/capsule -> ``radius, length``; box -> ``x, y, z``; sphere -> ``radius``.
    """

    kind: str
    size: dict[str, float]
    pos: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)


@dataclass
class Body:
    name: str
    mass: float = 1.0
    inertia: Vec3 = (1e-3, 1e-3, 1e-3)  # principal moments about the COM, in the body frame
    com: Vec3 = (0.0, 0.0, 0.0)
    shapes: list[Shape] = field(default_factory=list)


@dataclass
class Joint:
    name: str
    type: str  # "revolute" | "continuous" | "prismatic" | "fixed"
    parent: str
    child: str
    pos: Vec3 = (0.0, 0.0, 0.0)  # joint origin in the parent body frame
    rpy: Vec3 = (0.0, 0.0, 0.0)
    axis: Vec3 = (0.0, 0.0, 1.0)  # in the joint (= child body) frame
    lower: float = -math.pi
    upper: float = math.pi
    effort: float = 10.0
    velocity: float = 10.0

    @property
    def movable(self) -> bool:
        return self.type != "fixed"


@dataclass
class RobotDescription:
    name: str
    bodies: list[Body]
    joints: list[Joint]

    # ------------------------------------------------------------------ queries
    @property
    def body_names(self) -> list[str]:
        return [b.name for b in self.bodies]

    @property
    def joint_names(self) -> list[str]:
        """Names of movable joints, in description order."""
        return [j.name for j in self.joints if j.movable]

    @property
    def root_body(self) -> str:
        children = {j.child for j in self.joints}
        roots = [b.name for b in self.bodies if b.name not in children]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one root body, found {roots}")
        return roots[0]

    def body(self, name: str) -> Body:
        for b in self.bodies:
            if b.name == name:
                return b
        raise KeyError(name)

    def joint(self, name: str) -> Joint:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(name)

    def total_mass(self) -> float:
        return sum(b.mass for b in self.bodies)

    def _ordered_joints(self) -> list[Joint]:
        """Joints in topological order (parents before children)."""
        done = {self.root_body}
        ordered: list[Joint] = []
        remaining = list(self.joints)
        while remaining:
            progress = False
            for j in list(remaining):
                if j.parent in done:
                    ordered.append(j)
                    done.add(j.child)
                    remaining.remove(j)
                    progress = True
            if not progress:
                raise ValueError(f"Joints do not form a tree: {[j.name for j in remaining]}")
        return ordered

    # ------------------------------------------------------------- kinematics
    def forward_kinematics(
        self, root_pos: torch.Tensor, root_rot: torch.Tensor, joint_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Body frame poses from the root pose and movable joint positions.

        Args:
            root_pos: ``[..., 3]`` root body position (world).
            root_rot: ``[..., 3, 3]`` root body orientation (world).
            joint_pos: ``[..., num_movable_joints]`` in ``joint_names`` order.

        Returns:
            ``(body_pos [..., num_bodies, 3], body_rot [..., num_bodies, 3, 3])`` in ``body_names`` order.
        """
        dtype, device = root_pos.dtype, root_pos.device
        poses: dict[str, tuple[torch.Tensor, torch.Tensor]] = {self.root_body: (root_pos, root_rot)}
        q_index = {name: i for i, name in enumerate(self.joint_names)}
        for j in self._ordered_joints():
            p_pos, p_rot = poses[j.parent]
            o_pos = torch.tensor(j.pos, dtype=dtype, device=device)
            o_rot = rpy_to_matrix(torch.tensor(j.rpy, dtype=dtype, device=device))
            c_rot = p_rot @ o_rot
            c_pos = p_pos + (p_rot @ o_pos)
            if j.movable:
                q = joint_pos[..., q_index[j.name]]
                axis = torch.tensor(j.axis, dtype=dtype, device=device).expand(q.shape + (3,))
                if j.type == "prismatic":
                    c_pos = c_pos + (c_rot @ (axis * q.unsqueeze(-1)).unsqueeze(-1)).squeeze(-1)
                else:
                    c_rot = c_rot @ axis_angle_to_matrix(axis, q)
            poses[j.child] = (c_pos, c_rot)
        pos = torch.stack([poses[n][0].expand_as(root_pos) for n in self.body_names], dim=-2)
        rot = torch.stack([poses[n][1].expand_as(root_rot) for n in self.body_names], dim=-3)
        return pos, rot

    # -------------------------------------------------------------------- URDF
    def to_urdf(self) -> str:
        robot = ET.Element("robot", name=self.name)
        fmt = lambda v: " ".join(f"{x:.6g}" for x in v)  # noqa: E731
        for b in self.bodies:
            link = ET.SubElement(robot, "link", name=b.name)
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", xyz=fmt(b.com), rpy="0 0 0")
            ET.SubElement(inertial, "mass", value=f"{b.mass:.6g}")
            ixx, iyy, izz = b.inertia
            ET.SubElement(inertial, "inertia", ixx=f"{ixx:.6g}", iyy=f"{iyy:.6g}", izz=f"{izz:.6g}",
                          ixy="0", ixz="0", iyz="0")
            for tag in ("visual", "collision"):
                for s in b.shapes:
                    el = ET.SubElement(link, tag)
                    ET.SubElement(el, "origin", xyz=fmt(s.pos), rpy=fmt(s.rpy))
                    geom = ET.SubElement(el, "geometry")
                    if s.kind in ("cylinder", "capsule"):
                        # URDF has no capsule primitive; Isaac's importer can convert cylinders to capsules.
                        ET.SubElement(geom, "cylinder", radius=f"{s.size['radius']:.6g}",
                                      length=f"{s.size['length']:.6g}")
                    elif s.kind == "box":
                        ET.SubElement(geom, "box", size=fmt((s.size["x"], s.size["y"], s.size["z"])))
                    elif s.kind == "sphere":
                        ET.SubElement(geom, "sphere", radius=f"{s.size['radius']:.6g}")
                    else:
                        raise ValueError(f"Unsupported shape kind {s.kind}")
        for j in self.joints:
            joint = ET.SubElement(robot, "joint", name=j.name, type=j.type)
            ET.SubElement(joint, "parent", link=j.parent)
            ET.SubElement(joint, "child", link=j.child)
            ET.SubElement(joint, "origin", xyz=fmt(j.pos), rpy=fmt(j.rpy))
            if j.movable:
                ET.SubElement(joint, "axis", xyz=fmt(j.axis))
                ET.SubElement(joint, "limit", lower=f"{j.lower:.6g}", upper=f"{j.upper:.6g}",
                              effort=f"{j.effort:.6g}", velocity=f"{j.velocity:.6g}")
        ET.indent(robot)
        return '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode")

    def write_urdf(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_urdf())
        return path

    # -------------------------------------------------------------------- MJCF
    def to_mjcf(self, **kwargs) -> str:
        """MuJoCo XML (see :func:`somato.robots.mjcf.description_to_mjcf` for the options)."""
        from somato.robots.mjcf import description_to_mjcf  # lazy: mjcf imports this module

        return description_to_mjcf(self, **kwargs)

    def write_mjcf(self, path: str | Path, **kwargs) -> Path:
        from somato.robots.mjcf import write_mjcf

        return write_mjcf(self, path, **kwargs)

    @classmethod
    def from_urdf(cls, source: str | Path) -> RobotDescription:
        """Parse a URDF file (or XML string). Mesh geometry is skipped (no tactile auto-placement)."""
        text = str(source)
        if not text.lstrip().startswith("<"):
            text = Path(source).read_text()
        root = ET.fromstring(text)

        def vec(el, attr, default):
            if el is None or el.get(attr) is None:
                return default
            return tuple(float(v) for v in el.get(attr).split())

        bodies = []
        for link in root.findall("link"):
            inertial = link.find("inertial")
            mass, inertia, com = 1.0, (1e-3, 1e-3, 1e-3), (0.0, 0.0, 0.0)
            if inertial is not None:
                m = inertial.find("mass")
                mass = float(m.get("value")) if m is not None else mass
                it = inertial.find("inertia")
                if it is not None:
                    inertia = (float(it.get("ixx", 1e-3)), float(it.get("iyy", 1e-3)), float(it.get("izz", 1e-3)))
                com = vec(inertial.find("origin"), "xyz", com)
            shapes = []
            for col in link.findall("collision"):
                origin = col.find("origin")
                geom = col.find("geometry")
                pos, rpy = vec(origin, "xyz", (0.0, 0.0, 0.0)), vec(origin, "rpy", (0.0, 0.0, 0.0))
                if geom is None:
                    continue
                if (c := geom.find("cylinder")) is not None:
                    shapes.append(Shape("cylinder", {"radius": float(c.get("radius")),
                                                     "length": float(c.get("length"))}, pos, rpy))
                elif (b := geom.find("box")) is not None:
                    sx, sy, sz = (float(v) for v in b.get("size").split())
                    shapes.append(Shape("box", {"x": sx, "y": sy, "z": sz}, pos, rpy))
                elif (s := geom.find("sphere")) is not None:
                    shapes.append(Shape("sphere", {"radius": float(s.get("radius"))}, pos, rpy))
            bodies.append(Body(link.get("name"), mass, inertia, com, shapes))
        joints = []
        for jel in root.findall("joint"):
            origin = jel.find("origin")
            limit = jel.find("limit")
            joints.append(
                Joint(
                    name=jel.get("name"),
                    type=jel.get("type"),
                    parent=jel.find("parent").get("link"),
                    child=jel.find("child").get("link"),
                    pos=vec(origin, "xyz", (0.0, 0.0, 0.0)),
                    rpy=vec(origin, "rpy", (0.0, 0.0, 0.0)),
                    axis=vec(jel.find("axis"), "xyz", (1.0, 0.0, 0.0)),
                    lower=float(limit.get("lower", -math.pi)) if limit is not None else -math.pi,
                    upper=float(limit.get("upper", math.pi)) if limit is not None else math.pi,
                    effort=float(limit.get("effort", 10.0)) if limit is not None else 10.0,
                    velocity=float(limit.get("velocity", 10.0)) if limit is not None else 10.0,
                )
            )
        return cls(root.get("name", "robot"), bodies, joints)
