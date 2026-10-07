"""Procedural snake robot: a serial chain of capsule links joined by yaw and/or pitch joints.

Frames: link ``i``'s frame sits on the link axis at its proximal (head-side) end; the link extends
along its local +x axis. ``link_0`` is the head. Joint ``i`` connects ``link_i`` (parent) to
``link_{i+1}`` (child) at ``x = link_length`` in the parent frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from somato.robots.description import Body, Joint, RobotDescription, Shape

_AXES = {"z": (0.0, 0.0, 1.0), "y": (0.0, 1.0, 0.0)}


@dataclass
class SnakeConfig:
    num_links: int = 12
    link_length: float = 0.10
    radius: float = 0.03
    link_mass: float = 0.25
    # Pattern of joint axes, repeated along the body. ["z"]: planar (yaw only); ["z", "y"]: yaw/pitch
    # alternating (needed for sidewinding/rolling on isotropic friction, e.g. in Isaac Sim).
    joint_axes: list[str] = field(default_factory=lambda: ["z"])
    joint_limit_deg: float = 70.0
    joint_effort: float = 6.0
    joint_velocity: float = 8.0
    name: str = "snake"

    @property
    def num_joints(self) -> int:
        return self.num_links - 1

    def joint_axis(self, j: int) -> str:
        return self.joint_axes[j % len(self.joint_axes)]


def link_name(i: int) -> str:
    return f"link_{i}"


def joint_name(i: int) -> str:
    return f"joint_{i}"


def default_snake_sensor_spec(
    n_rings: int = 3, n_per_ring: int = 8, skin: float = 0.002, imu_body: str = "link_0"
) -> dict:
    """Default sensor spec for :func:`somato.geometry.build_layout`: taxel rings on every link,
    one sensor per joint and an IMU in the head."""
    return {
        "tactile": {"placements": [{"generator": "cylinder", "bodies": r"link_\d+", "n_rings": n_rings,
                                    "n_per_ring": n_per_ring, "skin": skin}]},
        "joint": {"placements": [{"generator": "joints"}]},
        "imu": {"placements": [{"generator": "point", "body": imu_body, "pos": (0.05, 0.0, 0.0),
                                "name": "imu"}]},
    }


def make_snake_description(cfg: SnakeConfig) -> RobotDescription:
    L, r, m = cfg.link_length, cfg.radius, cfg.link_mass
    # Capsule whose total length equals the link length (cylinder section + two hemispheres).
    cyl_len = max(L - 2.0 * r, 1e-3)
    shape = Shape("capsule", {"radius": r, "length": cyl_len}, pos=(L / 2.0, 0.0, 0.0), rpy=(0.0, math.pi / 2, 0.0))
    ixx = 0.5 * m * r * r
    iyy = izz = m * (3.0 * r * r + L * L) / 12.0
    bodies = [Body(link_name(i), m, (ixx, iyy, izz), (L / 2.0, 0.0, 0.0), [shape]) for i in range(cfg.num_links)]
    lim = math.radians(cfg.joint_limit_deg)
    joints = [
        Joint(
            name=joint_name(i),
            type="revolute",
            parent=link_name(i),
            child=link_name(i + 1),
            pos=(L, 0.0, 0.0),
            axis=_AXES[cfg.joint_axis(i)],
            lower=-lim,
            upper=lim,
            effort=cfg.joint_effort,
            velocity=cfg.joint_velocity,
        )
        for i in range(cfg.num_joints)
    ]
    return RobotDescription(cfg.name, bodies, joints)
