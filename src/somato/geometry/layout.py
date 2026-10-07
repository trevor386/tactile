"""Sensor layouts: where every sensor sits on the robot.

A :class:`SensorLayout` is a set of named *sensor groups*. Every group shares one temporal encoder
(stage 1) and one sensor model, so a group is "a modality on a given technology", e.g. ``tactile``
(FSR taxels), ``joint`` (motor encoders + current sensing) or ``imu``. Each sensor is rigidly
attached to a body with a fixed local pose; world poses at runtime come from body poses supplied by
the simulator or by forward kinematics on hardware.

Node order everywhere in the model is the concatenation of groups in layout order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from somato.geometry.generators import PLACEMENT_GENERATORS, Placement

if TYPE_CHECKING:
    from somato.robots.description import RobotDescription


@dataclass
class SensorGroup:
    name: str
    kind: str  # "tactile" | "joint" | "imu"
    body_index: torch.Tensor  # [n] long, index into SensorLayout.body_names
    local_pos: torch.Tensor  # [n, 3]
    local_rot: torch.Tensor  # [n, 3, 3]
    area: torch.Tensor  # [n] (taxel area in m^2; 1 for point sensors)
    names: list[str]
    joint_names: list[str] | None = None  # for kind == "joint": simulator joint per sensor

    def __len__(self) -> int:
        return self.body_index.shape[0]


@dataclass
class SensorLayout:
    body_names: list[str]
    groups: dict[str, SensorGroup] = field(default_factory=dict)

    # ---------------------------------------------------------------- shape info
    @property
    def group_names(self) -> list[str]:
        return list(self.groups)

    @property
    def num_sensors(self) -> int:
        return sum(len(g) for g in self.groups.values())

    def slices(self) -> dict[str, slice]:
        out, start = {}, 0
        for name, g in self.groups.items():
            out[name] = slice(start, start + len(g))
            start += len(g)
        return out

    def group_id(self) -> torch.Tensor:
        """``[N]`` index of each node's group (layout order)."""
        return torch.cat([torch.full((len(g),), i, dtype=torch.long) for i, g in enumerate(self.groups.values())])

    def _cat(self, attr: str) -> torch.Tensor:
        return torch.cat([getattr(g, attr) for g in self.groups.values()])

    @property
    def body_index(self) -> torch.Tensor:
        return self._cat("body_index")

    @property
    def area(self) -> torch.Tensor:
        return self._cat("area")

    # ---------------------------------------------------------------- kinematics
    def world_poses(
        self, body_pos: torch.Tensor, body_rot: torch.Tensor, group: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sensor world poses from body poses.

        Args:
            body_pos: ``[..., num_bodies, 3]``; body_rot: ``[..., num_bodies, 3, 3]``.
            group: restrict to one group (default: all groups concatenated).

        Returns:
            ``(pos [..., N, 3], rot [..., N, 3, 3])``.
        """
        if group is None:
            idx, lpos, lrot = self.body_index, self._cat("local_pos"), self._cat("local_rot")
        else:
            g = self.groups[group]
            idx, lpos, lrot = g.body_index, g.local_pos, g.local_rot
        idx, lpos, lrot = idx.to(body_pos.device), lpos.to(body_pos), lrot.to(body_rot)
        bp = body_pos.index_select(-2, idx)
        br = body_rot.index_select(-3, idx)
        pos = bp + (br @ lpos.unsqueeze(-1)).squeeze(-1)
        rot = br @ lrot
        return pos, rot

    def rest_poses(self, desc: RobotDescription) -> tuple[torch.Tensor, torch.Tensor]:
        """Sensor poses with all joints at zero and the root at the origin."""
        q = torch.zeros(len(desc.joint_names))
        bpos, brot = desc.forward_kinematics(torch.zeros(3), torch.eye(3), q)
        bpos, brot = _reorder_bodies(desc.body_names, self.body_names, bpos, brot)
        return self.world_poses(bpos, brot)

    # ---------------------------------------------------------------- I/O
    def to_dict(self) -> dict[str, Any]:
        return {
            "body_names": list(self.body_names),
            "groups": {
                n: {
                    "kind": g.kind, "body_index": g.body_index, "local_pos": g.local_pos, "local_rot": g.local_rot,
                    "area": g.area, "names": g.names, "joint_names": g.joint_names,
                }
                for n, g in self.groups.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SensorLayout:
        groups = {n: SensorGroup(name=n, **g) for n, g in d["groups"].items()}
        return cls(list(d["body_names"]), groups)

    def save(self, path: str | Path) -> None:
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> SensorLayout:
        return cls.from_dict(torch.load(path, weights_only=False))

    def summary(self) -> str:
        parts = [f"{n}({g.kind}): {len(g)}" for n, g in self.groups.items()]
        return f"SensorLayout[{self.num_sensors} sensors on {len(self.body_names)} bodies | " + ", ".join(parts) + "]"


def _reorder_bodies(src_names, dst_names, pos, rot):
    index = torch.tensor([src_names.index(n) for n in dst_names], dtype=torch.long)
    return pos.index_select(-2, index), rot.index_select(-3, index)


def group_from_placements(name: str, kind: str, placements: list[Placement], body_names: list[str]) -> SensorGroup:
    body_index = torch.cat(
        [torch.full((p.local_pos.shape[0],), body_names.index(p.body), dtype=torch.long) for p in placements]
    )
    joint_names = None
    if any(p.joint_names for p in placements):
        joint_names = [jn for p in placements for jn in (p.joint_names or [None] * p.local_pos.shape[0])]
    return SensorGroup(
        name=name,
        kind=kind,
        body_index=body_index,
        local_pos=torch.cat([p.local_pos for p in placements]).float(),
        local_rot=torch.cat([p.local_rot for p in placements]).float(),
        area=torch.cat([p.area for p in placements]).float(),
        names=[n for p in placements for n in p.names],
        joint_names=joint_names,
    )


def build_layout(desc: RobotDescription, spec: dict[str, Any]) -> SensorLayout:
    """Build a layout from a robot description and a YAML-style spec.

    ``spec`` maps group name -> ``{kind: ..., placements: [{generator: ..., **params}, ...]}``.
    ``kind`` defaults to the group name. Example::

        {"tactile": {"placements": [{"generator": "cylinder", "bodies": "link_.*", "n_rings": 3}]},
         "joint": {"placements": [{"generator": "joints"}]},
         "imu": {"placements": [{"generator": "point", "body": "link_0"}]}}
    """
    layout = SensorLayout(desc.body_names)
    for name, gspec in spec.items():
        kind = gspec.get("kind", name)
        placements: list[Placement] = []
        for pspec in gspec["placements"]:
            pspec = dict(pspec)
            gen = PLACEMENT_GENERATORS.get(pspec.pop("generator"))
            placements.extend(gen(desc, **pspec))
        layout.groups[name] = group_from_placements(name, kind, placements, desc.body_names)
    return layout


# ---------------------------------------------------------------------------- clustering


def farthest_point_sampling(pos: torch.Tensor, k: int, start: int = 0) -> torch.Tensor:
    """Indices of ``k`` farthest-point samples from ``pos [N, 3]`` (deterministic)."""
    n = pos.shape[0]
    k = min(k, n)
    chosen = torch.empty(k, dtype=torch.long)
    chosen[0] = start
    dist = (pos - pos[start]).norm(dim=-1)
    for i in range(1, k):
        chosen[i] = int(dist.argmax())
        dist = torch.minimum(dist, (pos - pos[chosen[i]]).norm(dim=-1))
    return chosen


def compute_clusters(layout: SensorLayout, rest_pos: torch.Tensor, mode: str = "body",
                     num_clusters: int | None = None) -> torch.Tensor:
    """Static cluster assignment ``[N]`` used by cluster-level attention heads.

    * ``body``: one cluster per body that carries sensors (natural for segmented robots).
    * ``fps``: ``num_clusters`` farthest-point centroids on the rest pose, nearest-centroid assignment.
    """
    if mode == "body":
        _, cluster = torch.unique(layout.body_index, return_inverse=True)
        return cluster
    if mode == "fps":
        if not num_clusters:
            raise ValueError("fps clustering needs num_clusters")
        centers = rest_pos[farthest_point_sampling(rest_pos, num_clusters)]
        return torch.cdist(rest_pos, centers).argmin(dim=-1)
    raise ValueError(f"Unknown cluster mode {mode}")


@dataclass
class LayoutInfo:
    """Per-layout tensors the model needs at runtime (moved to the model's device).

    Passing this alongside each batch, instead of baking it into the model, is what lets one set of
    weights run on robots with different sensor counts and geometry.
    """

    group_names: list[str]
    group_kinds: list[str]
    slices: dict[str, slice]
    group_id: torch.Tensor  # [N]
    body_index: torch.Tensor  # [N]
    area: torch.Tensor  # [N]
    rest_pos: torch.Tensor  # [N, 3]
    rest_rot: torch.Tensor  # [N, 3, 3]
    cluster_id: torch.Tensor  # [N]

    @property
    def num_nodes(self) -> int:
        return self.group_id.shape[0]

    @property
    def num_clusters(self) -> int:
        return int(self.cluster_id.max()) + 1

    def to(self, device) -> LayoutInfo:
        return LayoutInfo(
            self.group_names, self.group_kinds, self.slices,
            *(getattr(self, k).to(device) for k in ("group_id", "body_index", "area", "rest_pos", "rest_rot",
                                                     "cluster_id")),
        )

    @classmethod
    def from_layout(cls, layout: SensorLayout, desc: RobotDescription, cluster_mode: str = "body",
                    num_clusters: int | None = None) -> LayoutInfo:
        rest_pos, rest_rot = layout.rest_poses(desc)
        return cls(
            group_names=layout.group_names,
            group_kinds=[g.kind for g in layout.groups.values()],
            slices=layout.slices(),
            group_id=layout.group_id(),
            body_index=layout.body_index,
            area=layout.area,
            rest_pos=rest_pos,
            rest_rot=rest_rot,
            cluster_id=compute_clusters(layout, rest_pos, cluster_mode, num_clusters),
        )
