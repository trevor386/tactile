"""Episode-level quantities derived from stored episodes (e.g. physical targets for property estimation)."""

from __future__ import annotations

import torch

from somato.data.dataset import EpisodeStore
from somato.geometry.rotations import quat_to_matrix, rpy_to_matrix


def capsule_segments(desc) -> tuple[torch.Tensor, torch.Tensor]:
    """Per body, the end points ``[Nb, 2, 3]`` (link frame) of its collision capsule's axis segment, and its radius
    ``[Nb]``. Cylinder shapes count as capsules, as the simulators collide them (``robots/mjcf.py``)."""
    ends, radii = [], []
    for b in desc.bodies:
        s = next((s for s in b.shapes if s.kind in ("capsule", "cylinder")), None)
        if s is None:
            raise ValueError(f"body {b.name} has no capsule or cylinder shape")
        rot = rpy_to_matrix(torch.tensor(s.rpy, dtype=torch.float32))
        half = rot @ torch.tensor([0.0, 0.0, 0.5 * s.size["length"]])  # the capsule axis is the shape's local z
        pos = torch.tensor(s.pos, dtype=torch.float32)
        ends.append(torch.stack([pos - half, pos + half]))
        radii.append(float(s.size["radius"]))
    return torch.stack(ends), torch.tensor(radii)


def add_measured_sinkage(store: EpisodeStore, key: str = "sinkage_mm") -> None:
    """``ep.params[key]``: mean penetration (mm) of the loaded links into the terrain over the episode.

    A link's penetration is how far the lowest point of its collision capsule lies below the ground plane z = 0: the
    radius minus the height of the lower end of the capsule's axis segment, from the stored link poses. It is averaged
    over the steps and links flagged ``in_contact``: how far this robot physically sinks into the terrain (simulation
    v1 compliance), the quantity a property estimator should recover. (Before 2026-10-10 the depth of the link frame
    origin was used, which reads link tilt as sinkage: ~0.8 mm on rigid ground instead of ~0.05 mm; no finding used
    it.)
    """
    ends, radius = capsule_segments(store.desc)
    for ep in store.episodes:
        if key in ep.params:
            continue
        contact = ep.labels["in_contact"].float() > 0.5  # [T, Nb]
        rot_z = quat_to_matrix(ep.body_quat.float())[..., 2, :]  # world z row of each link rotation [T, Nb, 3]
        z = torch.einsum("tbj,bkj->tbk", rot_z, ends) + ep.body_pos[..., 2:3].float()  # segment end heights [T, Nb, 2]
        depth = (radius - z.min(-1).values).clamp_min(0.0)
        ep.params[key] = float(1e3 * (depth * contact).sum() / contact.sum().clamp_min(1))
