"""Episode-level quantities derived from stored episodes (e.g. physical targets for property estimation)."""

from __future__ import annotations

from somato.data.dataset import EpisodeStore


def add_measured_sinkage(store: EpisodeStore, key: str = "sinkage_mm") -> None:
    """``ep.params[key]``: mean depth (mm) of the loaded links below the ground over the episode.

    The depth of a link is ``radius - z`` of its frame origin (the capsule axis passes through it; the ground is the
    plane z = 0), averaged over the steps and links flagged ``in_contact``: how far this robot physically sinks into
    the terrain (simulation v1 compliance), the quantity a property estimator should recover.
    """
    radius = max((s.size.get("radius", 0.0) for b in store.desc.bodies for s in b.shapes), default=0.0)
    for ep in store.episodes:
        if key in ep.params:
            continue
        contact = ep.labels["in_contact"].float() > 0.5  # [T, Nb]
        depth = (radius - ep.body_pos[..., 2].float()).clamp_min(0.0)
        ep.params[key] = float(1e3 * (depth * contact).sum() / contact.sum().clamp_min(1))
