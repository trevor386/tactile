"""Collect simulated episodes from any :class:`SimRunner` (mock or Isaac Lab)."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch

from somato.data.episode import DatasetMeta, DatasetWriter, Episode
from somato.robots.description import RobotDescription
from somato.sensors.base import STIMULUS_CHANNELS
from somato.sim.runner import SimRunner


def collect_dataset(
    runner: SimRunner,
    desc: RobotDescription,
    out_dir: str | Path,
    num_rounds: int,
    steps_per_episode: int,
    warmup_steps: int = 25,
    terrain_names: list[str] | None = None,
    generator: torch.Generator | None = None,
    info: dict | None = None,
    log_every: int = 1,
) -> Path:
    """Run ``num_rounds`` rounds of ``num_envs`` parallel episodes and write them to ``out_dir``.

    Terrain classes are assigned round-robin across environments (with a random rotation per round),
    so the dataset stays class-balanced. The first ``warmup_steps`` latent steps (gait ramp-up) are
    simulated but not stored.
    """
    layout = runner.pipeline.layout
    E = runner.backend.num_envs
    catalog_names = terrain_names or [str(i) for i in range(int(runner.backend.terrain.class_id.max()) + 1)]
    n_classes = len(catalog_names)
    meta = DatasetMeta(
        data_kind="stimulus",
        rates=asdict(runner.rates),
        groups={n: {"kind": g.kind, "count": len(g), "channels": list(STIMULUS_CHANNELS[g.kind])}
                for n, g in layout.groups.items()},
        terrain_names=list(catalog_names),
        info=info or {},
    )
    writer = DatasetWriter(out_dir, meta, desc, layout)
    t0 = time.time()
    for r in range(num_rounds):
        shift = int(torch.randint(n_classes, (1,), generator=generator))
        classes = (torch.arange(E) + shift) % n_classes
        runner.reset(terrain_class=classes)
        for _ in range(warmup_steps):
            runner.step_latent()
        frames = [runner.step_latent() for _ in range(steps_per_episode)]
        terrain = runner.backend.terrain
        for e in range(E):
            ep = Episode(
                data={g: torch.stack([f.stimuli[g][e] for f in frames]).cpu() for g in layout.group_names},
                body_pos=torch.stack([f.body_pos[e] for f in frames]).cpu(),
                body_quat=torch.stack([f.body_quat[e] for f in frames]).cpu(),
                labels={"terrain": terrain.class_id[e].cpu(),
                        **{k: torch.stack([f.labels[k][e] for f in frames]).cpu() for k in frames[0].labels}},
                params={k: float(v[e]) for k, v in terrain.params.items()},
            )
            writer.write(ep)
        if log_every and (r + 1) % log_every == 0:
            print(f"[collect] round {r + 1}/{num_rounds}: {writer.count} episodes, {time.time() - t0:.1f}s")
    writer.close()
    return Path(out_dir)
