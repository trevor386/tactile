"""Windowed datasets over stored episodes, plus splitting utilities for data-efficiency studies."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from somato.data.episode import DatasetMeta, Episode, episode_paths, load_robot, read_meta


class EpisodeStore:
    """All episodes of a dataset directory, loaded into memory (float16)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta: DatasetMeta = read_meta(root)
        self.desc, self.layout = load_robot(root)
        self.episodes: list[Episode] = [Episode.load(p) for p in episode_paths(root)]
        if not self.episodes:
            raise ValueError(f"No episodes found in {root}")

    def __len__(self) -> int:
        return len(self.episodes)

    def labels(self, key: str = "terrain") -> torch.Tensor:
        return torch.stack([ep.labels[key].long() for ep in self.episodes])


class WindowDataset(Dataset):
    """Fixed-length windows of ``window`` latent steps from a subset of episodes.

    Returns dicts with ``data/<group> [L, S, N, C]`` (float32), ``body_pos [L, Nb, 3]``,
    ``body_quat [L, Nb, 4]``, scalar labels and per-step labels sliced to the window.
    """

    def __init__(self, store: EpisodeStore, episode_ids: list[int] | torch.Tensor, window: int, stride: int | None = None,
                 random_offset: bool = False, generator: torch.Generator | None = None):
        self.store, self.window = store, window
        self.random_offset, self.generator = random_offset, generator
        stride = stride or window
        self.index: list[tuple[int, int]] = []
        for e in [int(i) for i in episode_ids]:
            T = store.episodes[e].num_steps
            if T < window:
                continue
            for start in range(0, T - window + 1, stride):
                self.index.append((e, start))
        self.stride = stride

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        e, start = self.index[i]
        ep = self.store.episodes[e]
        if self.random_offset:
            jitter = int(torch.randint(self.stride, (1,), generator=self.generator))
            start = min(start + jitter, ep.num_steps - self.window)
        sl = slice(start, start + self.window)
        out = {f"data/{g}": v[sl].float() for g, v in ep.data.items()}
        out["body_pos"] = ep.body_pos[sl].float()
        out["body_quat"] = ep.body_quat[sl].float()
        for k, v in ep.labels.items():
            out[f"label/{k}"] = v[sl].float() if v.dim() > 0 else v.long()
        for k, v in ep.params.items():
            out[f"param/{k}"] = torch.tensor(v, dtype=torch.float32)
        return out


def stratified_split(labels: torch.Tensor, fractions: tuple[float, ...] = (0.7, 0.15, 0.15), seed: int = 0
                     ) -> list[torch.Tensor]:
    """Split episode indices into parts with (approximately) equal class proportions."""
    gen = torch.Generator().manual_seed(seed)
    parts: list[list[int]] = [[] for _ in fractions]
    for c in labels.unique():
        idx = (labels == c).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=gen)]
        bounds = torch.tensor([0.0, *torch.tensor(fractions).cumsum(0).tolist()]) * len(idx)
        bounds = bounds.round().long()
        for p in range(len(fractions)):
            parts[p].extend(idx[bounds[p]:bounds[p + 1]].tolist())
    return [torch.tensor(sorted(p), dtype=torch.long) for p in parts]


def stratified_subset(indices: torch.Tensor, labels: torch.Tensor, fraction: float, seed: int = 0,
                      min_per_class: int = 1) -> torch.Tensor:
    """A class-balanced random ``fraction`` of ``indices`` (for learning curves)."""
    gen = torch.Generator().manual_seed(seed)
    chosen = []
    sub_labels = labels[indices]
    for c in sub_labels.unique():
        idx = indices[sub_labels == c]
        k = max(min_per_class, int(round(fraction * len(idx))))
        chosen.append(idx[torch.randperm(len(idx), generator=gen)[:k]])
    return torch.cat(chosen).sort().values
