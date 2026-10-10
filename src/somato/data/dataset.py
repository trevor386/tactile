"""Windowed datasets over stored episodes, plus splitting utilities for data-efficiency studies."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from somato.data.episode import DatasetMeta, Episode, episode_paths, load_robot, read_meta
from somato.geometry.layout import SensorLayout

CACHE_VERSION = 1  # bump when the cache layout changes (it is part of the default directory name)

_Plan = dict[str, tuple[tuple[int, ...], np.dtype]]  # array key -> (shape of one episode's array, dtype)


def _npz_specs(path: Path) -> _Plan:
    """``{key: (shape, dtype)}`` of every array of an episode file, from the .npy headers only (no decompression)."""
    specs: _Plan = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as fh:
                read_header = (np.lib.format.read_array_header_1_0 if np.lib.format.read_magic(fh) == (1, 0)
                               else np.lib.format.read_array_header_2_0)
                shape, _, dtype = read_header(fh)
            specs[name.removesuffix(".npy")] = (shape, dtype)
    return specs


def _fingerprint(paths: list[Path]) -> list[dict]:
    """Name, size and mtime of every episode file: the cache is valid only while this is unchanged."""
    stats = [p.stat() for p in paths]
    return [{"name": p.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns} for p, st in zip(paths, stats)]


def _consolidation_plan(paths: list[Path]) -> _Plan | None:
    """Shape and (widest) dtype of each array, or ``None`` (with a note) if the episodes differ in keys or shapes."""
    plan = _npz_specs(paths[0])
    for p in paths[1:]:
        specs = _npz_specs(p)
        if specs.keys() != plan.keys():
            print(f"[dataset] {p.name} has different arrays than {paths[0].name}: loading episodes into memory")
            return None
        for key, (shape, dtype) in specs.items():
            if shape != plan[key][0]:
                print(f"[dataset] '{key}' has shape {shape} in {p.name} but {plan[key][0]} in {paths[0].name}: "
                      "loading episodes into memory")
                return None
            plan[key] = (shape, np.promote_types(plan[key][1], dtype))  # float16 + float32 episodes -> float32
    return plan


def _build_cache(paths: list[Path], plan: _Plan, fingerprint: list[dict], cache_dir: Path) -> None:
    """Stream the episodes one at a time into ``<key>.npy`` files ``[E, ...]`` and publish them atomically.

    The files are written in a temporary sibling directory that is renamed to ``cache_dir`` only when
    complete, so an interrupted build never leaves a half-written cache.
    """
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{cache_dir.name}.tmp-", dir=cache_dir.parent))
    try:
        tmp.chmod(cache_dir.parent.stat().st_mode & 0o777)  # mkdtemp makes it private; share like the dataset
        arrays = {key: {"file": key.replace("/", "__") + ".npy", "dtype": dtype.str, "shape": [len(paths), *shape]}
                  for key, (shape, dtype) in plan.items()}
        with contextlib.ExitStack() as stack:
            files, offsets = {}, {}
            for key, spec in arrays.items():
                # open_memmap lays out the .npy header; the rows are then filled with plain writes (not through
                # the mapping), so the build never holds mapped pages and its memory stays flat
                mm = np.lib.format.open_memmap(tmp / spec["file"], mode="w+", dtype=plan[key][1],
                                               shape=tuple(spec["shape"]))
                offsets[key] = mm.offset
                del mm
                files[key] = stack.enter_context(open(tmp / spec["file"], "r+b"))
            for i, path in enumerate(paths):
                with np.load(path) as f:
                    for key, fh in files.items():
                        row = f[key]
                        if row.shape != plan[key][0]:
                            raise ValueError(f"{path.name} changed while the cache was being built")
                        row = np.ascontiguousarray(row, dtype=plan[key][1])
                        fh.seek(offsets[key] + i * row.nbytes)
                        fh.write(row.data)
            for fh in files.values():
                fh.flush()
                os.fsync(fh.fileno())
        (tmp / "manifest.json").write_text(json.dumps({"version": CACHE_VERSION, "episodes": fingerprint,
                                                       "arrays": arrays}))
        stale = None
        if cache_dir.exists():  # outdated cache: move it aside first (rename cannot replace a non-empty directory)
            stale = cache_dir.with_name(f"{cache_dir.name}.stale-{os.getpid()}")
            os.rename(cache_dir, stale)
        try:
            os.rename(tmp, cache_dir)
        finally:
            if stale is not None:
                shutil.rmtree(stale, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # no-op after a successful rename


def _open_cache(cache_dir: Path, fingerprint: list[dict]) -> dict[str, np.ndarray] | None:
    """The cache arrays, memory-mapped copy-on-write, or ``None`` if missing, stale or damaged."""
    try:
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        if manifest["version"] != CACHE_VERSION or manifest["episodes"] != fingerprint:
            return None
        arrays = {}
        for key, spec in manifest["arrays"].items():
            arrays[key] = np.load(cache_dir / spec["file"], mmap_mode="c")  # "c": private pages, writable tensors
            if list(arrays[key].shape) != spec["shape"] or arrays[key].dtype != np.dtype(spec["dtype"]):
                return None
        return arrays
    except (OSError, ValueError, KeyError):
        return None


def _cached_episodes(paths: list[Path], cache_dir: Path) -> list[Episode] | None:
    """Episodes backed by the consolidated cache (built on first use), or ``None`` to load them one by one."""
    fingerprint = _fingerprint(paths)
    arrays = _open_cache(cache_dir, fingerprint)
    if arrays is None:
        plan = _consolidation_plan(paths)
        if plan is None:
            return None
        print(f"[dataset] building cache {cache_dir} ({len(paths)} episodes)")
        try:
            _build_cache(paths, plan, fingerprint, cache_dir)
        except (OSError, ValueError) as e:  # read-only or full disk, or another process published a cache first
            print(f"[dataset] cannot write cache {cache_dir} ({e})")
        arrays = _open_cache(cache_dir, fingerprint)
        if arrays is None:
            print("[dataset] no usable cache: loading episodes into memory")
            return None
    tensors = {key: torch.from_numpy(a) for key, a in arrays.items()}  # views of the mapping, no copies
    episodes = []
    for i in range(len(paths)):
        ep = Episode(data={}, body_pos=tensors["body_pos"][i], body_quat=tensors["body_quat"][i])
        for key, t in tensors.items():
            kind, _, name = key.partition("/")
            if kind == "data":
                ep.data[name] = t[i]
            elif kind == "label":
                ep.labels[name] = t[i]
            elif kind == "param":
                ep.params[name] = float(t[i])
        episodes.append(ep)
    return episodes


class EpisodeStore:
    """All episodes of a dataset directory.

    With ``cache=True`` the episodes are consolidated once into uncompressed ``.npy`` files (one per array,
    ``[E, ...]``) in ``cache_dir`` (default ``root/cache_v1``) and memory-mapped copy-on-write. Episode tensors are
    views of the mapping: nothing is decompressed or copied, resident memory is the shared, reclaimable page cache,
    and several processes can use one dataset. The cache is rebuilt when the set, size or mtime of the episode files
    changes; it is skipped (episodes loaded into RAM instead) if the episodes differ in shape, or the cache cannot
    be written. Use one ``cache_dir`` per dataset, e.g. when ``root`` is read-only.
    """

    def __init__(self, root: str | Path, cache: bool = True, cache_dir: str | Path | None = None):
        self.root = Path(root)
        self.meta: DatasetMeta = read_meta(root)
        self.desc, self.layout = load_robot(root)
        paths = episode_paths(root)
        if not paths:
            raise ValueError(f"No episodes found in {root}")
        episodes = None
        if cache:
            episodes = _cached_episodes(paths, Path(cache_dir) if cache_dir is not None
                                        else self.root / f"cache_v{CACHE_VERSION}")
        self.episodes: list[Episode] = episodes if episodes is not None else [Episode.load(p) for p in paths]

    def __len__(self) -> int:
        return len(self.episodes)

    def labels(self, key: str = "terrain") -> torch.Tensor:
        return torch.stack([ep.labels[key].long() for ep in self.episodes])

    def select_groups(self, names: list[str]) -> EpisodeStore:
        """A view with only the sensor groups ``names`` (modality ablations); episodes share tensors."""
        missing = [n for n in names if n not in self.layout.groups]
        if missing:
            raise ValueError(f"Unknown sensor groups {missing} (dataset has {self.layout.group_names})")
        keep = [n for n in self.layout.group_names if n in names]  # layout order fixes the node order
        view = copy.copy(self)
        view.meta = replace(self.meta, groups={n: self.meta.groups[n] for n in keep})
        view.layout = SensorLayout(self.layout.body_names, {n: self.layout.groups[n] for n in keep})
        view.episodes = [replace(ep, data={n: ep.data[n] for n in keep}) for ep in self.episodes]
        return view


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


def param_holdout(values: torch.Tensor, labels: torch.Tensor, frac: float, side: str = "high") -> torch.Tensor:
    """Within every class, the ``frac`` of episodes with the highest (``side="high"``) or lowest parameter ``values``:
    a held-out part of each class's parameter range (E-9)."""
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")
    held = []
    for c in labels.unique():
        idx = (labels == c).nonzero().flatten()
        order = idx[torch.argsort(values[idx], descending=side == "high", stable=True)]
        held.extend(order[: round(frac * len(idx))].tolist())
    return torch.tensor(sorted(held), dtype=torch.long)


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
