"""The memory-mapped episode cache (``EpisodeStore``): same tensors and windows as per-episode loading."""

import json
import os
import warnings

import numpy as np
import pytest
import torch

from somato.data import DatasetMeta, DatasetWriter, EpisodeStore, WindowDataset
from somato.data.episode import Episode, episode_paths

CACHE = "cache_v1"
STEPS = 8


def make_episode(layout, seed: int, steps: int = STEPS, overflow: bool = False) -> Episode:
    """A random episode; ``overflow`` pushes the tactile values past float16 so ``Episode.save`` stores float32."""
    gen = torch.Generator().manual_seed(seed)
    nb, n = len(layout.body_names), {g: len(layout.groups[g]) for g in layout.group_names}
    tactile = torch.randn(steps, 2, n["tactile"], 3, generator=gen)
    if overflow:
        tactile[0, 0, 0, 0] = 7.0e4
    return Episode(
        data={"tactile": tactile, "joint": torch.randn(steps, 2, n["joint"], 4, generator=gen),
              "imu": torch.randn(steps, 2, n["imu"], 6, generator=gen)},
        body_pos=torch.randn(steps, nb, 3, generator=gen), body_quat=torch.randn(steps, nb, 4, generator=gen),
        labels={"terrain": torch.tensor(seed % 3), "slip_speed": torch.rand(steps, nb, generator=gen)},
        params={"friction": 0.1 * seed, "sinkage": 1.0 / (seed + 1)})


def write_dataset(root, desc, layout, episodes) -> None:
    meta = DatasetMeta(data_kind="stimulus", rates={}, terrain_names=["a", "b", "c"],
                       groups={n: {"kind": n, "count": len(g), "channels": []} for n, g in layout.groups.items()})
    writer = DatasetWriter(root, meta, desc, layout)
    for ep in episodes:
        writer.write(ep)
    writer.close()


@pytest.fixture
def ds_dir(tmp_path, small_desc, small_layout):
    """Six episodes; episode 3 has float32 tactile data (the others float16)."""
    root = tmp_path / "ds"
    write_dataset(root, small_desc, small_layout, [make_episode(small_layout, s, overflow=s == 3) for s in range(6)])
    return root


def assert_same_episodes(a: EpisodeStore, b: EpisodeStore, consolidated: bool = True) -> None:
    """Identical values for every key. ``a`` is consolidated, so each array has the widest dtype any episode of
    ``b`` has; with ``consolidated=False`` it is loaded per episode like ``b`` and the dtypes match exactly."""
    assert len(a) == len(b)
    for ea, eb in zip(a.episodes, b.episodes):
        assert list(ea.data) == list(eb.data) and list(ea.labels) == list(eb.labels)
        assert ea.params == eb.params
        assert ea.num_steps == eb.num_steps
    pairs = {"body_pos": ([e.body_pos for e in a.episodes], [e.body_pos for e in b.episodes]),
             "body_quat": ([e.body_quat for e in a.episodes], [e.body_quat for e in b.episodes])}
    for g in b.episodes[0].data:
        pairs[f"data/{g}"] = ([e.data[g] for e in a.episodes], [e.data[g] for e in b.episodes])
    for k in b.episodes[0].labels:
        pairs[f"label/{k}"] = ([e.labels[k] for e in a.episodes], [e.labels[k] for e in b.episodes])
    for key, (ta, tb) in pairs.items():
        dtype = tb[0].dtype
        for t in tb[1:]:
            dtype = torch.promote_types(dtype, t.dtype)
        for x, y in zip(ta, tb):
            want = dtype if consolidated else y.dtype
            assert x.dtype == want and x.shape == y.shape and torch.equal(x, y.to(want)), key


def test_cache_is_created_and_reused(ds_dir, small_layout):
    nb, n_tactile = len(small_layout.body_names), len(small_layout.groups["tactile"])
    assert not (ds_dir / CACHE).exists()
    store = EpisodeStore(ds_dir)
    cache = ds_dir / CACHE
    manifest = json.loads((cache / "manifest.json").read_text())
    assert [e["name"] for e in manifest["episodes"]] == [p.name for p in episode_paths(ds_dir)]
    assert (cache / "data__tactile.npy").exists() and (cache / "body_pos.npy").exists()
    assert np.load(cache / "data__tactile.npy", mmap_mode="r").shape == (6, STEPS, 2, n_tactile, 3)
    assert np.load(cache / "label__terrain.npy", mmap_mode="r").shape == (6,)
    assert np.load(cache / "label__slip_speed.npy", mmap_mode="r").shape == (6, STEPS, nb)
    assert np.load(cache / "param__friction.npy", mmap_mode="r").shape == (6,)
    assert [p.name for p in ds_dir.iterdir() if p.name.startswith(CACHE)] == [CACHE]  # no temporary leftovers
    assert cache.stat().st_mode & 0o777 == ds_dir.stat().st_mode & 0o777  # not mkdtemp's private 0700

    before = {p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in cache.iterdir()}
    again = EpisodeStore(ds_dir)  # warm: nothing is rewritten
    assert {p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in cache.iterdir()} == before
    assert_same_episodes(again, store)


def test_cache_matches_per_episode_loader(ds_dir):
    cached, plain = EpisodeStore(ds_dir), EpisodeStore(ds_dir, cache=False)
    assert_same_episodes(cached, plain)
    assert [ep.data["tactile"].dtype for ep in plain.episodes].count(torch.float32) == 1  # mixed on disk ...
    assert {ep.data["tactile"].dtype for ep in cached.episodes} == {torch.float32}  # ... widest in the cache
    assert {ep.data["joint"].dtype for ep in cached.episodes} == {torch.float16}
    assert {ep.labels["terrain"].dtype for ep in cached.episodes} == {torch.int64}
    assert torch.equal(cached.labels(), plain.labels())
    assert cached.episodes[0].labels["terrain"].dim() == 0 and isinstance(cached.episodes[0].params["friction"], float)


def test_cached_tensors_are_writable_and_copy_on_write(ds_dir):
    EpisodeStore(ds_dir)  # build
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*not writable.*")  # torch warns about read-only numpy arrays
        store = EpisodeStore(ds_dir)
    original = store.episodes[1].body_pos.clone()
    store.episodes[1].body_pos.add_(1.0)
    store.episodes[1].data["joint"].zero_()
    fresh = EpisodeStore(ds_dir)
    assert torch.equal(fresh.episodes[1].body_pos, original) and fresh.episodes[1].data["joint"].abs().sum() > 0


@pytest.mark.parametrize("kwargs", [{}, {"random_offset": True, "stride": 3}])
def test_window_samples_identical_with_and_without_cache(ds_dir, small_layout, kwargs):
    cached, plain = EpisodeStore(ds_dir), EpisodeStore(ds_dir, cache=False)
    ds_c = WindowDataset(cached, range(6), window=4, generator=torch.Generator().manual_seed(1), **kwargs)
    ds_p = WindowDataset(plain, range(6), window=4, generator=torch.Generator().manual_seed(1), **kwargs)
    assert len(ds_c) == len(ds_p) > 6
    for i in range(len(ds_c)):
        a, b = ds_c[i], ds_p[i]
        assert a.keys() == b.keys() and "label/slip_speed" in a and "param/friction" in a
        for k in a:
            assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k
    batch = torch.utils.data.default_collate([ds_c[0], ds_c[1]])
    assert batch["data/tactile"].shape == (2, 4, 2, len(small_layout.groups["tactile"]), 3)
    assert batch["label/terrain"].dtype == torch.long


def test_changed_episode_set_rebuilds(ds_dir, small_desc, small_layout):
    EpisodeStore(ds_dir)
    manifest = (ds_dir / CACHE / "manifest.json").read_text()

    write_dataset(ds_dir, small_desc, small_layout, [make_episode(small_layout, 10)])  # append a 7th episode
    grown = EpisodeStore(ds_dir)
    assert len(grown) == 7 and len(json.loads((ds_dir / CACHE / "manifest.json").read_text())["episodes"]) == 7
    assert_same_episodes(grown, EpisodeStore(ds_dir, cache=False))

    path = episode_paths(ds_dir)[2]  # overwrite an episode in place (mtime bumped, in case the clock is coarse)
    make_episode(small_layout, 20).save(path)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    changed = EpisodeStore(ds_dir)
    assert_same_episodes(changed, EpisodeStore(ds_dir, cache=False))
    assert torch.equal(changed.episodes[2].body_pos, make_episode(small_layout, 20).body_pos)

    episode_paths(ds_dir)[-1].unlink()  # remove an episode
    shrunk = EpisodeStore(ds_dir)
    assert len(shrunk) == 6 and (ds_dir / CACHE / "manifest.json").read_text() != manifest
    assert [p.name for p in ds_dir.iterdir() if p.name.startswith(CACHE)] == [CACHE]  # stale/temporary dirs removed


def test_damaged_cache_is_rebuilt(ds_dir):
    store = EpisodeStore(ds_dir)
    del store
    target = ds_dir / CACHE / "data__joint.npy"
    target.write_bytes(target.read_bytes()[:100])  # truncated
    assert_same_episodes(EpisodeStore(ds_dir), EpisodeStore(ds_dir, cache=False))
    assert target.stat().st_size > 100


def test_select_groups_with_cache(ds_dir):
    cached, plain = EpisodeStore(ds_dir), EpisodeStore(ds_dir, cache=False)
    view, ref = cached.select_groups(["imu", "tactile"]), plain.select_groups(["imu", "tactile"])
    assert view.layout.group_names == ref.layout.group_names == ["tactile", "imu"]
    assert all(list(ep.data) == ["tactile", "imu"] for ep in view.episodes)
    assert set(cached.episodes[0].data) == {"tactile", "joint", "imu"}  # original untouched
    assert_same_episodes(view, ref)
    sample = WindowDataset(view, [0, 3], window=4)[1]
    assert set(sample) >= {"data/tactile", "data/imu"} and "data/joint" not in sample
    with pytest.raises(ValueError):
        cached.select_groups(["nope"])


def test_cache_can_be_disabled_or_relocated(tmp_path, ds_dir):
    EpisodeStore(ds_dir, cache=False)
    assert not (ds_dir / CACHE).exists()
    elsewhere = tmp_path / "scratch" / "mycache"
    store = EpisodeStore(ds_dir, cache_dir=elsewhere)
    assert (elsewhere / "manifest.json").exists() and not (ds_dir / CACHE).exists()
    assert_same_episodes(store, EpisodeStore(ds_dir, cache=False))


def test_unwritable_cache_dir_falls_back(tmp_path, ds_dir, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")  # a cache directory below a regular file can never be created
    store = EpisodeStore(ds_dir, cache_dir=blocker / "cache")
    assert "cannot write cache" in capsys.readouterr().out
    assert_same_episodes(store, EpisodeStore(ds_dir, cache=False), consolidated=False)
    assert not any(p.name.startswith("cache") for p in tmp_path.iterdir())


def test_episodes_of_different_length_are_not_consolidated(tmp_path, small_desc, small_layout, capsys):
    root = tmp_path / "ragged"
    write_dataset(root, small_desc, small_layout,
                  [make_episode(small_layout, 0), make_episode(small_layout, 1, steps=STEPS + 3)])
    store = EpisodeStore(root)
    assert "loading episodes into memory" in capsys.readouterr().out
    assert not (root / CACHE).exists()
    assert [ep.num_steps for ep in store.episodes] == [STEPS, STEPS + 3]
