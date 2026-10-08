import json

import pytest
import torch

from somato.control import GaitConfig
from somato.data import EpisodeStore, WindowDataset, collect_dataset, stratified_split, stratified_subset
from somato.data.episode import Episode
from somato.geometry import LayoutInfo
from somato.runtime import OnlineEncoder
from somato.sim import RateConfig
from somato.sim.setup import CollectConfig, make_mock_runner
from somato.sources import ReplaySource
from somato.training import AugmentConfig, BatchPreparer, ExperimentConfig, TrainConfig, run_experiment
from somato.training.experiment import build_suite, group_specs, load_trained_model

ROBOT = {"robot": {"type": "snake", "params": {"num_links": 4}},
         "sensors": {"tactile": {"placements": [{"generator": "cylinder", "bodies": r"link_\d+", "n_rings": 2,
                                                 "n_per_ring": 4, "skin": 0.002}]},
                     "joint": {"placements": [{"generator": "joints"}]},
                     "imu": {"placements": [{"generator": "point", "body": "link_0", "name": "imu"}]}}}
TINY_MODEL = {"dim": 16, "temporal": {"latent_dim": 16, "params": {"hidden": 16, "conv_channels": 8}},
              "spatial": {"graph": {"k": 6, "k_per_group": {"tactile": 4, "joint": 2, "imu": 1}},
                          "params": {"num_basis": 4, "kernel_hidden": 16}},
              "heads": {"terrain": {"hidden": 16, "cluster_layers": 1}}}
IDEAL = {"tactile": {"model": "ideal_tactile"}, "joint": {"model": "ideal_joint"}, "imu": {"model": "ideal_imu"}}


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("ds")
    cfg = CollectConfig(robot=ROBOT, out=str(out), num_envs=5, rounds=3, steps_per_episode=12, warmup_steps=2,
                        rates=RateConfig(physics_hz=500, control_hz=100, latent_hz=50,
                                         sensors={"tactile": 250, "joint": 100, "imu": 100}),
                        gait=GaitConfig())
    gen = torch.Generator().manual_seed(0)
    runner, desc, layout, catalog = make_mock_runner(cfg, gen)
    collect_dataset(runner, desc, out, cfg.rounds, cfg.steps_per_episode, cfg.warmup_steps, catalog.names, gen,
                    log_every=0)
    return out


@pytest.fixture(scope="module")
def store(dataset_dir):
    return EpisodeStore(dataset_dir)


def test_dataset_contents(store):
    assert len(store) == 15
    assert store.meta.terrain_names[0] == "glare_ice"
    ep = store.episodes[0]
    assert ep.data["tactile"].shape == (12, 5, 32, 3)
    assert ep.data["joint"].shape == (12, 2, 3, 4)
    assert ep.body_pos.shape == (12, 4, 3)
    assert ep.labels["slip_speed"].shape == (12, 4)
    assert "friction" in ep.params
    assert sorted(store.labels().tolist()) == sorted(list(range(5)) * 3)  # class-balanced


def test_episode_roundtrip(tmp_path, store):
    ep = store.episodes[0]
    ep.save(tmp_path / "e.npz")
    back = Episode.load(tmp_path / "e.npz")
    assert torch.equal(back.data["tactile"], ep.data["tactile"])
    assert int(back.labels["terrain"]) == int(ep.labels["terrain"])


def test_episode_keeps_values_beyond_float16(tmp_path):
    # Isaac Sim produces taxel pressure peaks above the float16 range (65504 Pa); they must not become inf.
    ep = Episode(data={"tactile": torch.tensor([[1.0, 7.0e4]]), "imu": torch.tensor([[0.5, 9.81]])},
                 body_pos=torch.zeros(1, 2, 3), body_quat=torch.zeros(1, 2, 4))
    ep.save(tmp_path / "ep.npz")
    back = Episode.load(tmp_path / "ep.npz")
    assert torch.equal(back.data["tactile"].float(), ep.data["tactile"])
    assert back.data["imu"].dtype == torch.float16


def test_windows_and_splits(store):
    ds = WindowDataset(store, [0, 1], window=5, stride=3)
    assert len(ds) == 2 * 3
    s = ds[0]
    assert s["data/tactile"].shape == (5, 5, 32, 3) and s["label/terrain"].dtype == torch.long
    labels = torch.arange(5).repeat(8)
    a, b, c = stratified_split(labels, (0.5, 0.25, 0.25))
    assert len(set(a.tolist()) & set(b.tolist())) == 0 and len(a) + len(b) + len(c) == 40
    assert torch.bincount(labels[a]).tolist() == [4] * 5
    sub = stratified_subset(a, labels, 0.5)
    assert torch.bincount(labels[sub]).tolist() == [2] * 5


def test_batch_preparer_augmentation(store):
    info = LayoutInfo.from_layout(store.layout, store.desc)
    suite = build_suite(store, IDEAL)
    prep = BatchPreparer(store.layout, info, suite, "cpu", AugmentConfig(sensor_dropout=0.5))
    sample = torch.utils.data.default_collate([WindowDataset(store, [0, 1], 5)[i] for i in range(2)])
    plain, aug = prep(sample, train=False), prep(sample, train=True)
    assert torch.equal(plain.readings["imu"], aug.readings["imu"])  # yaw about gravity keeps IMU valid
    exact = "donot_use_mm_for_euclid_dist"
    d_plain = torch.cdist(plain.pos[0, 0], plain.pos[0, 0], compute_mode=exact)
    d_aug = torch.cdist(aug.pos[0, 0], aug.pos[0, 0], compute_mode=exact)
    assert torch.allclose(d_plain, d_aug, atol=1e-4) and not torch.allclose(plain.pos, aug.pos)
    tac = info.slices["tactile"]
    assert aug.node_mask is not None and not aug.node_mask[:, tac].all() and aug.node_mask[:, tac.stop:].all()


def test_run_experiment_and_reload(tmp_path, dataset_dir, store):
    cfg = ExperimentConfig(name="smoke", dataset=str(dataset_dir), output_dir=str(tmp_path), model=TINY_MODEL,
                           train=TrainConfig(epochs=2, batch_size=4, output_steps=2), window=6, stride=3, eval_stride=6,
                           split=(1 / 3, 1 / 3, 1 / 3))
    result = run_experiment(cfg, store, verbose=False)
    assert 0.0 <= result["test/terrain/acc_last"] <= 1.0
    assert json.loads((tmp_path / "smoke" / "result.json").read_text())["params"] == result["params"]
    model, ckpt = load_trained_model(tmp_path / "smoke" / "model.pt")
    assert ckpt["terrain_names"] == store.meta.terrain_names
    assert model.heads["terrain"].out[-1].out_features == len(store.meta.terrain_names)


@pytest.mark.parametrize("groups", [["tactile"], ["joint", "imu"]])
def test_input_groups_ablation(tmp_path, dataset_dir, store, groups):
    view = store.select_groups(groups)
    assert view.layout.group_names == groups and list(view.meta.groups) == groups
    assert set(view.episodes[0].data) == set(groups) and len(store.layout.groups) == 3  # original untouched
    for name, model in (("hier", TINY_MODEL), ("flat", {"architecture": "flat_recurrent", "hidden": 16})):
        cfg = ExperimentConfig(name=name, dataset=str(dataset_dir), output_dir=str(tmp_path), model=model, sensors=IDEAL,
                               train=TrainConfig(epochs=1, batch_size=4, output_steps=2), window=6, stride=6,
                               split=(1 / 3, 1 / 3, 1 / 3), input_groups=groups, save_checkpoint=False)
        assert 0.0 <= run_experiment(cfg, store, verbose=False)["test/terrain/acc_last"] <= 1.0


def test_flat_baseline_experiment(tmp_path, dataset_dir, store):
    cfg = ExperimentConfig(name="flat", dataset=str(dataset_dir), output_dir=str(tmp_path), save_checkpoint=False,
                           model={"architecture": "flat_recurrent", "hidden": 16},
                           train=TrainConfig(epochs=1, batch_size=4, output_steps=2), window=6, stride=6,
                           split=(1 / 3, 1 / 3, 1 / 3))
    assert "test/loss" in run_experiment(cfg, store, verbose=False)


def test_online_encoder_matches_offline(store):
    torch.manual_seed(0)
    from somato.models import build_model

    suite = build_suite(store, IDEAL)
    model = build_model(TINY_MODEL, group_specs(store, suite)).eval()
    info = LayoutInfo.from_layout(store.layout, store.desc)
    online = OnlineEncoder(model, store.layout, info, suite)
    source = ReplaySource(store.episodes[:2], store.layout)
    outs = []
    while (frame := source.read()) is not None:
        outs.append(online.step(frame)["terrain"])
    prep = BatchPreparer(store.layout, info, suite, "cpu")
    sample = torch.utils.data.default_collate([WindowDataset(store, [i], 12)[0] for i in range(2)])
    offline, _ = model(prep(sample))
    assert torch.allclose(torch.stack(outs, 1), offline["terrain"], atol=1e-4)
    assert len(online.latency_ms) == 12


def test_online_encoder_sliding_window(store):
    torch.manual_seed(0)
    from somato.models import build_model

    suite = build_suite(store, IDEAL)
    model = build_model(TINY_MODEL, group_specs(store, suite)).eval()
    info = LayoutInfo.from_layout(store.layout, store.desc)
    W = 5
    online = OnlineEncoder(model, store.layout, info, suite, window=W)
    source = ReplaySource(store.episodes[:2], store.layout)
    outs = []
    while (frame := source.read()) is not None:
        outs.append(online.step(frame)["terrain"])
    prep = BatchPreparer(store.layout, info, suite, "cpu")
    sample = torch.utils.data.default_collate([WindowDataset(store, [i], 12)[0] for i in range(2)])
    full = prep(sample)  # sensor state persists over the whole episode, as online
    with torch.no_grad():
        for t in range(12):
            offline, _ = model(full.steps(max(0, t + 1 - W), t + 1), output_steps=1)
            assert torch.allclose(outs[t], offline["terrain"][:, -1], atol=1e-4)


def test_truncated_bptt_experiment(tmp_path, dataset_dir, store):
    # 12-step sequences in 4-step chunks: the backward pass of the 2nd and 3rd chunk only works if the carried
    # state is detached at chunk boundaries.
    cfg = ExperimentConfig(name="tbptt", dataset=str(dataset_dir), output_dir=str(tmp_path), model=TINY_MODEL,
                           sensors=IDEAL, train=TrainConfig(epochs=2, batch_size=4, output_steps=2), window=4,
                           stride=4, train_sequence=12, split=(1 / 3, 1 / 3, 1 / 3))
    result = run_experiment(cfg, store, verbose=False)
    assert 0.0 <= result["test/terrain/acc_last"] <= 1.0
    log = [json.loads(line) for line in (tmp_path / "tbptt" / "log.jsonl").read_text().splitlines()]
    assert len(log) == 2 and all("train/terrain/loss" in r for r in log)
    _, ckpt = load_trained_model(tmp_path / "tbptt" / "model.pt")
    assert ckpt["experiment"]["train"]["tbptt_chunk"] == 4 and ckpt["experiment"]["train_sequence"] == 12


def test_data_efficiency_study(tmp_path, dataset_dir):
    from somato.training.data_efficiency import run_data_efficiency
    from somato.utils.config import save_yaml

    save_yaml({"name": "base", "dataset": str(dataset_dir), "sensors": IDEAL, "window": 6, "stride": 6,
               "eval_stride": 6, "split": [1 / 3, 1 / 3, 1 / 3],
               "train": {"epochs": 1, "batch_size": 4, "output_steps": 2}}, tmp_path / "base.yaml")
    save_yaml({"base": str(tmp_path / "base.yaml"), "output_dir": str(tmp_path / "out"), "fractions": [0.5, 1.0],
               "seeds": [0], "match_params": True, "reference_model": "hier",
               "models": {"hier": TINY_MODEL, "flat": {"architecture": "flat_recurrent", "hidden": 8}}},
              tmp_path / "study.yaml")
    results = run_data_efficiency(tmp_path / "study.yaml")
    assert len(results) == 4
    params = {r["model"]: r["params"] for r in results}
    assert abs(params["flat"] - params["hier"]) / params["hier"] < 0.15
    assert (tmp_path / "out" / "summary.md").exists() and (tmp_path / "out" / "results.csv").exists()
