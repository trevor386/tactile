"""Config-driven experiments: dataset -> sensor suite -> model -> train -> test metrics."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from somato.data.dataset import EpisodeStore, WindowDataset, stratified_split, stratified_subset
from somato.data.derived import add_measured_sinkage
from somato.geometry.layout import LayoutInfo
from somato.models.batch import GroupSpec
from somato.models.baselines import count_parameters
from somato.models.builder import build_model, parse_model_config
from somato.sensors.suite import SensorSuite
from somato.sim.runner import RateConfig
from somato.training.prepare import AugmentConfig, BatchPreparer
from somato.training.tasks import ClassificationTask, PropertyRegressionTask, build_tasks
from somato.training.trainer import Trainer, TrainConfig, resolve_device, save_checkpoint
from somato.utils.config import deep_merge, from_dict, load_yaml, to_dict
from somato.utils.seeding import seed_everything


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    dataset: str = "datasets/mock_terrain"
    output_dir: str = "runs"
    sensors: dict[str, Any] = field(default_factory=lambda: {
        "tactile": {"model": "fsr"}, "joint": {"model": "motor"}, "imu": {"model": "mems_imu"}})
    model: dict[str, Any] = field(default_factory=dict)
    tasks: dict[str, Any] = field(default_factory=lambda: {"terrain": {"type": "classification", "label": "terrain"}})
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    window: int = 25  # latent steps per training window
    # > 0: train on crops of this many latent steps with truncated BPTT in ``window``-step chunks (the recurrent
    # state is carried across chunks, as in streaming deployment); validation and test then stream through
    # sequences of the same length and average the metrics over all chunk ends. 0 = independent windows.
    train_sequence: int = 0
    stride: int = 10
    eval_stride: int = 25
    split: tuple[float, float, float] = (0.7, 0.15, 0.15)
    split_seed: int = 0
    train_fraction: float = 1.0  # fraction of training episodes used (learning curves)
    input_groups: list[str] = field(default_factory=list)  # sensor groups fed to the model (empty = all)
    save_checkpoint: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
        d = load_yaml(path)
        if overrides:
            # A nested override of a section given as a YAML path (e.g. ``model.heads.terrain.pool``) merges
            # into that file's contents instead of replacing the path.
            for k, v in overrides.items():
                if isinstance(v, dict) and isinstance(d.get(k), str):
                    d[k] = load_yaml(d[k])
            d = deep_update(d, overrides)
        return from_dict(cls, d)


def deep_update(base: dict, upd: dict) -> dict:
    return deep_merge(copy.deepcopy(base), upd)


def group_specs(store: EpisodeStore, suite: SensorSuite | None) -> dict[str, GroupSpec]:
    rates = RateConfig(**store.meta.rates)
    specs = {}
    for name, g in store.meta.groups.items():
        channels = suite.models[name].num_outputs if suite is not None else len(g["channels"])
        specs[name] = GroupSpec(g["kind"], channels, rates.substeps(name))
    return specs


def build_suite(store: EpisodeStore, sensors_cfg: dict[str, Any]) -> SensorSuite | None:
    if store.meta.data_kind != "stimulus":
        return None
    return SensorSuite.from_config(sensors_cfg, store.layout, RateConfig(**store.meta.rates).sensors)


def resolve_config_refs(cfg: ExperimentConfig) -> ExperimentConfig:
    """Load ``sensors`` / ``model`` given as YAML paths."""
    cfg = copy.deepcopy(cfg)
    if isinstance(cfg.sensors, (str, Path)):
        cfg.sensors = load_yaml(cfg.sensors)
    if isinstance(cfg.model, (str, Path)):
        cfg.model = load_yaml(cfg.model)
    return cfg


def prepare_store_for_tasks(store: EpisodeStore, tasks: list) -> None:
    """Add the derived episode labels the tasks need (e.g. measured sinkage for property regression)."""
    if any(isinstance(t, PropertyRegressionTask) and "sinkage_mm" in t.targets for t in tasks):
        add_measured_sinkage(store)


def ensure_task_heads(model_cfg, tasks: list, num_classes: int) -> None:
    """Give every task a head of the right size: classification heads match the dataset's class count, and a task
    whose head the model config lacks gets a copy of the model's first head (same type and width)."""
    template = next(iter(model_cfg.heads.values()))
    for t in tasks:
        if isinstance(t, ClassificationTask):
            out_dim = num_classes
        elif isinstance(t, PropertyRegressionTask):
            out_dim = t.out_dim
        else:
            continue
        if t.head in model_cfg.heads:
            model_cfg.heads[t.head].out_dim = out_dim
        else:
            model_cfg.heads[t.head] = replace(template, out_dim=out_dim)


def run_experiment(cfg: ExperimentConfig, store: EpisodeStore | None = None, verbose: bool = True) -> dict[str, Any]:
    cfg = resolve_config_refs(cfg)
    seed_everything(cfg.train.seed)
    store = store or EpisodeStore(cfg.dataset)
    if cfg.input_groups:
        store = store.select_groups(cfg.input_groups)
    labels = store.labels("terrain")
    train_ids, val_ids, test_ids = stratified_split(labels, tuple(cfg.split), cfg.split_seed)
    if cfg.train_fraction < 1.0:
        train_ids = stratified_subset(train_ids, labels, cfg.train_fraction, seed=cfg.train.seed)
    gen = torch.Generator().manual_seed(cfg.train.seed)
    if cfg.train_sequence > 0:
        cfg.train.tbptt_chunk = cfg.window
        seq = cfg.train_sequence
        train_ds = WindowDataset(store, train_ids, seq, cfg.stride, random_offset=True, generator=gen)
        val_ds = WindowDataset(store, val_ids, seq, seq)
        test_ds = WindowDataset(store, test_ids, seq, seq)
    else:
        train_ds = WindowDataset(store, train_ids, cfg.window, cfg.stride, random_offset=True, generator=gen)
        val_ds = WindowDataset(store, val_ids, cfg.window, cfg.eval_stride)
        test_ds = WindowDataset(store, test_ids, cfg.window, cfg.eval_stride)
    for split_name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        if len(ds) == 0:
            raise ValueError(f"The {split_name} split has no windows (episodes per class too few for split "
                             f"{tuple(cfg.split)}, or window {cfg.window} longer than the episodes)")

    suite = build_suite(store, cfg.sensors)
    groups = group_specs(store, suite)
    model_cfg = parse_model_config(copy.deepcopy(cfg.model) if isinstance(cfg.model, dict) else cfg.model)
    task_list = build_tasks(cfg.tasks)
    prepare_store_for_tasks(store, task_list)
    for t in task_list:  # standardize regression targets on the training episodes; keep the stats for evaluation
        if isinstance(t, PropertyRegressionTask):
            t.fit(torch.tensor([[store.episodes[int(e)].params[k] for k in t.targets] for e in train_ids]))
            cfg.tasks[t.name] = {**cfg.tasks[t.name], "stats": t.stats}
    num_classes = len(store.meta.terrain_names)
    ensure_task_heads(model_cfg, task_list, num_classes)
    info = LayoutInfo.from_layout(store.layout, store.desc, getattr(model_cfg, "cluster_mode", "body"),
                                  getattr(model_cfg, "num_clusters", None))
    model = build_model(model_cfg, groups, info)
    device = resolve_device(cfg.train.device)
    preparer = BatchPreparer(store.layout, info, suite, device, cfg.augment)
    run_dir = Path(cfg.output_dir) / cfg.name
    trainer = Trainer(model, task_list, preparer, cfg.train, log_path=run_dir / "log.jsonl")
    if verbose:
        print(f"[{cfg.name}] params={count_parameters(model)} train_eps={len(train_ids)} windows={len(train_ds)} "
              f"val={len(val_ds)} test={len(test_ds)} device={device}")
    fit = trainer.fit(train_ds, val_ds)
    test = trainer.evaluate(test_ds)
    result = {
        "name": cfg.name,
        "params": count_parameters(model),
        "train_episodes": len(train_ids),
        "train_fraction": cfg.train_fraction,
        "seed": cfg.train.seed,
        "best_epoch": fit["best_epoch"],
        "best_step": fit["best_step"],
        **{f"test/{k}": v for k, v in test.items()},
    }
    cls_tasks = [t for t in task_list if isinstance(t, ClassificationTask)]
    if cls_tasks:
        result["confusion"] = trainer.confusion_matrix(test_ds, cls_tasks[0], num_classes).tolist()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    if cfg.save_checkpoint:
        save_checkpoint(run_dir / "model.pt", model, {
            "model_cfg": to_dict(model_cfg), "groups": {k: asdict(v) for k, v in groups.items()},
            "sensors": cfg.sensors, "terrain_names": store.meta.terrain_names, "rates": store.meta.rates,
            "experiment": to_dict(cfg),
        })
    if verbose:
        print(f"[{cfg.name}] test: " + ", ".join(f"{k}={v:.4f}" for k, v in result.items()
                                                  if k.startswith("test/")))
    return result


def load_trained_model(path: str | Path, map_location="cpu", info: LayoutInfo | None = None):
    """Load a checkpoint written by :func:`run_experiment`. Returns ``(model, checkpoint_dict)``.

    Layout-specific baselines (e.g. ``flat_recurrent``) also need the sensor layout ``info`` they were trained on.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    groups = {k: GroupSpec(**v) for k, v in ckpt["groups"].items()}
    model_cfg = dict(ckpt["model_cfg"])
    # Checkpoints from before version 1 predate these keys and were trained with the version-0 behaviour.
    arch = model_cfg.get("architecture", "hierarchical")
    if arch == "hierarchical":
        model_cfg.setdefault("fusion", "mixed")
    elif arch == "flat_recurrent":
        model_cfg.setdefault("kinematics", False)
    cfg = parse_model_config(model_cfg)
    if cfg.architecture != "hierarchical" and info is None:
        raise ValueError(f"'{cfg.architecture}' models are tied to a sensor layout: pass the LayoutInfo")
    model = build_model(cfg, groups, info)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
