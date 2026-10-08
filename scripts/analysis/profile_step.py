"""Time training steps of one model on real data, broken down by phase and top-level module.

    python scripts/analysis/profile_step.py --model configs/models/hierarchical_meanmax.yaml \
        --dataset datasets/mjlab_terrain_2400 [--width 100] [--steps 30] [--amp bf16] [--table]

Phases: loader (CPU batch assembly), prepare (sensor models + poses + augmentation on the device), forward
(with a per-module split from synchronizing hooks), backward, optimizer. ``--table`` adds a torch.profiler
table of the most expensive CUDA kernels over a few steps.
"""

import argparse
import time
from collections import defaultdict

import torch

from somato.data.dataset import EpisodeStore, WindowDataset, stratified_split
from somato.geometry.layout import LayoutInfo
from somato.models.baselines import count_parameters
from somato.models.builder import build_model, parse_model_config
from somato.training.data_efficiency import _with_width
from somato.training.experiment import ExperimentConfig, build_suite, group_specs, resolve_config_refs
from somato.training.prepare import BatchPreparer
from somato.training.tasks import ClassificationTask, build_tasks
from somato.training.trainer import Trainer, resolve_device


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """Top-level modules to time: direct children, expanding ModuleDicts one level."""
    out = {}
    for name, child in model.named_children():
        if isinstance(child, torch.nn.ModuleDict):
            out.update({f"{name}.{k}": m for k, m in child.items()})
        else:
            out[name] = child
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/experiments/terrain_mock.yaml")
    p.add_argument("--dataset", default="datasets/mjlab_terrain_2400")
    p.add_argument("--model", required=True)
    p.add_argument("--width", type=int, default=0, help="override the model width (as width matching does)")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--amp", default="bf16")
    p.add_argument("--batch_size", type=int, default=0)
    p.add_argument("--table", action="store_true")
    args = p.parse_args()

    train_over = {"amp": args.amp}
    if args.batch_size:
        train_over["batch_size"] = args.batch_size
    cfg = resolve_config_refs(ExperimentConfig.from_yaml(
        args.config, {"dataset": args.dataset, "model": args.model, "train": train_over}))
    if args.width:
        cfg.model = _with_width(cfg.model, args.width)
    store = EpisodeStore(cfg.dataset)
    labels = store.labels("terrain")
    train_ids, _, _ = stratified_split(labels, tuple(cfg.split), cfg.split_seed)
    train_ds = WindowDataset(store, train_ids, cfg.window, cfg.stride, random_offset=True,
                             generator=torch.Generator().manual_seed(0))
    suite = build_suite(store, cfg.sensors)
    groups = group_specs(store, suite)
    model_cfg = parse_model_config(cfg.model)
    tasks = build_tasks(cfg.tasks)
    for t in tasks:
        if isinstance(t, ClassificationTask) and t.head in model_cfg.heads:
            model_cfg.heads[t.head].out_dim = len(store.meta.terrain_names)
    info = LayoutInfo.from_layout(store.layout, store.desc, getattr(model_cfg, "cluster_mode", "body"),
                                  getattr(model_cfg, "num_clusters", None))
    model = build_model(model_cfg, groups, info)
    device = resolve_device("auto")
    preparer = BatchPreparer(store.layout, info, suite, device, cfg.augment)
    trainer = Trainer(model, tasks, preparer, cfg.train)
    trainer.fit_normalizers(train_ds)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    print(f"model={args.model} width={args.width or '-'} params={count_parameters(model)} "
          f"batch={cfg.train.batch_size} amp={args.amp} device={device}")

    fwd_t: dict[str, float] = defaultdict(float)
    starts: dict[str, float] = {}
    def pre_hook(name):
        def hook(mod, inp):  # must return None (a returned value replaces the inputs)
            sync()
            starts[name] = time.perf_counter()
        return hook

    def post_hook(name):
        def hook(mod, inp, out):
            sync()
            fwd_t[name] += time.perf_counter() - starts[name]
        return hook

    for name, m in timed_modules(model).items():
        m.register_forward_pre_hook(pre_hook(name))
        m.register_forward_hook(post_hook(name))

    phase: dict[str, float] = defaultdict(float)
    loader = trainer._loader(train_ds, shuffle=True)
    it = iter(loader)
    n = 0
    for i in range(args.warmup + args.steps):
        if i == args.warmup:
            phase.clear(); fwd_t.clear(); torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        t0 = time.perf_counter()
        sample = next(it)
        t1 = time.perf_counter()
        batch = preparer(sample, train=True); sync()
        t2 = time.perf_counter()
        with trainer._autocast():
            outputs, _ = model(batch, None, output_steps=cfg.train.output_steps)
            loss, _ = trainer._loss(outputs, batch)
        sync()
        t3 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward(); sync()
        t4 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sync()
        t5 = time.perf_counter()
        if i >= args.warmup:
            n += 1
            for k, v in (("loader", t1 - t0), ("prepare", t2 - t1), ("forward", t3 - t2), ("backward", t4 - t3),
                         ("optimizer", t5 - t4)):
                phase[k] += v
    total = sum(phase.values())
    print(f"per step: {1000 * total / n:.1f} ms")
    for k, v in phase.items():
        print(f"  {k:10s} {1000 * v / n:8.1f} ms  {100 * v / total:5.1f} %")
    print("forward by module (synchronized):")
    for k, v in sorted(fwd_t.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {1000 * v / n:8.1f} ms")
    if device.type == "cuda":
        print(f"peak memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    if args.table:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                batch = preparer(next(it), train=True)
                with trainer._autocast():
                    outputs, _ = model(batch, None, output_steps=cfg.train.output_steps)
                    loss, _ = trainer._loss(outputs, batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sync()
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))


if __name__ == "__main__":
    main()
