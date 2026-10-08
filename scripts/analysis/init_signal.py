"""Signal propagation at initialization: does each stage's output still depend on the input sample?

    python scripts/analysis/init_signal.py --dataset datasets/mjlab_terrain_600_dc [--model configs/models/hierarchical.yaml]

For one training batch it reports, per stage, the spread of the per-sample mean feature across samples
("between") relative to the spread inside a sample ("within"), the spread of the logits across samples, and
the gradient norm of each top-level module after one backward pass. A model whose logits barely vary across
samples at initialization starts on a loss plateau (uniform prediction).
"""

import argparse
import copy

import torch

from somato.data.dataset import EpisodeStore, WindowDataset, stratified_split
from somato.geometry import LayoutInfo
from somato.models import build_model
from somato.models.builder import parse_model_config
from somato.training.experiment import ExperimentConfig, build_suite, group_specs, resolve_config_refs
from somato.training.prepare import BatchPreparer
from somato.training.tasks import build_tasks
from somato.training.trainer import Trainer, resolve_device


def spread(x: torch.Tensor) -> tuple[float, float]:
    """``x [B, ..., D]`` -> (std across samples of the per-sample mean, mean within-sample std)."""
    flat = x.reshape(x.shape[0], -1, x.shape[-1]).float()
    return float(flat.mean(1).std(0).mean()), float(flat.std(1).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/experiments/terrain_mock.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    over = {"dataset": args.dataset, **({"model": args.model} if args.model else {})}
    cfg = resolve_config_refs(ExperimentConfig.from_yaml(args.config, over))
    torch.manual_seed(args.seed)
    store = EpisodeStore(cfg.dataset)
    train_ids = stratified_split(store.labels("terrain"), tuple(cfg.split), cfg.split_seed)[0]
    ds = WindowDataset(store, train_ids, cfg.window, cfg.stride)
    suite = build_suite(store, cfg.sensors)
    model_cfg = parse_model_config(copy.deepcopy(cfg.model))
    for h in getattr(model_cfg, "heads", {}).values():
        h.out_dim = len(store.meta.terrain_names)
    info = LayoutInfo.from_layout(store.layout, store.desc, getattr(model_cfg, "cluster_mode", "body"))
    model = build_model(model_cfg, group_specs(store, suite), info)
    device = resolve_device("auto")
    prep = BatchPreparer(store.layout, info, suite, device, cfg.augment)
    trainer = Trainer(model, build_tasks(cfg.tasks), prep, cfg.train)
    trainer.fit_normalizers(ds)
    sample = next(iter(trainer._loader(ds, shuffle=True)))
    batch = prep(sample, train=True)
    stages = {}
    hooks = []
    if hasattr(model, "encode_temporal"):
        orig = model.encode_temporal

        def enc(b, s=None):
            h, st = orig(b, s)
            stages["stage1 (temporal)"] = h.detach()
            return h, st
        model.encode_temporal = enc
        hooks.append(model.spatial.register_forward_hook(lambda m, i, o: stages.__setitem__("stage2 (spatial)", o.detach())))
    model.train()
    outputs, _ = model(batch, output_steps=cfg.train.output_steps)
    logits = outputs["terrain"]
    for name, x in stages.items():
        x = x.view(batch.batch_size, -1, *x.shape[-2:])  # stage 2 runs on [B * k, N, D]
        b, w = spread(x)
        print(f"  {name:20s} between-sample {b:.4f}   within-sample {w:.4f}   ratio {b / max(w, 1e-12):.4f}")
        for g in info.group_names:  # per-sensor dependence on the sample, by group: std over the batch of each feature
            xg = x[:, :, info.slices[g]].float()
            print(f"      {g:8s} per-sensor std over samples {float(xg.std(0).mean()):.4f}   "
                  f"feature RMS {float(xg.pow(2).mean().sqrt()):.4f}")
    lb = float(logits[:, -1].detach().std(0).mean())
    print(f"  logits               std across samples {lb:.4f}   |mean logit| {float(logits.mean().abs()):.4f}")
    loss, _ = trainer._loss(outputs, batch)
    loss.backward()
    print(f"  loss {float(loss):.4f}")
    for name, mod in model.named_children():
        g = [p.grad.norm() for p in mod.parameters() if p.grad is not None]
        if g:
            print(f"  grad norm {name:12s} {float(torch.stack(g).norm()):.4e}")


if __name__ == "__main__":
    main()
