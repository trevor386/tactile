"""Out-of-distribution evaluation (E-7): how do trained terrain classifiers respond to terrains they never saw?

    python scripts/analysis/ood_eval.py --runs "runs/v1/main_curves/*/frac1.0_seed*/model.pt" \
        --id_dataset datasets/mjlab_v1_2400 --ood_dataset datasets/mjlab_v1_ood_earth_600 [--out runs/v1/ood_eval.csv]

For every checkpoint and every OOD class (e.g. cold arctic ice, dry sand, gravel) it reports:
* the histogram of predicted training classes;
* the mean max-softmax confidence and predictive entropy, compared with the same quantities on the in-distribution
  test split.

An open-set-aware model should be *less* confident on unseen terrain (an AUROC of ID-vs-OOD by confidence above 0.5).
"""

import argparse
import csv
import glob
from pathlib import Path

import torch

from somato.data.dataset import EpisodeStore, WindowDataset, stratified_split
from somato.geometry import LayoutInfo
from somato.training.experiment import ExperimentConfig, build_suite, load_trained_model
from somato.training.prepare import BatchPreparer
from somato.training.trainer import resolve_device
from somato.utils.config import from_dict


@torch.no_grad()
def softmax_outputs(model, store, ids, cfg, info, sensors, device, batch_size=64):
    """Last-step class probabilities ``[W, C]`` and episode class ids ``[W]`` for the windows of episodes ``ids``."""
    ds = WindowDataset(store, ids, cfg.window, cfg.eval_stride)
    prep = BatchPreparer(store.layout, info, build_suite(store, sensors), device, cfg.augment)
    probs, labels = [], []
    for sample in torch.utils.data.DataLoader(ds, batch_size=batch_size):
        batch = prep(sample, train=False)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out, _ = model(batch, output_steps=1)
        probs.append(out["terrain"][:, -1].float().softmax(-1).cpu())
        labels.append(batch.labels["terrain"].long().cpu())
    return torch.cat(probs), torch.cat(labels)


def auroc(id_score: torch.Tensor, ood_score: torch.Tensor) -> float:
    """P(score_ID > score_OOD), i.e. how well confidence separates in- from out-of-distribution windows."""
    return float((id_score[:, None] > ood_score[None, :]).float().mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True)
    p.add_argument("--id_dataset", required=True)
    p.add_argument("--ood_dataset", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    device = resolve_device("auto")
    id_store, ood_store = EpisodeStore(args.id_dataset), EpisodeStore(args.ood_dataset)
    rows = []
    for path in sorted(glob.glob(args.runs)):
        info = LayoutInfo.from_layout(id_store.layout, id_store.desc)
        model, ckpt = load_trained_model(path, map_location=device, info=info)
        model = model.to(device)  # load_trained_model returns the model on the CPU
        cfg = from_dict(ExperimentConfig, ckpt["experiment"])
        torch.manual_seed(0)
        test_ids = stratified_split(id_store.labels("terrain"), tuple(cfg.split), cfg.split_seed)[2]
        pid, _ = softmax_outputs(model, id_store, test_ids, cfg, info, ckpt["sensors"], device)
        pood, yood = softmax_outputs(model, ood_store, torch.arange(len(ood_store)), cfg, info, ckpt["sensors"], device)
        conf_id, conf_ood = pid.max(-1).values, pood.max(-1).values
        ent = lambda q: -(q * q.clamp_min(1e-12).log()).sum(-1)  # noqa: E731
        run = Path(path).parent
        for c, name in enumerate(ood_store.meta.terrain_names):
            sel = yood == c
            hist = torch.bincount(pood[sel].argmax(-1), minlength=pid.shape[1]).float() / max(1, int(sel.sum()))
            rows.append({"model": run.parent.name, "run": run.name, "ood_class": name,
                         "conf_ood": float(conf_ood[sel].mean()), "conf_id": float(conf_id.mean()),
                         "entropy_ood": float(ent(pood[sel]).mean()), "entropy_id": float(ent(pid).mean()),
                         "auroc": auroc(conf_id, conf_ood[sel]),
                         **{f"pred_{n}": float(h) for n, h in zip(ckpt["terrain_names"], hist)}})
            r = rows[-1]
            print(f"{r['model']:14s} {r['run']:16s} {name:16s} conf {r['conf_ood']:.2f} (ID {r['conf_id']:.2f}) "
                  f"AUROC {r['auroc']:.2f} | " + " ".join(f"{n[:6]} {h:.2f}" for n, h in zip(ckpt["terrain_names"], hist)),
                  flush=True)
    if args.out and rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
