"""Streaming accuracy vs time since reset (E-18a): do models trained on 0.5 s windows keep working when their recurrent
state is carried over a whole episode, as in real-time streaming deployment (F-31)?

    python scripts/analysis/streaming_eval.py --runs "runs/v1/sensor_random/[fs]*/frac1.0_seed*/model.pt" \
        --dataset datasets/mjlab_v1_2400 [--sensors configs/sensors/fsr_v1.yaml] [--out runs/v1/e18a_streaming.csv]

Each test episode (150 latent steps, 3 s) runs in one pass from a fresh state, which equals step-by-step streaming
with the state carried. Accuracy is recorded at every output step (every step for the flat GRU, every brain step for
the hierarchical models) and summarized per time bin. The transformer has no recurrent state (window mode only).
Compare with window-mode accuracy (e.g. the "nominal" row of robustness.py).
"""

import argparse
import csv
import glob
from pathlib import Path

import torch

from somato.data.dataset import EpisodeStore, WindowDataset, stratified_split
from somato.geometry.layout import LayoutInfo
from somato.training.experiment import ExperimentConfig, build_suite, load_trained_model
from somato.training.prepare import BatchPreparer
from somato.training.trainer import resolve_device
from somato.utils.config import from_dict, load_yaml

BINS = [(0, 25), (25, 50), (50, 100), (100, 150)]  # latent steps since reset (50 Hz); training windows are 25


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True)
    p.add_argument("--dataset", default="datasets/mjlab_v1_2400")
    p.add_argument("--sensors", default="configs/sensors/fsr_v1.yaml")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    device = resolve_device("auto")
    store = EpisodeStore(args.dataset)
    info = LayoutInfo.from_layout(store.layout, store.desc)
    suite = build_suite(store, load_yaml(args.sensors))
    rows = []
    for path in sorted(glob.glob(args.runs)):
        model, ckpt = load_trained_model(path, map_location=device, info=info)
        if type(model).__name__ == "TransformerBaseline":
            continue
        model = model.to(device).eval()
        cfg = from_dict(ExperimentConfig, ckpt["experiment"])
        test_ids = stratified_split(store.labels("terrain"), tuple(cfg.split), cfg.split_seed)[2]
        T = store.episodes[int(test_ids[0])].num_steps
        ds = WindowDataset(store, test_ids, T, T)  # one full episode per sample
        prep = BatchPreparer(store.layout, info, suite, device, cfg.augment)
        correct, count = torch.zeros(T), torch.zeros(T)
        torch.manual_seed(0)
        with torch.no_grad():
            for sample in torch.utils.data.DataLoader(ds, batch_size=args.batch_size):
                batch = prep(sample, train=False)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out, _ = model(batch)
                logits = out["terrain"].float()  # [B, K, C]
                steps = out["steps"][0].long().cpu() if "steps" in out else torch.arange(T - logits.shape[1], T)
                hit = (logits.argmax(-1) == batch.labels["terrain"].long()[:, None]).float().sum(0).cpu()  # [K]
                correct.index_add_(0, steps, hit)
                count.index_add_(0, steps, torch.full_like(hit, logits.shape[0]))
        run = Path(path).parent
        row = {"model": run.parent.name, "run": run.name, "checkpoint": path}
        for lo, hi in BINS:
            sel = (count > 0) & (torch.arange(T) >= lo) & (torch.arange(T) < hi)
            row[f"acc_{lo}_{hi}"] = float(correct[sel].sum() / count[sel].sum()) if sel.any() else float("nan")
        rows.append(row)
        print(f"{row['model']:16s} {row['run']:16s} " + " | ".join(
            f"steps {lo}-{hi}: {row[f'acc_{lo}_{hi}']:.3f}" for lo, hi in BINS), flush=True)
    if args.out and rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
