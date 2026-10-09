"""Evaluate many checkpoints under sensor perturbations they were not trained on (sim-to-real proxy).

    python scripts/analysis/robustness.py --runs "runs/v3/seed0/*/frac*_seed0/model.pt" \
        --dataset datasets/mjlab_terrain_2400 --suite configs/sensors/robustness_fsr.yaml --out runs/robustness/v3.csv

The dataset and test split are loaded once; each (checkpoint, perturbation) pair is evaluated with the same
random seed, so all models see the same sensor noise and gain draws. Prints test accuracy per perturbation and
the drop from nominal; writes one CSV row per pair.
"""

import argparse
import csv
import glob
from pathlib import Path

import torch

from somato.data.dataset import EpisodeStore, WindowDataset, stratified_split
from somato.geometry import LayoutInfo
from somato.training.experiment import (
    ExperimentConfig, build_suite, deep_update, load_trained_model, prepare_store_for_tasks,
)
from somato.training.prepare import BatchPreparer
from somato.training.tasks import build_tasks
from somato.training.trainer import Trainer, resolve_device
from somato.utils.config import from_dict, load_yaml


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="glob of model.pt checkpoints")
    p.add_argument("--dataset", required=True)
    p.add_argument("--suite", default="configs/sensors/robustness_fsr.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    paths = sorted(glob.glob(args.runs))
    if not paths:
        raise SystemExit(f"no checkpoints match {args.runs}")
    suite_cfg = load_yaml(args.suite)
    base_sensors = load_yaml(suite_cfg["base"])
    perturbations = {k: deep_update(load_yaml(suite_cfg["base"]), v or {}) for k, v in suite_cfg["perturbations"].items()}
    device = resolve_device("auto")
    store = EpisodeStore(args.dataset)
    labels = store.labels("terrain")
    datasets: dict[tuple, WindowDataset] = {}
    rows = []
    for path in paths:
        ckpt0 = torch.load(path, map_location="cpu", weights_only=False)
        info = LayoutInfo.from_layout(store.layout, store.desc, ckpt0["model_cfg"].get("cluster_mode", "body"),
                                      ckpt0["model_cfg"].get("num_clusters"))
        model, ckpt = load_trained_model(path, map_location=device, info=info)
        if ckpt["sensors"] != base_sensors:
            print(f"  warning: {path} was trained with sensors {ckpt['sensors']}, suite base is {base_sensors}")
        cfg = from_dict(ExperimentConfig, ckpt["experiment"])
        cfg.train.tbptt_chunk = 0
        cfg.train.batch_size = args.batch_size
        key = (tuple(cfg.split), cfg.split_seed, cfg.window, cfg.eval_stride)
        if key not in datasets:
            test_ids = stratified_split(labels, tuple(cfg.split), cfg.split_seed)[2]
            datasets[key] = WindowDataset(store, test_ids, cfg.window, cfg.eval_stride)
        ds = datasets[key]
        tasks = build_tasks(cfg.tasks)
        prepare_store_for_tasks(store, tasks)
        run = Path(path).parent
        result = {"model": run.parent.name, "run": run.name, "train_fraction": cfg.train_fraction}
        for name, sensors in perturbations.items():
            trainer = Trainer(model, tasks, BatchPreparer(store.layout, info, build_suite(store, sensors), device,
                                                          cfg.augment), cfg.train)
            torch.manual_seed(args.seed)
            m = trainer.evaluate(ds)
            rows.append({**result, "perturbation": name, "acc_last": m["terrain/acc_last"],
                         "nll_last": m["terrain/nll_last"], "checkpoint": path})
        nominal = next(r["acc_last"] for r in rows if r["checkpoint"] == path and r["perturbation"] == "nominal")
        print(f"{result['model']:22s} {result['run']:18s} nominal {nominal:.3f} | " + "  ".join(
            f"{r['perturbation']} {r['acc_last'] - nominal:+.3f}" for r in rows
            if r["checkpoint"] == path and r["perturbation"] != "nominal"), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
