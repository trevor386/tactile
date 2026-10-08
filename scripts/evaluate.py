"""Evaluate a trained checkpoint on a dataset, e.g. a model trained on Isaac Sim data tested on MuJoCo data.

    python scripts/evaluate.py --checkpoint runs/isaac600_hier/model.pt --dataset datasets/mjlab_terrain_600

By default the test split is used, recomputed with the checkpoint's split settings; for datasets collected
with the same seed (episode-paired, as collect_isaac.yaml and collect_mjlab.yaml are) these are the same
episodes in both simulators. Prints the metrics and the confusion matrix (rows: true class).
"""

import argparse
import json

import torch

from somato.data.dataset import EpisodeStore, WindowDataset, stratified_split
from somato.geometry import LayoutInfo
from somato.training.experiment import ExperimentConfig, build_suite, load_trained_model
from somato.training.prepare import BatchPreparer
from somato.training.tasks import ClassificationTask, build_tasks
from somato.training.trainer import Trainer, resolve_device
from somato.utils.config import from_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=["test", "all"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None, help="optional JSON file for the results")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, ckpt = load_trained_model(args.checkpoint, map_location=device)
    cfg = from_dict(ExperimentConfig, ckpt["experiment"])
    store = EpisodeStore(args.dataset)
    names = store.meta.terrain_names
    if names != ckpt["terrain_names"]:
        raise ValueError(f"dataset classes {names} != model classes {ckpt['terrain_names']}")
    labels = store.labels("terrain")
    ids = torch.arange(len(labels)) if args.split == "all" else stratified_split(labels, tuple(cfg.split),
                                                                                  cfg.split_seed)[2]
    ds = WindowDataset(store, ids, cfg.window, cfg.eval_stride)
    suite = build_suite(store, ckpt["sensors"])
    info = LayoutInfo.from_layout(store.layout, store.desc, ckpt["model_cfg"].get("cluster_mode", "body"),
                                  ckpt["model_cfg"].get("num_clusters"))
    tasks = build_tasks(cfg.tasks)
    trainer = Trainer(model, tasks, BatchPreparer(store.layout, info, suite, device, cfg.augment), cfg.train)
    metrics = trainer.evaluate(ds)
    cls = [t for t in tasks if isinstance(t, ClassificationTask)]
    confusion = trainer.confusion_matrix(ds, cls[0], len(names)).tolist() if cls else None
    print(f"{args.checkpoint} on {args.dataset} ({args.split}: {len(ids)} episodes, {len(ds)} windows)")
    print("  " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    if confusion:
        width = max(len(n) for n in names)
        print("  confusion (rows: true):")
        for n, row in zip(names, confusion):
            print(f"    {n:>{width}s} " + " ".join(f"{int(v):4d}" for v in row))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "dataset": args.dataset, "split": args.split,
                       "metrics": metrics, "confusion": confusion, "terrain_names": names}, f, indent=2)


if __name__ == "__main__":
    main()
