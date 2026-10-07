"""Learning-curve study: test accuracy vs. amount of training data, per model.

This operationalizes the Phase 1 completion criterion ("matches or exceeds an unstructured
baseline, with a comparably sized network, using measurably less training data").
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import torch

from somato.data.dataset import EpisodeStore
from somato.geometry.layout import LayoutInfo
from somato.models.baselines import count_parameters, match_parameter_count
from somato.models.builder import build_model
from somato.training.experiment import (
    ExperimentConfig, build_suite, deep_update, group_specs, resolve_config_refs, run_experiment,
)
from somato.utils.config import load_yaml

WIDTH_KEY = {"hierarchical": "dim", "flat_recurrent": "hidden"}


def _with_width(model_cfg: dict, width: int) -> dict:
    cfg = copy.deepcopy(model_cfg)
    cfg[WIDTH_KEY[cfg.get("architecture", "hierarchical")]] = width
    return cfg


def match_widths(models: dict[str, dict], reference: str, store: EpisodeStore, sensors: dict,
                 num_classes: int) -> dict[str, dict]:
    """Scale every model's width so its parameter count matches ``models[reference]``."""
    suite = build_suite(store, sensors)
    groups = group_specs(store, suite)
    info = LayoutInfo.from_layout(store.layout, store.desc)

    def build(cfg):
        cfg = copy.deepcopy(cfg)
        for h in cfg.get("heads", {}).values():
            h["out_dim"] = num_classes
        return build_model(cfg, groups, info)

    target = count_parameters(build(models[reference]))
    out = {}
    for name, cfg in models.items():
        if name == reference:
            out[name] = cfg
            continue
        step = 4 if cfg.get("architecture", "hierarchical") == "hierarchical" else 1  # keep dims divisible by heads
        width, model = match_parameter_count(lambda w: build(_with_width(cfg, step * w)), target, 1, 1024)
        out[name] = _with_width(cfg, step * width)
        print(f"[match] {name}: width {step * width}, params {count_parameters(model)} (target {target})")
    return out


def run_data_efficiency(study_path: str | Path, overrides: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    study = load_yaml(study_path)
    if overrides:
        study = deep_update(study, overrides)
    base = resolve_config_refs(ExperimentConfig.from_yaml(study["base"], study.get("base_overrides")))
    store = EpisodeStore(base.dataset)
    models = {n: load_yaml(m) if isinstance(m, str) else m for n, m in study["models"].items()}
    if study.get("match_params", False):
        models = match_widths(models, study.get("reference_model", next(iter(models))), store, base.sensors,
                              len(store.meta.terrain_names))
    out_dir = Path(study.get("output_dir", "runs/data_efficiency"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    results: list[dict[str, Any]] = []
    for frac in study["fractions"]:
        for seed in study["seeds"]:
            for name, mcfg in models.items():
                cfg = copy.deepcopy(base)
                cfg.model, cfg.train_fraction, cfg.train.seed = mcfg, float(frac), int(seed)
                cfg.name, cfg.output_dir, cfg.save_checkpoint = f"{name}/frac{frac}_seed{seed}", str(out_dir), False
                r = run_experiment(cfg, store)
                r.pop("confusion", None)
                r["model"] = name
                results.append(r)
                _write_csv(csv_path, results)
    summary = summarize(results)
    (out_dir / "summary.md").write_text(summary)
    print(summary)
    _plot(results, out_dir / "learning_curves.png")
    return results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(results: list[dict[str, Any]], metric: str = "test/terrain/acc_last") -> str:
    models = sorted({r["model"] for r in results})
    fracs = sorted({r["train_fraction"] for r in results})
    lines = ["| model | params | " + " | ".join(f"{f:g}" for f in fracs) + " |",
             "|---|---|" + "---|" * len(fracs)]
    for m in models:
        rows = [r for r in results if r["model"] == m]
        cells = []
        for f in fracs:
            vals = torch.tensor([r[metric] for r in rows if r["train_fraction"] == f and metric in r])
            cells.append(f"{vals.mean():.3f} ± {vals.std(unbiased=False):.3f}" if len(vals) else "-")
        lines.append(f"| {m} | {rows[0]['params']} | " + " | ".join(cells) + " |")
    return f"Test {metric} vs. training-data fraction (mean ± std over seeds)\n\n" + "\n".join(lines) + "\n"


def _plot(results, path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    metric = "test/terrain/acc_last"
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in sorted({r["model"] for r in results}):
        fr = sorted({r["train_fraction"] for r in results if r["model"] == m})
        means = [torch.tensor([r[metric] for r in results if r["model"] == m and r["train_fraction"] == f]).mean()
                 for f in fr]
        ax.plot(fr, means, marker="o", label=m)
    ax.set_xscale("log")
    ax.set_xlabel("fraction of training episodes")
    ax.set_ylabel("test accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
