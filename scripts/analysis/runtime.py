"""Online inference cost per model (runtime matching, E-23): latency per output at batch 1, and FLOPs.

    python scripts/analysis/runtime.py --runs "runs/v1/main_curves/*/frac1.0_seed0/model.pt" \
        [--model transformer=configs/models/transformer.yaml:108,216] [--device cuda|cpu] [--threads 4] \
        [--amp bf16] [--out runs/v1/runtime.csv]

A deployed model gets one new latent step every 20 ms (50 Hz). Two ways to run online:
* ``window``: re-run the trained 0.5 s window (25 latent steps) from a fresh state at every step. This is the regime
  the models were trained in, and the only one for the transformer. It is the comparable cost per output.
* ``stream``: process one new latent step with the recurrent state carried (recurrent models only). Its accuracy
  beyond the training window is untested (E-18). The hierarchical models emit an output only every ``brain_stride``
  steps, so their stream latency varies by step (mean and p95 reported).
Excluded: sensor-model simulation (a robot reads real sensors) and computing sensor poses from body poses (cheap
forward kinematics). FLOPs are for one window forward, from ``torch.utils.flop_counter`` (with the fused attention
fast path disabled, which it cannot see) plus an analytic count for ``nn.GRU`` layers, which it does not see either.
"""

import argparse
import csv
import glob
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from somato.data.dataset import EpisodeStore, WindowDataset, stratified_split
from somato.geometry.layout import LayoutInfo
from somato.models.batch import SomatoBatch
from somato.models.baselines import count_parameters
from somato.models.builder import build_model, parse_model_config
from somato.training.data_efficiency import _with_width
from somato.training.experiment import (ExperimentConfig, build_suite, group_specs, load_trained_model,
                                        resolve_config_refs)
from somato.training.prepare import BatchPreparer
from somato.utils.config import load_yaml


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_ms(fn, device, reps: int, warmup: int) -> torch.Tensor:
    for _ in range(warmup):
        fn()
    sync(device)
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(device)
        out.append(1e3 * (time.perf_counter() - t0))
    return torch.tensor(out)


def gru_flops(model: torch.nn.Module):
    """Forward hooks adding the FLOPs of every ``nn.GRU`` call to ``total[0]`` (3 gates, input and hidden matmuls)."""
    total = [0]

    def hook(mod, inp, _):
        x = inp[0]
        steps = x.shape[0] * x.shape[1]  # batch x time (any order)
        for layer in range(mod.num_layers):
            d_in = mod.input_size if layer == 0 else mod.hidden_size
            total[0] += 2 * 3 * (d_in + mod.hidden_size) * mod.hidden_size * steps

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.GRU)]
    return total, handles


def is_recurrent(model) -> bool:
    return type(model).__name__ in ("HierarchicalSomatoModel", "FlatRecurrentModel")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=None, help="glob of model.pt checkpoints (trained widths)")
    p.add_argument("--model", action="append", default=[],
                   help="NAME=CONFIG.yaml[:W1,W2,...]: untrained models at these widths (as width matching sets them)")
    p.add_argument("--config", default="configs/experiments/terrain_v1.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=4, help="CPU threads (CPU runs)")
    p.add_argument("--amp", default="none", help="none | bf16 (GPU)")
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    cfg = resolve_config_refs(ExperimentConfig.from_yaml(args.config))
    store = EpisodeStore(cfg.dataset)
    suite = build_suite(store, cfg.sensors)
    groups = group_specs(store, suite)
    info = LayoutInfo.from_layout(store.layout, store.desc)
    test_ids = stratified_split(store.labels("terrain"), tuple(cfg.split), cfg.split_seed)[2]
    sample = next(iter(torch.utils.data.DataLoader(WindowDataset(store, test_ids[:1], cfg.window, cfg.window))))
    batch = BatchPreparer(store.layout, info, suite, device, cfg.augment)(sample, train=False)  # [1, L, ...]
    L = batch.pos.shape[1]
    steps = [SomatoBatch({g: x[:, t:t + 1] for g, x in batch.readings.items()}, batch.pos[:, t:t + 1],
                         batch.rot[:, t:t + 1], batch.info) for t in range(L)]

    models = []  # (name, width, model)
    for path in sorted(glob.glob(args.runs)) if args.runs else []:
        m, ckpt = load_trained_model(path, info=info)
        mc = ckpt["model_cfg"]
        width = mc.get("dim", mc.get("hidden"))
        models.append((Path(path).parent.parent.name, width, m))
    for spec in args.model:
        name, rest = spec.split("=", 1)
        path, _, widths = rest.partition(":")
        base = load_yaml(path)
        for w in [int(x) for x in widths.split(",")] if widths else [None]:
            mcfg = _with_width(base, w) if w else base
            for h in mcfg.get("heads", {}).values():
                h["out_dim"] = len(store.meta.terrain_names)
            m = build_model(parse_model_config(mcfg), groups, info)
            models.append((name, w or mcfg.get("dim", mcfg.get("hidden")), m))

    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp == "bf16" and device.type == "cuda")
    rows = []
    for name, width, model in models:
        model = model.to(device).eval()
        with torch.no_grad():
            total, handles = gru_flops(model)
            torch.backends.mha.set_fastpath_enabled(False)  # the fused attention kernel is invisible to the counter
            with FlopCounterMode(display=False) as fc:
                model(batch, output_steps=1)
            torch.backends.mha.set_fastpath_enabled(True)  # timed as deployed, with the fast path
            for h in handles:
                h.remove()
            gflops = (fc.get_total_flops() + total[0]) / 1e9

            def window():
                with torch.autocast(**amp):
                    model(batch, output_steps=1)

            t_win = time_ms(window, device, args.reps, args.warmup)
            t_str = None
            if is_recurrent(model):
                it = {"t": 0, "state": None}

                def stream():
                    t = it["t"] % L
                    with torch.autocast(**amp):
                        _, it["state"] = model(steps[t], None if t == 0 else it["state"])
                    it["t"] += 1

                t_str = time_ms(stream, device, max(args.reps, 4 * L), L)
        row = {"model": name, "width": width, "params": count_parameters(model), "device": args.device,
               "threads": args.threads if device.type == "cpu" else "", "amp": args.amp, "gflops_window": gflops,
               "window_median_ms": float(t_win.median()), "window_p95_ms": float(t_win.quantile(0.95)),
               "stream_mean_ms": float(t_str.mean()) if t_str is not None else "",
               "stream_p95_ms": float(t_str.quantile(0.95)) if t_str is not None else ""}
        rows.append(row)
        s = (f" | stream mean {row['stream_mean_ms']:.2f} ms, p95 {row['stream_p95_ms']:.2f} ms"
             if t_str is not None else " | stream n/a")
        print(f"{name:16s} w={width:<5} params={row['params']:>8d} {gflops:7.3f} GFLOP/window | window median "
              f"{row['window_median_ms']:.2f} ms, p95 {row['window_p95_ms']:.2f} ms{s}", flush=True)
    if args.out and rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        new = not Path(args.out).exists()
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
