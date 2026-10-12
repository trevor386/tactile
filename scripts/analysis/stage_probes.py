"""What do stage 1 and stage 2 encode (E-24)? Linear probes from per-taxel features to physical quantities.

    python scripts/analysis/stage_probes.py --runs "runs/v1/sensor_random/[rs]*/frac1.0_seed*/model.pt" \
        [--dataset datasets/mjlab_v1_2400] [--sensors configs/sensors/fsr_v1.yaml] [--out runs/v1/e24_probes.csv]

For every taxel at the last step of each test window, ridge regression (fitted on half of the test episodes,
scored by R^2 on the other half) predicts:
* ``pressure``: the true normal pressure (kPa), which the FSR reports through hysteresis, creep and saturation;
* ``shear``: the true shear magnitude (kPa), which the FSR does not sense at all;
* ``slip``: the slip speed of the taxel's link (m/s).
Pressure is scored on all taxels and on taxels in contact (true pressure > 1 kPa); shear and slip on taxels in contact.
Link level: each feature set's mean and max over a link's taxels predict the link's contact flag, and on links in
contact its slip speed and normal force (what stage 2's neighbourhood integration should make accessible).
Feature sets:
* ``raw_now``: the taxel's FSR samples in the last latent step (10);
* ``raw_hist``: its FSR samples over the whole 0.5 s window (250), i.e. the best linear filter of its own history;
* ``stage1``: the taxel's stage-1 latent (its own history, through the trained receptor encoder);
* ``stage2``: its stage-2 latent (plus its neighbours, through the 3D kernel).
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

CONTACT_KPA = 1.0


def _ridge(x_fit, y_fit, x_eval, y_eval, lam: float) -> float:
    mu, sd = x_fit.mean(0), x_fit.std(0).clamp_min(1e-8)
    a = torch.cat([(x_fit - mu) / sd, torch.ones(len(x_fit), 1, dtype=x_fit.dtype)], 1)
    b = torch.cat([(x_eval - mu) / sd, torch.ones(len(x_eval), 1, dtype=x_eval.dtype)], 1)
    w = torch.linalg.solve(a.T @ a + lam * len(a) * torch.eye(a.shape[1], dtype=a.dtype), a.T @ y_fit)
    resid = y_eval - b @ w
    return float(1 - resid.pow(2).sum() / (y_eval - y_eval.mean()).pow(2).sum())


def ridge_r2(x_fit, y_fit, x_eval, y_eval) -> float:
    """R^2 on (x_eval, y_eval) of a ridge regression fitted on (x_fit, y_fit), features standardized; the ridge
    strength is chosen on a 20 % hold-out of the fitting rows, so high-dimensional features are not penalized by
    overfitting."""
    n = int(0.8 * len(x_fit))
    lam = max((1e-4, 1e-3, 1e-2, 1e-1, 1.0), key=lambda l: _ridge(x_fit[:n], y_fit[:n], x_fit[n:], y_fit[n:], l))
    return _ridge(x_fit, y_fit, x_eval, y_eval, lam)


@torch.no_grad()
def collect(model, store, ids, prep, info, per_window: int, gen) -> dict[str, torch.Tensor]:
    """Per-taxel features and targets at the last step of every window of episodes ``ids`` (random taxel subset)."""
    tac = info.slices["tactile"]
    body = info.body_index[tac]
    rows: dict[str, list] = {}
    for sample in torch.utils.data.DataLoader(WindowDataset(store, ids, 25, 25), batch_size=16):
        batch = prep(sample, train=False)
        h, _ = model.encode_temporal(batch, None)  # [B, L, N, D]
        out, _ = model(batch, output_steps=1, return_features=True)
        stim = sample["data/tactile"][:, -1]  # [B, S, Nt, 3] stimulus (Pa) in the last latent step
        fsr = batch.readings["tactile"][..., 0].cpu()  # [B, L, S, Nt]
        B, nt = stim.shape[0], stim.shape[2]
        pick = torch.stack([torch.randperm(nt, generator=gen)[:per_window] for _ in range(B)])  # [B, k]

        def take(x):  # [B, Nt, F] -> [B, k, F] at the picked taxels
            return torch.gather(x, 1, pick[..., None].expand(-1, -1, x.shape[-1]))

        taxel = {
            "raw_now": fsr[:, -1].transpose(1, 2),  # [B, Nt, S]
            "raw_hist": fsr.permute(0, 3, 1, 2).reshape(B, nt, -1),  # [B, Nt, L*S]
            "stage1": h[:, -1, tac].float().cpu(),
            "stage2": out["features"][:, -1, tac].float().cpu(),
        }
        feats = {k: take(v) for k, v in taxel.items()}
        feats.update(pressure=take(stim[..., 0].mean(1)[..., None] / 1e3),
                     shear=take(stim[..., 1:].mean(1).norm(dim=-1, keepdim=True) / 1e3),
                     slip=take(sample["label/slip_speed"][:, -1][:, body][..., None]))
        # Link level: mean and max of each feature set over the link's taxels; the link's slip, force and contact.
        nb = int(body.max()) + 1
        onehot = torch.nn.functional.one_hot(body, nb).float()  # [Nt, Nb]
        for k, v in taxel.items():
            mean = torch.einsum("bnf,nl->blf", v, onehot) / onehot.sum(0)[None, :, None]
            mx = torch.stack([v[:, body == b].amax(1) for b in range(nb)], 1)
            feats[f"link_{k}"] = torch.cat([mean, mx], -1)  # [B, Nb, 2F]
        for t in ("slip_speed", "normal_force", "in_contact"):
            feats[f"link_{t}"] = sample[f"label/{t}"][:, -1][..., None]  # [B, Nb, 1]
        for k, v in feats.items():
            rows.setdefault(k, []).append(v.reshape(-1, v.shape[-1]).double())
    return {k: torch.cat(v) for k, v in rows.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True)
    p.add_argument("--dataset", default="datasets/mjlab_v1_2400")
    p.add_argument("--sensors", default="configs/sensors/fsr_v1.yaml")
    p.add_argument("--taxels_per_window", type=int, default=64)
    p.add_argument("--random_init", action="store_true",
                   help="control: re-initialize all weights (input normalizers kept), i.e. untrained features")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    device = resolve_device("auto")
    store = EpisodeStore(args.dataset)
    info = LayoutInfo.from_layout(store.layout, store.desc)
    suite = build_suite(store, load_yaml(args.sensors))
    rows = []
    for path in sorted(glob.glob(args.runs)):
        model, ckpt = load_trained_model(path, map_location=device, info=info)
        if args.random_init:
            torch.manual_seed(0)
            for m in model.modules():
                if m is not model and hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        model = model.to(device).eval()
        cfg = from_dict(ExperimentConfig, ckpt["experiment"])
        test_ids = stratified_split(store.labels("terrain"), tuple(cfg.split), cfg.split_seed)[2]
        prep = BatchPreparer(store.layout, info, suite, device, cfg.augment)
        torch.manual_seed(0)
        k = args.taxels_per_window
        fit = collect(model, store, test_ids[0::2], prep, info, k, torch.Generator().manual_seed(0))
        ev = collect(model, store, test_ids[1::2], prep, info, k, torch.Generator().manual_seed(1))
        run = Path(path).parent
        for feat in ("raw_now", "raw_hist", "stage1", "stage2"):
            name = run.parent.name + ("_untrained" if args.random_init else "")
            row = {"model": name, "run": run.name, "features": feat, "dim": fit[feat].shape[1]}
            cf, ce = fit["pressure"][:, 0] > CONTACT_KPA, ev["pressure"][:, 0] > CONTACT_KPA
            row["pressure_all"] = ridge_r2(fit[feat], fit["pressure"], ev[feat], ev["pressure"])
            for t in ("pressure", "shear", "slip"):
                row[f"{t}_contact"] = ridge_r2(fit[feat][cf], fit[t][cf], ev[feat][ce], ev[t][ce])
            row["contact_fraction"] = float(ce.double().mean())
            lf, le = f"link_{feat}", fit["link_in_contact"][:, 0] > 0.5
            le_ev = ev["link_in_contact"][:, 0] > 0.5
            row["link_in_contact"] = ridge_r2(fit[lf], fit["link_in_contact"], ev[lf], ev["link_in_contact"])
            for t in ("slip_speed", "normal_force"):  # on links in contact
                row[f"link_{t}"] = ridge_r2(fit[lf][le], fit[f"link_{t}"][le], ev[lf][le_ev], ev[f"link_{t}"][le_ev])
            rows.append(row)
            print(f"{row['model']:14s} {row['run']:15s} {feat:9s} (dim {row['dim']:3d}) R2: pressure all "
                  f"{row['pressure_all']:.3f} | in contact: pressure {row['pressure_contact']:.3f}, shear "
                  f"{row['shear_contact']:.3f}, slip {row['slip_contact']:.3f} | link: contact "
                  f"{row['link_in_contact']:.3f}, slip {row['link_slip_speed']:.3f}, force "
                  f"{row['link_normal_force']:.3f}", flush=True)
    if args.out and rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
