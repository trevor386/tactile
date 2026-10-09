"""Best achievable terrain-classification accuracy from the terrain catalog alone (Bayes ceiling).

    python scripts/analysis/task_ceiling.py [--terrains configs/terrains/ice_snow_v1.yaml] [--texture_filter 0.0035]

Episodes sample every terrain parameter uniformly within their class's range, so a model that inferred some of the
parameters *exactly* could at best pick the most probable class given them. That ceiling is estimated here by
nearest-neighbour classification of a large sample of parameter vectors (k-NN converges to the Bayes rate), for
subsets of what the robot can sense:

* friction (proprioception, shear/normal ratios, locomotion);
* compliance (sinkage, pressure level, impact response): the soft-contact parameters of simulation v1;
* felt roughness: texture amplitude x the skin's attenuation at the texture wavelength (exp(-(2 pi/lambda)^2 sigma^2 / 2));
  texture finer than the skin can resolve counts as absent.

"exact" assumes perfect inference of those quantities (a loose upper bound); "noisy" adds the observation noise given
by --noise (relative, per feature: an assumption about how well 0.5 s of sensing pins them down).
"""

import argparse
import math

import torch

from somato.sim.terrain import TerrainCatalog

COMPLIANCE = ["contact_dmin", "contact_width", "contact_timeconst", "contact_dampratio"]


def sample_features(catalog: TerrainCatalog, n: int, sigma: float, gen: torch.Generator) -> tuple[dict, torch.Tensor]:
    cls = torch.randint(len(catalog), (n,), generator=gen)
    batch = catalog.sample(cls, gen)
    p = batch.params
    lam = p["texture_wavelength"]
    felt = p["texture_amp"] * torch.exp(-0.5 * (2 * math.pi / lam * sigma) ** 2)
    feats = {"friction": p["friction"], "felt_roughness": felt, "sinkage(v0)": p["sinkage"]}
    for k in COMPLIANCE:
        feats[k] = p[k]
    return feats, cls


def knn_accuracy(train: torch.Tensor, ytr: torch.Tensor, test: torch.Tensor, yte: torch.Tensor, k: int = 25,
                 num_classes: int = 5) -> float:
    mu, sd = train.mean(0), train.std(0).clamp_min(1e-9)
    train, test = (train - mu) / sd, (test - mu) / sd
    correct = 0
    for chunk in torch.split(torch.arange(len(test)), 2048):
        d = torch.cdist(test[chunk], train)
        idx = d.topk(k, largest=False).indices
        votes = torch.zeros(len(chunk), num_classes).scatter_add_(1, ytr[idx], torch.ones_like(idx, dtype=torch.float))
        correct += int((votes.argmax(1) == yte[chunk]).sum())
    return correct / len(test)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--terrains", default="configs/terrains/ice_snow_v1.yaml")
    p.add_argument("--texture_filter", type=float, default=0.0035, help="skin aperture sigma [m] (0: unfiltered)")
    p.add_argument("--noise", type=float, default=0.15, help="relative observation noise for the 'noisy' ceiling")
    p.add_argument("--n", type=int, default=40000)
    args = p.parse_args()
    catalog = TerrainCatalog.from_yaml(args.terrains)
    gen = torch.Generator().manual_seed(0)
    ftr, ytr = sample_features(catalog, args.n, args.texture_filter, gen)
    fte, yte = sample_features(catalog, args.n // 4, args.texture_filter, gen)
    # A constant parameter has a ~1e-8 float32 spread; standardizing it would turn rounding into a noise feature.
    varies = lambda keys: [k for k in keys if float(ftr[k].std()) > 1e-6 * float(ftr[k].abs().mean()) + 1e-12]  # noqa: E731
    subsets = {
        "friction only": ["friction"],
        "compliance only": COMPLIANCE + ["sinkage(v0)"],
        "felt roughness only": ["felt_roughness"],
        "friction + compliance": ["friction", "sinkage(v0)"] + COMPLIANCE,
        "friction + compliance + felt roughness": ["friction", "sinkage(v0)", "felt_roughness"] + COMPLIANCE,
    }
    print(f"classes: {catalog.names}; skin sigma {args.texture_filter * 1e3:.1f} mm; noisy = {args.noise:.0%} noise")
    for label, keys in subsets.items():
        keys = varies(keys)
        if not keys:
            print(f"  {label:42s} (no varying parameter)")
            continue
        tr = torch.stack([ftr[k] for k in keys], -1)
        te = torch.stack([fte[k] for k in keys], -1)
        exact = knn_accuracy(tr, ytr, te, yte, num_classes=len(catalog))
        g2 = torch.Generator().manual_seed(1)
        scale = tr.abs().mean(0)  # noise relative to each feature's typical magnitude
        noisy = knn_accuracy(tr + args.noise * scale * torch.randn(tr.shape, generator=g2), ytr,
                             te + args.noise * scale * torch.randn(te.shape, generator=g2), yte,
                             num_classes=len(catalog))
        print(f"  {label:42s} exact {exact:.3f}   noisy {noisy:.3f}   ({', '.join(keys)})")


if __name__ == "__main__":
    main()
