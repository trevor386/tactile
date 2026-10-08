"""Bayes-optimal terrain-classification accuracy from the terrain catalog alone.

    python scripts/analysis/task_ceiling.py [--terrains configs/terrains/ice_forms.yaml]

Episodes sample every terrain parameter uniformly within their class's range, so with *perfect* knowledge of a
subset of parameters the best possible classifier picks the class of highest density at the observed values.
This gives the ceiling a model can reach if it infers those parameters exactly: e.g. friction alone (what
proprioception and locomotion reveal) vs. friction + sinkage/texture (what the tactile model adds).
"""

import argparse

import torch

from somato.sim.terrain import TerrainCatalog, default_catalog_path

SUBSETS = {
    "friction only": ["friction"],
    "friction + sinkage": ["friction", "sinkage"],
    "mjlab-enacted (friction, sinkage, texture amp/wavelength)": ["friction", "sinkage", "texture_amp",
                                                                  "texture_wavelength"],
    "Isaac-enacted (+ restitution)": ["friction", "sinkage", "texture_amp", "texture_wavelength", "restitution"],
}


def bayes_accuracy(catalog: TerrainCatalog, dims: list[str], n: int = 200_000, seed: int = 0) -> float:
    gen = torch.Generator().manual_seed(seed)
    classes = catalog.classes
    K = len(classes)
    cls = torch.randint(K, (n,), generator=gen)
    logp = torch.zeros(n, K)
    for d in dims:
        lo = torch.tensor([getattr(c, d)[0] for c in classes])
        hi = torch.tensor([getattr(c, d)[1] for c in classes])
        theta = lo[cls] + (hi[cls] - lo[cls]) * torch.rand(n, generator=gen)
        for k in range(K):
            width = float(hi[k] - lo[k])
            if width > 0:
                inside = (theta >= lo[k]) & (theta <= hi[k])
                logp[:, k] += torch.where(inside, -torch.log(torch.tensor(width)), torch.tensor(-1e9))
            else:
                logp[:, k] += torch.where((theta - lo[k]).abs() < 1e-12, 0.0, -1e9)
    ties = (logp >= logp.max(1, keepdim=True).values - 1e-9).float()  # random tie-break
    pred = torch.multinomial(ties, 1, generator=gen).squeeze(1)
    return float((pred == cls).float().mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--terrains", default=default_catalog_path())
    args = parser.parse_args()
    catalog = TerrainCatalog.from_yaml(args.terrains)
    print(f"classes: {catalog.names}")
    for label, dims in SUBSETS.items():
        print(f"  {label:60s} {bayes_accuracy(catalog, dims):.3f}")


if __name__ == "__main__":
    main()
