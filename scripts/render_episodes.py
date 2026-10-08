"""Render recorded episodes as an animated GIF (top and side view), colouring links by ground contact force.

    python scripts/render_episodes.py datasets/isaac_terrain --out outputs/isaac_snake.gif [--per_class 1]

Uses only the stored body poses and taxel stimuli, so it works for any simulator and needs no renderer
(Isaac Sim's RTX renderer is not required). Needs matplotlib and Pillow.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from somato.data.episode import Episode  # noqa: E402
from somato.geometry.layout import SensorLayout  # noqa: E402
from somato.geometry.rotations import quat_to_matrix  # noqa: E402
from somato.utils.config import load_yaml  # noqa: E402


def link_segments(pos: np.ndarray, quat: np.ndarray, length: float) -> np.ndarray:
    """Link centre lines ``[T, Nb, 2, 3]``: from each link frame origin along its local x axis."""
    x_axis = quat_to_matrix(torch.from_numpy(quat).float())[..., :, 0].numpy()
    return np.stack([pos, pos + length * x_axis], axis=-2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--out", default="outputs/snake.gif")
    parser.add_argument("--per_class", type=int, default=1, help="episodes per terrain class")
    parser.add_argument("--link_length", type=float, default=0.10)
    parser.add_argument("--every", type=int, default=2, help="render every n-th latent step")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    root = Path(args.dataset)
    meta = load_yaml(root / "meta.yaml")
    names, latent_hz = meta["terrain_names"], meta["rates"]["latent_hz"]
    layout = SensorLayout.load(root / "layout.pt")
    area = layout.groups["tactile"].area.numpy()
    body_of_taxel = layout.groups["tactile"].body_index.numpy()
    picked: dict[int, list[Episode]] = {}
    for f in sorted((root / "episodes").glob("*.npz")):
        ep = Episode.load(f)
        c = int(ep.labels["terrain"])
        if len(picked.setdefault(c, [])) < args.per_class:
            picked[c].append(ep)
    eps = [(c, ep) for c in sorted(picked) for ep in picked[c]]

    panels = []
    for c, ep in eps:
        pos = ep.body_pos.float().numpy()
        seg = link_segments(pos, ep.body_quat.float().numpy(), args.link_length)
        p = ep.data["tactile"].float().numpy()[:, :, :, 0].mean(1)  # [T, N] mean pressure over the latent step
        force = np.zeros((p.shape[0], pos.shape[1]))
        np.add.at(force, (slice(None), body_of_taxel), p * area)  # per-link normal force [T, Nb]
        panels.append((names[c], ep.params.get("friction", float("nan")), seg, force))

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5.2), gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
    fmax = max(np.percentile(f, 99) for *_, f in panels)
    artists = []
    for i, (name, mu, seg, force) in enumerate(panels):
        top, side = axes[0, i], axes[1, i]
        allxy = seg[..., :2].reshape(-1, 2)
        lo, hi = allxy.min(0) - 0.15, allxy.max(0) + 0.15
        span = (hi - lo).max()
        mid = (lo + hi) / 2
        top.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        top.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
        top.set_aspect("equal")
        top.set_title(f"{name}\nmu = {mu:.2f}", fontsize=9)
        top.tick_params(labelsize=6)
        trail, = top.plot([], [], color="0.7", lw=1)
        lc_top = LineCollection([], linewidths=4, cmap="viridis", capstyle="round")
        lc_top.set_clim(0, fmax)
        top.add_collection(lc_top)
        side.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        side.set_ylim(-0.01, 0.15)
        side.axhline(0, color="0.5", lw=0.8)
        side.tick_params(labelsize=6)
        side.set_xlabel("x [m] (side view, z up)", fontsize=7)
        lc_side = LineCollection([], linewidths=3, cmap="viridis", capstyle="round")
        lc_side.set_clim(0, fmax)
        side.add_collection(lc_side)
        artists.append((trail, lc_top, lc_side, seg, force))
    fig.colorbar(artists[0][1], ax=axes[0, :].tolist(), shrink=0.6, label="link normal force [N]")
    title = fig.suptitle("")
    T = min(a[3].shape[0] for a in artists)
    frames = range(0, T, args.every)

    def update(t):
        for trail, lc_top, lc_side, seg, force in artists:
            centroid = seg[: t + 1, :, 0, :2].mean(1)
            trail.set_data(centroid[:, 0], centroid[:, 1])
            lc_top.set_segments(seg[t, :, :, :2])
            lc_top.set_array(force[t])
            lc_side.set_segments(seg[t][:, :, [0, 2]])
            lc_side.set_array(force[t])
        title.set_text(f"{root.name}   t = {t / latent_hz:.2f} s")
        return []

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, update, frames=frames).save(args.out, writer=PillowWriter(fps=args.fps), dpi=80)
    print(f"wrote {args.out} ({len(frames)} frames, {n} episodes)")


if __name__ == "__main__":
    main()
