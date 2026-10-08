"""Compare simulator datasets recorded with the same robot, gait and terrain classes.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python scripts/compare_datasets.py \
        datasets/isaac_terrain_600 datasets/mjlab_terrain_600 --out outputs/sim_compare [--every 1] [--workers 6]

CPU only (numpy; the GPU is never touched). Episodes are streamed one at a time by a small worker pool, so peak memory
is a few hundred MB per worker; amplitude percentiles come from log-spaced histograms (about 2 percent relative
resolution) rather than from holding every sample. ``--every N`` analyses every N-th episode of each terrain class (the
same episode indices in all datasets, so paired analyses stay paired).

Writes ``report.md`` (side-by-side tables, cells are ``dataset1 / dataset2``), per-episode CSVs and PNG plots to --out:

    1  amplitude percentiles per channel and exceedance of "suspicious" levels
    2  top-10 episodes by joint velocity / gyro peak (flips, vertical excursions, contact loss)
    3  contact pattern per terrain class and effective friction (shear / normal) vs the friction parameter
    4  locomotion (centroid speed) per class
    5  power spectra of IMU |acc| and per-link normal force (contact chatter)
    6  class separability from engineered features (nearest centroid, multinomial logistic regression)
    7  paired comparison (episode i of every dataset has identical terrain parameters and joint targets)

If ``<out>/notes.md`` exists (or ``--notes`` is given) its text is appended to the report as hand-written notes.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process

import argparse  # noqa: E402
import csv  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from somato.data.episode import episode_paths  # noqa: E402
from somato.geometry.layout import SensorLayout  # noqa: E402
from somato.utils.config import load_yaml  # noqa: E402

G = 9.81
# "suspicious" levels (tested on the vector norm for acc and gyro, per element for joint velocity and pressure)
VEL_LIM, GYRO_LIM, ACC_LIM, PRESSURE_LIM = 8.0, 20.0, 50.0, 1.0e5
CONTACT_PA, ACTIVE_PA = 1.0, 50.0  # a taxel is "in contact" above 1 Pa and "active" above 50 Pa
Z_HIGH = 0.20  # [m] body z counted as a large vertical excursion (typical link height is 0.04-0.11 m)
TRACK_BIG = 0.5  # [rad] |pos - target| counted as a large tracking error
SAT_FRAC = 0.99  # a joint velocity / torque is "at its limit" when within 1% of the limit (pile-up at a clamp)
LO, PER, NDEC = -6, 100, 15  # log10 histogram: 1e-6 .. 1e9, 100 bins per decade
NB = PER * NDEC
PCTS = (50.0, 90.0, 99.0, 99.9, 99.99)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]  # blue, orange, aqua, magenta (categorical slots 1, 2, 3, 5)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2dd"
MARKERS = ["o", "^", "s", "D"]
LINESTYLES = ["-", "--", ":", "-."]


# --------------------------------------------------------------------------------------------- numerics helpers
def hist_abs(x) -> np.ndarray:
    """Counts of |x| in log-spaced bins; element 0 collects |x| < 1e-6 (including exact zeros)."""
    a = np.abs(np.asarray(x, dtype=np.float64)).ravel()
    out = np.zeros(NB + 1, np.int64)
    big = a >= 10.0 ** LO
    out[0] = a.size - int(big.sum())
    if big.any():
        k = np.clip(np.floor((np.log10(a[big]) - LO) * PER).astype(np.int64), 0, NB - 1)
        out[1:] += np.bincount(k, minlength=NB)
    return out


def hist_percentile(h: np.ndarray, q: float) -> float:
    """q-th percentile (0-100) of a ``hist_abs`` histogram, log-interpolated inside the bin."""
    c = np.cumsum(h)
    if c[-1] == 0:
        return float("nan")
    target = q / 100.0 * c[-1]
    i = int(np.searchsorted(c, target))
    if i == 0:
        return 0.0
    frac = (target - c[i - 1]) / max(h[i], 1)
    return float(10.0 ** (LO + (i - 1 + min(max(frac, 0.0), 1.0)) / PER))


def hist_cdf(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(upper bin edges, cumulative fraction) of a ``hist_abs`` histogram, without the underflow bin."""
    edges = 10.0 ** (LO + (np.arange(NB) + 1) / PER)
    return edges, (h[0] + np.cumsum(h[1:])) / h.sum()


def welch(x: np.ndarray, fs: float, nper: int) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch PSD (Hann window, 50 percent overlap, mean removed per segment) along the last axis."""
    x = np.asarray(x, dtype=np.float64)
    hop = nper // 2
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nper) / nper)
    seg = np.stack([x[..., s:s + nper] for s in range(0, x.shape[-1] - nper + 1, hop)], axis=-2)
    seg = seg - seg.mean(-1, keepdims=True)
    spec = np.abs(np.fft.rfft(seg * w, axis=-1)) ** 2 / (fs * (w ** 2).sum())
    spec[..., 1:-1] *= 2.0
    return np.fft.rfftfreq(nper, 1.0 / fs), spec.mean(-2)


def longest_run(b: np.ndarray) -> int:
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)  # noqa: E731  (no ties expected for float data)
    return pearson(rank(a[m]), rank(b[m]))


def slope_origin(x, y) -> float:
    """Least-squares slope of y = k x through the origin."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return float((x[m] * y[m]).sum() / (x[m] ** 2).sum())


# --------------------------------------------------------------------------------------------- per-episode worker
_CTX: dict = {}


def _init(ctx: dict) -> None:
    _CTX.update(ctx)


def analyze_episode(path: Path) -> dict:
    """Everything the report needs from one episode: histograms, maxima, counts, per-episode features, spectra."""
    c = _CTX
    area, M, weight = c["area"], c["M"], c["weight"]
    ch_t, ch_j, ch_i = c["channels"]["tactile"], c["channels"]["joint"], c["channels"]["imu"]
    fs_t, fs_j, fs_i, latent = c["fs_tactile"], c["fs_joint"], c["fs_imu"], c["latent_hz"]
    r: dict = {"ep": int(Path(path).stem.split("_")[1])}
    hist: dict[str, np.ndarray] = {}
    mx: dict[str, float] = {}

    with np.load(path) as f:
        tac = f["data/tactile"].astype(np.float32)
        jraw = f["data/joint"]
        imu = f["data/imu"].astype(np.float32)
        pos, quat = f["body_pos"].astype(np.float32), f["body_quat"].astype(np.float32)
        in_contact, nforce = f["label/in_contact"].astype(np.float32), f["label/normal_force"].astype(np.float32)
        r["terrain"], r["friction"] = int(f["label/terrain"]), float(f["param/friction"])
        jnt = jraw.astype(np.float32)
    r["target"] = jraw[..., ch_j.index("target")].reshape(-1, jraw.shape[2])  # raw dtype, for the paired comparison
    T, S, N, _ = tac.shape
    Sj, Si = jnt.shape[1], imu.shape[1]
    r["nonfinite"] = int(sum((~np.isfinite(a)).sum() for a in (tac, jnt, imu, pos, quat)))
    if r["nonfinite"]:
        tac, jnt, imu = (np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0) for a in (tac, jnt, imu))

    # ---- tactile (500 Hz): shape [T*S, N]
    p = tac[..., ch_t.index("normal")].reshape(T * S, N)
    sx = tac[..., ch_t.index("shear_x")].reshape(T * S, N)
    sy = tac[..., ch_t.index("shear_y")].reshape(T * S, N)
    contact, active = p > CONTACT_PA, p > ACTIVE_PA
    sh = np.hypot(sx, sy)
    hist["tactile/normal"], hist["tactile/shear_x"], hist["tactile/shear_y"] = (
        hist_abs(p[contact]), hist_abs(sx[contact]), hist_abs(sy[contact]))
    mx["tactile/normal"] = float(p.max())
    mx["tactile/shear_x"] = float(np.abs(sx[contact]).max()) if contact.any() else 0.0
    mx["tactile/shear_y"] = float(np.abs(sy[contact]).max()) if contact.any() else 0.0
    r["n_tac"], r["n_contact"], r["n_p_hi"] = p.size, int(contact.sum()), int((p > PRESSURE_LIM).sum())
    r["n_shear_no_normal"] = int(((~contact) & (sh > 1.0)).sum())  # shear > 1 Pa on a taxel with p <= 1 Pa
    r["max_shear_no_normal"] = float(sh[~contact].max()) if (~contact).any() else 0.0
    r["active_mean"] = float(active.sum() / (T * S))
    fl = p @ M  # [T*S, links] normal force per link [N] from the taxels
    r["f_over_w"] = float(fl.sum(1).mean() / weight)
    r["f_over_w_label"] = float(nforce.sum(1).mean() / weight)
    if active.any():
        pm, sm, am = p[active], sh[active], np.broadcast_to(area, p.shape)[active]
        r["ratio_mean"] = float((sm / pm).mean())  # mean shear/normal over active taxels
        r["ratio_fw"] = float((sm * am).sum() / (pm * am).sum())  # force-weighted (sum |shear| A / sum normal A)
    else:
        r["ratio_mean"] = r["ratio_fw"] = float("nan")
    r["ratio_mean_1pa"] = float((sh[contact] / p[contact]).mean()) if contact.any() else float("nan")
    hi = p > 5000.0  # isolated dropouts: a sample <= 1 Pa between two samples above 5 kPa on the same taxel
    sandwich = hi[:-2] & hi[2:]
    r["n_sandwich"], r["n_dropout"] = int(sandwich.sum()), int((sandwich & (p[1:-1] <= CONTACT_PA)).sum())
    mxp = np.maximum(p[1:], p[:-1])  # relative sample-to-sample pressure change where either sample is active
    pair = mxp > ACTIVE_PA
    r["rel_sum"] = float((np.abs(np.diff(p, axis=0))[pair] / mxp[pair]).sum())
    r["rel_n"] = int(pair.sum())
    ever = active.any(0)  # on/off chatter of the active-taxel state at 500 Hz
    r["toggle_rate"] = float((active[1:] != active[:-1])[:, ever].sum() / max(ever.sum() * (T * S / fs_t), 1e-9))

    # ---- joints (250 Hz): [T*Sj, 15]
    jflat = jnt.reshape(T * Sj, jnt.shape[2], jnt.shape[3])
    jpos, jvel, jtq, jtgt = (jflat[..., ch_j.index(k)] for k in ("pos", "vel", "torque", "target"))
    err = np.abs(jpos - jtgt)
    for name, a in (("pos", jpos), ("vel", jvel), ("torque", jtq), ("target", jtgt), ("track_err", err)):
        hist[f"joint/{name}"] = hist_abs(a)
        mx[f"joint/{name}"] = float(np.abs(a).max())
    av = np.abs(jvel)
    kv, jv = divmod(int(av.argmax()), av.shape[1])
    at_limit = lambda a, lim: (a >= SAT_FRAC * lim) & (a <= (2 - SAT_FRAC) * lim)  # noqa: E731  (within 1% of lim)
    vel_sat, tq_sat, err_big = at_limit(av, c["vel_limit"]), at_limit(np.abs(jtq), c["effort_limit"]), err > TRACK_BIG
    r.update(max_vel=float(av.max()), t_max_vel=kv / fs_j, joint_max_vel=jv, n_vel_hi=int((av > VEL_LIM).sum()),
             n_joint=av.size, torque_abs_mean=float(np.abs(jtq).mean()), track_err_mean=float(err.mean()),
             n_vel_sat=int(vel_sat.sum()), n_tq_sat=int(tq_sat.sum()), n_err_big=int(err_big.sum()),
             vel_hi_j=(av > VEL_LIM).sum(0), vel_max_j=av.max(0), vel_sat_j=vel_sat.sum(0), tq_sat_j=tq_sat.sum(0),
             err_big_j=err_big.sum(0))

    # ---- IMU (250 Hz): [T*Si, 6]
    iflat = imu.reshape(T * Si, -1)
    acc, gyro = iflat[:, [ch_i.index(k) for k in ("acc_x", "acc_y", "acc_z")]], iflat[:, [
        ch_i.index(k) for k in ("gyro_x", "gyro_y", "gyro_z")]]
    an, gn = np.linalg.norm(acc, axis=1), np.linalg.norm(gyro, axis=1)
    for i, k in enumerate(ch_i):
        hist[f"imu/{k}"], mx[f"imu/{k}"] = hist_abs(iflat[:, i]), float(np.abs(iflat[:, i]).max())
    hist["imu/acc_norm"], hist["imu/gyro_norm"] = hist_abs(an), hist_abs(gn)
    mx["imu/acc_norm"], mx["imu/gyro_norm"] = float(an.max()), float(gn.max())
    kg, ka = int(gn.argmax()), int(an.argmax())
    r.update(max_gyro=float(gn.max()), t_max_gyro=kg / fs_i, max_acc=float(an.max()), t_max_acc=ka / fs_i,
             n_gyro_hi=int((gn > GYRO_LIM).sum()), n_acc_hi=int((an > ACC_LIM).sum()), n_imu=an.size,
             acc_std=float(an.std()))

    # ---- poses and contact labels (latent rate)
    upz = 1.0 - 2.0 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)  # z component of every link's local z axis (wxyz)
    z = pos[..., 2]
    links = in_contact.sum(1)  # expected number of links in contact per latent step
    unloaded = nforce.sum(1) < 0.5 * weight
    cen = pos.mean(1)
    disp = cen[-1, :2] - cen[0, :2]
    r.update(min_up_head=float(upz[:, 0].min()), min_up_any=float(upz.min()), max_z=float(z.max()),
             links_in_contact=float(links.mean()), airborne_frac=float(unloaded.mean()),
             airborne_run_s=longest_run(unloaded) / latent, no_contact_frac=float((links == 0).mean()),
             speed=float(np.linalg.norm(disp) / (T / latent)), disp=disp.astype(np.float64),
             head_xy=pos[:, 0, :2].copy())
    for tag, k, sens in (("vel", kv, Sj), ("gyro", kg, Si), ("acc", ka, Si)):  # state at the time of each peak
        s = min(k // sens, T - 1)
        r[f"links_at_{tag}"], r[f"up_at_{tag}"], r[f"z_at_{tag}"] = float(links[s]), float(upz[s, 0]), float(z[s].max())

    # ---- spectra
    fq_i, r["psd_imu"] = welch(an, fs_i, c["nper_imu"])
    fq_f, P = welch(fl.T, fs_t, c["nper_force"])
    used = fl.mean(0) > 0.1  # links carrying load (> 0.1 N on average) in this episode
    r["psd_force_sum"], r["psd_force_n"] = P[used].sum(0), int(used.sum())
    r["hist"], r["max"] = hist, mx
    mx["tactile/shear_no_normal"] = r["max_shear_no_normal"]
    return r


def select_episodes(root: str, every: int) -> list[int]:
    """Episode numbers to analyse: every ``every``-th episode *within each terrain class* (classes cycle with a short
    period in the stored order, so a plain stride could hit a single class)."""
    by_class: dict[int, list[int]] = {}
    for p in episode_paths(root):
        with np.load(p) as f:
            by_class.setdefault(int(f["label/terrain"]), []).append(int(p.stem.split("_")[1]))
    return sorted(i for eps in by_class.values() for i in eps[::every])


# --------------------------------------------------------------------------------------------- dataset container
class Dataset:
    """Per-episode results of one dataset plus the merged histograms."""

    def __init__(self, root: str, label: str, episodes: list[int], workers: int, link_mass: float,
                 vel_limit: float, effort_limit: float):
        self.root = Path(root)
        self.meta = load_yaml(self.root / "meta.yaml")
        layout = SensorLayout.load(self.root / "layout.pt")
        g = layout.groups["tactile"]
        area, body = g.area.numpy().astype(np.float64), g.body_index.numpy()
        n_links = int(body.max()) + 1
        M = np.zeros((len(area), n_links), np.float32)
        M[np.arange(len(area)), body] = area
        rates = self.meta["rates"]
        self.fs = {k: float(v) for k, v in rates["sensors"].items()}
        self.latent_hz = float(rates["latent_hz"])
        self.control_hz = float(rates.get("control_hz", 0.0))
        self.terrain_names = list(self.meta["terrain_names"])
        self.simulator = str(self.meta.get("info", {}).get("simulator", self.root.name))
        self.label, self.vel_limit, self.effort_limit = label, vel_limit, effort_limit
        self.weight = n_links * link_mass * G
        paths = [self.root / "episodes" / f"ep_{i:06d}.npz" for i in episodes]
        ctx = dict(area=area, M=M, weight=self.weight, fs_tactile=self.fs["tactile"], fs_joint=self.fs["joint"],
                   fs_imu=self.fs["imu"], latent_hz=self.latent_hz, nper_imu=256, nper_force=512,
                   vel_limit=vel_limit, effort_limit=effort_limit,
                   channels={k: self.meta["groups"][k]["channels"] for k in ("tactile", "joint", "imu")})
        t0 = time.time()
        with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(ctx,)) as pool:
            self.results = list(pool.imap(analyze_episode, paths, chunksize=2))
        self.seconds = time.time() - t0
        self.hist = {k: np.sum([r["hist"][k] for r in self.results], axis=0) for k in self.results[0]["hist"]}
        self.max = {k: max(r["max"][k] for r in self.results) for k in self.results[0]["max"]}
        self.freq_imu = np.fft.rfftfreq(ctx["nper_imu"], 1.0 / self.fs["imu"])
        self.freq_force = np.fft.rfftfreq(ctx["nper_force"], 1.0 / self.fs["tactile"])
        self.duration = len(self.results[0]["head_xy"]) / self.latent_hz

    def __len__(self) -> int:
        return len(self.results)

    def col(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.results])

    def total(self, key: str) -> int:
        return int(self.col(key).sum())

    def cls(self, c: int) -> np.ndarray:
        return self.col("terrain") == c

    def vec(self, key: str) -> np.ndarray:
        """Sum of a per-episode vector (e.g. per joint) over the analysed episodes, or its max for ``*max*`` keys."""
        a = np.array([r[key] for r in self.results])
        return a.max(0) if "max" in key else a.sum(0)

    def episodes_with(self, key: str) -> int:
        return int((self.col(key) > 0).sum())

    def psd_imu(self, mask=None) -> np.ndarray:
        P = np.array([r["psd_imu"] for r in self.results])
        return P[slice(None) if mask is None else mask].mean(0)

    def psd_force(self, mask=None) -> np.ndarray:
        m = slice(None) if mask is None else mask
        S = np.array([r["psd_force_sum"] for r in self.results])[m].sum(0)
        return S / max(self.col("psd_force_n")[m].sum(), 1)


# --------------------------------------------------------------------------------------------- classifiers
def standardize(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-12
    f = lambda X: np.nan_to_num((X - mu) / sd, nan=0.0)  # noqa: E731  (missing values -> training mean)
    return f(Xtr), f(Xte)


def fit_logreg(X: np.ndarray, y: np.ndarray, k: int, lam: float = 1e-2, iters: int = 3000, lr: float = 0.5):
    """Multinomial logistic regression with L2 penalty, full-batch gradient descent (convex, so this converges)."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    W, Y = np.zeros((d + 1, k)), np.eye(k)[y]
    for _ in range(iters):
        Z = Xb @ W
        P = np.exp(Z - Z.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        grad = Xb.T @ (P - Y) / n
        grad[:-1] += lam * W[:-1]
        W -= lr * grad
    return W


def predict_logreg(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (np.hstack([X, np.ones((len(X), 1))]) @ W).argmax(1)


def predict_centroid(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, k: int) -> np.ndarray:
    cents = np.stack([Xtr[ytr == c].mean(0) for c in range(k)])
    return ((Xte[:, None, :] - cents[None]) ** 2).sum(-1).argmin(1)


def stratified_folds(y: np.ndarray, nfold: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(y), int)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        fold[idx] = np.arange(len(idx)) % nfold
    return fold


def cross_validate(X: np.ndarray, y: np.ndarray, k: int, nfold: int = 5, seed: int = 0) -> dict:
    fold = stratified_folds(y, nfold, seed)
    acc = {"centroid": [], "logreg": []}
    pred = {"centroid": np.zeros_like(y), "logreg": np.zeros_like(y)}
    for f in range(nfold):
        tr, te = fold != f, fold == f
        Xtr, Xte = standardize(X[tr], X[te])
        pred["centroid"][te] = predict_centroid(Xtr, y[tr], Xte, k)
        pred["logreg"][te] = predict_logreg(fit_logreg(Xtr, y[tr], k), Xte)
        for m in acc:
            acc[m].append(float((pred[m][te] == y[te]).mean()))
    return {"acc": acc, "pred": pred}


def confusion(y: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y, pred), 1)
    return cm


FEATURES = ["shear/normal ratio", "mean |joint torque| [Nm]", "active taxels", "std of IMU |acc| [m/s2]",
            "centroid speed [m/s]"]


def feature_matrix(ds: Dataset) -> np.ndarray:
    return np.stack([ds.col("ratio_mean"), ds.col("torque_abs_mean"), ds.col("active_mean"), ds.col("acc_std"),
                     ds.col("speed")], axis=1)


# --------------------------------------------------------------------------------------------- markdown helpers
def g3(v, n: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{n}g}"


def pct(v: float, n: int = 3) -> str:
    return "n/a" if not np.isfinite(v) else f"{100 * v:.{n}g}%"


def table(headers: list[str], rows: list[list[str]]) -> str:
    esc = lambda c: str(c).replace("|", "\\|")  # noqa: E731  (a literal | would end the markdown cell)
    out = ["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def sidebyside(vals, fmt=g3) -> str:
    return " / ".join(fmt(v) for v in vals)


def mean_std(a: np.ndarray, n: int = 3) -> str:
    a = a[np.isfinite(a)]
    return "n/a" if a.size == 0 else f"{a.mean():.{n}g} ± {a.std():.{n}g}"


# --------------------------------------------------------------------------------------------- plotting
def style_axes(ax) -> None:
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def new_fig(nrows: int, ncols: int, w: float, h: float):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), facecolor="white", squeeze=False)
    for ax in axes.ravel():
        style_axes(ax)
    return fig, axes


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def plot_psd(dsets: list[Dataset], path: Path) -> None:
    fig, axes = new_fig(1, 2, 11, 4.2)
    specs = [("IMU |acc| (250 Hz)", lambda d: (d.freq_imu, d.psd_imu()), "PSD [(m/s$^2$)$^2$/Hz]"),
             ("Summed normal force per link (500 Hz, loaded links)", lambda d: (d.freq_force, d.psd_force()),
              "PSD [N$^2$/Hz]")]
    for ax, (title, getter, ylabel) in zip(axes[0], specs):
        for i, d in enumerate(dsets):
            fq, P = getter(d)
            ax.loglog(fq[1:-1], P[1:-1], color=COLORS[i], ls=LINESTYLES[i], lw=1.6, label=d.label)  # no DC / Nyquist bin
        ax.axvline(dsets[0].latent_hz / 2, color=INK2, lw=0.8, ls=":")
        ax.text(dsets[0].latent_hz / 2 * 0.96, 0.97, "latent Nyquist 25 Hz", color=INK2, fontsize=7, rotation=90,
                va="top", ha="right", transform=ax.get_xaxis_transform())
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, path)


def plot_friction_ratio(dsets: list[Dataset], path: Path) -> None:
    rows = [("ratio_mean", "mean shear/normal over active taxels"), ("ratio_fw", "force-weighted shear/normal")]
    fig, axes = new_fig(len(rows), len(dsets), 4.6 * len(dsets), 8.2)
    for i, d in enumerate(dsets):
        mu = d.col("friction")
        for j, (key, ylabel) in enumerate(rows):
            ax, y = axes[j, i], d.col(key)
            ax.scatter(mu, y, s=9, color=COLORS[i], alpha=0.45, linewidths=0)
            lim = max(np.nanmax(mu), np.nanmax(y)) * 1.05
            ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--", label="ratio = friction")
            k = slope_origin(mu, y)
            ax.plot([0, lim], [0, k * lim], color=INK, lw=1.2, label=f"fit through 0: {k:.2f} x")
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel("friction parameter")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{d.label}: r = {pearson(mu, y):.3f}, Spearman = {spearman(mu, y):.3f}", fontsize=9)
            ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, path)


def plot_speed(dsets: list[Dataset], path: Path) -> None:
    fig, axes = new_fig(1, 2, 11, 4.2)
    ax, nd, names = axes[0, 0], len(dsets), dsets[0].terrain_names
    rng = np.random.default_rng(0)
    for i, d in enumerate(dsets):
        off = (i - (nd - 1) / 2) * 0.28
        for c in range(len(names)):
            v = d.col("speed")[d.cls(c)]
            ax.scatter(c + off + rng.uniform(-0.07, 0.07, len(v)), v, s=6, color=COLORS[i], alpha=0.3, linewidths=0)
            ax.errorbar(c + off, v.mean(), yerr=v.std(), color=INK, marker=MARKERS[i], ms=5, mfc=COLORS[i], mec=INK,
                        capsize=3, lw=1.0, label=d.label if c == 0 else None)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("centroid speed [m/s]")
    ax.set_title("Centroid speed per class (mean ± std, points = episodes)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 1]
    for i, d in enumerate(dsets):
        mu, v = d.col("friction"), d.col("speed")
        ax.scatter(mu, v, s=9, color=COLORS[i], alpha=0.5, marker=MARKERS[i], linewidths=0,
                   label=f"{d.label} (r = {pearson(mu, v):.2f})")
    ax.set_xlabel("friction parameter")
    ax.set_ylabel("centroid speed [m/s]")
    ax.set_title("Centroid speed vs friction", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    save(fig, path)


def plot_paired(a: Dataset, b: Dataset, path: Path) -> None:
    fig, axes = new_fig(2, 2, 9.5, 8.4)
    panels = [("speed", "centroid displacement / duration [m/s]"), ("torque_abs_mean", "mean |joint torque| [Nm]"),
              ("ratio_mean", "mean shear/normal ratio (active taxels)")]
    for ax, (key, title) in zip(axes.ravel()[:3], panels):
        x, y = a.col(key), b.col(key)
        ax.scatter(x, y, s=9, color=COLORS[1], alpha=0.5, linewidths=0)
        lim = np.nanmax([x, y]) * 1.05
        ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel(a.label)
        ax.set_ylabel(b.label)
        ax.set_title(f"{title}\nr = {pearson(x, y):.3f}", fontsize=9)
    ax = axes[1, 1]
    for i, d in enumerate((a, b)):
        e, cdf = hist_cdf(d.hist["joint/track_err"])
        ax.semilogx(e, cdf, color=COLORS[i], ls=LINESTYLES[i], lw=1.6, label=d.label)
    ax.set_xlim(1e-4, 2)
    ax.set_xlabel("joint tracking error |pos - target| [rad]")
    ax.set_ylabel("cumulative fraction of samples")
    ax.set_title("Joint tracking error (all samples)", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, path)


def pick_examples(a: Dataset) -> list[int]:
    """Positions (in the analysed list) of the episodes closest to the 10th, 50th and 90th friction percentile."""
    mu = a.col("friction")
    return [int(np.abs(mu - np.quantile(mu, q)).argmin()) for q in (0.1, 0.5, 0.9)]


def plot_trajectories(dsets: list[Dataset], picks: list[int], path: Path) -> None:
    fig, axes = new_fig(1, len(picks), 4.2 * len(picks), 4.6)
    for ax, i in zip(axes[0], picks):
        r0 = dsets[0].results[i]
        for j, d in enumerate(dsets):
            xy = d.results[i]["head_xy"]
            ax.plot(xy[:, 0], xy[:, 1], color=COLORS[j], ls=LINESTYLES[j], lw=1.6, label=d.label)
            ax.plot(*xy[0], marker="o", ms=5, color=COLORS[j], mec=INK)
            ax.plot(*xy[-1], marker="s", ms=5, color=COLORS[j], mec=INK)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"ep {r0['ep']}: {dsets[0].terrain_names[r0['terrain']]}, friction {r0['friction']:.2f}",
                     fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    save(fig, path)


# --------------------------------------------------------------------------------------------- report sections
CHANNELS = [  # (histogram key, display name, unit)
    ("tactile/normal", "tactile normal (p > 1 Pa only)", "Pa"), ("tactile/shear_x", "tactile shear_x (p > 1 Pa)", "Pa"),
    ("tactile/shear_y", "tactile shear_y (p > 1 Pa)", "Pa"), ("joint/pos", "joint pos", "rad"),
    ("joint/vel", "joint vel", "rad/s"), ("joint/torque", "joint torque", "Nm"), ("joint/target", "joint target", "rad"),
    ("joint/track_err", "joint tracking error |pos - target|", "rad"), ("imu/acc_x", "imu acc_x", "m/s2"),
    ("imu/acc_y", "imu acc_y", "m/s2"), ("imu/acc_z", "imu acc_z", "m/s2"), ("imu/acc_norm", "imu |acc| (norm)", "m/s2"),
    ("imu/gyro_x", "imu gyro_x", "rad/s"), ("imu/gyro_y", "imu gyro_y", "rad/s"), ("imu/gyro_z", "imu gyro_z", "rad/s"),
    ("imu/gyro_norm", "imu |gyro| (norm)", "rad/s")]


def section_amplitude(ds: list[Dataset]) -> str:
    rows = []
    for key, name, unit in CHANNELS:
        cells = [sidebyside([hist_percentile(d.hist[key], q) for d in ds]) for q in PCTS]
        rows.append([name, unit, *cells, sidebyside([d.max[key] for d in ds])])
    s = "## 1. Amplitude statistics of |x|\n\n"
    s += (f"Order in every cell: {' / '.join(d.label for d in ds)}. Percentiles are over all samples at the native "
          "sensor rate (tactile 500 Hz, joint and IMU 250 Hz), except tactile channels, which use in-contact taxel "
          "samples only; they come from log-spaced histograms (about 2% relative resolution), the max is exact. "
          "Joint and IMU values are stored as float16, so tracking-error values near 1e-3 are quantisation-limited.\n\n")
    s += table(["channel", "unit", "p50", "p90", "p99", "p99.9", "p99.99", "max"], rows)
    exc = []

    def row(name, nkey, dkey, ep_key):
        n = [d.total(nkey) for d in ds]
        den = [d.total(dkey) for d in ds]
        exc.append([name, sidebyside([a / b for a, b in zip(n, den)], pct), sidebyside(n, lambda v: f"{v:d}"),
                    sidebyside([d.episodes_with(ep_key) for d in ds], lambda v: f"{v:d}") + f" of {len(ds[0])}"])

    row(f"|joint vel| > {VEL_LIM:g} rad/s", "n_vel_hi", "n_joint", "n_vel_hi")
    row(f"|gyro| (norm) > {GYRO_LIM:g} rad/s", "n_gyro_hi", "n_imu", "n_gyro_hi")
    row(f"|acc| (norm) > {ACC_LIM:g} m/s2", "n_acc_hi", "n_imu", "n_acc_hi")
    row(f"taxel p > {PRESSURE_LIM / 1e3:g} kPa (of in-contact samples)", "n_p_hi", "n_contact", "n_p_hi")
    row(f"taxel p > {PRESSURE_LIM / 1e3:g} kPa (of all taxel samples)", "n_p_hi", "n_tac", "n_p_hi")
    row(f"|joint vel| within 1% of the velocity limit ({ds[0].vel_limit:g} rad/s)", "n_vel_sat", "n_joint", "n_vel_sat")
    row(f"|joint torque| within 1% of the effort limit ({ds[0].effort_limit:g} Nm)", "n_tq_sat", "n_joint", "n_tq_sat")
    row(f"|joint pos - target| > {TRACK_BIG:g} rad", "n_err_big", "n_joint", "n_err_big")
    s += "\n### Exceedance of suspicious levels\n\n"
    s += table(["level", "fraction of samples", "samples", "episodes affected"], exc)
    jr = []
    for j in range(len(ds[0].vec("vel_max_j"))):
        n = ds[0].total("n_joint") // len(ds[0].vec("vel_max_j"))
        jr.append([j, sidebyside([d.vec("vel_hi_j")[j] for d in ds], lambda v: f"{int(v)}"),
                   sidebyside([d.vec("vel_max_j")[j] for d in ds]),
                   sidebyside([d.vec("vel_sat_j")[j] / n for d in ds], pct),
                   sidebyside([d.vec("tq_sat_j")[j] / n for d in ds], pct),
                   sidebyside([d.vec("err_big_j")[j] / n for d in ds], pct)])
    s += ("\n### Per joint\n\nSamples per joint: " + f"{ds[0].total('n_joint') // len(ds[0].vec('vel_max_j'))}. "
          "`vel at limit` / `torque at limit` = within 1% of the limits given by `--velocity_limit` / `--effort_limit`.\n\n")
    s += table(["joint", f"samples |vel| > {VEL_LIM:g}", "max |vel| [rad/s]", "vel at limit", "torque at limit",
                f"|pos - target| > {TRACK_BIG:g} rad"], jr)
    other = [
        ["in-contact (p > 1 Pa) fraction of taxel samples", sidebyside([d.total("n_contact") / d.total("n_tac")
                                                                       for d in ds], pct)],
        ["shear > 1 Pa on a taxel with p <= 1 Pa (fraction of non-contact samples)", sidebyside(
            [d.total("n_shear_no_normal") / (d.total("n_tac") - d.total("n_contact")) for d in ds], pct)],
        ["number of such samples", sidebyside([d.total("n_shear_no_normal") for d in ds], lambda v: f"{v:d}")],
        ["largest shear on such a taxel [Pa]", sidebyside([d.max["tactile/shear_no_normal"] for d in ds])],
        ["median over episodes of the per-episode max |acc| [m/s2]", sidebyside([np.median(d.col("max_acc")) for d in ds])],
        ["median over episodes of the per-episode max |gyro| [rad/s]", sidebyside([np.median(d.col("max_gyro")) for d in ds])],
        ["median over episodes of the per-episode max |joint vel| [rad/s]", sidebyside([np.median(d.col("max_vel")) for d in ds])],
        ["non-finite values in the analysed episodes", sidebyside([d.total("nonfinite") for d in ds], lambda v: f"{v:d}")],
    ]
    s += "\n### Other checks\n\n" + table(["check", "value"], other)
    return s


def top10_table(d: Dataset, key: str, tkey: str, tag: str, unit: str, extra: str | None = None) -> tuple[str, list[int]]:
    order = np.argsort(-d.col(key))[:10]
    names = d.terrain_names
    headers = ["rank", "ep", "class", "friction", f"max [{unit}]", "t [s]"] + (["joint"] if extra else []) + [
        "min up(head)", "max z [m]", "airborne %", "links in contact at peak", "up(head) at peak"]
    rows = []
    for rank, i in enumerate(order, 1):
        r = d.results[i]
        rows.append([rank, r["ep"], names[r["terrain"]], f"{r['friction']:.3f}", g3(r[key]), f"{r[tkey]:.2f}"]
                    + ([r[extra]] if extra else []) + [f"{r['min_up_head']:.2f}", f"{r['max_z']:.3f}",
                                                       f"{100 * r['airborne_frac']:.0f}", f"{r['links_at_' + tag]:.1f}",
                                                       f"{r['up_at_' + tag]:.2f}"])
    return table(headers, rows), [d.results[i]["ep"] for i in order]


def section_events(ds: list[Dataset]) -> str:
    s = ("## 2. Extreme events\n\nPer episode: `min up(head)` is the minimum over time of the world-z component of the "
         "head link's (link_0) local z axis from `body_quat` (+1 upright, < 0 upside-down); `max z` is the highest "
         "body-origin z of any link; `airborne %` is the fraction of latent steps whose summed per-link normal force "
         f"(`label/normal_force`) is below half the robot weight. Large vertical excursion = max z > {Z_HIGH:g} m.\n\n")
    names = ds[0].terrain_names
    ctx_rows = []
    for d in ds:
        ctx_rows.append([d.label, f"{(d.col('min_up_head') < 0).sum()}", f"{(d.col('min_up_any') < 0).sum()}",
                         g3(np.median(d.col("max_z"))), g3(np.percentile(d.col("max_z"), 95)), g3(d.col("max_z").max()),
                         f"{(d.col('max_z') > Z_HIGH).sum()}", f"{(d.col('airborne_frac') > 0.1).sum()}",
                         g3(d.col("airborne_run_s").max()), f"{(d.col('no_contact_frac') > 0).sum()}"])
    s += "### Episode-level context (all analysed episodes)\n\n"
    s += table(["dataset", "episodes head flipped (min up < 0)", "any link flipped", "median max z [m]",
                "p95 max z [m]", "max max z [m]", f"episodes with max z > {Z_HIGH:g} m", "episodes airborne > 10% of steps",
                "longest airborne run [s]", "episodes with a step of zero links in contact"], ctx_rows)
    fl = [[names[c], sidebyside([int(((d.col("min_up_head") < 0) & d.cls(c)).sum()) for d in ds], lambda v: f"{v:d}"),
           sidebyside([int(((d.col("max_z") > Z_HIGH) & d.cls(c)).sum()) for d in ds], lambda v: f"{v:d}"),
           sidebyside([g3(np.median(d.col("max_z")[d.cls(c)])) for d in ds], str)] for c in range(len(names))]
    s += "\n### Flips and tall excursions per class\n\n" + table(
        ["class", "head flipped", f"max z > {Z_HIGH:g} m", "median max z [m]"], fl)
    s += "\nRank correlations over all analysed episodes (" + "; ".join(
        f"{d.label}: max |joint vel| vs airborne fraction {spearman(d.col('max_vel'), d.col('airborne_frac')):.2f}, "
        f"max |gyro| vs flip depth (-min up(head)) {spearman(d.col('max_gyro'), -d.col('min_up_head')):.2f}" for d in ds)
    s += ").\n"
    tops: dict[str, list[list[int]]] = {"vel": [], "gyro": []}
    for d in ds:
        s += f"\n### {d.label}: top-10 episodes by max |joint vel|\n\n"
        t, eps = top10_table(d, "max_vel", "t_max_vel", "vel", "rad/s", extra="joint_max_vel")
        s += t
        tops["vel"].append(eps)
        s += f"\n### {d.label}: top-10 episodes by max |gyro| (norm)\n\n"
        t, eps = top10_table(d, "max_gyro", "t_max_gyro", "gyro", "rad/s")
        s += t
        tops["gyro"].append(eps)
    if len(ds) >= 2:
        s += ("\nOverlap of the top-10 lists between the first two datasets (the episodes are paired): "
              f"joint vel {len(set(tops['vel'][0]) & set(tops['vel'][1]))}/10, "
              f"gyro {len(set(tops['gyro'][0]) & set(tops['gyro'][1]))}/10.\n")
    return s


def section_contact(ds: list[Dataset]) -> str:
    names = ds[0].terrain_names
    a_rows, b_rows = [], []
    for c, n in enumerate(names):
        m = [d.cls(c) for d in ds]
        a_rows.append([n, sidebyside([d.col("links_in_contact")[k].mean() for d, k in zip(ds, m)]),
                       sidebyside([d.col("active_mean")[k].mean() for d, k in zip(ds, m)]),
                       sidebyside([d.col("f_over_w")[k].mean() for d, k in zip(ds, m)]),
                       sidebyside([d.col("f_over_w_label")[k].mean() for d, k in zip(ds, m)])])
    a_rows.append(["all", sidebyside([d.col("links_in_contact").mean() for d in ds]),
                   sidebyside([d.col("active_mean").mean() for d in ds]),
                   sidebyside([d.col("f_over_w").mean() for d in ds]),
                   sidebyside([d.col("f_over_w_label").mean() for d in ds])])
    for c, n in enumerate(names):
        m = [d.cls(c) for d in ds]
        b_rows.append([n, f"{ds[0].col('friction')[m[0]].mean():.3f} ({ds[0].col('friction')[m[0]].min():.2f}-"
                          f"{ds[0].col('friction')[m[0]].max():.2f})",
                       sidebyside([mean_std(d.col("ratio_mean")[k]) for d, k in zip(ds, m)], str),
                       sidebyside([mean_std(d.col("ratio_fw")[k]) for d, k in zip(ds, m)], str),
                       sidebyside([np.nanmean(d.col("ratio_mean")[k]) / d.col("friction")[k].mean()
                                   for d, k in zip(ds, m)], lambda v: f"{v:.2f}")])
    s = "## 3. Contact pattern and effective friction\n\n"
    s += (f"Order in every cell: {' / '.join(d.label for d in ds)}. Weight = {ds[0].weight:.2f} N "
          f"(16 links x 0.25 kg x {G} m/s2). `links in contact` is the per-step sum of `label/in_contact` "
          f"(fraction of physics sub-steps in contact), `active taxels` the mean number of taxels with p > {ACTIVE_PA:g} Pa "
          "per 500 Hz sample, `F/W (taxels)` sum(normal x area) / weight, `F/W (labels)` the summed "
          "`label/normal_force` / weight.\n\n")
    s += table(["class", "links in contact", "active taxels", "F/W (taxels)", "F/W (labels)"], a_rows)
    s += ("\n### Effective friction\n\n"
          f"Shear/normal = sqrt(shear_x^2 + shear_y^2) / normal over *active* taxels (p > {ACTIVE_PA:g} Pa; "
          "the 1 Pa in-contact set gives nearly the same mean, see below), averaged per episode; "
          "`force-weighted` = sum(|shear| A) / sum(normal A). `ratio / friction` uses the class means.\n\n")
    s += table(["class", "friction param mean (range)", "mean shear/normal (mean ± std over episodes)",
                "force-weighted shear/normal", "ratio / friction"], b_rows)
    corr = []
    for d in ds:
        mu = d.col("friction")
        corr.append([d.label, f"{pearson(mu, d.col('ratio_mean')):.3f}", f"{spearman(mu, d.col('ratio_mean')):.3f}",
                     f"{slope_origin(mu, d.col('ratio_mean')):.3f}", f"{pearson(mu, d.col('ratio_fw')):.3f}",
                     f"{slope_origin(mu, d.col('ratio_fw')):.3f}", g3(np.nanmean(d.col("ratio_mean_1pa"))),
                     g3(np.nanmean(d.col("ratio_mean")))])
    s += "\n" + table(["dataset", "Pearson r (mean ratio vs friction)", "Spearman", "slope through 0",
                       "Pearson r (force-weighted)", "slope through 0 (force-weighted)",
                       "mean ratio, p > 1 Pa", "mean ratio, p > 50 Pa"], corr)
    s += "\nPlot: `friction_vs_shear_ratio.png`.\n"
    return s


def section_locomotion(ds: list[Dataset]) -> str:
    names = ds[0].terrain_names
    rows = [[n, sidebyside([mean_std(d.col("speed")[d.cls(c)], 3) for d in ds], str)] for c, n in enumerate(names)]
    rows.append(["all", sidebyside([mean_std(d.col("speed")) for d in ds], str)])
    s = "## 4. Locomotion\n\n"
    s += (f"Centroid speed = xy displacement of the mean body position between the first and last stored latent step, "
          f"divided by {ds[0].duration:g} s (order: {' / '.join(d.label for d in ds)}).\n\n")
    s += table(["class", "speed [m/s], mean ± std"], rows)
    corr = [[d.label, f"{pearson(d.col('friction'), d.col('speed')):.3f}", f"{spearman(d.col('friction'), d.col('speed')):.3f}",
             sidebyside([pearson(d.col("friction")[d.cls(c)], d.col("speed")[d.cls(c)]) for c in range(len(names))],
                        lambda v: f"{v:.2f}")] for d in ds]
    s += "\n" + table(["dataset", "Pearson r (speed vs friction)", "Spearman", "within-class Pearson r (" +
                       ", ".join(names) + ")"], corr)
    s += "\nPlot: `speed_by_class.png`.\n"
    return s


def peak_ratio(freq: np.ndarray, P: np.ndarray, f0: float) -> float:
    """Highest PSD within 1.5 Hz of f0 over the mean PSD 4-12 Hz away from f0 (narrow-band peak indicator)."""
    near, far = np.abs(freq - f0) <= 1.5, (np.abs(freq - f0) >= 4) & (np.abs(freq - f0) <= 12)
    return float(P[near].max() / P[far].mean())


def calm_mask(ds: list[Dataset]) -> np.ndarray:
    """Paired episodes in which no dataset shows a flipped head, > 2% unloaded steps or joint velocity above 1.1 x limit."""
    m = np.ones(len(ds[0]), bool)
    for d in ds:
        m &= (d.col("min_up_head") > 0) & (d.col("airborne_frac") < 0.02) & (d.col("max_vel") <= 1.1 * d.vel_limit)
    return m


def band_fraction(freq: np.ndarray, P: np.ndarray, f0: float) -> float:
    return float(P[freq > f0].sum() / P[1:].sum())


def section_spectra(ds: list[Dataset]) -> str:
    s = "## 5. Temporal character\n\n"
    s += (f"Welch PSD (Hann, 50% overlap, per-segment mean removed), averaged over episodes. IMU: |acc| at "
          f"{ds[0].fs['imu']:g} Hz, 256-sample segments (Nyquist {ds[0].fs['imu'] / 2:g} Hz). Force: summed taxel normal "
          f"force of each link at {ds[0].fs['tactile']:g} Hz, 512-sample segments, averaged over (episode, link) series "
          "whose mean force exceeds 0.1 N (Nyquist 250 Hz). The latent rate is 50 Hz (Nyquist 25 Hz). "
          f"Order in every cell: {' / '.join(d.label for d in ds)}.\n\n")
    freqs = [1, 2, 5, 10, 20, 25, 50, 100, 200]
    rows = []
    for name, get, fmax in (("IMU |acc| [(m/s2)^2/Hz]", lambda d: (d.freq_imu, d.psd_imu()), 125),
                            ("link force [N^2/Hz]", lambda d: (d.freq_force, d.psd_force()), 250)):
        for f in freqs:
            if f >= fmax:
                continue
            v = [float(np.interp(f, *get(d))) for d in ds]
            ratio = f"{v[1] / v[0]:.2f}" if len(v) > 1 and v[0] > 0 else "n/a"
            rows.append([name, f"{f}", sidebyside(v, lambda x: f"{x:.2e}"), ratio])
    s += table(["signal", "frequency [Hz]", "PSD", f"ratio {ds[1].label}/{ds[0].label}" if len(ds) > 1 else ""], rows)
    brow = []
    for name, get, f0s in (("IMU |acc|", lambda d: (d.freq_imu, d.psd_imu()), (25, 60)),
                           ("link force", lambda d: (d.freq_force, d.psd_force()), (25, 100))):
        for f0 in f0s:
            brow.append([name, f"> {f0} Hz", sidebyside([band_fraction(*get(d), f0) for d in ds], pct)])
    s += "\n### Fraction of (non-DC) power above a frequency\n\n" + table(["signal", "band", "fraction"], brow)
    f0 = ds[0].control_hz
    s += ("\n### Contact on/off chatter and control-rate line\n\n" + table(
        ["metric", "value"], [["active-taxel (p > 50 Pa) state toggles per taxel-second, taxels active at least once",
                               sidebyside([d.col("toggle_rate").mean() for d in ds], lambda v: f"{v:.1f}")],
                              ["isolated taxel dropouts: fraction of samples <= 1 Pa whose neighbours (+-2 ms) are both > 5 kPa",
                               sidebyside([d.total("n_dropout") / max(d.total("n_sandwich"), 1) for d in ds], pct)],
                              ["mean relative pressure change |dp| / max(p) between consecutive 500 Hz samples (active taxels)",
                               sidebyside([d.col("rel_sum").sum() / max(d.col("rel_n").sum(), 1) for d in ds],
                                          lambda v: f"{v:.2f}")],
                              [f"IMU |acc| PSD peak at the control rate ({f0:g} Hz) / PSD 4-12 Hz away",
                               sidebyside([peak_ratio(d.freq_imu, d.psd_imu(), f0) for d in ds], lambda v: f"{v:.2f}")],
                              [f"link-force PSD peak at {f0:g} Hz / PSD 4-12 Hz away",
                               sidebyside([peak_ratio(d.freq_force, d.psd_force(), f0) for d in ds], lambda v: f"{v:.2f}")]]))
    s += "\nPlot: `psd.png`.\n"
    return s


def section_separability(ds: list[Dataset]) -> tuple[str, dict]:
    names, k = ds[0].terrain_names, len(ds[0].terrain_names)
    feats = {d.label: feature_matrix(d) for d in ds}
    ys = {d.label: d.col("terrain").astype(int) for d in ds}
    s = "## 6. Class separability from engineered features\n\n"
    s += ("Features per episode: " + ", ".join(FEATURES) + ". Standardised with training statistics; 5-fold stratified "
          "cross-validation (seed 0) over episodes; nearest centroid (Euclidean on standardised features) and "
          f"multinomial logistic regression (L2 1e-2, numpy gradient descent). Chance = {100 / k:.0f}%.\n\n")
    cv = {d.label: cross_validate(feats[d.label], ys[d.label], k) for d in ds}
    rows = []
    for d in ds:
        a = cv[d.label]["acc"]
        rows.append([d.label, f"{100 * np.mean(a['centroid']):.1f}% ± {100 * np.std(a['centroid']):.1f}",
                     f"{100 * np.mean(a['logreg']):.1f}% ± {100 * np.std(a['logreg']):.1f}"])
    s += table(["dataset", "nearest centroid (5-fold mean ± std)", "logistic regression"], rows)
    s += "\n### Per-class recall of logistic regression (pooled over folds)\n\n"
    rows = []
    cms = {d.label: confusion(ys[d.label], cv[d.label]["pred"]["logreg"], k) for d in ds}
    for c, n in enumerate(names):
        rows.append([n, sidebyside([cms[d.label][c, c] / cms[d.label][c].sum() for d in ds], pct)])
    s += table(["class", "recall"], rows)
    for d in ds:
        s += f"\nConfusion matrix, {d.label} (rows = true class, columns = predicted):\n\n"
        s += table(["true \\ pred"] + names, [[n] + [int(v) for v in cms[d.label][c]] for c, n in enumerate(names)])
    s += "\n### Feature means by class\n\n"
    rows = []
    for j, fn in enumerate(FEATURES):
        rows.append([fn] + [sidebyside([np.nanmean(feats[d.label][ys[d.label] == c, j]) for d in ds]) for c in range(k)])
    s += f"Order in every cell: {' / '.join(d.label for d in ds)}.\n\n" + table(["feature"] + names, rows)
    cross = []
    for a in ds:
        for b in ds:
            if a is b:
                continue
            Xa, Xb = standardize(feats[a.label], feats[b.label])
            ya, yb = ys[a.label], ys[b.label]
            acc_c = float((predict_centroid(Xa, ya, Xb, k) == yb).mean())
            acc_l = float((predict_logreg(fit_logreg(Xa, ya, k), Xb) == yb).mean())
            cross.append([f"{a.label} -> {b.label}", pct(acc_c, 3), pct(acc_l, 3)])
    if cross:
        s += ("\n### Cross-simulator transfer (extra)\n\nTrain on all episodes of one dataset (standardised with its "
              "statistics), test on all episodes of the other (same terrain configurations, different physics).\n\n")
        s += table(["train -> test", "nearest centroid", "logistic regression"], cross)
    return s, {"cv": cv, "cross": cross}


def section_paired(ds: list[Dataset], picks: list[int]) -> str:
    a, b = ds[0], ds[1]
    s = f"## 7. Paired comparison ({a.label} vs {b.label})\n\n"
    same_t = bool(np.array_equal(a.col("terrain"), b.col("terrain")))
    same_f = bool(np.allclose(a.col("friction"), b.col("friction")))
    tdiff = np.array([np.abs(ra["target"].astype(np.float32) - rb["target"].astype(np.float32)).max()
                      for ra, rb in zip(a.results, b.results)])
    s += (f"Episodes {len(a)}; terrain class identical in all pairs: {same_t}; friction identical: {same_f}; "
          f"joint targets: max |target difference| over all pairs = {tdiff.max():.3g} rad "
          f"({int((tdiff == 0).sum())} of {len(tdiff)} pairs bit-identical).\n\n")
    rows = []
    for key, name in (("speed", "centroid speed [m/s]"), ("torque_abs_mean", "mean |joint torque| [Nm]"),
                      ("ratio_mean", "mean shear/normal ratio"), ("ratio_fw", "force-weighted shear/normal"),
                      ("track_err_mean", "mean |pos - target| [rad]"), ("active_mean", "active taxels"),
                      ("acc_std", "std of IMU |acc| [m/s2]")):
        x, y = a.col(key), b.col(key)
        m = np.isfinite(x) & np.isfinite(y)
        rows.append([name, g3(np.mean(x[m])), g3(np.mean(y[m])), g3(np.mean(y[m] - x[m])), g3(np.mean(np.abs(y[m] - x[m]))),
                     f"{pearson(x, y):.3f}", f"{spearman(x, y):.3f}"])
    s += table(["per-episode quantity", f"mean {a.label}", f"mean {b.label}", f"mean difference ({b.label} - {a.label})",
                "mean |difference|", "Pearson r", "Spearman"], rows)
    da, db = np.stack(a.col("disp")), np.stack(b.col("disp"))
    cos = (da * db).sum(1) / (np.linalg.norm(da, axis=1) * np.linalg.norm(db, axis=1) + 1e-12)
    dist = np.linalg.norm(da - db, axis=1)
    s += (f"\nNet centroid displacement vectors (world xy, same initial heading): median cosine between the two vectors "
          f"{np.median(cos):.3f} (10th percentile {np.percentile(cos, 10):.3f}); mean norm of the vector difference "
          f"{dist.mean():.3f} m ({a.label} mean displacement {np.linalg.norm(da, axis=1).mean():.3f} m, "
          f"{b.label} {np.linalg.norm(db, axis=1).mean():.3f} m).\n")
    slow = b.col("speed") < 0.6 * a.col("speed")
    s += (f"\n### Episodes where {b.label} is much slower than {a.label}\n\nSlow = {b.label} centroid speed < 60% of the "
          f"paired {a.label} speed: {int(slow.sum())} of {len(slow)} episodes (per class: "
          + ", ".join(f"{n} {int((slow & a.cls(c)).sum())}" for c, n in enumerate(a.terrain_names))
          + "). Cells: slow episodes / other episodes.\n\n")
    metrics = [("head flipped (min up < 0)", lambda d: d.col("min_up_head") < 0, pct),
               ("> 10% of steps below half the weight", lambda d: d.col("airborne_frac") > 0.1, pct),
               ("a step with zero links in contact", lambda d: d.col("no_contact_frac") > 0, pct),
               ("links in contact (mean)", lambda d: d.col("links_in_contact"), g3),
               ("max |joint vel| [rad/s] (mean)", lambda d: d.col("max_vel"), g3)]
    s += table(["metric"] + [d.label for d in (a, b)],
               [[n] + [f"{f(fn(d)[slow].mean())} / {f(fn(d)[~slow].mean())}" for d in (a, b)] for n, fn, f in metrics])
    calm = calm_mask(ds)
    s += (f"\n### Calm subset\n\nCalm = no dataset shows a flipped head link, more than 2% unloaded steps or a joint velocity above "
          f"1.1 x the velocity limit: {int(calm.sum())} of {len(calm)} episodes. Cells: all episodes / calm episodes "
          f"({a.label}, then {b.label}).\n\n")
    crow = []
    for name, key, f in (("centroid speed [m/s]", "speed", g3), ("links in contact", "links_in_contact", g3),
                         ("active taxels", "active_mean", g3), ("active-taxel toggles per taxel-second", "toggle_rate", g3),
                         ("std of IMU |acc| [m/s2]", "acc_std", g3), ("mean |joint torque| [Nm]", "torque_abs_mean", g3),
                         ("mean shear/normal ratio", "ratio_mean", g3)):
        crow.append([name] + [f"{f(np.nanmean(d.col(key)))} / {f(np.nanmean(d.col(key)[calm]))}" for d in (a, b)])
    s += table(["metric", a.label, b.label], crow)
    if calm.sum() > 3:
        ds_all, ds_calm = b.col("speed") - a.col("speed"), (b.col("speed") - a.col("speed"))[calm]
        s += (f"\nPaired centroid speed, all / calm / non-calm episodes: Pearson r {pearson(a.col('speed'), b.col('speed')):.3f} / "
              f"{pearson(a.col('speed')[calm], b.col('speed')[calm]):.3f} / "
              f"{pearson(a.col('speed')[~calm], b.col('speed')[~calm]):.3f}; mean difference ({b.label} - {a.label}) "
              f"{ds_all.mean():+.3f} / {ds_calm.mean():+.3f} / {ds_all[~calm].mean():+.3f} m/s. "
              f"IMU |acc| PSD at 1 Hz ({a.label} vs {b.label}), all / calm: "
              f"{np.interp(1, a.freq_imu, a.psd_imu()):.2f} vs {np.interp(1, b.freq_imu, b.psd_imu()):.2f} / "
              f"{np.interp(1, a.freq_imu, a.psd_imu(calm)):.2f} vs {np.interp(1, b.freq_imu, b.psd_imu(calm)):.2f}.\n")
    rows = [[f"p{q:g}" if q < 100 else "max", *[g3(hist_percentile(d.hist["joint/track_err"], q)) for d in ds]]
            for q in PCTS]
    rows.append(["max", *[g3(d.max["joint/track_err"]) for d in ds]])
    s += "\n### Joint tracking error |pos - target| [rad] (all joint samples)\n\n"
    s += table(["statistic"] + [d.label for d in ds], rows)
    s += "\n### Example episodes (head link xy trajectory)\n\n"
    ex = [[a.results[i]["ep"], a.terrain_names[a.results[i]["terrain"]], f"{a.results[i]['friction']:.3f}",
           g3(a.results[i]["speed"]), g3(b.results[i]["speed"])] for i in picks]
    s += ("Chosen as the analysed episodes closest to the 10th, 50th and 90th percentile of the friction parameter "
          "(not selected on outcome).\n\n")
    s += table(["episode", "class", "friction", f"speed {a.label} [m/s]", f"speed {b.label} [m/s]"], ex)
    s += "\nPlots: `paired_scatter.png`, `trajectories.png`.\n"
    return s


def write_csv(d: Dataset, path: Path) -> None:
    keys = [k for k, v in d.results[0].items() if np.isscalar(v) or isinstance(v, (int, float))]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in d.results:
            w.writerow([r[k] for k in keys])


# --------------------------------------------------------------------------------------------- key differences
def key_differences(ds: list[Dataset], sep: dict) -> str:
    """Bullets generated from the numbers of the first two datasets (descriptive only; hypotheses go in notes.md)."""
    if len(ds) < 2:
        return ""
    a, b = ds[0], ds[1]
    la, lb = a.label, b.label
    calm = calm_mask(ds[:2])

    def two(x, y, n=3, f=None):  # "x (la) vs y (lb)"
        f = f or (lambda v: f"{v:.{n}g}")
        return f"{f(x)} ({la}) vs {f(y)} ({lb})"

    def mean(d, k):
        return float(np.nanmean(d.col(k)))

    out = []
    jb = int(np.argmax(b.vec("vel_hi_j")))
    top_joints = np.argsort(-b.vec("vel_hi_j"))[:4]
    out.append(
        f"**Joint velocity piles up at the {a.vel_limit:g} rad/s limit in {la}, not in {lb}.** Max |joint vel| {two(a.max['joint/vel'], b.max['joint/vel'])} rad/s; "
        f"samples within 1% of the {a.vel_limit:g} rad/s velocity limit {two(a.total('n_vel_sat') / a.total('n_joint'), b.total('n_vel_sat') / b.total('n_joint'), f=pct)}; "
        f"samples > {VEL_LIM:g} rad/s {two(a.total('n_vel_hi'), b.total('n_vel_hi'), f=lambda v: f'{int(v)}')} in "
        f"{two(a.episodes_with('n_vel_hi'), b.episodes_with('n_vel_hi'), f=lambda v: f'{int(v)}')} episodes; in {lb} "
        f"{b.vec('vel_hi_j')[top_joints].sum() / max(b.vec('vel_hi_j').sum(), 1):.0%} of them are on joints "
        f"{', '.join(str(int(j)) for j in sorted(top_joints))} (joint {jb}: max {b.vec('vel_max_j')[jb]:.0f} rad/s). "
        f"Joint torque is within 1% of the {a.effort_limit:g} Nm effort limit in {two(a.total('n_tq_sat') / a.total('n_joint'), b.total('n_tq_sat') / b.total('n_joint'), f=pct)} of samples.")
    out.append(
        f"**{lb} has less ground contact and more tumbling.** Links in contact {two(mean(a, 'links_in_contact'), mean(b, 'links_in_contact'))}, "
        f"active taxels per sample {two(mean(a, 'active_mean'), mean(b, 'active_mean'))}, episodes with > 10% of steps below half the robot weight "
        f"{two(int((a.col('airborne_frac') > 0.1).sum()), int((b.col('airborne_frac') > 0.1).sum()), f=lambda v: f'{int(v)}')}, episodes with the head upside-down "
        f"{two(int((a.col('min_up_head') < 0).sum()), int((b.col('min_up_head') < 0).sum()), f=lambda v: f'{int(v)}')} (of {len(a)}). "
        f"Total taxel force / weight is about 1 in both ({two(mean(a, 'f_over_w'), mean(b, 'f_over_w'))}).")
    f_ratio = [float(np.interp(f, b.freq_force, b.psd_force()) / np.interp(f, a.freq_force, a.psd_force())) for f in (20, 50, 100)]
    out.append(
        f"**Different impulsive / spectral character.** IMU |acc| p99.9 {two(hist_percentile(a.hist['imu/acc_norm'], 99.9), hist_percentile(b.hist['imu/acc_norm'], 99.9), 3)} m/s2, "
        f"max {two(a.max['imu/acc_norm'], b.max['imu/acc_norm'])}, but |acc| > {ACC_LIM:g} m/s2 in {two(a.total('n_acc_hi') / a.total('n_imu'), b.total('n_acc_hi') / b.total('n_imu'), f=pct)} of samples; "
        f"share of IMU |acc| power above 25 Hz {two(band_fraction(a.freq_imu, a.psd_imu(), 25), band_fraction(b.freq_imu, b.psd_imu(), 25), f=pct)}, "
        f"at 1 Hz the {lb}/{la} PSD ratio is {np.interp(1, b.freq_imu, b.psd_imu()) / np.interp(1, a.freq_imu, a.psd_imu()):.1f}. "
        f"Link-force PSD {lb}/{la} at 20/50/100 Hz: {f_ratio[0]:.1f}/{f_ratio[1]:.1f}/{f_ratio[2]:.1f}; active-taxel on/off toggles per taxel-second "
        f"{two(mean(a, 'toggle_rate'), mean(b, 'toggle_rate'))}; isolated one-sample taxel dropouts to <= 1 Pa between two > 5 kPa samples "
        f"{two(a.total('n_dropout') / max(a.total('n_sandwich'), 1), b.total('n_dropout') / max(b.total('n_sandwich'), 1), f=pct)}; taxel p99.9 {two(hist_percentile(a.hist['tactile/normal'], 99.9) / 1e3, hist_percentile(b.hist['tactile/normal'], 99.9) / 1e3)} kPa, "
        f"max {two(a.max['tactile/normal'] / 1e3, b.max['tactile/normal'] / 1e3, f=lambda v: f'{v:.0f}')} kPa.")
    out.append(
        f"**Friction readout and gait are close, speed is not identical.** Mean shear/normal vs friction parameter: Pearson r "
        f"{two(pearson(a.col('friction'), a.col('ratio_mean')), pearson(b.col('friction'), b.col('ratio_mean')))}, slope through 0 "
        f"{two(slope_origin(a.col('friction'), a.col('ratio_mean')), slope_origin(b.col('friction'), b.col('ratio_mean')))}; paired per-episode ratio r "
        f"{pearson(a.col('ratio_mean'), b.col('ratio_mean')):.3f}. Centroid speed {two(mean(a, 'speed'), mean(b, 'speed'))} m/s "
        f"(paired r {pearson(a.col('speed'), b.col('speed')):.2f}, mean difference {np.mean(b.col('speed') - a.col('speed')):+.3f} m/s); "
        f"speed vs friction r {two(pearson(a.col('friction'), a.col('speed')), pearson(b.col('friction'), b.col('speed')), 2)}."
        + (f" In the {int(calm.sum())} calm episodes (no flip, < 2% unloaded steps, no velocity spike in either simulator) the paired speed r is "
           f"{pearson(a.col('speed')[calm], b.col('speed')[calm]):.3f} and the mean difference {np.mean((b.col('speed') - a.col('speed'))[calm]):+.3f} m/s; "
           f"in the other {int((~calm).sum())} episodes {np.mean((b.col('speed') - a.col('speed'))[~calm]):+.3f} m/s." if calm.sum() > 3 else ""))
    ac = [sep["cv"][d.label]["acc"] for d in ds[:2]]
    cross = {c[0]: c for c in sep["cross"]}
    xs = "; ".join(f"{k} {v[2]}" for k, v in cross.items() if k.split(" -> ")[0] in (la, lb) and k.split(" -> ")[1] in (la, lb))
    out.append(
        f"**Class separability from 5 engineered features is similar within each simulator, less so across them.** 5-fold accuracy (logistic regression) "
        f"{two(100 * np.mean(ac[0]['logreg']), 100 * np.mean(ac[1]['logreg']), 3, f=lambda v: f'{v:.1f}%')}, nearest centroid "
        f"{two(100 * np.mean(ac[0]['centroid']), 100 * np.mean(ac[1]['centroid']), 3, f=lambda v: f'{v:.1f}%')} (chance 20%); "
        f"transfer without retraining (logistic regression): {xs}.")
    return ("## Key differences (generated from the numbers below; " + f"{la} vs {lb})\n\n"
            + "\n".join(f"- {o}" for o in out) + "\n")


# --------------------------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", help="two or more dataset directories (the first is the reference)")
    parser.add_argument("--out", default="outputs/sim_compare")
    parser.add_argument("--every", type=int, default=1, help="analyse every n-th episode of each terrain class")
    parser.add_argument("--workers", type=int, default=6, help="worker processes (CPU only)")
    parser.add_argument("--labels", nargs="*", default=None, help="display names (default: from meta.yaml)")
    parser.add_argument("--link_mass", type=float, default=0.25, help="mass per link [kg] used for the robot weight")
    parser.add_argument("--velocity_limit", type=float, default=8.0, help="joint velocity limit [rad/s] for saturation checks")
    parser.add_argument("--effort_limit", type=float, default=6.0, help="joint effort limit [Nm] for saturation checks")
    parser.add_argument("--notes", default=None, help="markdown file appended to the report (default: <out>/notes.md)")
    args = parser.parse_args()
    if len(args.datasets) < 2:
        parser.error("need at least two datasets")

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pretty = {"isaaclab": "Isaac", "isaac": "Isaac", "mjlab": "mjlab"}
    ds: list[Dataset] = []
    episodes = select_episodes(args.datasets[0], args.every)  # chosen on the first dataset, reused for all (paired)
    for i, root in enumerate(args.datasets):
        sim = str(load_yaml(Path(root) / "meta.yaml").get("info", {}).get("simulator", Path(root).name))
        label = args.labels[i] if args.labels and i < len(args.labels) else pretty.get(sim, sim)
        missing = [e for e in episodes if not (Path(root) / "episodes" / f"ep_{e:06d}.npz").exists()]
        if missing:
            raise SystemExit(f"{root} lacks episodes {missing[:5]}...; datasets must be paired")
        d = Dataset(root, label, episodes, args.workers, args.link_mass, args.velocity_limit, args.effort_limit)
        print(f"{label}: {len(d)} episodes analysed in {d.seconds:.0f} s", flush=True)
        ds.append(d)
    if len({len(d) for d in ds}) != 1:
        raise SystemExit("datasets have different numbers of episodes; pairing is impossible")

    picks = pick_examples(ds[0])
    plot_psd(ds, out / "psd.png")
    plot_friction_ratio(ds, out / "friction_vs_shear_ratio.png")
    plot_speed(ds, out / "speed_by_class.png")
    plot_paired(ds[0], ds[1], out / "paired_scatter.png")
    plot_trajectories(ds[:2], picks, out / "trajectories.png")
    for d in ds:
        write_csv(d, out / f"episode_metrics_{d.label}.csv")

    sep_md, sep = section_separability(ds)
    head = "# Simulator comparison\n\n"
    head += table(["dataset", "directory", "simulator", "episodes analysed", "classes", "rates (latent / tactile / joint / imu Hz)"],
                  [[d.label, d.root, d.simulator, f"{len(d)} (every {args.every})", ", ".join(d.terrain_names),
                    f"{d.latent_hz:g} / {d.fs['tactile']:g} / {d.fs['joint']:g} / {d.fs['imu']:g}"] for d in ds])
    head += (f"\nEach episode is {ds[0].duration:g} s. Command: `python scripts/compare_datasets.py "
             f"{' '.join(args.datasets)} --out {args.out} --every {args.every}`. Cells of the form `a / b` are "
             f"{' / '.join(d.label for d in ds)}.\n\n")
    parts = [head, key_differences(ds, sep), section_amplitude(ds), section_events(ds), section_contact(ds),
             section_locomotion(ds), section_spectra(ds), sep_md, section_paired(ds, picks)]
    notes = Path(args.notes) if args.notes else out / "notes.md"
    if notes.exists():
        parts.append("## Notes and hypotheses (hand-written)\n\n" + notes.read_text().strip() + "\n")
    parts.append(f"\n---\nRuntime: {time.time() - t0:.0f} s (CPU only, {args.workers} workers).\n")
    (out / "report.md").write_text("\n".join(parts))
    print(f"wrote {out / 'report.md'} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
