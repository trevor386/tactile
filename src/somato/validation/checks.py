"""Backend-agnostic validation of a simulator integration.

These checks catch the integration bugs that silently ruin tactile datasets: body/joint name or
frame mismatches between the robot description and the simulator, wrong force sign conventions,
broken per-environment terrain assignment, and non-finite values. Run them on the mock in CI
(``tests/test_validation.py``) and on Isaac Lab with ``scripts/isaac/validate_isaac.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import torch

from somato.control.gaits import GaitConfig, SerpenoidGait
from somato.geometry.layout import SensorLayout
from somato.robots.description import RobotDescription
from somato.sim.backend import SimBackend
from somato.sim.contact_model import ContactModelConfig
from somato.sim.runner import RateConfig, SimRunner
from somato.sim.stimulus import StimulusPipeline

G = 9.81


@dataclass
class CheckResult:
    name: str
    passed: bool | None  # None = skipped
    value: float | None = None
    criterion: str = ""
    detail: str = ""

    def line(self) -> str:
        status = {True: "PASS", False: "FAIL", None: "SKIP"}[self.passed]
        val = "" if self.value is None else f"{self.value:.4g}"
        return f"[{status}] {self.name:28s} {val:>10s}  {self.criterion}  {self.detail}"


class _Constant:
    def __init__(self, targets: torch.Tensor):
        self.targets = targets

    def __call__(self, t):
        return self.targets


def _midpoint_gait(cfg: GaitConfig) -> GaitConfig:
    """``cfg`` with every parameter range collapsed to its midpoint (the same gait in every env)."""
    ranges = {f.name: getattr(cfg, f.name) for f in fields(cfg) if isinstance(getattr(cfg, f.name), tuple)}
    return replace(cfg, **{k: ((lo + hi) / 2,) * 2 for k, (lo, hi) in ranges.items()})


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-12))


class BackendValidator:
    def __init__(self, backend: SimBackend, desc: RobotDescription, layout: SensorLayout, rates: RateConfig,
                 gait: GaitConfig | None = None, contact: ContactModelConfig | None = None, settle_time: float = 1.0,
                 run_time: float = 3.0, skin: float | None = None):
        self.backend, self.desc, self.layout, self.rates = backend, desc, layout, rates
        self.gait_cfg = gait or GaitConfig()
        self.pipeline = StimulusPipeline(layout, contact)
        self.settle_steps = max(1, int(settle_time * rates.latent_hz))
        self.run_steps = max(1, int(run_time * rates.latent_hz))
        self.skin = skin
        self.results: list[CheckResult] = []

    # ------------------------------------------------------------------ helpers
    def _runner(self, controller) -> SimRunner:
        return SimRunner(self.backend, self.pipeline, controller, self.rates)

    def _zeros(self) -> torch.Tensor:
        return torch.zeros(self.backend.num_envs, len(self.backend.joint_names), device=self.backend.device)

    def _add(self, name, passed, value=None, criterion="", detail=""):
        self.results.append(CheckResult(name, passed, value, criterion, detail))

    def _payload(self) -> torch.Tensor:
        p = getattr(self.backend, "payload", None)
        return torch.ones(self.backend.num_envs, device=self.backend.device) if p is None else p

    def _compliance_classes(self) -> tuple[int, int] | None:
        """``(stiffest, softest)`` class indices when the catalog's classes differ in contact compliance, else None.

        Compliant terrain (simulation v1) is physically penetrated, so the checks that assume a rigid support (taxels
        at ground level at rest) run on the stiffest class.
        """
        catalog = getattr(self.backend, "catalog", None)
        if catalog is None or not hasattr(catalog.classes[0], "contact_dmin"):
            return None
        # Softness grows with the impedance transition width and time constant and falls with the surface impedance.
        soft = [sum(c.contact_width) * sum(c.contact_timeconst) * (2 - sum(c.contact_dmin) / 2) for c in catalog.classes]
        if max(soft) - min(soft) < 1e-12:
            return None
        return min(range(len(soft)), key=soft.__getitem__), max(range(len(soft)), key=soft.__getitem__)

    def _capsule_depth(self, state) -> torch.Tensor:
        """``[E, Nb]`` depth of each body's lowest capsule point below the ground (m; negative above it)."""
        radius = max((s.size.get("radius", 0.0) for b in self.desc.bodies for s in b.shapes), default=0.0)
        return self.backend.terrain.ground_height[:, None] - (state.body_pos[..., 2] - radius)

    # ------------------------------------------------------------------ checks
    def run(self) -> list[CheckResult]:
        self.results = []
        try:
            self.pipeline.bind(self.backend.body_names, self.backend.joint_names)
            self._add("names_match", True, detail="all layout bodies and joints exist in the simulator")
        except ValueError as e:
            self._add("names_match", False, detail=str(e))
            return self.results
        self.check_static()
        self.check_compliance()
        self.check_joint_tracking()
        self.check_gait()
        return self.results

    def check_compliance(self) -> None:
        """On compliant terrain catalogs: the softest class is physically penetrated more than the stiffest."""
        classes = self._compliance_classes()
        E = self.backend.num_envs
        if classes is None or E < 2:
            return
        if not getattr(self.backend, "_num_pairs", 0):  # only mjlab with terrain_compliance enacts compliance
            self._add("soft_terrain_sinks_deeper", None, detail="backend does not enact terrain compliance")
            return
        runner = self._runner(_Constant(self._zeros()))
        cls = torch.tensor([classes[i % 2] for i in range(E)])
        runner.reset(terrain_class=cls)
        for _ in range(self.settle_steps):
            runner.step_latent()
        state = self.backend.get_state()
        touching = state.contact_normal_force > 1e-3
        depth = (self._capsule_depth(state) * touching).sum(1) / touching.sum(1).clamp_min(1)
        stiff, soft = float(depth[cls == classes[0]].mean()), float(depth[cls == classes[1]].mean())
        self._add("soft_terrain_sinks_deeper", soft > 2 * max(stiff, 0.0) and soft > 5e-4, soft,
                  "> 2x the stiffest class and > 0.5 mm",
                  f"resting sinkage of loaded capsules: softest class {1e3 * soft:.2f} mm, stiffest {1e3 * stiff:.2f} mm")

    def check_static(self) -> None:
        runner = self._runner(_Constant(self._zeros()))
        classes = self._compliance_classes()
        if classes is None:
            runner.reset()
        else:  # rigid-support checks on the stiffest terrain; compliance is checked separately
            runner.reset(terrain_class=torch.full((self.backend.num_envs,), classes[0], dtype=torch.long))
        frames = [runner.step_latent() for _ in range(self.settle_steps)]
        f = frames[-1]
        state = self.backend.get_state()
        self._check_shapes(f)
        self._check_finite("finite_static", frames)
        # Kinematic consistency: child body origin == parent origin + R_parent * joint origin.
        bpos, brot = self.pipeline.body_poses(state)
        names = self.layout.body_names
        err = 0.0
        for j in self.desc.joints:
            if j.parent in names and j.child in names:
                p, c = names.index(j.parent), names.index(j.child)
                expect = bpos[:, p] + (brot[:, p] @ torch.tensor(j.pos, device=bpos.device, dtype=bpos.dtype))
                err = max(err, float((bpos[:, c] - expect).norm(dim=-1).max()))
        self._add("kinematic_consistency", err < 2e-3, err, "< 2 mm",
                  "simulator link frames must match the URDF link frames")
        # Taxels should reach the ground at rest: lowest taxel per body within [-skin - 2 mm, +2 mm].
        for name, g in self.layout.groups.items():
            if g.kind != "tactile":
                continue
            pos, _ = self.layout.world_poses(bpos, brot, group=name)
            z = pos[..., 2] - self.backend.terrain.ground_height[:, None]
            low = torch.full((z.shape[0], len(names)), float("inf"), device=z.device)
            low = low.scatter_reduce(1, g.body_index.to(z.device).expand_as(z), z, "amin")
            touching = low[torch.isfinite(low)]
            skin = self.skin if self.skin is not None else 0.003
            ok = bool(((touching > -skin - 2e-3) & (touching < 2e-3)).float().mean() > 0.9)
            self._add(f"taxel_rest_height[{name}]", ok, float(touching.median()), f"in [-{skin + 2e-3:.3f}, 0.002] m",
                      "lowest taxel of each resting body should be at ground level")
            # Force conservation: integrated taxel pressure == reported normal force == weight.
            stim = f.stimuli[name][:, -1]  # [E, N, 3]
            tactile_force = (stim[..., 0] * g.area.to(stim.device)).sum(-1)
            if state.contact_normal_force is not None:
                reported = state.contact_normal_force.sum(-1)
                rel = float(((tactile_force - reported).abs() / reported.clamp_min(1e-6)).max())
                self._add(f"tactile_force_conservation[{name}]", rel < 0.35, rel, "< 0.35 relative",
                          "texture modulation makes this approximate")
                weight = self.desc.total_mass() * G * self._payload()
                relw = float(((reported - weight).abs() / weight).max())
                self._add("normal_force_equals_weight", relw < 0.1, relw, "< 0.1 relative",
                          "simulator contact reporting at rest")
        # IMU at rest reads +g along world up, no rotation. Averaged over the last latent step: a single sample of
        # a finite-difference accelerometer still carries rigid-contact solver jitter (~1 m/s^2 in PhysX).
        for name, g in self.layout.groups.items():
            if g.kind != "imu":
                continue
            stim = f.stimuli[name].mean(1)  # [E, N, 6]
            _, rot = self.layout.world_poses(bpos, brot, group=name)
            up_local = (rot.transpose(-1, -2) @ torch.tensor([0.0, 0.0, G], device=rot.device, dtype=rot.dtype))
            acc_err = float((stim[..., :3] - up_local).norm(dim=-1).max())
            gyro = float(stim[..., 3:].norm(dim=-1).max())
            self._add(f"imu_gravity[{name}]", acc_err < 0.5, acc_err, "< 0.5 m/s^2")
            self._add(f"imu_still[{name}]", gyro < 0.05, gyro, "< 0.05 rad/s")

    def _alternating_pose(self, amplitude: float = 0.25) -> torch.Tensor:
        """Joint targets alternating in sign along the chain *within each joint axis*. For a yaw/pitch snake a
        global alternation would bend every pitch joint the same way and coil the body off the ground."""
        count: dict[tuple, int] = {}
        sign: dict[str, float] = {}
        for name in self.desc.joint_names:
            axis = tuple(round(abs(a), 3) for a in self.desc.joint(name).axis)
            sign[name] = (-1.0) ** count.get(axis, 0)
            count[axis] = count.get(axis, 0) + 1
        return amplitude * torch.tensor([sign[n] for n in self.backend.joint_names], device=self.backend.device)

    def check_joint_tracking(self) -> None:
        nj = len(self.backend.joint_names)
        pattern = self._alternating_pose()
        runner = self._runner(_Constant(pattern.expand(self.backend.num_envs, nj).clone()))
        runner.reset()
        for _ in range(self.settle_steps):
            runner.step_latent()
        state = self.backend.get_state()
        err = float((state.joint_pos - state.joint_target).abs().max())
        self._add("joint_tracking", err < 0.05, err, "< 0.05 rad",
                  "static alternating pose; checks actuator wiring and joint ordering")

    def check_gait(self) -> None:
        E = self.backend.num_envs
        gen = torch.Generator().manual_seed(0)
        # The same gait in every env, so friction is the only per-env difference: with randomized gaits,
        # inertial joint torques (~ A w^2) dominate the friction-dependent part in dynamic simulators.
        gait = SerpenoidGait(self.desc, _midpoint_gait(self.gait_cfg), E, self.backend.device, gen)
        runner = self._runner(gait)
        n_classes = int(self.backend.terrain.class_id.max().item()) + 1 if E else 1
        runner.reset(terrain_class=torch.arange(E) % max(n_classes, 1))
        gait.params["phase"].zero_()
        for _ in range(self.settle_steps):
            runner.step_latent()
        torque, shear, dots, frames = [], [], [], []
        for _ in range(self.run_steps):
            frame = runner.step_latent()
            frames.append(frame)
            state = self.backend.get_state()
            torque.append(state.joint_torque.abs().mean(-1))
            for name, g in self.layout.groups.items():
                if g.kind == "tactile":
                    shear.append(frame.stimuli[name][..., 1:].norm(dim=-1).mean((1, 2)))
            if state.contact_friction_force is not None:
                dots.append(self._friction_power(state))
        self._check_finite("finite_gait", frames)
        mu = self.backend.terrain.friction
        if E >= 3 and float(mu.std()) > 0:
            rho_t = _spearman(mu.cpu(), torch.stack(torque).mean(0).cpu())
            self._add("torque_increases_with_friction", rho_t > 0.5, rho_t, "Spearman > 0.5",
                      "per-env mean |joint torque| vs. friction coefficient")
            if shear:
                rho_s = _spearman(mu.cpu(), torch.stack(shear).mean(0).cpu())
                self._add("shear_increases_with_friction", rho_s > 0.5, rho_s, "Spearman > 0.5")
        else:
            self._add("torque_increases_with_friction", None, detail="needs >= 3 envs with different friction")
        if dots:
            d = torch.cat(dots)
            frac = float((d < 0).float().mean()) if d.numel() else math.nan
            self._add("friction_opposes_sliding", frac > 0.8, frac, "> 0.8 of sliding contacts",
                      "if this fails in Isaac, flip IsaacSnakeConfig.friction_sign")
        else:
            self._add("friction_opposes_sliding", None, detail="backend reports no friction forces")

    # ------------------------------------------------------------------ utils
    def _friction_power(self, state) -> torch.Tensor:
        """``F . v`` at contact points of sliding bodies (should be negative: friction dissipates)."""
        m = self.pipeline._body_map.to(state.body_pos.device)
        F = state.contact_friction_force[:, m]
        p, v, w = state.body_pos[:, m], state.body_lin_vel[:, m], state.body_ang_vel[:, m]
        c = state.contact_point[:, m] if state.contact_point is not None else p
        vc = v + torch.cross(w, c - p, dim=-1)
        vc[..., 2] = 0
        sliding = (vc.norm(dim=-1) > 0.01) & (F.norm(dim=-1) > 0.01)
        return (F * vc).sum(-1)[sliding]

    def _check_shapes(self, frame) -> None:
        bad = []
        for name, g in self.layout.groups.items():
            x = frame.stimuli[name]
            want = (self.backend.num_envs, self.rates.substeps(name), len(g))
            if tuple(x.shape[:3]) != want:
                bad.append(f"{name}: {tuple(x.shape)} != {want}+(C,)")
        self._add("stimulus_shapes", not bad, detail="; ".join(bad) or "rates and sensor counts consistent")

    def _check_finite(self, name, frames) -> None:
        ok = all(torch.isfinite(x).all() for f in frames for x in f.stimuli.values())
        self._add(name, bool(ok), detail="" if ok else "NaN/inf in stimuli")


def format_report(results: list[CheckResult]) -> str:
    lines = [r.line() for r in results]
    n_fail = sum(r.passed is False for r in results)
    lines.append(f"{len(results)} checks, {n_fail} failed, {sum(r.passed is None for r in results)} skipped")
    return "\n".join(lines)
