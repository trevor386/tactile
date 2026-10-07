"""A lightweight planar snake simulator (pure torch, CPU or GPU) for tests and quick experiments.

Physics model: the joint shape follows the gait through first-order servos; the base motion is
*quasi-static*, i.e. at every step the base velocity ``(v_x, v_y, omega)`` is the one for which the
ground friction forces on all links balance (net force and moment zero). Friction per link is
anisotropic regularized Coulomb (axial ``mu``, lateral ``mu * anisotropy``, like snake scales), plus a
rotational friction moment, solved by iteratively reweighted least squares. Joint torques are the
moments the actuators must supply to hold the distal chain against friction, so motor sensing
reflects the terrain implicitly. Normal load is the link weight times a random payload scale, shared
between links according to the terrain's macro undulation.

This is deliberately simple: no inertia, no vertical motion, no obstacles. Its job is to produce
terrain-dependent multimodal signals fast and deterministically. Use Isaac Lab for anything physical.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from somato.robots.description import RobotDescription
from somato.sim.backend import RawSimState, SimBackend
from somato.sim.terrain import TerrainBatch, TerrainCatalog


@dataclass
class PlanarSnakeConfig:
    physics_dt: float = 1e-3
    servo_tau: float = 0.03  # s, joint tracking time constant
    irls_max_iters: int = 40  # IRLS iterations cap (stops early at irls_tol)
    irls_tol: float = 1e-4  # m/s, rad/s (force-balance residual ~0.01 N)
    velocity_eps: float = 0.005  # m/s, Coulomb regularization
    payload_scale: tuple[float, float] = (0.8, 1.3)  # random load multiplier per environment
    gravity: float = 9.81


def _cross2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _perp(v: torch.Tensor) -> torch.Tensor:
    return torch.stack([-v[..., 1], v[..., 0]], dim=-1)


def _suffix_sum(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return x.flip(dim).cumsum(dim).flip(dim)


class PlanarSnakeBackend(SimBackend):
    def __init__(self, desc: RobotDescription, catalog: TerrainCatalog, num_envs: int,
                 cfg: PlanarSnakeConfig | None = None, device: str = "cpu", generator: torch.Generator | None = None):
        self.cfg = cfg or PlanarSnakeConfig()
        self.desc, self.catalog = desc, catalog
        self._num_envs, self._device, self.generator = num_envs, torch.device(device), generator
        self._parse(desc)
        E, Nj, dev = num_envs, len(self._joints), self._device
        self.p0 = torch.zeros(E, 2, device=dev)
        self.theta0 = torch.zeros(E, device=dev)
        self.qdot = torch.zeros(E, 3, device=dev)
        self.phi = torch.zeros(E, Nj, device=dev)
        self.phid = torch.zeros(E, Nj, device=dev)
        self.target = torch.zeros(E, Nj, device=dev)
        self.payload = torch.ones(E, device=dev)
        self.time = torch.zeros(E, device=dev)
        self._terrain = catalog.sample(torch.zeros(E, dtype=torch.long), generator, dev)
        self._out: dict[str, torch.Tensor] = {}
        self.reset()

    # ------------------------------------------------------------------ setup
    def _parse(self, desc: RobotDescription) -> None:
        """Extract the serial chain (link lengths, masses, radius); requires yaw-only joints."""
        chain = [desc.root_body]
        joints = []
        while True:
            nxt = [j for j in desc.joints if j.parent == chain[-1] and j.movable]
            if not nxt:
                break
            if len(nxt) > 1 or abs(nxt[0].axis[2]) < 0.999:
                raise ValueError("PlanarSnakeBackend needs a serial chain of yaw (z-axis) joints")
            joints.append(nxt[0])
            chain.append(nxt[0].child)
        self._bodies, self._joints = chain, [j.name for j in joints]
        bodies = [desc.body(n) for n in chain]
        dev = self._device
        self.ell = torch.tensor([2.0 * b.com[0] for b in bodies], device=dev)
        self.mass = torch.tensor([b.mass for b in bodies], device=dev)
        self.radius = bodies[0].shapes[0].size["radius"]
        self.lim_lo = torch.tensor([j.lower for j in joints], device=dev)
        self.lim_hi = torch.tensor([j.upper for j in joints], device=dev)
        self.vmax = torch.tensor([j.velocity for j in joints], device=dev)

    # ------------------------------------------------------------------ interface
    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def physics_dt(self) -> float:
        return self.cfg.physics_dt

    @property
    def body_names(self) -> list[str]:
        return list(self._bodies)

    @property
    def joint_names(self) -> list[str]:
        return list(self._joints)

    @property
    def terrain(self) -> TerrainBatch:
        return self._terrain

    @property
    def device(self) -> torch.device:
        return self._device

    def reset(self, env_ids=None, terrain_class=None) -> None:
        dev, E = self._device, self._num_envs
        ids = torch.arange(E, device=dev) if env_ids is None else env_ids.to(dev)
        n = len(ids)
        if terrain_class is None:
            terrain_class = torch.randint(len(self.catalog), (n,), generator=self.generator)
        self._terrain.update(ids, self.catalog.sample(terrain_class, self.generator, dev))
        lo, hi = self.cfg.payload_scale
        self.payload[ids] = (lo + (hi - lo) * torch.rand(n, generator=self.generator)).to(dev)
        self.p0[ids] = 0.0
        self.theta0[ids] = (2 * torch.pi * torch.rand(n, generator=self.generator)).to(dev)
        self.qdot[ids] = 0.0
        self.phi[ids] = 0.0
        self.phid[ids] = 0.0
        self.target[ids] = 0.0
        self.time[ids] = 0.0
        self._solve()

    def step(self, joint_targets: torch.Tensor) -> None:
        dt = self.cfg.physics_dt
        self.target = joint_targets.to(self._device)
        self.phid = ((self.target - self.phi) / self.cfg.servo_tau).clamp(-self.vmax, self.vmax)
        self.phi = (self.phi + self.phid * dt).clamp(self.lim_lo, self.lim_hi)
        self._solve()
        self.p0 = self.p0 + self.qdot[:, :2] * dt
        self.theta0 = self.theta0 + self.qdot[:, 2] * dt
        self.time = self.time + dt

    def get_state(self) -> RawSimState:
        o = self._out
        th, p, _, _, _ = self._kinematics()
        E, Nb = th.shape
        z = torch.full((E, Nb, 1), self.radius, device=self._device)
        half = th / 2
        quat = torch.stack([half.cos(), torch.zeros_like(half), torch.zeros_like(half), half.sin()], -1)
        zeros = torch.zeros(E, Nb, 1, device=self._device)
        return RawSimState(
            body_pos=torch.cat([p, z], -1),
            body_quat=quat,
            body_lin_vel=torch.cat([o["v_frame"], zeros], -1),
            body_ang_vel=torch.cat([zeros, zeros, o["omega"].unsqueeze(-1)], -1),
            joint_pos=self.phi.clone(),
            joint_vel=self.phid.clone(),
            joint_torque=o["torque"],
            joint_target=self.target.clone(),
            contact_normal_force=o["normal"],
            contact_friction_force=torch.cat([o["force"], zeros], -1),
            contact_point=torch.cat([o["com"], zeros], -1),
        )

    # ------------------------------------------------------------------ physics
    def _kinematics(self):
        """Link angles, frame origins, COMs and shape-induced velocities at the current state."""
        E = self._num_envs
        zero = torch.zeros(E, 1, device=self._device)
        th = self.theta0[:, None] + torch.cat([zero, self.phi.cumsum(1)], 1)  # [E, Nb]
        u = torch.stack([th.cos(), th.sin()], -1)
        seg = self.ell[None, :, None] * u
        zero2 = torch.zeros(E, 1, 2, device=self._device)
        p = self.p0[:, None] + torch.cat([zero2, seg[:, :-1].cumsum(1)], 1)
        com = p + 0.5 * seg
        thd_s = torch.cat([zero, self.phid.cumsum(1)], 1)
        ju = _perp(u)
        pd_s = torch.cat([zero2, (self.ell[None, :, None] * thd_s[..., None] * ju)[:, :-1].cumsum(1)], 1)
        cd_s = pd_s + 0.5 * self.ell[None, :, None] * thd_s[..., None] * ju
        return th, p, com, (u, ju, thd_s, pd_s), cd_s

    def _solve(self) -> None:
        cfg, tb = self.cfg, self._terrain
        th, p, com, (u, ju, thd_s, pd_s), cd_s = self._kinematics()
        E, Nb = th.shape
        # Normal load sharing from macro undulation of the terrain.
        und = tb.undulation(com)  # [E, Nb]
        raw = self.mass * (1.0 + tb.undulation_amp[:, None] * und).clamp_min(0.1)
        normal = raw / raw.sum(1, keepdim=True) * (self.mass.sum() * cfg.gravity * self.payload)[:, None]
        mu_t = tb.friction[:, None]
        mu_n = mu_t * tb.anisotropy[:, None]
        r = com - self.p0[:, None]
        A = torch.zeros(E, Nb, 2, 3, device=self._device)
        A[..., 0, 0] = 1.0
        A[..., 1, 1] = 1.0
        A[..., 0, 2] = -r[..., 1]
        A[..., 1, 2] = r[..., 0]
        aniso = mu_t[..., None, None] * u[..., :, None] * u[..., None, :] + mu_n[..., None, None] * ju[..., :, None] * ju[..., None, :]
        lr = self.ell[None] / 4.0
        qdot = self.qdot
        for _ in range(cfg.irls_max_iters):
            v = torch.einsum("enij,ej->eni", A, qdot) + cd_s
            omega = qdot[:, 2:3] + thd_s
            C = (normal / torch.sqrt(v.pow(2).sum(-1) + cfg.velocity_eps**2))[..., None, None] * aniso
            c_rot = mu_n * normal * lr**2 / torch.sqrt((omega * lr).pow(2) + cfg.velocity_eps**2)
            H = torch.einsum("enai,enab,enbj->eij", A, C, A)
            g = torch.einsum("enai,enab,enb->ei", A, C, cd_s)
            H[:, 2, 2] += c_rot.sum(1)
            g[:, 2] += (c_rot * thd_s).sum(1)
            H = H + 1e-9 * torch.eye(3, device=self._device)
            new = -torch.linalg.solve(H, g)
            done = float((new - qdot).abs().max()) < cfg.irls_tol
            qdot = new
            if done:
                break
        self.qdot = qdot
        v = torch.einsum("enij,ej->eni", A, qdot) + cd_s
        omega = qdot[:, 2:3] + thd_s
        C = (normal / torch.sqrt(v.pow(2).sum(-1) + cfg.velocity_eps**2))[..., None, None] * aniso
        force = -(C @ v.unsqueeze(-1)).squeeze(-1)  # [E, Nb, 2]
        moment = -mu_n * normal * lr**2 * omega / torch.sqrt((omega * lr).pow(2) + cfg.velocity_eps**2)
        # Actuator torque at joint j (located at p[j+1]) holding the distal links j+1.. against friction.
        s_f = _suffix_sum(force)[:, 1:]
        s_cf = _suffix_sum(_cross2(com, force))[:, 1:]
        s_m = _suffix_sum(moment)[:, 1:]
        torque = -(s_cf - _cross2(p[:, 1:], s_f) + s_m)
        v_frame = qdot[:, None, :2] + qdot[:, 2:3, None] * _perp(p - self.p0[:, None]) + pd_s
        self._out = {"normal": normal, "force": force, "torque": torque, "omega": omega, "v_frame": v_frame,
                     "com": com}
