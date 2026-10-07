"""Raw simulator state -> ideal sensor stimuli for every sensor group, plus ground-truth labels."""

from __future__ import annotations

import torch

from somato.geometry.layout import SensorLayout
from somato.geometry.rotations import quat_to_matrix
from somato.sim.backend import RawSimState
from somato.sim.contact_model import ContactModelConfig, TaxelContactModel
from somato.sim.terrain import TerrainBatch

GRAVITY = torch.tensor([0.0, 0.0, -9.81])


class StimulusPipeline:
    """Computes stimuli in the channel conventions of :mod:`somato.sensors.base`.

    Call :meth:`bind` once with the backend's body/joint names to resolve name -> index maps (the
    layout and the simulator may order bodies differently), and :meth:`reset` on episode resets.
    """

    def __init__(self, layout: SensorLayout, contact: ContactModelConfig | None = None,
                 contact_threshold: float = 0.05):
        self.layout = layout
        self.contact_model = TaxelContactModel(contact)
        self.contact_threshold = contact_threshold
        self._body_map: torch.Tensor | None = None
        self._joint_maps: dict[str, torch.Tensor] = {}
        self._prev_imu_vel: dict[str, torch.Tensor] = {}
        self._fresh: dict[str, torch.Tensor] = {}

    def bind(self, body_names: list[str], joint_names: list[str]) -> StimulusPipeline:
        missing = [b for b in self.layout.body_names if b not in body_names]
        if missing:
            raise ValueError(f"Simulator is missing bodies {missing}")
        self._body_map = torch.tensor([body_names.index(b) for b in self.layout.body_names])
        for name, g in self.layout.groups.items():
            if g.kind == "joint":
                miss = [j for j in g.joint_names if j not in joint_names]
                if miss:
                    raise ValueError(f"Simulator is missing joints {miss}")
                self._joint_maps[name] = torch.tensor([joint_names.index(j) for j in g.joint_names])
        return self

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Forget velocity history (IMU accelerations restart from a zero-acceleration assumption)."""
        for name in self._prev_imu_vel:
            if env_ids is None:
                self._fresh[name][:] = True
            else:
                self._fresh[name][env_ids] = True

    def body_poses(self, state: RawSimState) -> tuple[torch.Tensor, torch.Tensor]:
        """Body poses reordered to the layout's body order: ``[E, Nb, 3]``, ``[E, Nb, 3, 3]``."""
        m = self._body_map.to(state.body_pos.device)
        return state.body_pos[:, m], quat_to_matrix(state.body_quat[:, m])

    def __call__(self, state: RawSimState, terrain: TerrainBatch, dt: float
                 ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Returns ``(stimuli[group] [E, N_g, C], labels)``; labels hold per-body contact quantities."""
        if self._body_map is None:
            raise RuntimeError("Call bind() before using the pipeline")
        dev = state.body_pos.device
        m = self._body_map.to(dev)
        bpos, brot = state.body_pos[:, m], quat_to_matrix(state.body_quat[:, m])
        bvel, bomega = state.body_lin_vel[:, m], state.body_ang_vel[:, m]
        nb = len(self.layout.body_names)
        normal = None if state.contact_normal_force is None else state.contact_normal_force[:, m]
        friction = None if state.contact_friction_force is None else state.contact_friction_force[:, m]

        stimuli: dict[str, torch.Tensor] = {}
        labels: dict[str, torch.Tensor] = {}
        for name, g in self.layout.groups.items():
            pos, rot = self.layout.world_poses(bpos, brot, group=name)
            bi = g.body_index.to(dev)
            r = pos - bpos[:, bi]
            vel = bvel[:, bi] + torch.cross(bomega[:, bi], r, dim=-1)
            if g.kind == "tactile":
                stim, w = self.contact_model(pos, rot, vel, g.area.to(dev), bi, nb, terrain, normal, friction)
                stimuli[name] = stim
                labels.update(self._contact_labels(pos, vel, w, bi, nb, normal))
            elif g.kind == "joint":
                j = self._joint_maps[name].to(dev)
                stimuli[name] = torch.stack(
                    [state.joint_pos[:, j], state.joint_vel[:, j], state.joint_torque[:, j], state.joint_target[:, j]],
                    dim=-1,
                )
            elif g.kind == "imu":
                stimuli[name] = self._imu(name, g, state, rot, vel, bomega[:, bi], dt)
            else:
                raise ValueError(f"No stimulus model for sensor kind '{g.kind}'")
        if normal is not None:
            labels["normal_force"] = normal
        return stimuli, labels

    def _imu(self, name, group, state, rot, vel, omega, dt) -> torch.Tensor:
        native = (state.imu or {}).get(name)
        if native is not None:
            return torch.cat([native["acc"], native["gyro"]], dim=-1).unsqueeze(1).expand(-1, len(group), -1)
        if name not in self._prev_imu_vel or self._prev_imu_vel[name].shape != vel.shape:
            self._prev_imu_vel[name] = vel.clone()
            self._fresh[name] = torch.ones(vel.shape[0], dtype=torch.bool, device=vel.device)
        fresh = self._fresh[name][:, None, None]
        acc = torch.where(fresh, torch.zeros_like(vel), (vel - self._prev_imu_vel[name]) / dt)
        self._prev_imu_vel[name] = vel.clone()
        self._fresh[name][:] = False
        specific = acc - GRAVITY.to(vel)
        rt = rot.transpose(-1, -2)
        f_local = (rt @ specific.unsqueeze(-1)).squeeze(-1)
        w_local = (rt @ omega.unsqueeze(-1)).squeeze(-1)
        return torch.cat([f_local, w_local], dim=-1)

    def _contact_labels(self, pos, vel, w, body_index, nb, normal) -> dict[str, torch.Tensor]:
        """Per-body contact centroid sliding speed (slip) from the footprint-weighted taxel velocities."""
        E = pos.shape[0]
        idx = body_index.expand(E, -1)
        w_sum = torch.zeros(E, nb, device=pos.device, dtype=pos.dtype).scatter_add_(1, idx, w)
        v_t = vel[..., :2]
        v_mean = torch.zeros(E, nb, 2, device=pos.device, dtype=pos.dtype).scatter_add_(
            1, idx.unsqueeze(-1).expand(-1, -1, 2), v_t * w.unsqueeze(-1)) / w_sum.clamp_min(1e-12).unsqueeze(-1)
        in_contact = (normal > self.contact_threshold) if normal is not None else (w_sum > 1e-6)
        slip = v_mean.norm(dim=-1) * in_contact
        return {"slip_speed": slip, "in_contact": in_contact.to(pos.dtype)}
