"""Isaac Lab implementation of :class:`somato.sim.backend.SimBackend`.

Usage (inside a script that has launched the app with ``isaaclab.app.AppLauncher``)::

    backend = IsaacLabBackend(desc, layout, catalog, IsaacSnakeConfig(num_envs=64))
    runner = SimRunner(backend, StimulusPipeline(layout), gait, rates)

Per-environment terrain: friction and restitution of the robot's collision shapes are set per
environment through the PhysX tensor API (the ground uses ``multiply`` combine mode with unit
coefficients, so the robot's material is the effective one). Other terrain parameters (texture,
sinkage) act through the taxel contact model.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from somato.geometry.layout import SensorLayout
from somato.robots.description import RobotDescription
from somato.sim.backend import RawSimState, SimBackend
from somato.sim.isaaclab.config import IsaacSnakeConfig
from somato.sim.isaaclab.scene import ROBOT, contact_sensor_name, imu_sensor_name, make_scene_cfg
from somato.sim.terrain import TerrainBatch, TerrainCatalog


class IsaacLabBackend(SimBackend):
    def __init__(self, desc: RobotDescription, layout: SensorLayout, catalog: TerrainCatalog,
                 cfg: IsaacSnakeConfig | None = None, generator: torch.Generator | None = None,
                 sim: sim_utils.SimulationContext | None = None):
        self.cfg = cfg or IsaacSnakeConfig()
        self.desc, self.layout, self.catalog, self.generator = desc, layout, catalog, generator
        self.sim = sim or sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=self.cfg.physics_dt, device=self.cfg.device, render_interval=self.cfg.render_interval)
        )
        self.scene = InteractiveScene(make_scene_cfg(desc, layout, self.cfg))
        self.sim.reset()
        # Pre-populate asset/sensor buffers (as Isaac Lab's envs do); the IMU needs a dt before its first read.
        self.scene.update(self.sim.get_physics_dt())
        self.robot = self.scene[ROBOT]
        self._device = torch.device(self.sim.device)
        self._body_names = list(self.robot.body_names)
        self._joint_names = list(self.robot.joint_names)
        self._imu_groups = [n for n, g in layout.groups.items() if g.kind == "imu"] if self.cfg.use_native_imu else []
        if self.cfg.contact_mode == "net":
            sensor = self.scene["contact_all"]
            self._net_map = torch.tensor([sensor.body_names.index(b) for b in self._body_names], device=self._device)
        # With a GUI, refresh the viewport every `render_interval` physics steps (as Isaac Lab's envs do).
        self._render = self.sim.has_gui()
        self._step_count = 0
        if self._render:
            o = self.scene.env_origins[0].tolist()
            self.sim.set_camera_view([o[0] + 2.0, o[1] + 2.0, o[2] + 1.5], o)
        E = self.num_envs
        self._terrain = catalog.sample(torch.zeros(E, dtype=torch.long), generator, self._device)
        self.reset()

    # ------------------------------------------------------------------ properties
    @property
    def num_envs(self) -> int:
        return self.scene.num_envs

    @property
    def physics_dt(self) -> float:
        return self.sim.get_physics_dt()

    @property
    def body_names(self) -> list[str]:
        return self._body_names

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def terrain(self) -> TerrainBatch:
        return self._terrain

    @property
    def device(self) -> torch.device:
        return self._device

    # ------------------------------------------------------------------ control
    def reset(self, env_ids=None, terrain_class=None) -> None:
        E = self.num_envs
        ids = torch.arange(E, device=self._device) if env_ids is None else env_ids.to(self._device)
        n = len(ids)
        if terrain_class is None:
            terrain_class = torch.randint(len(self.catalog), (n,), generator=self.generator)
        self._terrain.update(ids, self.catalog.sample(terrain_class, self.generator, self._device))
        self._apply_materials(ids)

        root = self.robot.data.default_root_state[ids].clone()
        root[:, :3] += self.scene.env_origins[ids]
        yaw = 2 * torch.pi * torch.rand(n, generator=self.generator).to(self._device)
        root[:, 3:7] = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw),
                                    torch.sin(yaw / 2)], -1)
        root[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root[:, :7], ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], ids)
        q = self.robot.data.default_joint_pos[ids].clone()
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        self.robot.set_joint_position_target(q, env_ids=ids)
        self.scene.reset(ids)
        self.scene.write_data_to_sim()

    def _apply_materials(self, ids: torch.Tensor) -> None:
        view = self.robot.root_physx_view
        materials = view.get_material_properties()  # [E, num_shapes, 3] on CPU
        ids_cpu = ids.cpu()
        mu = self._terrain.friction[ids].cpu()
        materials[ids_cpu, :, 0] = (mu * self.cfg.static_friction_ratio)[:, None]
        materials[ids_cpu, :, 1] = mu[:, None]
        materials[ids_cpu, :, 2] = self._terrain.restitution[ids].cpu()[:, None]
        view.set_material_properties(materials, ids_cpu)

    def step(self, joint_targets: torch.Tensor) -> None:
        self.robot.set_joint_position_target(joint_targets.to(self._device))
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self._step_count += 1
        if self._render and self._step_count % self.cfg.render_interval == 0:
            self.sim.render()
        self.scene.update(self.physics_dt)

    def render(self) -> None:
        self.sim.render()

    def close(self) -> None:
        """Release the scene and simulation context (as Isaac Lab's envs do); without this, closing the app
        can hang in PhysX teardown."""
        del self.scene
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    # ------------------------------------------------------------------ state
    def get_state(self) -> RawSimState:
        d = self.robot.data
        origins = self.scene.env_origins[:, None, :]
        normal, friction, point = self._contacts()
        imu = None
        if self._imu_groups:
            imu = {}
            for g in self._imu_groups:
                sd = self.scene[imu_sensor_name(g)].data
                imu[g] = {"acc": sd.lin_acc_b, "gyro": sd.ang_vel_b}
        return RawSimState(
            body_pos=d.body_link_pos_w - origins,
            body_quat=d.body_link_quat_w,
            body_lin_vel=d.body_link_lin_vel_w,
            body_ang_vel=d.body_link_ang_vel_w,
            joint_pos=d.joint_pos,
            joint_vel=d.joint_vel,
            joint_torque=d.applied_torque,
            joint_target=d.joint_pos_target,
            contact_normal_force=normal,
            contact_friction_force=friction,
            contact_point=None if point is None else point - origins,
            imu=imu,
        )

    def _contacts(self):
        if self.cfg.contact_mode == "net":
            net = self.scene["contact_all"].data.net_forces_w[:, self._net_map]  # [E, Nb, 3]
            return net[..., 2].abs(), None, None
        normal, friction, point = [], [], []
        for body in self._body_names:
            data = self.scene[contact_sensor_name(body)].data
            fm = data.force_matrix_w[:, 0, 0]  # [E, 3] normal force against the ground
            normal.append(fm.norm(dim=-1))
            friction.append(self.cfg.friction_sign * torch.nan_to_num(data.friction_forces_w[:, 0, 0]))
            point.append(torch.nan_to_num(data.contact_pos_w[:, 0, 0]))
        return torch.stack(normal, 1), torch.stack(friction, 1), torch.stack(point, 1)
