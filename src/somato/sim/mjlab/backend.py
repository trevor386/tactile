"""mjlab (MuJoCo-Warp) implementation of :class:`somato.sim.backend.SimBackend`.

The robot is loaded from the same :class:`RobotDescription` as in Isaac (exported with ``to_mjcf``: identical
link frames, inertias and capsule geometry) and driven by the same explicit PD actuator, so differences in the
data come from the contact models: MuJoCo's soft, convex contacts vs. PhysX's rigid contacts.

Per-environment terrain: the sliding friction of the robot geoms is written per world (``geom_friction`` is
expanded to one copy per env). MuJoCo has no restitution coefficient and no static/dynamic friction split, so
``restitution`` and Isaac's ``static_friction_ratio`` have no counterpart here. Other terrain parameters
(texture, sinkage) act through the taxel contact model, as in Isaac.

Contacts come from one netforce ``mjSENS_CONTACT`` sensor over all links against the ground, which reports
the force exerted *by the link on the ground*, so it is negated here.
"""

from __future__ import annotations

import re

import mujoco
import torch
from mjlab.actuator import DcMotorActuatorCfg, IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import BuiltinSensorCfg, ContactMatch, ContactSensorCfg, ObjRef
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from somato.geometry.layout import SensorLayout
from somato.geometry.rotations import matrix_to_quat
from somato.robots.description import RobotDescription
from somato.sim.backend import RawSimState, SimBackend
from somato.sim.mjlab.config import MjlabSnakeConfig
from somato.sim.terrain import TerrainBatch, TerrainCatalog

ROBOT = "robot"
CONTACT = "ground_contact"


def imu_site_name(group: str) -> str:
    return f"imu_{group}"


class MjlabBackend(SimBackend):
    def __init__(self, desc: RobotDescription, layout: SensorLayout, catalog: TerrainCatalog,
                 cfg: MjlabSnakeConfig | None = None, generator: torch.Generator | None = None):
        self.cfg = cfg = cfg or MjlabSnakeConfig()
        self.desc, self.layout, self.catalog, self.generator = desc, layout, catalog, generator
        self._imu_groups = [n for n, g in layout.groups.items() if g.kind == "imu"] if cfg.use_native_imu else []
        self.scene = Scene(self._scene_cfg(), cfg.device)
        sim_cfg = SimulationCfg(
            nconmax=cfg.nconmax, njmax=cfg.njmax,
            mujoco=MujocoCfg(timestep=cfg.physics_dt, solver=cfg.solver, iterations=cfg.iterations,
                             ls_iterations=cfg.ls_iterations, cone=cfg.cone, impratio=cfg.impratio),
        )
        self.sim = Simulation(num_envs=cfg.num_envs, cfg=sim_cfg, spec=self.scene.spec, device=cfg.device)
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self.sim.expand_model_fields(("geom_friction",))
        self.robot = self.scene[ROBOT]
        self._device = torch.device(cfg.device)
        self._body_names = list(self.robot.body_names)
        self._joint_names = list(self.robot.joint_names)
        self._geom_ids = self.robot.indexing.geom_ids.long()
        sensor = self.scene[CONTACT]
        self._contact_map = torch.tensor([list(sensor.primary_names).index(b) for b in self._body_names],
                                         device=self._device)
        E = self.num_envs
        self._terrain = catalog.sample(torch.zeros(E, dtype=torch.long), generator, self._device)
        self.reset()

    def _scene_cfg(self) -> SceneCfg:
        desc, layout, cfg = self.desc, self.layout, self.cfg
        sites = []
        for name in self._imu_groups:
            g = layout.groups[name]
            if len(g) != 1:
                raise ValueError("Native MuJoCo IMUs support one sensor per IMU group; use use_native_imu=False")
            body = layout.body_names[int(g.body_index[0])]
            quat = matrix_to_quat(g.local_rot[0].double()).tolist()
            sites.append((imu_site_name(name), body, tuple(g.local_pos[0].tolist()), tuple(quat)))
        xml = desc.to_mjcf(floating_base=True, self_collision=cfg.self_collision, sites=sites)
        radius = max((s.size.get("radius", 0.0) for b in desc.bodies for s in b.shapes), default=0.05)

        def spec_fn() -> mujoco.MjSpec:
            spec = mujoco.MjSpec.from_string(xml)
            # mujoco-warp sizes its contact buffers at qpos0, so keep the default pose clear of the ground.
            spec.body(desc.root_body).pos = [0.0, 0.0, 0.5]
            for geom in spec.geoms:
                geom.priority = 1
                geom.solref = list(cfg.contact_solref)
                geom.margin = cfg.contact_margin
                geom.gap = cfg.contact_gap
            return spec

        robot = EntityCfg(
            spec_fn=spec_fn,
            init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, radius + cfg.spawn_clearance), joint_pos={".*": 0.0}),
            articulation=EntityArticulationInfoCfg(actuators=(self._actuator_cfg(),)),
        )
        sensors = [ContactSensorCfg(
            name=CONTACT,
            primary=ContactMatch(mode="body", pattern=tuple(re.escape(b) for b in desc.body_names), entity=ROBOT),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force", "pos"),
            reduce="netforce",
        )]
        for name in self._imu_groups:
            site = ObjRef(type="site", name=imu_site_name(name), entity=ROBOT)
            sensors += [BuiltinSensorCfg(name=f"{imu_site_name(name)}_acc", sensor_type="accelerometer", obj=site),
                        BuiltinSensorCfg(name=f"{imu_site_name(name)}_gyro", sensor_type="gyro", obj=site)]
        return SceneCfg(num_envs=cfg.num_envs, env_spacing=cfg.env_spacing,
                        terrain=TerrainEntityCfg(terrain_type="plane"),
                        entities={ROBOT: robot}, sensors=tuple(sensors))

    def _actuator_cfg(self) -> IdealPdActuatorCfg:
        cfg = self.cfg
        common = dict(target_names_expr=tuple(re.escape(j) for j in self.desc.joint_names), stiffness=cfg.stiffness,
                      damping=cfg.damping, effort_limit=cfg.effort_limit, armature=cfg.armature)
        if cfg.actuator == "dc_motor":
            return DcMotorActuatorCfg(**common, saturation_effort=cfg.saturation_effort,
                                      velocity_limit=cfg.velocity_limit)
        if cfg.actuator == "ideal_pd":
            return IdealPdActuatorCfg(**common)
        raise ValueError(f"Unknown actuator {cfg.actuator!r} (dc_motor | ideal_pd)")

    # ------------------------------------------------------------------ properties
    @property
    def num_envs(self) -> int:
        return self.cfg.num_envs

    @property
    def physics_dt(self) -> float:
        return self.cfg.physics_dt

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
        self.sim.reset(ids)
        self.scene.reset(ids)
        self.sim.model.geom_friction[ids[:, None], self._geom_ids[None, :], 0] = self._terrain.friction[ids][:, None]

        root = self.robot.data.default_root_state[ids].clone()
        root[:, :3] += self.scene.env_origins[ids]
        yaw = 2 * torch.pi * torch.rand(n, generator=self.generator).to(self._device)
        root[:, 3:7] = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw),
                                    torch.sin(yaw / 2)], -1)
        root[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root, env_ids=ids)
        q = self.robot.data.default_joint_pos[ids].clone()
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        self.robot.set_joint_position_target(q, env_ids=ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(self.physics_dt)

    def step(self, joint_targets: torch.Tensor) -> None:
        self.robot.set_joint_position_target(joint_targets.to(self._device))
        self.scene.write_data_to_sim()
        self.sim.step()
        if self.cfg.forward_after_step:
            self.sim.forward()
        self.scene.update(self.physics_dt)

    # ------------------------------------------------------------------ state
    def get_state(self) -> RawSimState:
        d = self.robot.data
        origins = self.scene.env_origins[:, None, :]
        c = self.scene[CONTACT].data
        F = c.force[:, self._contact_map]  # [E, Nb, 3] world, force by the link on the ground
        found = c.found[:, self._contact_map] > 0
        friction = -F.clone()
        friction[..., 2] = 0.0
        point = torch.where(found[..., None], c.pos[:, self._contact_map] - origins, torch.zeros_like(F))
        imu = None
        if self._imu_groups:
            imu = {g: {"acc": self.scene[f"{ROBOT}/{imu_site_name(g)}_acc"].data,
                       "gyro": self.scene[f"{ROBOT}/{imu_site_name(g)}_gyro"].data} for g in self._imu_groups}
        return RawSimState(
            body_pos=d.body_link_pos_w - origins,
            body_quat=d.body_link_quat_w,
            body_lin_vel=d.body_link_lin_vel_w,
            body_ang_vel=d.body_link_ang_vel_w,
            joint_pos=d.joint_pos,
            joint_vel=d.joint_vel,
            joint_torque=d.qfrc_actuator,
            joint_target=d.joint_pos_target,
            contact_normal_force=(-F[..., 2]).clamp_min(0.0),
            contact_friction_force=friction,
            contact_point=point,
            imu=imu,
        )
