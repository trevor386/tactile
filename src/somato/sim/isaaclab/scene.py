"""Isaac Lab scene construction for a robot described by :class:`RobotDescription` + :class:`SensorLayout`.

Requires Isaac Lab (``isaaclab``) and a running simulation app (see ``scripts/isaac/*``).
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.terrains import TerrainImporterCfg

from somato.geometry.layout import SensorLayout
from somato.geometry.rotations import matrix_to_quat
from somato.robots.description import RobotDescription
from somato.sim.isaaclab.config import IsaacSnakeConfig

ROBOT = "robot"


def contact_sensor_name(body: str) -> str:
    return f"contact_{body}"


def imu_sensor_name(group: str) -> str:
    return f"imu_{group}"


def make_articulation_cfg(desc: RobotDescription, cfg: IsaacSnakeConfig) -> ArticulationCfg:
    urdf_path = Path(cfg.urdf_dir) / f"{desc.name}.urdf"
    desc.write_urdf(urdf_path)
    radius = max((s.size.get("radius", 0.0) for b in desc.bodies for s in b.shapes), default=0.05)
    drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
    )
    spawn = sim_utils.UrdfFileCfg(
        asset_path=str(urdf_path.resolve()),
        usd_dir=str((Path(cfg.urdf_dir) / "usd").resolve()),
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=True,
        self_collision=cfg.self_collision,
        activate_contact_sensors=True,
        joint_drive=drive,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=cfg.self_collision,
            solver_position_iteration_count=cfg.solver_position_iterations,
            solver_velocity_iteration_count=cfg.solver_velocity_iterations,
        ),
    )
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, radius + cfg.spawn_clearance),
                                                   joint_pos={".*": 0.0}),
        actuators={"joints": make_actuator_cfg(cfg)},
    )


def make_actuator_cfg(cfg: IsaacSnakeConfig) -> IdealPDActuatorCfg:
    common = dict(joint_names_expr=[".*"], stiffness=cfg.stiffness, damping=cfg.damping, effort_limit=cfg.effort_limit,
                  velocity_limit=cfg.velocity_limit, armature=cfg.armature)
    if cfg.actuator == "dc_motor":
        return DCMotorCfg(**common, saturation_effort=cfg.saturation_effort, velocity_limit_sim=cfg.velocity_limit_sim)
    if cfg.actuator == "ideal_pd":
        return IdealPDActuatorCfg(**common)
    raise ValueError(f"Unknown actuator {cfg.actuator!r} (dc_motor | ideal_pd)")


def make_scene_cfg(desc: RobotDescription, layout: SensorLayout, cfg: IsaacSnakeConfig) -> InteractiveSceneCfg:
    scene = InteractiveSceneCfg(num_envs=cfg.num_envs, env_spacing=cfg.env_spacing, replicate_physics=True)
    # Ground with friction/restitution 1 and "multiply" combine mode: the effective coefficients are
    # then exactly the robot's material, which the backend sets per environment.
    scene.terrain = TerrainImporterCfg(
        prim_path=cfg.ground_prim,
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=1.0,
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
        ),
    )
    setattr(scene, ROBOT, make_articulation_cfg(desc, cfg))
    if cfg.contact_mode == "per_body":
        for body in layout.body_names:
            setattr(scene, contact_sensor_name(body), ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}",
                update_period=0.0,
                history_length=0,
                filter_prim_paths_expr=[cfg.ground_collision_prim],
                track_contact_points=True,
                track_friction_forces=True,
                max_contact_data_count_per_prim=cfg.max_contact_data_count_per_prim,
            ))
    elif cfg.contact_mode == "net":
        bodies = "|".join(layout.body_names)
        scene.contact_all = ContactSensorCfg(prim_path=f"{{ENV_REGEX_NS}}/Robot/({bodies})", update_period=0.0, history_length=0)
    else:
        raise ValueError(f"Unknown contact_mode {cfg.contact_mode}")
    if cfg.use_native_imu:
        for name, g in layout.groups.items():
            if g.kind != "imu":
                continue
            if len(g) != 1:
                raise ValueError("Native Isaac IMUs support one sensor per IMU group; use use_native_imu=False")
            body = layout.body_names[int(g.body_index[0])]
            quat = matrix_to_quat(g.local_rot[0]).tolist()
            setattr(scene, imu_sensor_name(name), ImuCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}",
                offset=ImuCfg.OffsetCfg(pos=tuple(g.local_pos[0].tolist()), rot=tuple(quat)),
                gravity_bias=(0.0, 0.0, 9.81),
                update_period=0.0,
            ))
    scene.light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    return scene
