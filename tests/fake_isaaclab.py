"""A minimal stand-in for the parts of the Isaac Lab API used by ``somato.sim.isaaclab``.

It returns random-but-correctly-shaped data so the adapter's plumbing (scene construction, name
mapping, material assignment, state packing) can be exercised without Isaac Sim. It knows nothing
about physics; real behavior is checked by scripts/isaac/validate_isaac.py on a machine with Isaac.
"""

from __future__ import annotations

import sys
import types

import torch

from somato.robots.description import RobotDescription


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def replace(self, **kw):
        return type(self)(**{**self.__dict__, **kw})


def _cfg_class(name, **nested):
    return type(name, (_Cfg,), nested)


class _PhysxView:
    def __init__(self, num_envs, num_shapes):
        self.materials = torch.ones(num_envs, num_shapes, 3)
        self.set_calls = []

    def get_material_properties(self):
        return self.materials.clone()

    def set_material_properties(self, materials, env_ids):
        assert materials.device.type == "cpu" and env_ids.device.type == "cpu"
        self.materials = materials.clone()
        self.set_calls.append(env_ids.clone())


class _Data:
    pass


class _Articulation:
    def __init__(self, cfg, num_envs):
        desc = RobotDescription.from_urdf(cfg.spawn.asset_path)
        # PhysX may order bodies differently from the URDF: use a reversed order to test name mapping.
        self.body_names = list(reversed(desc.body_names))
        self.joint_names = list(desc.joint_names)
        E, B, J = num_envs, len(self.body_names), len(self.joint_names)
        d = _Data()
        d.body_link_pos_w = torch.randn(E, B, 3)
        d.body_link_quat_w = torch.nn.functional.normalize(torch.randn(E, B, 4), dim=-1)
        d.body_link_lin_vel_w = torch.randn(E, B, 3)
        d.body_link_ang_vel_w = torch.randn(E, B, 3)
        d.joint_pos = torch.randn(E, J)
        d.joint_vel = torch.randn(E, J)
        d.applied_torque = torch.randn(E, J)
        d.joint_pos_target = torch.zeros(E, J)
        d.default_root_state = torch.zeros(E, 13)
        d.default_root_state[:, 3] = 1.0
        d.default_joint_pos = torch.zeros(E, J)
        self.data = d
        self.root_physx_view = _PhysxView(E, B)
        self.cfg = cfg

    def write_root_pose_to_sim(self, pose, env_ids):
        assert pose.shape == (len(env_ids), 7)

    def write_root_velocity_to_sim(self, vel, env_ids):
        assert vel.shape == (len(env_ids), 6)

    def write_joint_state_to_sim(self, pos, vel, joint_ids=None, env_ids=None):
        assert pos.shape == vel.shape

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        if env_ids is None:
            self.data.joint_pos_target = target.clone()


class _ContactSensor:
    def __init__(self, cfg, num_envs, robot_bodies):
        E = num_envs
        d = _Data()
        if getattr(cfg, "filter_prim_paths_expr", None):
            d.force_matrix_w = torch.randn(E, 1, 1, 3)
            d.friction_forces_w = torch.randn(E, 1, 1, 3)
            d.contact_pos_w = torch.randn(E, 1, 1, 3)
            d.contact_pos_w[0] = float("nan")  # not in contact
        else:
            self.body_names = list(robot_bodies)
            d.net_forces_w = torch.randn(E, len(robot_bodies), 3)
        self.data = d


class _Imu:
    def __init__(self, cfg, num_envs):
        d = _Data()
        d.lin_acc_b = torch.randn(num_envs, 3)
        d.ang_vel_b = torch.randn(num_envs, 3)
        self.data = d


class InteractiveScene:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.env_origins = torch.arange(cfg.num_envs).float()[:, None] * torch.tensor([cfg.env_spacing, 0.0, 0.0])
        self._entities = {}
        robot = None
        for name, c in cfg.__dict__.items():
            if isinstance(c, ArticulationCfg):
                robot = self._entities[name] = _Articulation(c, cfg.num_envs)
        for name, c in cfg.__dict__.items():
            if isinstance(c, ContactSensorCfg):
                self._entities[name] = _ContactSensor(c, cfg.num_envs, list(reversed(robot.body_names)))
            elif isinstance(c, ImuCfg):
                self._entities[name] = _Imu(c, cfg.num_envs)
        self.resets = []

    def __getitem__(self, key):
        return self._entities[key]

    def reset(self, env_ids=None):
        self.resets.append(env_ids)

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass


class SimulationContext:
    def __init__(self, cfg):
        self.cfg, self.device, self.steps = cfg, cfg.device, 0

    def reset(self):
        pass

    def step(self, render=False):
        self.steps += 1

    def render(self):
        pass

    def has_gui(self):
        return False

    def set_camera_view(self, eye, target):
        pass

    def get_physics_dt(self):
        return self.cfg.dt


ArticulationCfg = _cfg_class("ArticulationCfg", InitialStateCfg=_cfg_class("InitialStateCfg"))
ContactSensorCfg = _cfg_class("ContactSensorCfg")
ImuCfg = _cfg_class("ImuCfg", OffsetCfg=_cfg_class("OffsetCfg"))


def install() -> None:
    """Register fake ``isaaclab`` modules in ``sys.modules``."""
    root = types.ModuleType("isaaclab")
    sim = types.ModuleType("isaaclab.sim")
    sim.SimulationContext = SimulationContext
    sim.SimulationCfg = _cfg_class("SimulationCfg")
    sim.UrdfConverterCfg = _cfg_class(
        "UrdfConverterCfg", JointDriveCfg=_cfg_class("JointDriveCfg", PDGainsCfg=_cfg_class("PDGainsCfg")))
    for n in ("UrdfFileCfg", "RigidBodyPropertiesCfg", "ArticulationRootPropertiesCfg", "RigidBodyMaterialCfg",
              "DomeLightCfg"):
        setattr(sim, n, _cfg_class(n))
    actuators = types.ModuleType("isaaclab.actuators")
    actuators.IdealPDActuatorCfg = _cfg_class("IdealPDActuatorCfg")
    assets = types.ModuleType("isaaclab.assets")
    assets.ArticulationCfg = ArticulationCfg
    assets.AssetBaseCfg = _cfg_class("AssetBaseCfg")
    scene = types.ModuleType("isaaclab.scene")
    scene.InteractiveScene = InteractiveScene
    scene.InteractiveSceneCfg = _cfg_class("InteractiveSceneCfg")
    sensors = types.ModuleType("isaaclab.sensors")
    sensors.ContactSensorCfg = ContactSensorCfg
    sensors.ImuCfg = ImuCfg
    terrains = types.ModuleType("isaaclab.terrains")
    terrains.TerrainImporterCfg = _cfg_class("TerrainImporterCfg")
    for name, mod in {"isaaclab": root, "isaaclab.sim": sim, "isaaclab.actuators": actuators,
                      "isaaclab.assets": assets, "isaaclab.scene": scene, "isaaclab.sensors": sensors,
                      "isaaclab.terrains": terrains}.items():
        sys.modules[name] = mod
    root.sim = sim


def uninstall() -> None:
    for name in [n for n in sys.modules if n == "isaaclab" or n.startswith("isaaclab.")]:
        del sys.modules[name]
    for name in [n for n in sys.modules if n.startswith("somato.sim.isaaclab.") and not n.endswith(".config")]:
        del sys.modules[name]
