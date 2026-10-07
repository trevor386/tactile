# Isaac Sim / Isaac Lab

Target: **Isaac Lab 2.3.x on Isaac Sim 5.x**. The backend uses the Isaac Lab API surface:
`InteractiveScene`, `ArticulationCfg` from a URDF, `ContactSensorCfg` with `track_friction_forces` /
`track_contact_points` (added in 2.3), `ImuCfg`, and the PhysX material tensor API.

## Setup

```bash
# inside the Isaac Lab python environment (e.g. ./isaaclab.sh -p, or the conda env Isaac Lab created)
cd tactile
pip install -e .
python scripts/isaac/validate_isaac.py --headless
```

## How the scene is built (`somato/sim/isaaclab/scene.py`)

1. The robot description (snake generator or any URDF) is written to `outputs/urdf/<name>.urdf` and
   imported with `UrdfFileCfg` (cylinders → capsules, contact reporting enabled, no self-collision).
2. Joints are driven by an **explicit** `IdealPDActuatorCfg`, so `robot.data.applied_torque` contains the
   torque that motor sensing reports.
3. The ground is a `TerrainImporterCfg` plane with friction and restitution 1 and `multiply` combine
   mode. Per-environment terrain friction is then written to the *robot's* collision shapes with
   `root_physx_view.set_material_properties`, so each environment can be a different terrain class.
4. Contact reporting (`IsaacSnakeConfig.contact_mode`):
   * `per_body` (default): one `ContactSensor` per link, filtered against the ground. It gives the
     normal force, friction force and mean contact point. Isaac Lab only supports filtering for
     one sensor body per sensor, hence one sensor per link.
   * `net`: a single unfiltered sensor that gives net normal forces only. It is cheaper; tactile
     shear is then estimated from sliding velocity.
5. One `ImuCfg` per IMU group (`gravity_bias = (0, 0, 9.81)`, so it reads specific force like a real
   IMU).

Positions are reported relative to each environment's origin.

## Validate before collecting

`scripts/isaac/validate_isaac.py` runs `somato.validation.BackendValidator`:

| check | catches |
|---|---|
| `names_match` | layout bodies/joints missing in the imported articulation |
| `kinematic_consistency` | Isaac link frames ≠ URDF link frames (importer conventions) |
| `taxel_rest_height` | taxels not at the collision surface (radius/skin/frame mismatch) |
| `normal_force_equals_weight` | contact reporting not enabled / wrong prim paths / missing bodies |
| `tactile_force_conservation` | contact-model distribution bugs |
| `imu_gravity`, `imu_still` | IMU frame or gravity-bias convention |
| `joint_tracking` | actuator gains, joint ordering |
| `torque_increases_with_friction`, `shear_increases_with_friction` | per-env terrain materials not applied |
| `friction_opposes_sliding` | sign of `friction_forces_w`; if it fails, set `isaac.friction_sign: -1.0` |

**Not yet verified on Isaac Sim**, and the first things to check if anything fails:

* link prim paths `{ENV_REGEX_NS}/Robot/<link_name>` (the importer's flat hierarchy);
* the sign convention of `friction_forces_w` (exposed as `friction_sign`);
* capsule replacement keeping link frames (`kinematic_consistency`);
* physics stability of the snake at 1 kHz with the default PD gains (`stiffness`, `damping`).

## Collect, train, deploy

```bash
python scripts/isaac/collect_isaac.py --config configs/experiments/collect_isaac.yaml --headless
python scripts/train.py --config configs/experiments/terrain_mock.yaml --set dataset=datasets/isaac_terrain name=isaac_hier
python scripts/isaac/online_demo.py --checkpoint runs/isaac_hier/model.pt --headless
```

The default Isaac robot is `configs/robots/snake_3d.yaml` (yaw/pitch joints) with a **sidewinding**
gait, because PhysX friction is isotropic and lateral undulation barely propels without anisotropic
friction. The mock simulator models scale anisotropy and uses lateral undulation.

## Performance notes

* Cost is dominated by the number of environments × physics rate. Start with `num_envs: 64–256`.
* `contact_mode: net` avoids one sensor per link (much cheaper for long robots).
* The stimulus pipeline runs every physics step (IMU finite differences need it). Taxel stimuli are
  vectorized over environments on the GPU.
* Datasets are about 1 MB per 3 s episode for 288 taxels at 500 Hz (compressed float16).
