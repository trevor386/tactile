# MuJoCo-Warp (mjlab)

A second physics backend for comparing MuJoCo's contact model against Isaac Sim / PhysX on the same
robot, gait, sensor layout and terrain classes. [mjlab](https://github.com/mujocolab/mjlab) provides
Isaac Lab's scene / entity / sensor API on top of [MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp)
(GPU, many parallel worlds). Only that layer is used, not mjlab's RL managers, driven step by step like
the Isaac backend. Both backends write the same dataset format, so training and evaluation code is shared.

**Verified** with mjlab 1.6.0, mujoco / mujoco-warp 3.11.0, warp 1.18, torch 2.14.1+cu130, Python 3.12, on
an RTX 3060 Ti: all 14 validation checks pass.

## Setup

mjlab >= 1.5 needs mujoco >= 3.10 and warp >= 1.14, which conflict with Isaac Sim's pins, and importing
mjlab changes global Warp settings, so use a separate env (never import it in an Isaac Sim process):

```bash
conda create -n mjlab python=3.12 -y && conda activate mjlab
pip install "mjlab==1.6.0"
pip install -e ".[dev]"
python scripts/mjlab/validate_mjlab.py --num_envs 10
python scripts/mjlab/collect_mjlab.py --config configs/experiments/collect_mjlab.yaml
```

## How the scene is built (`somato/sim/mjlab/backend.py`)

1. The robot description is exported with `RobotDescription.to_mjcf` (`somato/robots/mjcf.py`): the same
   link frames, masses, inertias and capsule collision shapes that Isaac imports from the URDF (Isaac
   replaces cylinders by capsules), plus a free joint and one site per native IMU. The root body is
   placed at z = 0.5 m in the spec only because mujoco-warp sizes its contact buffers at `qpos0`.
2. Joints use mjlab's explicit `IdealPdActuatorCfg` with the same gains as Isaac's `IdealPDActuatorCfg`
   (stiffness 20, damping 0.5, effort 6 Nm, armature 0.002). `qfrc_actuator` is the reported torque.
   mjlab has no joint velocity limit (Isaac clips at 8 rad/s, reached in 0.2 % of samples).
3. Ground: mjlab's `TerrainEntityCfg(terrain_type="plane")` (body and geom `terrain`). Robot geoms get
   `priority = 1`, so their friction and `solref` override the plane's; MuJoCo otherwise combines two
   geoms' friction with **max**, which would make every env behave like the plane (mu = 1).
4. Per-env terrain: `geom_friction` is expanded to one copy per world and the robot geoms' sliding
   friction is written per env on reset. MuJoCo has **no restitution coefficient and no static/dynamic
   friction split**, so `restitution` and Isaac's `static_friction_ratio` have no counterpart.
5. Contacts: one `ContactSensorCfg` (MuJoCo's native `mjSENS_CONTACT`) over all links against the
   terrain with `reduce="netforce"`: per link, the summed contact force in the world frame and the
   force-weighted contact centroid. The sensor reports the force exerted **by the link on the ground**:
   the normal force is `-F_z`, the friction force on the link `-(F_x, F_y, 0)`.
6. IMU: native `accelerometer` (specific force, +g up at rest) and `gyro` at the IMU site.
7. After every `sim.step()` the backend calls `sim.forward()` (`forward_after_step`) so poses, contact
   forces and sensors describe the post-step state, as in Isaac Lab; otherwise they lag the joint state
   by one physics step.

### Solver settings (`MjlabSnakeConfig`)

* **Elliptic friction cones.** With the Newton solver and pyramidal cones (mjlab's default),
  mujoco-warp 3.11 returns NaN contact forces and accelerations for friction mu <= ~0.07 (glare ice)
  within ~20 ms of gait onset; more iterations do not help. Elliptic cones, or the CG solver, are
  stable at every friction tested (0.01 to 0.5).
* **Contact buffers** `nconmax = 64`, `njmax = 256` per world. A resting 16-link snake has 32 ground
  contacts (2 per capsule); mujoco-warp's default sizing (48 / 64) overflows `njmax` and silently drops
  constraints (only a device `printf` per step).
* Newton, 20 iterations, 10 line-search iterations, dt = 1 ms.

## Validation results (10 envs, `collect_mjlab.yaml`)

| check | mjlab | Isaac Sim 5.1 | criterion |
|---|---|---|---|
| `kinematic_consistency` | 2.4e-7 m | 5.5e-7 m | < 2 mm |
| `taxel_rest_height` | -0.002 m | -0.002 m | in [-0.005, 0.002] |
| `tactile_force_conservation` | 0.037 | 0.070 | < 0.35 |
| `normal_force_equals_weight` | 1.3e-6 | 0.015 | < 0.1 |
| `imu_gravity` / `imu_still` | 8e-6 m/s² / 6e-5 rad/s | 0.23 m/s² / 0.017 rad/s | < 0.5 / < 0.05 |
| `joint_tracking` | 0.015 rad | 0.017 rad | < 0.05 |
| `torque_increases_with_friction` / `shear_...` | 1.0 / 1.0 | 1.0 / 1.0 | > 0.5 |
| `friction_opposes_sliding` | 0.93 | 0.999 | > 0.8 |

At rest MuJoCo's soft contacts are smooth (no step-to-step chatter), unlike PhysX's rigid contacts, which
needed more solver iterations. `friction_opposes_sliding` is lower in MuJoCo (0.93 vs 0.999); not yet
investigated. Plausible causes are soft-contact creep and the netforce sensor combining a link's two
capsule-end contacts into one force at their weighted centroid.
