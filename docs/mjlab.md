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
4. Per-env terrain (simulation v1, `terrain_compliance: true`, the default): robot-ground contacts are explicit
   contact pairs, one per robot geom (the robot geoms' own collisions are disabled). `pair_friction`, `pair_solref`
   and `pair_solimp` are expanded to one copy per world and written per env on reset from the sampled terrain. See
   "Terrain compliance" below. With `terrain_compliance: false` (v0), `geom_friction` is expanded instead and only
   friction varies. MuJoCo has **no restitution coefficient and no static/dynamic friction split**, so
   `restitution` and Isaac's `static_friction_ratio` have no counterpart (the damping ratio plays restitution's role).
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

## Terrain compliance (simulation v1)

Soft terrain must be *physically* soft: the robot sinks into snow, the footprint on the skin widens and impacts are
damped. In v0 this was faked by a constant offset in the taxel model. A plain soft geom contact is not the answer
either: MuJoCo applies one solref to the normal and the friction rows, so softening the contact also softened
friction, and locomotion stopped depending on friction (speed-friction r 0.04, see below).

Explicit contact pairs separate the two:
* the **normal** row gets the terrain's softness:
  * `solref` time constant and damping ratio (`contact_timeconst`, `contact_dampratio`);
  * an impedance profile `solimp = (dmin, 0.95, width, 0.5, 2)` (`contact_dmin`, `contact_width`): low impedance at
    the surface rising over a transition depth, i.e. a ground that yields quickly and stiffens with depth;
* the **friction** rows keep a stiff `solreffriction` (`friction_solref`, 0.02 s).

Calibration (`scripts/mjlab/calibrate_compliance.py`, findings F-6, F-7, F-10), with resting and moving sinkage of the
loaded capsules:

| setting (τ, dmin, width) | rest | moving | flicker |
|---|---|---|---|
| 0.02, 0.9, 1 mm (hard ground) | 0.05 mm | 0.5–0.8 mm | ~3 % (bounce) |
| 0.02, 0.5, 5 mm | 0.8 mm | 1.0–1.4 mm | < 1 % |
| 0.02, 0.3, 15 mm | 2.5 mm | 2.3–2.7 mm | < 0.2 % |
| 0.02, 0.02, 50 mm | 13.8 mm | ~10 mm | 0 |
| 0.06, 0.02, 50 mm | 21 mm | ~12 mm | 0 |

Speed keeps increasing with friction on soft ground (r 0.67–0.81 over μ 0.1–0.6). The time constant alone is a poor
compliance knob: it gives deep resting sinkage but responds slowly, so a moving robot barely sinks (0.9 mm moving
vs 5.4 mm at rest at τ 0.2 s). Hard-ground damping ratios of 0.15–0.3 (bouncy skin, as the literature suggests) are
stable but raise one-step flicker to 3–10 %; catalog v1 uses 0.2–0.4. `impratio` (elliptic cones) made no difference.

The catalog (`configs/terrains/ice_snow_v1.yaml`) maps the literature ranges (`docs/references/terrain_physics_v2.md`)
onto these parameters. Validation: 15/15. Static rigid-support checks run on the stiffest class; the check
`soft_terrain_sinks_deeper` gives 11 mm on fresh snow vs 0.0 mm on concrete.

## Validation results (v0 setup: 10 envs, `configs/archive/v0/experiments/collect_mjlab.yaml`)

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

## Isaac vs MuJoCo with identical actuators (DC motor), and the simulator decision

With both simulators on the same DC-motor actuator (`actuator: dc_motor`), the paired 600-episode datasets
(`*_600_dc`) agree closely (Isaac / mjlab):

* **Locomotion:** centroid speed 0.443 / 0.448 m/s, paired per-episode speed r = 0.99.
* **Friction readout:** shear/normal ratio vs the friction parameter r = 0.991 / 0.992, paired r = 0.999.
* **Tumbling and saturation:** episodes with the head flipped 5 / 6 (32 with the earlier actuator
  mismatch); joint velocity > 8 rad/s in 152 / 164 episodes.

Remaining differences:

* **Isaac:** rigid impacts give heavy-tailed spikes (IMU |acc| up to 944 m/s², taxel pressure up to
  1.4 MPa), and it reports shear on taxels without normal pressure in 0.2 % of non-contact samples.
* **mjlab:** contact flicker. A link in contact loses all its force for 1–2 physics steps in ~4.5 % of
  steps. Replaying those states in CPU MuJoCo confirms the capsule really is 4–16 µm above the ground: a
  lightly loaded link rests at only ~50 µm of penetration with the default stiff soft-contact
  (solref 0.02 s), and gait jitter separates it.
  * A detection `margin` (with or without `gap`) does not help; MuJoCo only shifts the equilibrium.
  * A softer robot contact (solref 0.1 s, ~0.7 mm penetration, skin-like) cuts the flicker to 0.24 %. But
    it also softens friction (MuJoCo applies solref to the friction rows too): speed +22 %, speed no
    longer depends on friction (r 0.04), 52 episodes partly unloaded, and `friction_opposes_sliding`
    drops to 0.80. It is therefore not used. A compliant skin with stiff Coulomb friction needs explicit
    contact pairs with a separate `solreffriction`.
  * The 1–2 ms dropouts are largely smoothed by the FSR model's 20 ms unloading time constant.

**Decision: mjlab (MuJoCo-Warp) with the default contact and DC-motor actuator is the primary
simulator.** It agrees with Isaac on the quantities that matter here, collects data ~6× faster
(600 episodes in 2 min vs 11 min), validates in 20 s vs 2.5 min, and can represent terrain and skin
compliance through per-env contact parameters (PhysX is rigid). It also avoids this machine's Isaac issues
(GUI/RTX crash on driver 595, Kit shutdown quirks). Isaac stays validated for cross-simulator transfer
checks, a cheap proxy for sim-to-real robustness.

## Isaac vs MuJoCo: what the sensors see (earlier datasets, actuator mismatch)

`scripts/compare_datasets.py datasets/isaac_terrain_600 datasets/mjlab_terrain_600` compares 600 paired
episodes from each simulator. Episode *i* has the same terrain class, friction, heading and joint-target
trajectory in both (same seed), so the differences come from the physics. Main results (Isaac / mjlab):

* **Friction readout agrees.** Per-episode shear/normal ratio vs the friction parameter: r = 0.991 /
  0.992. Between the paired episodes: r = 0.999. Total contact force / weight: 1.00 / 0.995.
* **Locomotion mostly agrees.** Centroid speed is 0.443 / 0.412 m/s (paired r = 0.82). In the 384
  "calm" episodes (no flip, no unloading, no velocity spike in either simulator) the paired speeds give
  r = 0.994.
* **Contact patterns differ.** Links in contact: 3.0 / 2.2. Active taxels: 16 / 13. mjlab taxels
  flicker about 5× more (on/off toggles per taxel-second: 4 / 20), and its link-force spectrum carries
  1.6–2.2× more power at 20–100 Hz.
* **Tumbling and velocity spikes appear in mjlab.** Episodes with the head upside down: 5 / 32. Joint
  velocities in MuJoCo reach 43 rad/s, in 219 episodes, mostly on torque-saturated middle joints;
  Isaac clips at the actuator's 8 rad/s `velocity_limit`, and MuJoCo has no joint velocity limit.
  **This asymmetry in configuration confounds the comparison** and should be removed (the same
  velocity limit, or a torque-speed curve, in both) before drawing physics conclusions.
* **IMU.** Isaac has rarer but larger impact spikes: |acc| p99.9 is 173 / 129 m/s² and the max is
  902 / 416. In Isaac, 66 % of the |acc| power lies above 25 Hz; in mjlab, 40 %.
* **Separability.** From five engineered features (logistic regression, 5-fold CV, chance 20 %):
  77.7 % / 79.0 %. A model trained on Isaac reaches 75.2 % on mjlab; the reverse direction gives
  66.3 %. Glare ice and concrete are recognised at 91–100 %, while rough ice, packed snow and fresh snow
  are confused with each other. Their friction ranges overlap, and the remaining differences (texture,
  sinkage) act only through the taxel contact model.
