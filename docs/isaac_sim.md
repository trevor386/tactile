# Isaac Sim / Isaac Lab

Target: **Isaac Lab 2.3.x on Isaac Sim 5.x**. The backend uses the Isaac Lab API surface:
`InteractiveScene`, `ArticulationCfg` from a URDF, `ContactSensorCfg` with `track_friction_forces` /
`track_contact_points` (added in 2.3), `ImuCfg`, and the PhysX material tensor API.

**Verified** on Isaac Lab **2.3.2** + Isaac Sim **5.1.0** (pip), Python 3.11, torch 2.7.0+cu128, Ubuntu 24.04,
RTX 3060 Ti (8 GB), driver 595.71, headless: all 14 validation checks pass, data collection, training
and the online demo run. Isaac Lab 2.3 does **not** support Isaac Sim 6.x (that needs Isaac Lab 3.0,
whose API differs: warp arrays, XYZW quaternions, a new URDF importer).

## Setup

A separate conda env (Isaac Sim 5.1 needs Python 3.11):

```bash
conda create -n isaaclab23 python=3.11 -y && conda activate isaaclab23
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab && ./isaaclab.sh -i none
# flatdict (an Isaac Lab dependency) builds with pkg_resources, which setuptools >= 81 removed; if
# `import isaaclab` fails, build it against an older setuptools and install the core package again:
pip install "setuptools<81" && pip install --no-build-isolation flatdict==4.0.1 && pip install -e source/isaaclab
cd ~/tactile && pip install -e ".[dev]"
python -u scripts/isaac/validate_isaac.py --headless
```

Practical notes:

* Accept the Omniverse EULA once (interactive prompt) or set `OMNI_KIT_ACCEPT_EULA=YES`.
* If a ROS distribution is sourced in `~/.bashrc`, unset `PYTHONPATH` (and drop `/opt/ros` from
  `LD_LIBRARY_PATH`) before running Isaac: ROS Jazzy's Python 3.12 packages leak into the 3.11 env.
* Run scripts with `python -u`: `SimulationApp.close()` ends the process without flushing buffered
  stdout. It also exits with status 0, so `validate_isaac.py` exits through `os._exit(code)` instead.
* **Rendering (GUI or cameras) crashes on NVIDIA driver 595.x** (segfault in `librtx.scenedb.plugin`
  0.5 s after "app ready", also for Isaac Lab's empty-scene tutorial). This is a known Isaac Sim 5.1
  issue with the R590 driver branch; 5.1 is validated with the 580 production driver. Headless physics
  is unaffected. Without a GUI, `scripts/render_episodes.py` animates collected episodes from the
  stored body poses. **Update 2026-10-08:** the machine now runs driver 580.178 (kernel 7.0.0-38), the validated
  branch, so the GUI may work again. Isaac has not been re-run on it yet (task T-10 in `investigation_log.md`).
* Simulation v1 (physical terrain compliance) exists only in the mjlab backend so far. Isaac datasets use the v0
  terrain model until PhysX compliant contacts are wired up for sim-to-sim tests (T-10).

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

Results on Isaac Sim 5.1 (10 envs, `collect_isaac.yaml`):

| check | value | criterion |
|---|---|---|
| `kinematic_consistency` | 5.5e-7 m | < 2 mm |
| `taxel_rest_height` | -0.002 m | in [-0.005, 0.002] |
| `tactile_force_conservation` | 0.070 | < 0.35 |
| `normal_force_equals_weight` | 0.015 | < 0.1 |
| `imu_gravity` / `imu_still` | 0.23 m/s² / 0.017 rad/s | < 0.5 / < 0.05 |
| `joint_tracking` | 0.017 rad | < 0.05 |
| `torque_increases_with_friction` / `shear_increases_with_friction` | Spearman 1.0 / 1.0 | > 0.5 |
| `friction_opposes_sliding` | 0.999 | > 0.8 |

What this verified about the integration:

* Link prims are flat, `{ENV_REGEX_NS}/Robot/<link_name>` (rigid bodies with `PhysxContactReportAPI`),
  and the ground collision prim is `/World/ground/terrain/GroundPlane/CollisionPlane`.
* `friction_forces_w` is the force on the sensor body: `friction_sign: 1.0` is correct.
* `replace_cylinders_with_capsules` keeps the URDF link frames; masses (0.25 kg/link) are imported
  exactly; one collision shape per link (`get_material_properties()` is `[E, 16, 3]`).
* Per-link filtered contact sensors (`contact_mode: per_body`, `track_friction_forces`,
  `track_contact_points`) work on Isaac Lab 2.3.2. `contact_mode: net` was not exercised.

Physics stability at 1 kHz: the snake does not blow up with the default PD gains, but with PhysX's
8 position / 1 velocity solver iterations resting contacts **chatter every step** (normal force ±70 %,
IMU acceleration ±20 m/s² alternating between consecutive steps). `IsaacSnakeConfig` therefore uses
**16 / 4 iterations** (normal force within 4 % of the weight; ~5 % slower). PhysX `enable_stabilization`
has no effect on it; 4 / 0 is much worse.

Because physics-rate jitter would alias into sensor streams sampled below the physics rate,
`SimRunner` averages stimuli over each sensor sample period (`rates.anti_alias: true`, default). Two
checks were also corrected for dynamic simulators (thresholds unchanged): `joint_tracking` alternates
the pose within each joint axis (a global ±0.25 alternation bends all pitch joints of the yaw/pitch
snake the same way and lifts it off the ground), and the gait checks use the same gait in every env
(with randomized gaits inertial torques dominate the friction dependence). The IMU checks average over
the last latent step.

## Collect, train, deploy

```bash
python -u scripts/isaac/collect_isaac.py --config configs/experiments/collect_isaac.yaml --headless
python scripts/train.py --config configs/experiments/terrain_mock.yaml --set dataset=datasets/isaac_terrain name=isaac_hier
python -u scripts/isaac/online_demo.py --checkpoint runs/isaac_hier/model.pt --headless
python scripts/render_episodes.py datasets/isaac_terrain --out outputs/isaac_snake.gif
```

What the data looks like (64 episodes, 3 s each, sidewinding): the snake's centroid moves 1.1–1.5 m
per episode (0.4–0.5 m/s); mean taxel pressure is ~300 Pa for every class (weight) while mean shear
grows with friction (16 Pa on glare ice to 106 Pa on concrete). In-contact taxel pressure has a median
of 1.3 kPa but peaks above 65 kPa (0.01 % of samples), beyond float16: such arrays are stored as float32.
Joint velocity reaches the 8 rad/s `velocity_limit` in 0.2 % of samples and torque the 6 Nm effort limit
in 1.5 %. The IMU has heavy-tailed impact spikes (|acc| median 11.6, p99 33, max ~700 m/s²).

Terrain classification with the default hierarchical model and FSR skin (`terrain_mock.yaml`, 20 epochs,
RTX 3060 Ti; test split = 15 % of episodes):

| training data | test acc (last step) | notes |
|---|---|---|
| 64 episodes (`--rounds 1 --num_envs 64`) | 0.27 | 44 training episodes, too few |
| 600 episodes (`collect_isaac.yaml`) | 0.535 | training 7.5 s/epoch at 64 episodes, 75 s at 600; peak 6.6 GB at batch 32 |
| same model, tested on the paired mjlab episodes | 0.515 | small sim-to-sim gap (`scripts/evaluate.py`) |

Fresh snow is recognized perfectly (sinkage widens the taxel footprint); glare ice, rough ice and
concrete are confused. The FSR model reports normal pressure only, so the network sees friction only
indirectly (engineered features using the ideal shear/normal ratio separate concrete and glare ice at
91–100 %, see `docs/mjlab.md`).

Online (`online_demo.py`, 10 envs, streaming): **latency 19.7 ms median** per 20 ms latent step for all 10
envs (sensor models + model on the GPU), but accuracy drops to 0.31 because the stage-1 recurrent state
runs far beyond the 25-step (0.5 s) windows it was trained on. Offline the same drop appears with longer
windows: `scripts/evaluate.py --window 25 / 75 / 150` gives 0.53 / 0.43 / 0.32. Fixing this is a
training-protocol question (longer or random-length windows, state carried across windows, or a
sliding reset online), not a simulator issue.

The default Isaac robot is `configs/robots/snake_3d.yaml` (yaw/pitch joints) with a **sidewinding**
gait, because PhysX friction is isotropic and lateral undulation barely propels without anisotropic
friction. The mock simulator models scale anisotropy and uses lateral undulation.

## Performance notes

* Measured: 64 envs × 4 s of simulated time (4000 physics steps incl. warm-up) in 103 s, i.e. ~26 ms
  per physics step, nearly independent of `num_envs` at this scale (per-step Python overhead of 16
  contact sensors plus the stimulus pipeline dominates; the solver iterations cost ~5 %).
* Cost is dominated by the number of environments × physics rate. Start with `num_envs: 64–256`.
* `contact_mode: net` avoids one sensor per link (much cheaper for long robots).
* The stimulus pipeline runs every physics step (IMU finite differences need it). Taxel stimuli are
  vectorized over environments on the GPU.
* Datasets are about 1 MB per 3 s episode for 288 taxels at 500 Hz (compressed float16).
