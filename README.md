# somato — hierarchical somatosensory encoding for robots

PyTorch code for a hierarchical encoder of dense tactile, IMU and joint (motor) sensing, built for
reactive locomotion on icy terrain. It includes simulator integration (Isaac Lab on Isaac Sim 5.x,
MuJoCo-Warp through mjlab, plus a lightweight CPU mock) and the training and evaluation tooling for the
Phase 1 data-efficiency study.

```
sensor histories ──► Stage 1 "receptors" ──► Stage 2 "spinal cord" ──► Stage 3 "brain" ──► terrain class,
 (per modality,       per-sensor temporal      spatial interaction        region tokens per     properties, ...
  high rate)          encoder, shared within   *within* each modality:    (modality, body),
                      a modality, no position  3D kernel over a physical  current-distance
                                               support radius (current    attention, GRU over
                                               poses)                     time: modalities meet
```

* **Stage 1** (`somato/models/temporal.py`): every sensor of a group (all taxels, all joints, the IMU)
  shares one encoder. It sees only that sensor's own high-rate history (no position) and emits a latent per
  latent step. Options: `conv_gru` (default), `gru`, `receptor` (SA/RA-like filter bank) and `mlp` (ablation).
* **Stage 2** (`somato/models/spatial.py`): within each modality only (`fusion: segregated`). The default
  `kernel3d` is a continuous convolution: a learned kernel over the neighbour's position in the sensor's own
  frame, with a physical support radius, evaluated at the current poses (the 3D analogue of a CNN kernel). The
  graph alternatives (`geo_attention`, `egnn`) pass feature-dependent messages on a KNN graph.
* **Stage 3** (`somato/models/heads.py`, `type: brain`): the modalities meet here. Region tokens attend to each
  other with a bias from their current distances, then a GRU integrates over brain steps (10 Hz).

The model is invariant to the robot's global pose and does not depend on the number or arrangement of sensors,
so the same weights run on another sensor layout. Streaming in chunks gives exactly the window results.
**Baselines** (`somato/models/baselines.py`): a flat GRU and a transformer over (sensor, time patch) tokens. Both
get the same kinematics and are width-matched. Design: [docs/architecture.md](docs/architecture.md); plan and
status: [docs/investigation_log.md](docs/investigation_log.md); results: [docs/findings.md](docs/findings.md).

## Install

```bash
pip install -e ".[dev]"        # torch, numpy, pyyaml, pytest
pytest                         # ~30 s on CPU; Isaac tests are skipped without Isaac Lab
```

For Isaac Sim, install into the Isaac Lab Python environment (Isaac Lab 2.3, Isaac Sim 5.x);
see [docs/isaac_sim.md](docs/isaac_sim.md).

## Quick start (CPU mock simulator)

```bash
python scripts/validate_backend.py                                         # simulator integration checks
python scripts/collect.py --config configs/experiments/collect_mock.yaml   # ~400 episodes -> datasets/mock_terrain
python scripts/train.py --config configs/experiments/terrain_mock.yaml     # hierarchical model, terrain classification
python scripts/train.py --set model=configs/models/flat_gru.yaml name=flat # a baseline
python scripts/data_efficiency.py --config configs/experiments/data_efficiency_mock.yaml
```

Training on CPU is slow; use a GPU (`train.device: auto` picks CUDA when available). On Isaac Sim:

```bash
python scripts/isaac/validate_isaac.py --headless
python scripts/isaac/collect_isaac.py --config configs/experiments/collect_isaac.yaml --headless
python scripts/train.py --set dataset=datasets/isaac_terrain
python scripts/isaac/online_demo.py --checkpoint runs/terrain_hierarchical/model.pt --headless
```

The primary simulator is MuJoCo-Warp through mjlab (separate Python 3.12 env; see [docs/mjlab.md](docs/mjlab.md)),
with physical terrain compliance (simulation v1). Version-1 workflow:

```bash
python scripts/mjlab/validate_mjlab.py
python scripts/mjlab/collect_mjlab.py --config configs/experiments/collect_mjlab_large.yaml   # 2400 episodes
python scripts/train.py --config configs/experiments/terrain_v1.yaml                           # somato_v1
python scripts/data_efficiency.py --config configs/experiments/v1_signs_of_life.yaml           # vs. baselines
python scripts/render_episodes.py datasets/mjlab_v1_2400 --out outputs/mjlab_snake.gif
```

## What is swappable, and where

| To change... | Edit | Code |
|---|---|---|
| Tactile technology (FSR, capacitive, binary, ideal), IMU and motor models | `configs/sensors/*.yaml` | `somato/sensors/` (`@SENSOR_MODELS.register`) |
| Robot and sensor geometry | `configs/robots/*.yaml` (snake generator or any URDF + placement generators) | `somato/robots/`, `somato/geometry/generators.py` |
| Stage 1 / 2 / 3 variants | `configs/models/*.yaml` | registries in `somato/models/` |
| Terrain classes | `configs/terrains/ice_snow_v1.yaml` (v0: `ice_forms.yaml`) | `somato/sim/terrain.py` |
| Simulator | implement `SimBackend` | `somato/sim/backend.py` (mock, Isaac Lab, mjlab) |
| Data source at runtime | sim, replay or hardware | `somato/sources/` (`HardwareSource` template) |

Datasets store **ideal stimuli** (pressure/shear per taxel, true joint state, specific force). The
sensor technology is applied at training time with fresh noise every epoch, so comparing FSR and
capacitive skins is a config change, not a re-simulation:

```bash
python scripts/train.py --set sensors=configs/sensors/capacitive.yaml name=capacitive
```

## Repository layout

```
src/somato/
  robots/       RobotDescription (URDF in/out, forward kinematics), procedural snake, config factory
  geometry/     rotations, sensor placement generators, SensorLayout, LayoutInfo (clusters, rest pose)
  sensors/      sensor technology models (stimulus -> raw readings) and SensorSuite
  models/       stage 1 (temporal), graph + edge features, stage 2 (spatial), stage 3 (heads), baselines
  sim/          SimBackend API, terrain catalog, taxel contact model, stimulus pipeline, multi-rate runner
    mock/       planar snake simulator (pure torch) for tests and quick experiments
    isaaclab/   Isaac Lab scene + backend (imported only when Isaac is present)
  control/      serpenoid gaits (lateral undulation, sidewinding, rolling)
  data/         episode storage, collection, windowed datasets, stratified splits
  training/     batch preparation + augmentation, tasks, trainer, experiments, data-efficiency study
  validation/   simulator integration checks (frames, force reporting, friction sign, terrain wiring)
  sources/      SimSource, ReplaySource, HardwareSource template
  runtime/      OnlineEncoder (streaming inference with persistent state)
configs/        robots, sensors, terrains, models, experiments
scripts/        collect / train / data_efficiency / validate (+ scripts/isaac/*)
tests/          unit and integration tests (incl. a fake isaaclab module for the Isaac adapter)
docs/           architecture.md, investigation_log.md (plan), findings.md (results), mjlab.md, isaac_sim.md,
                references/ (literature), archive/ (version 0, with caveats)
```

See [docs/architecture.md](docs/architecture.md) for design decisions and the mapping to the research
plan.
