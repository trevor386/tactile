# somato — hierarchical somatosensory encoding for robots

PyTorch code for a hierarchical encoder of dense tactile, IMU and joint (motor) sensing, built for
reactive locomotion on icy terrain. It includes simulator integration (Isaac Lab on Isaac Sim 5.x,
plus a lightweight CPU mock) and the training and evaluation tooling for the Phase 1 data-efficiency
study.

```
sensor histories ──► Stage 1: temporal ──► Stage 2: geometric ──► Stage 3: task head ──► terrain class,
 (per sensor,         per-sensor shared     continuous-kernel        cluster attention       slip, ...
  high rate)          GRU encoder per       convolution over the     between body regions
                      sensor group          sensor graph (poses)     + task MLP
                      (no position info)
```

* **Stage 1** (`somato/models/temporal.py`): every sensor of a group (all taxels, all joints, the IMUs)
  shares one recurrent encoder. It sees only that sensor's own high-rate history (no position) and
  emits a latent at a lower rate. Options: `conv_gru` (default), `gru`, `receptor`
  (a mechanoreceptor-like multi-timescale filter bank) and `mlp` (no memory, for ablation).
* **Stage 2** (`somato/models/spatial.py`): sensors exchange information over a neighbor graph built from
  their *current* poses. The default operator is a **continuous kernel convolution**: a learned weight
  field over relative geometry, shared by every sensor (the 3D, irregular-grid analogue of a CNN
  kernel). Alternatives with the same interface: geometry-biased local attention, EGNN (the Jiang et
  al. 2025 baseline), full attention, and no interaction.
* **Stage 3** (`somato/models/heads.py`): pluggable task heads. Pooling ranges from mean/max to attention
  pooling to cluster attention, where body regions attend to each other with a distance bias.

The architecture is invariant to the robot's global pose (local-frame edge features). It does not
depend on the number or arrangement of sensors, so the same weights run on a robot with a different
sensor layout. Stage 1 runs in streaming mode with exactly the same results as window processing.

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

## What is swappable, and where

| To change... | Edit | Code |
|---|---|---|
| Tactile technology (FSR, capacitive, binary, ideal), IMU and motor models | `configs/sensors/*.yaml` | `somato/sensors/` (`@SENSOR_MODELS.register`) |
| Robot and sensor geometry | `configs/robots/*.yaml` (snake generator or any URDF + placement generators) | `somato/robots/`, `somato/geometry/generators.py` |
| Stage 1 / 2 / 3 variants | `configs/models/*.yaml` | registries in `somato/models/` |
| Terrain classes | `configs/terrains/ice_forms.yaml` | `somato/sim/terrain.py` |
| Simulator | implement `SimBackend` | `somato/sim/backend.py` (mock, Isaac Lab) |
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
docs/           architecture.md, isaac_sim.md
```

See [docs/architecture.md](docs/architecture.md) for design decisions and the mapping to the research
plan.
