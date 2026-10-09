# Archive: Phase 1, version 0 (2026-10-07 to 2026-10-08)

Documents and configs from the first prototype and its experiments. They are kept as a reference, but **none of
the results test the hypothesis as intended**. The setup differed from the research intent in the ways listed
below; the current plan is in `docs/research_plan.md`, the current log in `docs/investigation_log.md`.

| file | what it is |
|---|---|
| `investigation_log_v0.md` | running log of the v0 experiments (decisions, findings with numbers, task queue) |
| `architecture_v0.md` | design document of the v0 code |
| `architecture_review_v0.md` | review of the v0 code against the research intent, which motivated version 1 |
| `../../configs/archive/v0/` | experiment and model configs of the v0 studies |

## What v0 was, and why its results are caveated

**Model (v0 hierarchical).**
* Stage 2 was one graph over *all* sensor types. Each taxel connected to its 12 nearest taxels, 3 nearest joints and
  the IMU, so modalities were mixed at stage 2. The intent is to keep them segregated until stage 3.
* The continuous 3D kernel was evaluated on a KNN neighbourhood (count-based, not a physical support radius),
  rebuilt at every step.
* Stage 3 was mean+max pooling over all nodes (cluster attention was available but had a ~200-step
  initialization plateau), with no temporal processing after stage 2.

**Baselines.**
* The flat GRU used hand-crafted per-chunk statistics (mean, std, last) and received **no kinematics**.
* It also ignored the sensor-dropout mask, so it trained without the 5 % taxel dropout the other models got.
* There was no transformer baseline with kinematics. Flat-vs-structured differences therefore mixed four factors,
  not structure alone.

**Simulation (v0 terrain catalog `configs/terrains/ice_forms.yaml`).**
* Only sliding friction was physical; the terrain was a rigid plane.
* *Sinkage* was a constant virtual offset in the taxel contact model, a clean synthetic footprint-width cue that a
  local geometric operator detects by construction.
* Micro-texture was unfiltered (except in `mjlab_terrain_2400_skin`), so most of it was invisible to a real
  2 mm skin.
* Friction boxes covered only near-melting ice, so friction alone separated classes better than it would in the
  cold.
* The FSR sensor parameters were fixed and identical across taxels.

**Protocol.**
* v1: fixed 20 epochs, so small data fractions got fewer updates.
* v2: min_steps 3000.
* v3: a fixed 3000-step budget, which under-trained the full-data runs (10k steps changed the conclusion).
* All results are single-seed.

## Results worth remembering (with their caveats)

| result | caveat |
|---|---|
| Isaac v1 study: flat 0.80 vs hierarchical 0.63 at full data | invalid: training-budget artifact plus the cluster-attention plateau |
| v2 (600 episodes): hierarchical 0.887, flat 0.874, no_interaction 0.752 | one seed, 90-episode test set (±3 points) |
| Modality ablation (v2): touch carries most information; the structured model leads by 7 points on tactile-only input | mixed-modality graph, synthetic sinkage |
| Sensor physics: with shear sensing, per-taxel processing alone reaches 0.93 (stage 2 worth ~2 points vs 13 with normal-only FSR) | the value of spatial interaction depends on what each taxel measures; likely still true |
| v3 learning curves: geometric stage 2 +19–21 points over flat at 42 episodes, +8 at 168 | much of the gain is fresh snow and concrete; fresh snow's cue is the synthetic sinkage footprint; flat baseline confounded; one seed |
| v3 full data, 10k steps: hierarchical 0.946, flat 0.942, attn_dist 0.936; NLL 0.14 vs 0.27 | the 3k-step "flat wins at full data" was a budget artifact; calibration gap consistent across studies |
| Texture-reliance test: hierarchical −6 points without texture, flat ±0 | ±3.5 points SE: suggestive only |
| Robustness first look: flat (full data) −32 points at 3× sensor noise, −12 with slower FSR unloading | flat's std features are noise-sensitive by construction; one checkpoint |

Engineering findings that still hold (and live in the current docs/code):
* the Isaac vs MuJoCo comparison and the simulator decision (`docs/mjlab.md`, `docs/isaac_sim.md`);
* integrate-and-dump sensor sampling;
* float16 overflow handling;
* DC-motor actuator parity;
* elliptic friction cones;
* the exact-distance speed-up;
* vectorized sensor models.

## Using the archived configs

Paths inside the archived configs point to the pre-archive locations: `configs/models/X` is now
`configs/archive/v0/models/X`, and `configs/experiments/X` is now `configs/archive/v0/experiments/X`, except for
`terrain_mock.yaml`, `collect_mjlab.yaml` and `collect_isaac.yaml`, which stayed in `configs/experiments/`. The v0
model configs carry `fusion: mixed`, which reproduces the v0 stage 2 with the current code. Datasets and run
directories (`datasets/`, `runs/`) are not in git; the v0 runs are under `runs/v3/`, `runs/v3_*`, `runs/v2*` and
`runs/isaac600*`.
