# Architecture and design decisions

This document maps the research plan (hierarchical somatosensory encoder, Phase 1) onto the code
and records the choices made in the first prototype, so they can be revisited deliberately.

## 1. Data flow and rates

Everything is organized around the **latent step** (default 50 Hz). Each sensor group samples
`S_g = rate_g / latent_hz` times per latent step (defaults: tactile 500 Hz → 10, joints and IMU 250 Hz
→ 5; physics runs at 1 kHz).

| tensor | shape | meaning |
|---|---|---|
| `readings[g]` | `[B, L, S_g, N_g, C_g]` | raw readings of group `g` over `L` latent steps |
| `pos`, `rot` | `[B, L, N, 3]`, `[B, L, N, 3, 3]` | world pose of every sensor at each latent step |
| stage 1 output | `[B, L, N, D]` | per-sensor latents (groups concatenated in layout order) |
| stage 2 output | `[B, k, N, D]` | after geometric interaction (last `k` steps) |
| head output | `[B, k, ...]` | one prediction per output step |

A **sensor group** is "a sensor kind on a given technology": the stage-1 encoder and the sensor model
are shared within a group. If a robot carries FSRs on its belly and capacitive taxels on its sides,
make two groups (`tactile_belly`, `tactile_side`, both `kind: tactile`).

## 2. Stage 1: temporal, per sensor

*Proposal:* a sensor-specific local encoder with a hidden state turns a high-frequency history into a
lower-frequency, position-agnostic latent and absorbs dynamics (vibration) and artifacts (hysteresis).

* Weights are shared across all sensors of a group. The encoder never sees position or neighbors,
  and this is enforced by a test (`test_temporal_encoder_is_per_sensor`).
* `conv_gru` (default): 1D convolutions inside each chunk act as a learned filter bank; chunk
  statistics, including the raw mean and last sample that carry the DC load, feed a GRU at the
  latent rate.
* `receptor`: a bio-inspired variant. A learnable bank of first-order low-pass filters (log-spaced
  time constants) yields slowly-adapting (SA) features, and the high-pass residual energy yields
  rapidly-adapting (RA / Pacinian-like) vibration features. It is implemented as a causal FIR, and
  the filter history is part of the streaming state.
* `gru`: recurrence at the full sensor rate (most expressive, slowest). `mlp`: memoryless ablation.
* Input standardization statistics are fitted from data per group (`InputNormalizer`), since FSR
  volts, capacitance ratios and torques live on different scales.
* **Streaming**: running `L` steps at once equals `L` single-step calls (tested for every encoder).
  Deployment uses `somato.runtime.OnlineEncoder`.
* **Training horizon vs. deployment horizon.** By default each training window (25 latent steps = 0.5 s)
  starts from a zero state, so the recurrent state is never trained beyond 0.5 s of history. In streaming
  deployment it runs indefinitely, and accuracy decays with time since reset: on Isaac data it falls from
  0.58 at 0.5 s to 0.33 at 3 s. With a fresh 25-step state it stays at 0.55–0.60. The sensor models'
  own state (FSR hysteresis/creep, IMU bias walk) does not contribute: streaming vs. per-window sensor
  state changes accuracy by < 0.03. Two opt-in remedies leave the architecture unchanged:
  * `train_sequence: N` (training): crops of `N` steps processed in `window`-step chunks with the state
    carried, detached, between chunks (truncated BPTT). This trains predictions at every horizon up to `N`.
  * `OnlineEncoder(window=N)` / `online_demo.py --window N` (deployment): re-run the model on the last
    `N` steps from a fresh state at every step. This is exactly the training regime, at ~`N`× stage-1 cost.

## 3. Stage 2: geometry-informed interaction

### Graph
`knn_graph` builds a dense `[B, N, K]` neighbor index from the **current** sensor positions, rebuilt
every step (`graph.dynamic: true`, matching the "kinematic distance" of Jiang et al.). Alternatively
it is built once from the rest pose (`dynamic: false`, "geometric distance"), while edge features still
use current poses.

* `k_per_group` (e.g. `{tactile: 12, joint: 3, imu: 1}`) takes the nearest neighbors *from each group*.
  Without it, a few proprioceptive nodes among hundreds of taxels would never be selected, and
  proprioception would not fuse with touch locally.
* Regular layouts (taxel rings) produce exact distance ties. The KNN quantizes distances (0.1 mm),
  breaks ties by index, and keeps a few extra slots so that a tie group straddling the k-th neighbor
  is included whole. This keeps the neighbor sets invariant to rigid motions. In tests, a few rows per
  million still differ, when a distance lands within float error of a quantization boundary.
* Joints and IMUs are graph nodes like taxels. A joint sensor sits at the joint origin with +z along
  the axis; an IMU sits at its mount pose.

### Edge features (the inductive-bias axis)
The experimental progression suggested in the design discussion,
*no geometry → distance → relative pose → equivariant*, is a single config key:

| `edge_features` | invariance |
|---|---|
| `none` | – (neighbors are still chosen geometrically) |
| `distance` | E(3) |
| `rel_pos_world` | translation only (a deliberately weaker baseline) |
| `rel_pos_local` (default) | SE(3): `R_i^T (p_j - p_i)`, j's position in i's sensor frame |
| `rel_pose_local` | SE(3), adds `R_i^T R_j` (6D) |
| `ppf` | SE(3), independent of taxels' arbitrary in-plane x/y axes |

Using sensor *i*'s own frame realizes "sensor *i* only needs to know how *j* is situated relative to
it". Changing the global frame changes nothing, and tests verify this for the full model.

### Continuous kernel convolution (default operator)
`y_i = Σ_j W(e_ij) h_j` with `W(e) = Σ_b φ_b(e) W_b`: an MLP maps relative geometry to `num_basis`
coefficients that mix learned basis matrices. This is the "weight field" idea, a CNN kernel that is
continuous in 3D and shared by all sensors (similar relative geometry ⇒ similar interaction). The
basis factorization keeps per-edge memory at `num_basis` scalars instead of `D×D`. Options:

* `envelope`: a learnable Gaussian window in distance (a smooth locality prior).
* `aggregation: area`: weights neighbors by taxel area, a quadrature of the continuous convolution
  integral that reduces sensitivity to sensor density (the paper's "density" concern).

Alternatives behind the same interface (`spatial.type`): `geo_attention` (local attention with an
edge bias and a learnable `-λ d²` prior, the suggestion to bias attention by distance rather than
hard-code it), `egnn`, `full_attention` (fully connected, optionally distance-biased), and `none`.
Together these cover the progression *fully connected → local → hierarchical*.

### Not (yet) implemented
Latents with explicit vector/tensor channels transformed by `ρ(T)` (equivariant latents) are not in
this prototype. Following the design discussion, geometry currently enters only through the
interaction operator, so latents stay invariant. Adding an equivariant layer would mean a new
`SPATIAL_LAYERS` entry operating on `(scalar, vector)` features.

## 4. Stage 3: task heads and global attention

The proposal places a "task-specific global attention layer" at stage 3. Here it is a head option:
`pool: cluster_attention` pools sensors into fixed geometric clusters (one per link by default, or
farthest-point clusters), runs transformer layers between clusters with a learned inter-cluster
distance bias, then pools. This is the bipartite / hierarchical formulation, sensors → regions →
global, with O(N + C²) cost. Simpler pools (`mean`, `max`, `meanmax`, `attention`) are available.
Heads predict per output step. Window-level tasks average the loss over steps, and online use reads
the latest step. `per_body` heads give per-link outputs (e.g. slip).

## 5. Simulation layering

```
SimBackend (physics only) ──RawSimState──► StimulusPipeline ──stimuli──► SensorSuite ──readings──► model
 mock / Isaac Lab / mjlab                   taxel contact model,            FSR / capacitive /
                                            IMU specific force, joints      MEMS / motor models
```

* Backends report body poses and velocities, joint states and torques, and per-body ground contact
  (normal force, and friction force / contact point when available).
* The **taxel contact model** turns per-body contact into a dense taxel field. The footprint comes
  from geometry: a softplus of each taxel's indentation into the ground, so softer terrain with
  sinkage engages more taxels. The magnitude comes from physics, so the integrated pressure equals
  the body's contact force. On top of that it adds terrain micro-texture: sliding over a random
  texture field produces vibration at `v / λ`. Shear is the friction force distributed the same way,
  or a regularized Coulomb estimate. Rigid-body simulators do not resolve skin contact or
  ice/snow texture, so this layer is where most sim-to-real assumptions live. It is deliberately
  isolated and simulator-agnostic.
* The **mock planar snake** is quasi-static, with anisotropic Coulomb friction and actuator torques
  from distal friction moments. It is fast, deterministic and terrain-dependent, but has no inertia,
  no vertical dynamics and no obstacles. It is for tests and pipeline development only.

## 6. Data

Datasets store ideal stimuli (float16, compressed `.npz` per episode) and body poses, plus labels
and the sampled terrain parameters. Sensor models run at training time, which gives sensor-noise
augmentation for free and lets one dataset serve every sensor technology. Hardware recordings can
store readings instead (`data_kind: readings`).

## 7. Extending

* **New sensor technology**: subclass `SensorModel`, set `kind` and `output_channels`, decorate with
  `@SENSOR_MODELS.register("name")`, and reference it in a `configs/sensors/*.yaml`.
* **New robot**: point `robot: {type: urdf, path: ...}` at a URDF and describe sensors with the
  placement generators (`cylinder`, `sphere`, `grid_patch`, `explicit` from CAD files, `point`,
  `joints`). Joint sensors and FK come from the URDF.
* **New stage-1/2/3 module**: register in `TEMPORAL_ENCODERS`, `SPATIAL_LAYERS` or `HEADS`.
* **New task**: subclass `Task` in `training/tasks.py` and register it.
* **New simulator**: implement `SimBackend` (6 methods). `BackendValidator` then checks it.
* **Hardware**: subclass `HardwareSource` (`read_group` per sensor group). Poses come from FK of
  the joint encoders. The root can stay at the identity because stage 2 uses relative, local-frame
  geometry.

## 8. Open items and known limitations

* Sensor-model parameters (FSR curve, capacitive modulus, IMU noise) are plausible placeholders;
  calibrate them on the benchtop array.
* Taxel contact is localized only against the ground plane. Contact with obstacles keeps the
  correct per-body force but is placed on ground-facing taxels. Heightfields or SDF queries would
  lift this.
* Self-supervised objectives (masked-sensor prediction, reconstruction) are not implemented yet. The
  model already returns stage-2 features (`return_features=True`), and `node_mask` provides masking.
* The Isaac Lab backend is verified on Isaac Lab 2.3.2 / Isaac Sim 5.1 (see `isaac_sim.md`) and is
  exercised in CI against a fake API. The mjlab (MuJoCo-Warp) backend runs the same robot and checks for
  a PhysX vs. MuJoCo contact comparison (see `mjlab.md`). Run the validation script after any change.
* Sensor streams are sampled from the physics rate with integrate-and-dump averaging
  (`RateConfig.anti_alias`). Point sampling aliased PhysX's step-to-step contact chatter into the IMU and
  tactile channels.
