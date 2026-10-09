# Architecture (version 1)

How the research intent maps onto the code, and why. Version 0 is in `docs/archive/architecture_v0.md`; the review
that motivated version 1 is in `docs/archive/architecture_review_v0.md`. Hypotheses and experiments that test each
choice are in `docs/investigation_log.md` (H-*, E-*).

## 1. The model: receptors → spinal cord → brain

```
 readings per sensor group (modality)            [B, L, S, N_g, C_g]   L latent steps (50 Hz), S samples per step
   │
   │ STAGE 1, "receptors" (per sensor, temporal)
   │   one encoder per group, shared by every sensor of the group (all taxels are the same kind of receptor);
   │   it sees only its own sensor's samples: a conv filter bank within each 20 ms step (vibration,
   │   transients, sensor artifacts such as hysteresis) and a GRU across steps → a latent per sensor and step
   ▼
 per-sensor latents                               [B, L, N_g, D]
   │
   │ STAGE 2, "spinal cord" (spatial, within a modality only), at every brain step (10 Hz)
   │   tactile: 3D continuous kernel over a physical support radius (45 mm ≈ 2 taxel pitches, ~12 taxels)
   │   joint:   the same operator along the kinematic chain (a joint and its two neighbours)
   │   imu:     single sensor, no spatial operator
   ▼
 per-sensor features (still segregated)          [B, L/5, N_g, D]
   │
   │ STAGE 3, "brain" (where modalities first meet)
   │   region tokens: mean ‖ max of stage-2 features per (modality, body) + modality embedding
   │   attention over tokens, biased by their *current* pairwise distances (follows the moving body)
   │   short pooled path + GRU over brain steps → task output
   ▼
 prediction per brain step                        [B, L/5, out]
```

Code: `somato/models/hierarchical.py` (stages and stride), `temporal.py` (stage 1), `spatial.py` + `graph.py`
(stage 2), `heads.py` (`BrainHead` and simpler heads). Config: `configs/models/somato_v1.yaml`.

### Stage 1: receptors
Unchanged from version 0. Each sensor group (a modality on a technology: FSR taxels, joint encoders with current
sensing, the IMU) has its own encoder, so different modalities never share weights or a latent code. Within a group the
encoder is shared and position-blind, a mechanoreceptor model applied at every site. It has to absorb the sensor
physics: FSR v1 has per-taxel unloading time constants and rate-independent hysteresis (`configs/sensors/fsr_v1.yaml`).
Variants: `conv_gru` (default), `receptor` (SA/RA-like filter bank), `gru` (full-rate recurrence), `mlp` (memoryless).

### Stage 2: spatial structure within a modality
**Segregation** (`fusion: segregated`, the default). Each group gets its own stage-2 operator and graph, so taxels
interact only with taxels and joints only with joints, and modalities stay apart until stage 3. This mirrors the
nervous system, where receptor types travel separate pathways until they converge in cortex. It also avoids forcing
different modalities into one latent code. `fusion: mixed` (version 0: one graph over all groups, each taxel also
linked to its nearest joints and the IMU) remains as an ablation (`somato_v1_mixed.yaml`, H-7).

**Two approaches to the same problem**, kept separate:
* **3D kernel** (`type: kernel3d`, the default): a CNN kernel made continuous in space.
  `y_i = Σ_{j: |p_j−p_i|<r} W(R_iᵀ(p_j−p_i)) h_j`, with `W(e) = Σ_b φ_b(e) W_b`: the kernel weights are a learned
  function of *where the neighbour sits in sensor i's own frame*, shared by every sensor, linear in the neighbour's
  features. Its support is a physical radius (`graph.radius`; `graph.k` only caps the neighbour count for memory)
  with a learnable Gaussian envelope. A direction-selective kernel can respond to a pattern moving along the body in
  one direction (H-9).
* **Graph** (`type: geo_attention` or `egnn`, `somato_v1_graph.yaml`): message passing on a k-nearest-neighbour graph,
  where each message depends on both endpoints' features as well as the edge geometry (local attention with a
  relative-position bias and a learnable distance prior). More expressive, less constrained (H-6).

Implementation note: both need a neighbour set; a capped ball query (KNN with a radius cut) is simply the efficient
way to evaluate a kernel with finite support, not a separate design choice.

**Why the poses are taken at every step.** A 3D kernel is evaluated at the *current* relative position of each
neighbour. Within a rigid link the relative positions never change. Across a bending joint they do: when the body
curls, taxels on adjacent links approach, enter or leave each other's support, and the kernel weight they receive
changes with their relative position. Re-reading the poses at every brain step is exactly how the kernel "naturally
handles a sensor moving relative to another one". The poses come from kinematics (on hardware: forward kinematics
from the joint encoders plus the known skin layout), and only relative geometry in a sensor's own frame is used, so
the model is invariant to where the robot is in the world.

**Moving stimuli.** Stage 2 runs on stage-1 latents, which summarize each taxel's recent history (when it was
loaded, how the load changed). A force sliding along the body therefore appears at one brain step as a spatial
gradient of "time since contact" across neighbouring taxels, which an oriented kernel can detect, much like a
Reichardt detector built from temporal filters plus spatial comparison. Whether this suffices, or a spatio-temporal
kernel is needed, is H-9 / E-13.

**Different sensor types in one graph** (v0 only). In mixed mode, KNN neighbours were chosen per group
(`k_per_group`), so each taxel took its 12 nearest taxels, 3 nearest joints and the IMU, all projected into one
shared latent space with a group embedding. Version 1 does not do this by default.

### Stage 3: the brain
`BrainHead` (`heads.py`, `type: brain`):
1. **Region tokens:** mean and max of stage-2 features per (group, body), e.g. 16 tactile link tokens, 15 joint
   tokens and 1 IMU token, plus a learned modality embedding.
2. **Dynamic geometry:** each token sits at the current centroid of its sensors; each attention head adds a learned
   function of the current pairwise token distance. The attention pattern therefore changes as the body moves; there
   is no fixed attention map and no per-region identity, so the same weights run on robots with other link counts.
3. **Attention layers**, then mean and max over tokens, concatenated with the mean and max of the *input* tokens (a
   short path that trains from the first step; v0's attention-only cluster head started on a ~200-step loss
   plateau).
4. **Memory:** a GRU over brain steps carries body-wide information through time, then an MLP gives the task output.

**Rates.** Stage 1 runs at the latent rate (50 Hz), stages 2–3 every `brain_stride` = 5 latent steps (10 Hz),
counted from the start of the stream so a 25-step window ends on a brain step. Streaming in chunks is exact: the
state carries the stage-1 states, the brain's GRU state and the step counter (`test_brain_head_stride_and_streaming`).
`output_steps` selects how many of the last brain steps produce outputs (the loss uses them).

Ablations within the family (H-5, H-7, H-8): `somato_v1_no_spatial` (no stage 2), `somato_v1_mixed` (modalities mixed in
stage 2), `somato_v1_pool` (static mean+max head, no brain).

## 2. Baselines (same inputs, no imposed structure)

Both get the kinematics too, as every body's pose (or each sensor's position and normal) expressed in the robot
frame, i.e. the IMU frame (`baselines.py`). A difference between them and the structured model therefore measures the
inductive bias, not an input. Both apply the same sensor-dropout augmentation: dead sensors read zero.

* **Flat GRU** (`flat_gru.yaml`, `flat_gru_raw.yaml`): per latent step, every sensor's readings (per-step
  statistics, or every raw sample) and the body poses flattened into one vector → MLP → GRU → MLP. No weight sharing
  across sensors, no locality, early fusion of all modalities. Tied to one layout.
* **Transformer** (`transformer.yaml`): one token per (sensor, 100 ms time patch) from its raw samples (a linear
  embedding per group), plus a Fourier encoding of the sensor's current position and normal in the robot frame, a
  group embedding and a time embedding; global self-attention over all sensors, modalities and time (~2,000 tokens,
  flash attention); class-token readout once per window. The least structured standard model.

Widths are matched to the structured model's parameter count (~636k; `match_params` in study configs). Compute is not
matched: per step at batch 32, somato_v1 takes 270 ms, the transformer 168 ms and the flat GRU 50 ms.

## 3. Simulation and data (version 1)

```
mjlab physics (1 kHz) ──► taxel contact model ──► stored ideal stimuli ──► sensor models (train/test time) ──► model
 terrain compliance          footprint from real       tactile 500 Hz          FSR v1, motor, MEMS IMU
 (explicit contact pairs)    penetration, felt          joint/IMU 250 Hz
 friction, gait              roughness (skin-filtered)
```

* **Terrain** (`configs/terrains/ice_snow_v1.yaml`, `docs/mjlab.md`, findings F-6 to F-13):
  * Compliance is physical. Each robot-ground contact is an explicit MuJoCo contact pair whose normal softness (solref,
    and an impedance profile in solimp) is set per environment, while its friction rows stay stiff, so soft snow is
    really penetrated (up to ~2 cm) without softening friction.
  * Friction ranges cover cold and near-melting conditions.
  * Only roughness at ≥ 15 mm wavelengths reaches the taxels.
  * The v0 synthetic sinkage offset is gone.
* **Taxel contact model** (`sim/contact_model.py`): distributes each link's contact force over its taxels by a
  footprint from the taxels' depth below the ground (now including real penetration), modulated by the
  skin-filtered roughness field.
* **Sensor models** run at training and test time on stored ideal stimuli, so one dataset serves every sensor
  technology and robustness tests can change the sensor without re-simulating.
* **Data** is stored as compressed episodes and loaded through a memory-mapped cache (`<dataset>/cache_v1/`).

## 4. Training

* Supervised, end to end (the task loss on the last output steps of 0.5 s windows), protocol v4:
  * budget max(3,000 steps, 15 epochs);
  * validation every 250 steps, best-validation checkpoint;
  * bf16.
* Self-supervised objectives (masked sensor prediction, reconstruction, cross-modal prediction) and stage-wise
  training with only the last stage task-specific are planned as separate experiments (H-10, T-11, E-14, E-15).
  Stage-local objectives are natural for this hierarchy: stage 1 can predict its own sensor's next samples, stage 2 a
  masked taxel from its neighbours.

## 5. Extending

* New sensor technology: subclass `SensorModel`, register it, reference it in `configs/sensors/*.yaml`.
* New robot: `robot: {type: urdf, path: ...}` plus placement generators; the structured model runs unchanged on new
  layouts (the baselines need retraining).
* New stage variant: register in `TEMPORAL_ENCODERS`, `SPATIAL_LAYERS` or `HEADS`. Per-group stage-2 configs go in
  `spatial_overrides`.
* New task: subclass `Task` in `training/tasks.py`.
* New simulator: implement `SimBackend`; `BackendValidator` checks it.
