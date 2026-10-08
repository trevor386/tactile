# Architecture review: what the code actually does (2026-10-08)

A snapshot for checking design decisions against the research intent. It describes the code as of commit
`d58b156`, not the original plan. Section 5 lists the places where the two may differ.

## 1. Pipeline

```
 SIM (mjlab, 1 kHz)                    STORED DATASET (ideal stimuli)         TRAIN/TEST TIME
 ───────────────────                   ───────────────────────────            ────────────────────────────
 16-link snake, open-loop      ──►     per episode, 3 s @ 50 Hz latent:  ──►  sensor models (random per sample):
 sidewinding gait on a flat            tactile [150,10,384,3] Pa              FSR → 1 ch (normal only)
 rigid plane; per-env friction         joint   [150, 5, 15,4]                 motor → pos, vel, torque, error
        │                              imu     [150, 5,  1,6]                 MEMS IMU → acc, gyro
        ▼                              body poses, class label                        │
 per-link NET contact force    ──►     (taxel stimulus = contact                       ▼
 + taxel contact model                  model output, anti-aliased)           0.5 s windows (25 latent steps)
 (footprint · sinkage · texture)                                              + augmentation → model
```

The models (all ~324k parameters, widths matched):

```
 HIERARCHICAL FAMILY (shared stage 1 and head; stage 2 varies)          FLAT BASELINE
 ──────────────────────────────────────────────────────────            ─────────────────────────────────
 readings[g] [B,L,S,N,C]                                                readings, all sensors
   │ stage 1: per-sensor conv1d (within each 20 ms chunk)                 │ per chunk & sensor channel:
   │          → chunk stats → GRU across chunks                            │ mean, std, last (hand-crafted)
   │          weights shared by all sensors of a group, no position        │ concat → 1350-d vector
   ▼                                                                       ▼
 [B,L,N,64] + group embedding                                            2-layer MLP (width 133)
   │ stage 2 (last 4 steps only, each step independently):                 │
   │   no_interaction : per-node FFN                                       ▼
   │   attn_sid       : global attention + learned sensor-ID embedding   GRU over time (integrates
   │   attn_dist      : global attention, bias −λ·d²  (soft locality)      the whole body over time)
   │   hierarchical   : continuous-kernel conv on a dynamic KNN graph      │
   │                    (12 taxels + 3 joints + 1 IMU neighbours,          ▼
   │                    edge = neighbour position in sensor i's frame)   MLP → 5 logits per step
   ▼
 stage 3: masked mean ‖ max over nodes → MLP → 5 logits per step
          (no recurrence after stage 2)
```

## 2. Task and how it exists in simulation

* **Task.** 5-way terrain classification (glare ice, rough ice, packed snow, fresh snow, concrete) from a 0.5 s window.
  The loss covers the last 4 latent steps; the metric is accuracy at the last step. The label is per episode.
* **Data.** 2400 episodes (480 per class), each 3 s after a 1 s warm-up. Stratified 70/15/15 split by episode; the
  learning curves use 42 / 168 / 420 / 1680 training episodes. Same robot, same sensor layout and the same
  simulator in train and test.
* **Physical in mjlab:** only *sliding friction* (one isotropic coefficient per env). It changes how the snake moves
  and the per-link contact force.
* **Synthetic, added by our taxel model (`sim/contact_model.py`), not by physics:**
  * how each link's *net* force is spread over its 24 taxels: footprint ∝ facing × softplus(indentation / 1 mm);
  * **sinkage**: the robot does not sink. A virtual surface raised by 0–8 mm widens the footprint, so more
    taxels engage;
  * **micro-texture**: pressure × (1 + A·random field(x, y)), giving vibration at v/λ when sliding. It is
    spatially low-passed (σ 3.5 mm) only in the `_skin` dataset.
* **In the catalog but not enacted in mjlab:** restitution (Isaac only), friction anisotropy, undulation (the
  terrain is a flat plane).
* **Class boxes.** Friction overlaps for rough ice, packed snow and fresh snow. The Bayes ceiling is 0.795 from
  friction alone and 1.0 from friction + sinkage.
* **Gait.** Open-loop serpenoid with amplitude, frequency and phase random per episode, independent of the class.
  It is tracked by PD control with a DC-motor torque–speed limit.
* **Sensors.**
  * Layout: 384 taxels (3 rings × 8 per link, 2 mm skin), 15 joint sensors, 1 IMU on the head.
  * Rates: tactile 500 Hz, joint and IMU 250 Hz, period-averaged.
  * The FSR measures normal force only (no shear), with first-order hysteresis and creep, gain spread, noise
    and a 12-bit ADC. The sensor models run at training and test time on the stored ideal stimuli.

## 3. Training protocol

* Optimizer: AdamW (lr 1e-3, weight decay 1e-4), cosine schedule, batch 32, gradient clip 1, bf16.
* Checkpoint: best validation `acc_last`, evaluated every 250 steps; the test set is held out.
* Step budget:
  * v3 used 3000 steps, which was shown to under-train at full data.
  * v4 uses max(3000 steps, 15 epochs).
* Augmentation: random yaw and xy translation (has no effect on the invariant models), and 5 % taxel dropout.
* So far there is one seed; seeds 1–2 are queued.

## 4. Assumptions built in

1. Tactile information is local and stationary: one stage-1 encoder per sensor group, shared by every taxel.
   Stage 2 interactions depend only on relative geometry, not on which sensor it is.
2. Temporal processing happens before spatial mixing, per sensor. Space is mixed per step, and time is not
   revisited after mixing.
3. Sensor geometry is known exactly at every step. It comes from simulator body poses; on hardware this would
   be forward kinematics plus the skin layout. Only relative geometry is used (SE(3)-invariant edges, dynamic KNN).
4. The task is answerable from 0.5 s. There is no long-horizon memory (TBPTT is off; streaming uses a sliding
   window).
5. Contact is quasi-static and per link. The within-link pressure pattern follows from geometry plus the sinkage
   and texture parameters, not from contact mechanics. Skin dynamics exist only as sensor-level filters.

## 5. Decisions that may conflict with the intent or confound the hypothesis

The hypothesis is that imposed structure trades off against data efficiency, and the aim is to find the right
inductive bias for touch. Most important first:

1. **The spatial cues are designed by us, and in a form local geometric operators detect by construction.**
   * Sinkage (the main fresh-snow cue) is a wider footprint across neighbouring taxels, produced by the taxel
     model, not by physics.
   * Much of the structured models' low-data gain is on fresh snow (+26–29 points at 42 episodes). Concrete
     gains even more (+37–41), for reasons not yet understood. Part of the "structure helps" result may reflect
     the contact model's design.
   * Fix: physical terrain compliance in MuJoCo (per-env soft contact or a deformable layer) and/or calibrated
     footprints. Report results with and without synthetic cues (the notex and skin datasets do part of this).
2. **The flat baseline differs from the structured family in more than structure**, so flat-vs-structured
   mixes four factors:
   * its input front-end is hand-crafted chunk statistics (mean, std, last) instead of the learned per-sensor
     conv + GRU. The std features are what make it fragile to sensor noise (−32 points at 3× noise);
   * it **ignores the sensor-dropout mask**, so it trains without the 5 % taxel-dropout regularization the
     others get;
   * it receives no geometry at all;
   * it has a GRU *after* fusing all sensors, which the hierarchical head lacks.

   The within-family spectrum (no_interaction → attn_sid → attn_dist → local conv) shares stage 1, the head and
   the augmentation, so it is the clean comparison. Fix: add a "flat with shared stage 1" control (per-sensor
   latents concatenated in fixed order, then MLP and GRU) and make the flat model honour the dropout mask.
3. **Geometry is extra information, not only an inductive bias.** The structured models get exact sensor poses
   every step and the flat one gets none. Where flat loses, it is unclear how much is bias and how much is
   missing kinematic information. Fix: give flat the sensor positions (or forward kinematics) as inputs, or
   accept the comparison as "bias + kinematic knowledge".
4. **No test of what geometric structure is supposed to buy.** The robot, sensor layout and simulator are fixed
   and the test is in-distribution, so layout transfer, dead taxels and sim-to-real are untested. The sensor-ID
   model can memorize the layout. The robustness script (sensor perturbations) and Isaac→mjlab evaluation are a
   start.
5. **The hierarchical head is per-step (no recurrence after stage 2).** Cross-body temporal patterns (e.g. a
   contact wave travelling along the body) can only enter through each sensor's own history. The flat model
   integrates the whole body over time. This is a disadvantage for the structured side; `recurrent: true` on
   the head is available but unused.
6. **Stage 3 default changed.** The planned cluster-attention head is still implemented (pluggable), but the
   comparisons use mean+max pooling, because cluster attention has a ~200-step initialization plateau.
7. **The simulation is simple:**
   * a flat rigid plane, with friction the only physical terrain property;
   * a normal-only FSR;
   * quasi-static taxels;
   * hysteresis only as a sensor filter.

   Simple simulations may favour simple solutions. The value of stage 2 already depends strongly on sensor
   physics: +13 points with FSR vs small with shear sensing.
8. **Parameters are matched, compute is not.** The structured models use ~5× the compute per step. Most of the
   flat model's parameters sit in its layout-specific input layer.
9. **Protocol caveats.** The v3 full-data results were under-trained (fixed in v4). There is one seed so far and
   the test SE is ≈ ±1.2–2 points; differences of 1–2 points are noise.
