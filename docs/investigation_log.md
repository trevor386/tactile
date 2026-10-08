# Investigation log (Phase 1)

Running notes for the Phase 1 question: what inductive bias (structure) suits distributed tactile +
proprioceptive sensing, and does the somatosensory-inspired hierarchy (per-sensor temporal encoder →
geometric interaction → global head) learn from less data than an unstructured model of the same size?
Working hypothesis (user): there is a data-efficiency vs. imposed-structure trade-off; strongly
structured models need less data but may be less expressive.

Keep this file current: decisions with their reasons, measured findings (with numbers and dates),
open problems, and the task queue. The design docs are `architecture.md`, `isaac_sim.md` and `mjlab.md`.

## Decisions

| date | decision | why |
|---|---|---|
| 2026-10-07 | Isaac Lab 2.3.2 on Isaac Sim 5.1 in its own env (`isaaclab23`) | the existing Isaac Sim 6.0.1 is not supported by Isaac Lab 2.3 |
| 2026-10-07 | PhysX solver 16 position / 4 velocity iterations | 8/1 makes resting contacts chatter every step |
| 2026-10-07 | `SimRunner` averages stimuli over each sensor sample period | point sampling aliased physics-rate chatter into the sensor streams |
| 2026-10-08 | DC-motor actuator in both simulators (stall 12 Nm, no-load 8 rad/s, continuous 6 Nm) | Isaac imported the URDF velocity as a hard PhysX clamp that MuJoCo cannot express; this confounded the comparison |
| 2026-10-08 | **Primary simulator: mjlab (MuJoCo-Warp), default stiff contact** | agrees with Isaac (paired speed r 0.99, friction readout r 0.999), 6× faster, physically tunable contact; see `mjlab.md` |

## Findings

**Simulators (2026-10-08).** Same robot, gait and terrain, paired episodes, identical actuators.
* The two simulators agree on locomotion, friction readout and tumbling.
* Isaac has rigid-impact spikes (IMU up to 944 m/s², taxels up to 1.4 MPa).
* mjlab's default contact flickers: links lose contact for 1–2 ms at micron separation. A softer contact
  removes the flicker but also softens friction and changes locomotion (speed +22 %, speed no longer
  depends on friction), so it is not used. Details are in `mjlab.md`.

**Terrain classification on Isaac data (2026-10-07; FSR skin, 20 epochs, one seed).**
* 600 episodes: hierarchical 0.535 test accuracy. The same model reaches 0.515 on the paired MuJoCo
  episodes.
* Data-efficiency study, test acc_last vs fraction of the 420 training episodes:

  | model | 0.1 | 0.25 | 0.5 | 1.0 |
  |---|---|---|---|---|
  | flat_gru (unstructured) | 0.520 | 0.591 | 0.693 | **0.798** |
  | hierarchical | 0.396 | 0.359 | 0.407 | 0.628 |
  | no_interaction (no stage 2) | 0.254 | 0.319 | 0.420 | 0.674 |

  The opposite of the hypothesis. Before interpreting it, the benchmark and the training have to be
  checked: see the open problems below.

**Streaming (2026-10-07).**
* Online accuracy drops from 0.58 (0.5 s after reset) to 0.33 (3 s), because the stage-1 recurrent state
  is only trained on 0.5 s windows. The sensor-model state does not contribute.
* Opt-in fixes: `train_sequence` (truncated BPTT) and `OnlineEncoder(window=N)`.
* The TBPTT model (`runs/isaac600_hier_tbptt`) is trained but not yet evaluated, because the logs were
  lost in a reboot.

**Training adequacy (2026-10-08).** The data-efficiency comparison above is confounded by training budget.
* The study trained a fixed 20 epochs, so small fractions also got proportionally fewer optimizer steps.
* Learning curves on Isaac data (train loss start → epoch 10 → epoch 20):
  * hierarchical, 100 %: 1.61 → 1.06 → 0.71, still improving (best val at the last epoch).
  * hierarchical, 25 %: 1.62 → 1.34 → 1.28, barely learning.
  * flat_gru: train loss 0.02, memorized and then overfit.
* Cause: the **cluster-attention head starts on a loss plateau**. Loss stays at ln 5 for ~150–200
  optimizer steps at lr 1e-3; lr 3e-3 never escapes in 300 steps; lr 3e-4 behaves like 1e-3. Val acc
  after 300 steps (mjlab data):

  | variant | val acc |
  |---|---|
  | cluster-attention head | 0.38 |
  | same backbone, mean+max head | 0.54 (no plateau) |
  | no_interaction | 0.49 |
  | flat_gru | 0.76 |

  Stages 1–2 learn fine; the head's optimization is the bottleneck.
* The TBPTT run (`isaac600_hier_tbptt`, cluster-attention head) never left the plateau in 3,360 steps
  and stayed at chance.
* Fix for comparisons: `train.min_steps` (≥ ~3,000 updates at every fraction) + early stopping, with
  the head choice as an explicit factor.

**Task ceiling (2026-10-08).** In mjlab, the terrain parameters that reach the sensors are:
* friction, in the physics;
* sinkage (taxel footprint width), texture amplitude and texture wavelength (taxel vibration), in the
  taxel model.

Isaac adds restitution. Anisotropy and undulation act only in the mock.

Bayes-optimal accuracy with *perfect* knowledge of the parameters (uniform sampling within each class
box, from the catalog):

| parameters | Bayes accuracy |
|---|---|
| friction only | 0.795 |
| friction + sinkage | 1.000 (same with texture and restitution added) |

So the task is fully solvable in principle. Beyond ~0.80, a model must read sinkage or texture from the
tactile signals: sinkage is 0 / ≤ 0.2 / 0.5–1.5 / 3–8 mm across classes and widens the taxel footprint.
The Isaac flat baseline's 0.80 sits right at the friction-only ceiling. Whether any model passes it is
the key question for the modality ablation.

**Plateau location and TBPTT (2026-10-08, 300–600-step screens on mjlab data).**
* Head without attention blocks (`cluster_layers: 0`): escapes after ~100 steps, val 0.41 at 300 steps.
  The attention blocks cause most of the delay, the token/LayerNorm path the rest.
* TBPTT:
  * 50-step crops (2 chunks) with the mean+max head learn well: val 0.68 (averaged over chunk ends) at
    600 steps.
  * 150-step sequences (6 chunks) stay at uniform prediction with *either* head. Open issue in the TBPTT
    path. Differences: 14 distinct batches per epoch vs 145; 6 correlated consecutive updates per batch.
    To test: 4 chunks, 3 chunks.
* bf16 autocast: 1.3× faster per step with identical learning curves; used for protocol v2.

**Protocol v2, mjlab data, full data, seed 0 (2026-10-08).** min_steps 3000, early stopping, bf16, widths matched to
~400 k params:

| model | test acc | test NLL | notes |
|---|---|---|---|
| hierarchical_meanmax | **0.887** | **0.318** | best epoch 19/20, still improving |
| hierarchical (cluster-attention head) | 0.880 | 0.326 | |
| flat_gru | 0.874 | 0.469 | train loss 0.008, memorizes; best at epoch 12 |
| hierarchical_meanmax_sid (+ sensor-ID embedding) | 0.857 | 0.446 | |
| no_interaction | 0.752 | 0.652 | |

* With adequate training the hierarchical model matches or beats the flat baseline and is better calibrated. The
  v1 "flat wins" result was a training-budget artifact (plus Isaac data).
* Stage 2 (geometric interaction) is worth +13 points.
* Absolute sensor identity does not help (slightly worse).
* All models exceed the friction-only ceiling (0.795) except no_interaction.
* Caveat: one seed and a 90-episode test set (SE ≈ ±3 points), so the 1–2-point differences are noise. Next: a
  larger dataset (thousands of episodes; mjlab collects 600 in 2 min) and ≥ 3 seeds.

**Modality ablation, protocol v2, mjlab, full data, seed 0 (2026-10-08).** Tactile only (FSR):

| model | test acc | NLL |
|---|---|---|
| hierarchical_meanmax | **0.822** | 0.52 |
| flat_gru | 0.752 | 1.01 |
| no_interaction | 0.544 | 1.04 |

| inputs | hierarchical_meanmax | flat_gru | no_interaction |
|---|---|---|---|
| all | **0.887** | 0.874 | 0.752 |
| tactile only | **0.822** | 0.752 | 0.544 |
| joint + IMU only | 0.685 | 0.661 | 0.683 |

* Touch carries most of the terrain information. Proprioception alone (~0.68) stays below even the friction-only
  ceiling: 0.5 s windows do not pin down friction from torques and motion.
* Structure matters where there is spatial data. For the 16 proprioceptive sensors the architectures tie, and
  stage 2 adds nothing. For the 384-taxel skin the structured model leads by 7 points, and stage 2 is worth 28.
* The flat model fuses proprioception well (0.752 → 0.874 with all inputs), so at full data it nearly closes the
  gap. Data efficiency is the open question for the v3 learning curves.

**Sensor technology: ideal skin with shear vs FSR (protocol v2, mjlab, full data, seed 0, 2026-10-08).**

| model | ideal + shear | FSR (normal only) | Δ |
|---|---|---|---|
| hierarchical_meanmax | **0.954** (NLL 0.13) | 0.887 | +6.7 |
| flat_gru | 0.939 (NLL 0.28) | 0.874 | +6.5 |
| no_interaction | 0.930 (NLL 0.20) | 0.752 | **+17.8** |

**The value of spatial interaction depends on what each sensor measures.**
* With shear, each taxel reads friction locally (shear/normal), so per-sensor processing alone reaches 0.93.
* With FSR (normal only), single taxels are ambiguous. Friction and sinkage must be inferred from the spatial
  pressure pattern, and stage 2 is worth 13 points.

So the appropriate inductive bias is not fixed: it depends on the sensor physics. This matters for sim-to-real,
because real skins differ in what each taxel resolves.

**Initialization signal (`scripts/analysis/init_signal.py`).** Every group's features still depend on the
input after stage 2: the per-sensor std across samples is 11–17 % of the feature RMS. Stage-1 features
are dominated by a sample-independent component at init (85–95 %), which is typical for an untrained GRU.

## What is (not) modelled: hysteresis and contact dynamics

* **Simulators.** Both resolve each link's contact as rigid (PhysX) or soft-constraint (MuJoCo)
  contact with Coulomb friction: 2 contact points per capsule. There is no skin viscoelasticity, no
  rubber friction (velocity/pressure dependence), and no stick-slip beyond Coulomb. PhysX also has a
  static/dynamic ratio of 1.1; MuJoCo has a single coefficient.
* **Taxel contact model (`sim/contact_model.py`).** Quasi-static. It distributes each link's force over
  its taxels by a geometric footprint (a softplus of indentation, which widens with terrain sinkage),
  adds micro-texture vibration at v/λ while sliding, and distributes shear like pressure. It has no
  dynamics or memory.
* **Sensor models (`sensors/`), applied at training time:**
  * FSR: rate-dependent hysteresis (loading 2 ms / unloading 20 ms), creep (6 %, τ 1.5 s), power-law
    nonlinearity, per-taxel gain spread, noise, ADC.
  * Capacitive: viscoelastic dielectric (20 %, τ 0.3 s), baseline drift.
  * Motor: friction, gain error, torque filtering.
  * IMU: bias, bias walk, scale, noise.
* **Consequence.** Mechanical hysteresis and complex contact dynamics exist only as sensor-level
  first-order filters. The sim is "simple", so simple models may look better in sim than they would on
  hardware.

## Open problems and hypotheses

1. **Is the hierarchical model learning what it should?** The flat baseline sees every joint torque and
   taxel directly. The hierarchical model mixes 15 joint nodes and 1 IMU node into 384 taxel nodes and
   pools them, which may dilute proprioception.
   * Test: modality ablations (tactile only / joint+IMU only) for both models.
2. **Is the benchmark testing tactile structure at all?**
   * The classes differ in friction (overlapping ranges for rough ice, packed snow and fresh snow), and in
     texture, sinkage and undulation, which act only through the taxel model.
   * The FSR skin senses no shear.
   * If proprioception alone solves the task, the benchmark mostly measures proprioceptive regression.
3. **Training adequacy.** 20 fixed epochs, one seed, no early stopping: the hierarchical model's
   training loss was still falling at epoch 20. Seed variance is unknown.
4. **Layer choices vs. the abstraction itself.** A GRU may not be the best temporal operator, a
   continuous kernel may not be the best spatial one, and the three-stage split may be too restrictive.
   Long-term.
5. **Sim simplicity bias.** Prefer evaluations that perturb the sensor or contact model between
   training and test (e.g. train on FSR v1, test on stronger hysteresis or a different skin model) as
   a proxy for sim-to-real robustness. Isaac → MuJoCo transfer is another proxy.

## Task queue (current plan; keep statuses current)

**Running now (GPU queue, sequential; logs in `runs/logs/`):**
First v2 result: hierarchical (cluster head), mjlab data, full data → **test acc 0.880** (NLL 0.33). Well above
the friction-only ceiling 0.795, so tactile cues are being used. v1 on Isaac data gave 0.628.
1. `runs/logs/queue_v2.sh`:
   * protocol v2, all inputs, full data, seed 0 (`configs/experiments/protocol_v2_mjlab.yaml` → `runs/v2/all/`):
     hierarchical (cluster head), hierarchical_meanmax, hierarchical_meanmax_sid, no_interaction, flat_gru.
   * modality ablations (`protocol_v2_modality_mjlab.yaml`): tactile only → `runs/v2/tactile/`;
     joint+imu only → `runs/v2/joint_imu/`.
   * Results: each `runs/v2/*/summary.md` / `results.csv`; per-run `log.jsonl` and `result.json` (with confusion).
2. `runs/logs/queue_v2b.sh` (starts after 1): ideal skin with shear, all inputs → `runs/v2/ideal_all/`.
3. (done) Literature review → `docs/references/sensor_terrain_calibration.md`.

4. `runs/logs/queue_v2c.sh`: collect `datasets/mjlab_terrain_2400` (seed 1; `collect_mjlab_large.yaml`).
5. `runs/logs/queue_v3b.sh` (replaces queue_v3.sh, which was stopped while still waiting): **main study** `configs/experiments/curves_v3_mjlab.yaml`. Learning curves at fractions
   0.025/0.1/0.25/1.0 of 1,680 training episodes. Structure spectrum with shared stage 1 and mean+max head:
   flat_gru → attn_sid (transformer + sensor IDs, no geometry) → attn_dist (attention + distance prior) →
   hierarchical_meanmax (local continuous kernel) → no_interaction. Protocol v3: 3,000 steps for every run,
   validation every 250 steps, best checkpoint, checkpoints saved. Seed 0 → `runs/v3/seed0/`. Then the
   **texture-reliance test**: collect `datasets/mjlab_terrain_600_notex` (`collect_mjlab_notex.yaml`, texture
   amplitude 0, otherwise paired with `mjlab_terrain_600_dc`) and train hierarchical_meanmax / no_interaction /
   flat_gru → `runs/v2/notex/`; compare with `runs/v2/all/`. Then seeds 1, 2 → `runs/v3/seed12/`. About 4–5 h
   per v3 seed batch.

**Next, when the results are in:**
- [ ] Analyse v2: does any model beat the friction-only ceiling (0.795)? Hierarchical vs flat after adequate
      training? What does the sensor-ID embedding add? How do the heads compare? What does each modality carry?
      What does ideal (shear) sensing add over FSR?
- [ ] Decide the head for subsequent studies (meanmax vs cluster attention; or fix the cluster head's init plateau).
- [ ] 3 seeds at full data for the key models (variance estimate).
- [ ] Main experiment: learning curves (fractions 0.05–1.0 × 3 seeds, protocol v2) along a spectrum of spatial
      structure with stage 1 and head fixed: flat vector → transformer over sensors + ID embeddings (no geometry) →
      full attention + distance bias → local continuous-kernel graph → no interaction.
- [ ] Robustness proxy for sim-to-real: evaluate trained models with `scripts/evaluate.py --sensors
      configs/sensors/fsr_degraded.yaml` (stronger hysteresis/creep/gain spread) and on Isaac data (cross-sim).

**Sim realism (from `docs/references/sensor_terrain_calibration.md`, literature review 2026-10-08):**
- [ ] Texture reliance test (queued in queue_v3b): if accuracy drops a lot without texture, the models exploit a cue
      a real 2 mm skin with 1 cm taxels would mostly filter out.
- [ ] Contact model: spatial skin filter (average texture over the taxel area, smoothing with skin depth), and pressure
      levels consistent with snow (1–2 kPa) vs ice (~10 kPa).
- [ ] Sensor models: FSR `tau_unload` 20–100 ms per taxel, rate-independent hysteresis (7–17 % full scale),
      log-time drift; capacitive `visco_frac` 0.05–0.12.
- [ ] Terrain: ice friction is strongly temperature dependent (0.05–0.15 near 0 °C, 0.4–0.9 at −10 to −25 °C);
      fresh-snow sinkage 3–15 mm; static/dynamic ratio 1.1–1.5. The current boxes cover only near-melting ice.

**Later / open:**
- [ ] TBPTT fails with 6 chunks per sequence (2 chunks work): test 3/4-chunk variants and more distinct sequences per
      epoch; until then use sliding-window inference (`OnlineEncoder(window=25)`) for streaming.
- [ ] Use the literature review to recalibrate sensor-model parameters and terrain ranges; consider a skin-dynamics
      (viscoelastic) layer in the contact model.
- [ ] Compliant skin in MuJoCo with stiff friction (explicit contact pairs with `solreffriction`), or skin dynamics in
      the contact model; would also remove the contact flicker.
- [ ] Physically modelled terrain compliance (per-env contact softness) in mjlab instead of only the taxel-model sinkage.
- [ ] Long-term: alternative stage-1/2/3 operators (TCN, transformer, receptor; GAT, full attention, EGNN), and whether
      the three-stage abstraction itself is too restrictive.

**Done (2026-10-07/08):**
- Isaac Lab 2.3.2 / Isaac Sim 5.1 bring-up and validation.
- mjlab backend and paired comparison; simulator decision (mjlab).
- DC-motor actuator parity.
- Streaming diagnosis; anti-aliasing.
- float16 overflow fix.
- Plateau diagnosis (cluster-attention head).
- Protocol-v2 tools: min_steps, bf16, grad-norm logging, input_groups, evaluate --sensors.
- Task ceiling analysis.
