# Investigation log (Phase 1, v0) — ARCHIVED

> **Archived 2026-10-08.** These experiments used the v0 model (modalities mixed in stage 2), v0 baselines (flat
> GRU without kinematics) and the v0 simulation (synthetic sinkage, unfiltered texture, near-melting friction). Read
> `docs/archive/README.md` for the caveats before using any number here. The "RESUME HERE" section and task queue
> below are obsolete; the current ones are in `docs/investigation_log.md`.

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

**MAIN STUDY v3, seed 0 (2026-10-08): learning curves along a structure spectrum.** mjlab_terrain_2400, FSR skin,
shared stage 1 + mean+max head, ~324 k params each, 3,000 steps per run, best-val checkpoint, 360-episode test
set (`runs/v3/seed0/`). Test acc (training episodes):

| model (stage 2) | 42 | 168 | 420 | 1680 |
|---|---|---|---|---|
| flat_gru (no structure) | 0.419 | 0.756 | 0.877 | **0.937** |
| attn_sid (attention + sensor IDs, no geometry) | 0.457 | 0.806 | 0.855 | 0.885 |
| attn_dist (attention + distance prior) | **0.631** | 0.831 | 0.878 | 0.895 |
| hierarchical_meanmax (local continuous kernel) | 0.612 | **0.834** | 0.878 | 0.902 |
| no_interaction | 0.421 | 0.771 | 0.778 | 0.830 |

* **A data-efficiency vs. structure trade-off, as hypothesized.**
  * Geometric structure wins at low data: +19–21 points at 42 episodes, +8 at 168. Flat needs ~2–2.5× the data
    to match.
  * The curves cross at ~420 episodes. At 1,680 episodes the unstructured flat model is best.
* A soft distance prior (attn_dist) is as data-efficient as strict locality. Attention without geometry is only
  slightly better than flat at low data and worse at full data.
* NLL at 42 episodes: flat 3.9 vs hierarchical 2.0 (flat is badly overconfident).
* **Caveat:** at full data every model's best checkpoint was the last evaluation (~3,400 steps), so all were still
  improving. A 10,000-step budget check is queued, then seeds 1–2.
* **Per-class recall** (from the saved confusion matrices) locates both effects:

  | class | 42 eps: flat | 42 eps: geometric | 1680 eps: flat | 1680 eps: hierarchical |
  |---|---|---|---|---|
  | fresh snow | 0.54 | 0.80–0.83 | 0.97 | 0.98 |
  | concrete | 0.39 | 0.76–0.80 | 0.97 | 0.97 |
  | glare ice | 0.48 | 0.34–0.62 | 0.94 | 0.89 |
  | rough ice | 0.25 | 0.44–0.53 | 0.88 | 0.79 |
  | packed snow | 0.46 | 0.52–0.57 | 0.92 | 0.88 |

  * The low-data advantage of geometry is mostly fresh snow and concrete. Fresh snow is defined by sinkage, i.e.
    a wider footprint across neighbouring taxels: a multi-taxel spatial pattern the geometric interaction captures
    with few examples.
  * Spatial interaction is *necessary* for it: no_interaction never exceeds 0.77 on fresh snow, while every
    spatial model reaches 0.96–0.99 from 10 % of the data on.
  * The flat model's full-data edge is in glare ice, rough ice and packed snow. These differ by overlapping
    friction, texture amplitude and sub-mm sinkage: fine discrimination a position-indexed, expressive model finds
    with enough data. That is exactly the cue suspected to be unrealistically visible (unfiltered texture). The
    skin-filtered rerun (`runs/v3_skin/`) tests whether the flat model's full-data edge survives realistic texture.

**Step-budget check (2026-10-08, `runs/v3/budget10k/`).** Full data (1,680 episodes), seed 0, same protocol with
10,245 steps (15 epochs) instead of ~3,400 (4.4 epochs):

| model | test acc 3.4k → 10k | test NLL 10k | final train loss | best_step |
|---|---|---|---|---|
| flat_gru | 0.937 → 0.942 | 0.266 | 0.020 | 9,500 |
| attn_dist | 0.895 → 0.936 | 0.172 | 0.111 | 10,000 |
| hierarchical_meanmax | 0.902 → **0.946** | **0.144** | 0.085 | 10,245 (still improving) |

* **The full-data "flat wins" crossover was a step-budget artifact.** Given enough optimizer steps, all three are
  within ~1 point at full data (SE ≈ ±1.2 points on 360 test episodes). Hierarchical is nominally best and much better
  calibrated.
* Flat optimizes fastest per step: val 0.72 vs 0.57–0.61 at 500 steps. It also memorizes (train loss 0.02 vs val NLL
  0.21). The structured models fit more slowly and generalize with a much smaller train/val gap (hierarchical 0.085 vs
  0.116), so 3,000 steps understates them most on large training sets.
* Per-class recall at 10k: the structured models catch up on glare, rough and packed (hierarchical 0.947 / 0.870 /
  0.926 vs flat 0.924 / 0.868 / 0.944). The "fine friction/texture discrimination" edge of the flat model was also
  budget. Remaining errors at full data are rough↔packed and glare↔rough confusions (overlapping friction boxes).
* Cost per step, including validation: flat 0.062 s, hierarchical 0.28 s, attn_dist 0.39 s.
* At 0.1 and 0.25, v3's best checkpoints were also at the last evaluations (2,750–3,078 steps). The cosine schedule
  makes that partly expected, but adequacy there is unverified. → Protocol v4 (below).

**Texture-reliance test (2026-10-08).** Protocol v2, 600-episode sets, 420 training episodes, seed 0, 90-episode test
set (±3.5 points). `mjlab_terrain_600_notex` (texture amplitude 0) vs `mjlab_terrain_600_dc`:

| model | with texture | no texture |
|---|---|---|
| hierarchical_meanmax | 0.887 | 0.824 (−6) |
| flat_gru | 0.874 | 0.876 (±0) |
| no_interaction | 0.752 | 0.817 (+6) |

Suggestive, not conclusive: the structured model seems to exploit the micro-texture vibration cue and the flat
model does not. A real 2 mm skin with 1 cm taxels would mostly filter that cue out, so part of the structured
models' low-data advantage could be a simulator artifact.
* → `ContactModelConfig.texture_filter` (spatial low-pass, σ ≈ 3.5 mm) and a paired, physically filtered dataset
  `mjlab_terrain_2400_skin`.
* The main study is re-run on it (queue_v3d).

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

### RESUME HERE (state as of 2026-10-08 ~18:00)

**Running now (GPU, sequential, unattended):**
1. Step-budget check, started 16:26: `configs/experiments/budget_v3_mjlab.yaml` → `runs/v3/budget10k/`, log
   `runs/logs/v3_budget10k.log`. flat_gru, attn_dist and hierarchical_meanmax at full data (1,680 episodes, 10,000 steps).
   * Question: does the flat model's full-data edge (0.937 vs 0.90 at 3,400 steps) survive when the structured
     models get more steps?
   * Interim: flat val 0.94 at ~8k steps.
2. Then `runs/logs/queue_v3d.sh` (log `runs/logs/queue_v3d.log`), automatically:
   * collect `datasets/mjlab_terrain_2400_skin` (`collect_mjlab_large_skin.yaml`, texture_filter 3.5 mm,
     paired with `mjlab_terrain_2400`);
   * main study on it, seed 0 → `runs/v3_skin/seed0/`;
   * seeds 1, 2 → `runs/v3_skin/seed12/`;
   * main study seeds 1, 2 on the unfiltered data → `runs/v3/seed12/`.
   * Roughly 5 h per seed batch, ~16 h total.

**Check status:**
```bash
cat runs/logs/queue_v3d.log
pgrep -af "data_efficiency|collect_mjlab"
cat runs/v3/budget10k/summary.md
```
Each study writes `summary.md`, `results.csv` and `learning_curves.png`; per run there is a `log.jsonl` (step, val,
grad_norm), a `result.json` (test metrics, best_step, confusion) and a `model.pt` (v3 studies save checkpoints).

**Environment** (scratchpad helpers are wiped on reboot): `conda activate mjlab` after
`unset PYTHONPATH` and stripping `/opt/ros` from `LD_LIBRARY_PATH` (ROS Jazzy is sourced in `~/.bashrc`). Run GPU jobs
one at a time (8 GB card; hierarchical training peaks at ~6.6 GB).

**Analyse when each result lands** (done by the main Opus session, not delegated):
- Budget check: compare 10k-step test acc with the 3.4k-step v3 seed-0 numbers (flat 0.937, attn_dist 0.895,
  hierarchical 0.902). Check best_step: still improving at 10k? Is the full-data crossover a budget artifact?
- v3_skin seed 0 vs v3 seed 0 (paired data, only the texture filter differs):
  * Does the structured models' low-data advantage persist? It is mostly fresh snow and concrete (sinkage
    footprint), which the filter should not affect.
  * Does the flat model's full-data edge on glare/rough/packed shrink?
  * Per-class recall from the confusion matrices.
- Seeds 1, 2: means ± std. Are the differences beyond seed noise? The test set is 360 episodes (SE ~±2 points).
- Robustness: `scripts/evaluate.py --checkpoint <run>/model.pt --dataset datasets/mjlab_terrain_2400[_skin]
  --sensors configs/sensors/fsr_degraded.yaml` on the v3 checkpoints, comparing the accuracy drop per architecture
  (sim-to-real proxy). Also cross-simulator: evaluate on Isaac data. That needs a paired Isaac dataset with the same
  layout; `datasets/isaac_terrain_600_dc` exists (600 episodes, unfiltered texture).

**Decisions made, not to revisit without reason:**
* mjlab is the primary simulator.
* Protocol v3 (fixed step budget, step-based validation, best checkpoint, bf16).
* The mean+max head is the default for comparisons; the cluster-attention head has a ~200-step init plateau.
* Analysis is done in the main session; subagents only for bulky low-judgment or easy Sonnet tasks.
* File edits with the Edit/Write tools.

**Next (after the above):**
- [ ] Write up the Phase 1 picture so far for the user (structure vs data efficiency, the sensor-physics
      dependence, realism caveats).
- [ ] Robustness proxy runs (see above).
- [ ] Decide whether the primary dataset should be the skin-filtered one (likely yes, if the analysis supports it).

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
