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

## Task queue

- [x] Find why the hierarchical model learns slowly → cluster-attention head plateau (see findings).
- [ ] Locate the plateau inside the head (attention blocks vs. token norm/pooling); remedy (e.g. identity-init
      residual branches) if it is a pure initialization issue.
- [ ] Modality ablation on mjlab data with protocol v2 (min_steps, early stopping): flat_gru / hierarchical
      (both heads) / no_interaction × {all, tactile, joint+imu}; hierarchical + sensor_id_embedding.
- [ ] 3 seeds at full data with protocol v2.
- [ ] TBPTT with 6 chunks fails (see findings): test 3/4-chunk variants, and more distinct sequences per epoch.
- [ ] Main experiment once v2 checks pass: learning curves (fractions × 3 seeds) along a spectrum of spatial
      structure with stage 1 and the head fixed: flat vector → transformer over sensors with ID embeddings (no
      geometry) → full attention + distance bias → local continuous-kernel graph → no interaction.
- [ ] Inspect the hierarchical model's use of joint/IMU nodes (head pooling, normalizer statistics of
      zero-inflated FSR readings).
- [ ] Evaluate the TBPTT model (windows 25/75/150) and sliding-window inference; decide on a default.
- [ ] Re-run the data-efficiency study on mjlab data with ≥ 3 seeds, once the above checks pass.
- [ ] Ideal-shear skin vs FSR: how much does shear sensing help each architecture?
- [ ] Robustness to sensor-model perturbation (hysteresis/creep/gain) between training and test.
- [ ] Compliant skin in MuJoCo with stiff friction (explicit contact pairs with `solreffriction`), or a
      skin-dynamics layer in the contact model; would also remove the flicker.
- [ ] Physically modelled terrain compliance (per-env contact softness) in mjlab instead of only the
      taxel-model sinkage.
- [ ] Long-term: alternative stage-1/2/3 operators (TCN, transformer, receptor; GAT, full attention, EGNN).
