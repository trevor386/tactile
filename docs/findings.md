# Findings

One entry per experiment or measurement, newest first within each version:
* **why** it was run (the question);
* **setup**;
* **result** with numbers;
* **evidence**: what supports the conclusion;
* **caveats**;
* **implication**: what it changed.

The plan, hypotheses (H-*) and queue (E-*) live in `docs/investigation_log.md`. Version-0 entries (F0-*) refer to the
archived setup; read their caveats before citing them (`docs/archive/README.md`).

## Version 1 (2026-10-08)

**F-13. Bayes ceilings: catalog v1 needs every cue type; v0 was solvable from friction + the synthetic sinkage.**
*Why:* interpret E-1 and check what the task rewards (T-6).
*Setup:* `scripts/analysis/task_ceiling.py`, k-NN (k 25) on 40k sampled parameter vectors per catalog. "Felt roughness" is
the texture amplitude × the skin's attenuation at its wavelength (σ 3.5 mm in v1, unfiltered in v0). "Exact" means the
parameters are known perfectly (a loose upper bound); "noisy" adds 15 % relative noise (an assumption).
*Result:*

| cues known | v1 exact / noisy | v0 exact / noisy |
|---|---|---|
| friction only | 0.49 / 0.41 | 0.78 / 0.67 |
| compliance only | 0.78 / 0.72 | 0.80 / 0.62 (synthetic sinkage) |
| felt roughness only | 0.79 / 0.64 | 0.68 / 0.60 |
| friction + compliance | 0.88 / 0.81 | 1.00 / 0.92 |
| all three | **0.98 / 0.92** | 1.00 / 0.99 |

*Evidence:* the method reproduces the v0 box-uniform calculation (0.795 / 1.0, F0-5).
*Caveats:* the parameters are not all observable to the same degree in 0.5 s, and the noise level is assumed.
*Implication:*
* Under v1 a model must combine proprioceptive friction cues, physical compliance (sinkage, pressure level,
  impacts) and spatial roughness patterns. Proprioception-like cues alone cap at ~0.5.
* Expected confusions: glare ice ↔ concrete (same stiffness; glare ice's faint relief is the only difference), and
  packed ↔ fresh snow at the soft end of packed snow.

**F-12. Revised catalog v1 validates.** *Why:* the catalog was revised to the literature boxes (F-11).
*Result:* `validate_mjlab.py`, 10 envs: 15/15 pass, thresholds unchanged.
* Softest class (fresh snow) rests at 11.1 mm sinkage, the stiffest at 0.00 mm.
* friction_opposes_sliding 0.92.
* torque and shear increase with friction (Spearman 0.96 / 1.0).

**F-11. Literature: what distinguishes ice and snow physically** (`docs/references/terrain_physics_v2.md`, an agent
review; tags [M]/[S]/[E] in the file). *Why:* simulation v1 should differ between classes in what a real skin feels.
*Result:*
* On hard ground (ice at any temperature, concrete, hard packed snow) the 2 mm skin sets the contact stiffness:
  30–90 µm indentation, 6–10 kPa. These surfaces should share one compliance box.
* Hard ground is bouncy (damping ratio ~0.07–0.3, a guess).
* Snow is mostly plastic: fresh snow sinks 1.5–25 mm at 0.5–2 kPa (uncertain ~10×), packed snow 0.01–1.5 mm.
* Cold ice is grippy for slow rubber (μ ~0.4–1.5 at −30 °C); only near 0 °C is it slippery (0.05–0.2).
* Only relief at ≥ 15 mm wavelengths reaches a 1 cm taxel: weathered ice 6–7.5 mm RMS at ~65 mm. Frost, grains and
  concrete texture are invisible.
* Europa: g = 1.315 m/s², so a link presses 1.5–2.6 kPa on solid ice, the same as Earth snow or sand: a strong OOD case.
  Silicone is glassy there (Q-2).

*Caveats:* no silicone-on-ice friction data at any temperature; the snow numbers come from large plates; skin
restitution is unmeasured. *Implication:* catalog v1 was revised (F-12). Glare ice and concrete are now nearly
indistinguishable by touch (Q-3). This motivates property estimation (H-11) and OOD sets (T-9).

**F-10. Low damping ratios and snow extremes in mjlab.** *Why:* the literature suggests bouncy hard contacts (ζ 0.07–0.3)
and deep fresh-snow sinkage; check MuJoCo stability and calibration.
*Setup:* `calibrate_compliance.py`, ζ 0.15 and 0.3 on stiff settings, μ 0.1/0.5/1.0; snow settings at ζ 1.
*Result:*
* Stable (no NaN), but one-step contact flicker rises to 5–10 % at ζ 0.15 and 3–5 % at ζ 0.3 (the links bounce),
  against ~3 % at ζ 1.
* Centroid speed is non-monotonic in friction up to μ 1.0 (0.48 / 0.58 / 0.42 m/s), so the speed–friction correlation
  turns negative over that range; this is not a failure.
* Snow calibration (rest / moving sinkage):

  | τ, dmin, width | rest | moving |
  |---|---|---|
  | 0.02, 0.4, 1 mm | 0.44 mm | — |
  | 0.02, 0.4, 8 mm | 1.3 mm | — |
  | 0.02, 0.02, 8 mm | 3.1 mm | — |
  | 0.02, 0.02, 50 mm | 13.8 mm | 10 mm |
  | 0.06, 0.02, 50 mm | 21 mm | 12 mm |

*Implication:* hard-ground ζ set to 0.2–0.4 (a compromise between the literature and flicker); the snow boxes in catalog v1.

**F-9. Model v1 cost.** *Why:* plan the E-1 budget. *Setup:* `profile_step.py`, batch 32, bf16, `mjlab_v1_2400`.
*Result:*

| model | parameters | width | ms per step | peak GPU memory |
|---|---|---|---|---|
| somato_v1 | 636k | — | 270 (stage 2 tactile 60, stage 1 tactile 24, brain 3 forward) | 4.8 GB |
| flat_gru (matched) | — | 203 | 50 | — |
| transformer (matched) | — | dim 132 | 168 | — |

The transformer's first version ran out of memory: PyTorch expanded the key-padding mask for dead sensors to a dense
[B, heads, 2001, 2001] mask, which also disables flash attention. Fixed by letting dead sensors read zero, as in the
flat model.

**F-8. Validation of simulation v1, and a check-methodology artifact.** *Why:* the explicit contact pairs must not break the
physics. *Result:*
* First run: 14/15. friction_opposes_sliding was 0.785 (< 0.8), because I had moved the gait check onto the stiffest
  class (concrete, μ 0.5–1.0).
* *Evidence that it is the check, not the pairs:*
  * on the same class with plain geom contacts the check gives 0.80;
  * with the original class cycling and the pairs it gives 0.93 (v0: 0.93).

  At high friction, links classified as "sliding" are mostly near-stick and rolling, and the check's centroid-velocity
  estimate is noisy.
* *Implication:* gait check reverted to class cycling; 15/15 pass with no threshold changes.

**F-7. Terrain compliance through the impedance profile (solimp) works.** *Why:* the time-constant route (F-6) sinks too
slowly. *Setup:* solref τ 0.02 s; solimp dmin (surface impedance) 0.9/0.5/0.3/0.15/0.05; transition width 1–30 mm.
*Result:*
* The resting sinkage spans the literature range: 0.05 mm (0.9, 1 mm) → 0.8 (0.5, 5 mm) → 2.5 (0.3, 15 mm) →
  6.5 (0.15, 30 mm) → 11.6 mm (τ 0.04, 0.05, 30 mm).
* Sinkage while moving is 70–90 % of the resting value: it develops dynamically.
* Speed still increases with friction (r 0.67–0.81).
* Contact flicker falls from ~4 % to < 0.3 % on compliant ground.
* impratio 10 changed nothing.

*Implication:* the parameterization of catalog v1.

**F-6. Terrain compliance through the solref time constant: right physics, wrong dynamics.** *Why:* make sinkage
physical instead of synthetic (v0's virtual offset). *Setup:* explicit contact pairs with stiff friction rows
(solreffriction 0.02 s), normal τ 0.02–0.5 s.
*Result:*
* Friction coupling is kept (speed–friction r 0.77–0.92 at every τ), unlike v0's soft geom contact (r 0.04), so stiff
  friction rows fix the v0 problem.
* Resting sinkage is steeply nonlinear in τ: 0.05 mm at 0.02 s, 5.4 mm at 0.2 s, 16 mm at 0.3 s.
* But long time constants respond slowly: sinkage while moving is only 0.9 mm at τ 0.2 s against 5.4 mm at rest.

*Implication:* use the impedance profile instead (F-7).

**F-5. RAM freeze and the dataset cache.** *Why:* the machine froze (2026-10-08 evening).
*Evidence:*
* An uncached 2,400-episode `EpisodeStore` costs 13.7 GB RSS: 5.6 MiB per episode for 3.5 MiB of data
  (decompression leftovers).
* Training, robustness evaluation and an analysis process held about four copies of the 31 GB machine's memory.

*Fix:* a consolidated memory-mapped cache. On 600 episodes RSS falls from 3.9 GB to 0.5 GB, the data sit in shared,
reclaimable page cache, and all windows are checksum-identical. *Implication:* the memory-cap rule (D, 2026-10-08).

**F-4. Robustness first look (v0 checkpoint, before the RAM freeze).** *Why:* sim-to-real proxy (H-2a).
*Result:* the v0 flat GRU at full data (10k steps) loses:

| perturbation | accuracy change |
|---|---|
| 3× sensor noise | −32 points |
| slower FSR unloading | −12 points |
| more creep | −2 points |
| larger gain spread | −2 points |
| everything combined | −43 points |

*Caveat:* one v0 checkpoint; the structured models' runs were lost in the freeze. The flat model's per-step std
features are noise-sensitive by construction. *Implication:* robustness is a central measurement for H-2/H-3 (E-6).

**F-3. Speed: exact pairwise distances.** `torch.cdist` with `donot_use_mm_for_euclid_dist` took ~95 ms per call on
128×400×400 points. A per-coordinate exact computation takes 3.75 ms and agrees within 2.4e-7 m (tie quantum 1e-4 m).
Training step: hierarchical 299 → 210 ms, attention model ~390 → 236 ms.

**F-2. Speed: sensor-model loops.** The FSR conductance transfer, the motor velocity differencing and the capacitive
mix moved out of the per-sample loops: FSR 37 → 13.5 ms per batch. Tested to equal the old loops.

**F-1. The 0.25-fraction budget check was lost** in the reboot after the freeze, after only part of its flat_gru run. Its
partial log showed the flat GRU overfitting beyond ~3k steps (val loss rising, accuracy ~0.80–0.82). Re-queued as E-1b
under protocol v4.

## Version 0 (archived; 2026-10-07/08)

These used the v0 model (modalities mixed in stage 2), v0 baselines (flat GRU without kinematics and without dropout)
and the v0 simulation (only friction physical, synthetic sinkage, unfiltered texture, near-melting friction).
Details: `docs/archive/investigation_log_v0.md`.

**F0-1. Isaac Lab bring-up.** Isaac Lab 2.3.2 / Isaac Sim 5.1 validated 14/14 after targeted fixes:
* 16/4 solver iterations against contact chatter;
* IMU initialization after reset;
* backend close and exit codes;
* integrate-and-dump sampling against aliasing.

*Still valid.*

**F0-2. Isaac vs MuJoCo.** With identical DC-motor actuators the paired datasets agree:

| quantity | Isaac / MuJoCo agreement |
|---|---|
| speed, paired | r = 0.99 |
| friction readout, paired | r = 0.999 |
| tumbling | 5 / 6 episodes |

Isaac has rigid-impact spikes; MuJoCo's stiff contact flickers (micron separations). *Still valid; basis of the
simulator decision.*

**F0-3. First data-efficiency study (Isaac, 20 epochs): flat 0.80 vs hierarchical 0.63 at full data.** *Invalid:* a
training-budget artifact (small fractions got few updates) plus the cluster-attention head's ~200-step loss plateau.

**F0-4. Cluster-attention plateau.** The head stays at ln 5 for ~150–200 steps. Val accuracy after 300 steps:
cluster head 0.38, mean+max head on the same backbone 0.54, flat 0.76.
*Implication:* the brain head in v1 has a direct pooled path.

**F0-5. Task ceiling (v0 catalog).**

| parameters known exactly | Bayes accuracy |
|---|---|
| friction only | 0.795 |
| friction + sinkage | 1.0 |

*Caveat:* sinkage was the synthetic cue; the v1 ceilings come from T-6.

**F0-6. Protocol v2 (600 episodes):**

| model | test accuracy | NLL |
|---|---|---|
| hierarchical | 0.887 | 0.32 |
| flat | 0.874 | 0.47 |
| no_interaction | 0.752 | — |

*Caveats:* one seed, ±3 points.

**F0-7. Modality ablation and sensor physics.**
* Touch carried most of the information:

  | model | tactile only |
  |---|---|
  | hierarchical | 0.82 |
  | flat | 0.75 |

* Proprioception alone reached ~0.68.
* With shear-sensing taxels, per-sensor processing alone reached 0.93: stage 2 was worth ~2 points, against 13 with the
  normal-only FSR.

*Caveat:* mixed graph, synthetic sinkage. *Likely still true in kind (H-5).*

**F0-8. Initialization signal.** Stage-1 features at initialization are dominated by a sample-independent component
(85–95 %); after stage 2 the per-sensor std across samples is 11–17 % of the feature RMS.

**F0-9. v3 learning curves (2,400 episodes, 3,000 steps, seed 0).** Geometric stage-2 variants beat flat by 19–21 points
at 42 training episodes and by 8 at 168; the curves crossed at ~420. Much of the low-data gain was fresh snow
(synthetic-sinkage footprint) and concrete. *Caveats:* the confounded flat baseline, the synthetic cue, one seed, and
the full-data under-training (F0-10).

**F0-10. Budget check (10k steps, full data):**

| model | test accuracy | NLL |
|---|---|---|
| hierarchical | 0.946 | 0.14 |
| flat | 0.942 | 0.27 |
| attn_dist | 0.936 | — |

The 3k-step "flat wins at full data" was a budget artifact. Flat memorizes (train loss 0.02); the structured models fit
more slowly with a smaller generalization gap.

**F0-11. Budget adequacy.** At 3k steps every full-data run's best checkpoint was its last evaluation; at 0.1–0.25
fractions too, though the cosine schedule makes that partly expected. → protocol v4 and E-1b.

**F0-12. Texture-reliance test.** Without micro-texture the hierarchical model lost 6 points and flat lost nothing
(±3.5 points SE). This suggested reliance on a cue a real skin filters out, and led to the skin filter, now the default
in simulation v1.

**F0-13. Streaming.** Online accuracy decays from 0.58 (0.5 s after reset) to 0.33 (3 s), because the recurrent state
was trained only on 0.5 s windows. Sliding-window inference fixes it. TBPTT with 6 chunks per sequence stayed at chance
(T-17).
