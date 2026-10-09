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

**F-17. E-13, slide proxy: the 3D kernel detects motion across the skin (H-9 ✓), but the brain head stalls and then
memorizes (H-8 ✗ for now).**
*Why:* can stage 2, acting on stage-1 latents, tell which way a contact slides (H-9), and what does the brain add (H-8)?
*Setup:*
* `configs/experiments/proxy_slide.yaml`, 4 directions (chance 0.25);
* 84 and 840 training episodes;
* widths matched (pool dim 84, no_spatial 88, somato_v1 64);
* fixed FSR model.

*Result (test acc):*

| model | 84 eps | 840 eps |
|---|---|---|
| somato_v1_pool (static mean+max head) | **0.983** | **0.960** |
| somato_v1 (brain) | 0.250 | 0.811 |
| somato_v1_no_spatial (brain) | 0.461 | 0.446 |
| flat_gru | 0.201 | 0.542 |
| transformer | 0.253 | 0.308 |

*Evidence:*
* Stage 1 + kernel3d + pooling learns the direction from 84 episodes (val 0.96 already at the first quarter of the
  budget).
* Without stage 2, the model cannot (0.45); nor can the unstructured baselines, which see the same kinematics.
* The brain head's training loss sits at exactly ln 4 (uniform output) for ~1,500 steps. It then falls to 0.001 while
  validation stays at chance: it memorized the 84 episodes. At 840 episodes the brain escapes later (val 0.25 → 0.85).
  no_spatial (also brain) shows no stall but memorizes too.

*Caveats:* one seed; a synthetic stimulus-level proxy.
*Implication:*
* The spatial operator is essential for moving stimuli, and here unstructured models fail outright: strong support
  for H-9 and for structure in general.
* The brain head as implemented has an optimization/generalization problem (same symptom as v0's cluster head). It
  carries ~208k more head parameters than the pool head at the same width.
* Diagnose before tuning: E-19 (pool vs brain vs brain without GRU / without attention / with a direct node-pool path /
  smaller), on the proxy and on terrain at 10 %. E-1c waits for the stage-3 decision.
* Related, E-3 so far (terrain, 10 %): somato_v1 0.615, no_spatial 0.618, mixed 0.604. With the brain head, stage 2
  adds nothing on terrain at low data, so E-1's lead over the baselines may come from stage 1 and region pooling, not
  the kernel. Wait for the pool and graph results.

**F-16. E-1 complete: the structured model leads at both data sizes; no crossover yet (one seed, FSR bug).**
*Why:* signs of life and a first test of H-1/H-4. *Setup:* as F-14; full data = 1,680 episodes, 10,245 steps.
**Caveat: ran with the FSR hysteresis bug (F-15)**, identical for all models.
*Result:*

| model | 170 eps | 1,680 eps | NLL (full) | best step (full) |
|---|---|---|---|---|
| somato_v1 | **0.630** | **0.782** | **0.49** | 9,500 / 10,245 |
| flat_gru | 0.492 | 0.672 | 0.80 | 3,500 |
| transformer | 0.506 | 0.635 | 0.83 | 8,750 |

Recall at full data:

| class | somato | flat | transformer |
|---|---|---|---|
| glare | 0.55 | 0.33 | 0.16 |
| rough | **0.83** | 0.51 | 0.55 |
| packed | 0.76 | 0.70 | 0.61 |
| fresh | 1.00 | 0.98 | 0.99 |
| concrete | 0.80 | 0.85 | 0.88 |

*Evidence:*
* The gap is 11–15 points at full data and 12–14 at 10 %. The flat GRU stops improving after 3,500 steps (it
  memorizes), so it is data-limited, not step-limited.
* The structured model's largest advantage is rough ice (+28 to +31), whose cue is cm-scale relief across
  neighbouring taxels (F-13), the spatial pattern stage 2 is designed for.
* The baselines lump glare ice into concrete, the touch-ambiguous pair (Q-3); the structured model separates them
  better (probably through friction where glare ice is slippery).

*Caveats:*
* One seed.
* The FSR bug.
* **Baselines untuned:** lr 1e-3, warmup 1 epoch for all; the transformer learns slowly (val 0.56–0.61) and may need a
  lower learning rate.
* Everything is far from the ~0.92 noisy ceiling.

*Implication:* encouraging for H-1, nothing for H-4 yet. Before E-2: **E-1c** reruns E-1 with the fixed sensor and a
learning-rate sweep (3e-4, 1e-3, 3e-3) for all three models, so the baselines get a fair chance.

**F-15. Code review of the v1 changes (T-5): one bug that affects results, six latent ones.** *Why:* catch bugs before
long experiments. *Setup:* two independent Sonnet reviewers (models/training; sim/sensors/data). Every finding was
re-checked against the code before fixing. Commit "Code-review fixes".
*Result:*
* **Affects results:** the FSR v1 rate-independent hysteresis (play operator) settled at half its band when the load
  returned to zero. That left a permanent 4–9 % full-scale offset after every contact and a dead zone that hid light
  contacts. Fixed: the band narrows to zero with the output, so the loop closes at no load (as measured FSR loops
  do), with a test.
* **Latent** (no current config triggered them):
  * per-step labels were not aligned with strided brain steps;
  * a TBPTT chunk without a brain step crashed, and the online demo crashed on its first frames;
  * a window that is not a multiple of the stride was silently scored at an earlier step;
  * an override's `graph.k` was ignored when the base had `k_per_group`;
  * the Europa glassy-skin dmin above the default dmax was silently capped;
  * non-native IMU under non-Earth gravity was wrong;
  * the compliance check ran on backends without compliance.

  All fixed, with tests.

*Implication:* **E-1 ran with the hysteresis bug** (sensor model `fsr_v1`, all three models equally affected; it hid
light contacts and added history-dependent offsets). E-1 is still a valid sign of life, but its numbers are not
comparable with later runs; E-2 supersedes it. Queue v1a (E-13, E-3/E-4) starts new processes and uses the fixed
code.

**F-14. E-1, first half: at 10 % of the data the structured model leads both baselines (interim, one seed).**
*Why:* signs of life on simulation v1 and a first test of H-1. *Setup:* `configs/experiments/v1_signs_of_life.yaml`,
170 training episodes, 3,000 steps (protocol v4), widths matched (~636k), 360-episode test set (SE ≈ ±2.6 points at
0.6). `scripts/analysis/summarize_study.py runs/v1/signs_of_life`.
*Result:*

| model | acc | NLL | best/total steps | glare | rough | packed | fresh | concrete |
|---|---|---|---|---|---|---|---|---|
| somato_v1 | **0.630** | **1.12** | 3000/3010 | 0.50 | **0.56** | 0.54 | 0.91 | **0.60** |
| transformer | 0.506 | 1.53 | 3010/3010 | 0.28 | 0.42 | 0.53 | 0.92 | 0.35 |
| flat_gru | 0.492 | 4.02 | 2500/3010 | 0.39 | 0.28 | 0.42 | 0.85 | 0.51 |

*Evidence:*
* Every model trains; the brain head learns from the first evaluation (val 0.49 at 250 steps, no plateau).
* The somato lead is 12–14 points (~4–5 SE) and spread over classes: rough ice +14/+27 (cm-scale relief, a spatial
  cue), glare ice and concrete +10 to +25. All models get fresh snow (0.85–0.92): the physical sinkage cue is learnable
  without a spatial bias, unlike v0's synthetic footprint (H-3 concern addressed).
* The flat GRU is badly overconfident (NLL 4.0, as in v0); the transformer is calibrated reasonably.

*Caveats:*
* One seed.
* somato_v1 and the transformer were still improving at the end (best step = last evaluation), so 3,000 steps may
  under-train them at 10 % (E-1b).
* The ceiling is ~0.92 with realistic noise (F-13), so there is room above all three.

*Implication:* signs of life for H-1 at low data. Wait for full data and seeds before drawing conclusions.

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
