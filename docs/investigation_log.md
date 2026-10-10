# Investigation log (version 1)

The working plan: what is running, what we believe and want to test, what comes next, and what has to be built.
**Keep it current**: update statuses when they change, add items as they come up, and record every experiment's outcome
in `docs/findings.md` (why it was run, result, evidence, caveats, implication). Design: `docs/architecture.md`.
Simulators: `docs/mjlab.md`, `docs/isaac_sim.md`. Literature: `docs/references/`. Version 0 and its caveats:
`docs/archive/`. How to run things on this machine: `CLAUDE.md` and `tools/README.md`.

IDs are stable: H = hypothesis, E = experiment, T = task (code or setup). Status words: *running*, *next*, *queued*,
*blocked (by X)*, *done (date, finding)*, *dropped (why)*.

## RESUME HERE (state as of 2026-10-10 12:00)

* **Running:** E-5 (jobq `e5_properties`, ~5.5 h from 11:50, output `runs/v1/properties`). When done: summarize,
  then run `ood_eval.py` on its `frac1.0` checkpoints (T-21 is ready) and write the finding. Check with
  `python3 tools/jobq.py status`; after a reboot run `start`.
* **Next, in order:**
  1. **E-21**, sensor-model randomization against the hysteresis overfitting found in F-23 (T-22 first).
  2. E-13b and E-1b.

  Submit each with `python3 tools/jobq.py add NAME -- python -u scripts/data_efficiency.py --config ...` and wait in
  the background with `tools/jobq.py wait NAME`.

## 1. Hypotheses and conjectures

Central hypothesis (user, 2026-10-08): an intentionally structured hierarchy inspired by the somatosensory system is
(H-1) more data-efficient than unstructured models and (H-2) generalizes better. Context: short term, arctic monitoring
and robots on icy terrain in high-latitude cities; long term (motivation only), icy worlds like Europa with no training
data. The structure is a starting point, changed only on evidence.

| ID | hypothesis / conjecture | status |
|---|---|---|
| H-1 | The structured hierarchy is more data-efficient than the flat GRU and the transformer (equal parameters, same inputs incl. kinematics). | **supported in sim v1** (F-22: 2–3× fewer episodes, gaps ≥ 2 seed std at all sizes) |
| H-2 | It generalizes better under shift: (a) sensor non-idealities, (b) terrain outside the training range, (c) another simulator, (d) layout changes. | (a) mixed (F-23): equal drop under the combined shift (lead kept), more robust than the transformer to unloading and dead taxels, but stage 1 overfits the simulated hysteresis (E-21). (b) class posteriors give no OOD awareness for any model (F-24); test properties (E-5, T-21) and E-9. (c) E-8. (d) E-10. |
| H-3 | A simple simulator lets models exploit sim shortcuts that fail in reality. | open. v1 removed the synthetic sinkage shortcut; F-23 found a sensor-model shortcut (hysteresis signature) used by every model that reads raw dynamics (E-21). |
| H-4 | Structure trades expressiveness for efficiency: the advantage shrinks with data and may cross over. | partial (F-22: the transformer gap halves over 40× data; no crossover; the flat gap does not shrink). E-11 for more data. |
| H-5 | Stage-2 spatial interaction matters when single taxels are ambiguous; its value depends on sensor physics and task. | task-dependent: +1–2 points on terrain (F-18, F-22), decisive for moving stimuli (F-17). E-12 for sensor physics. |
| H-6 | The continuous 3D kernel beats feature-dependent graph message passing. | weakly supported (F-18, one seed) |
| H-7 | Keeping modalities segregated until stage 3 is at least as good and generalizes better. | no difference in-distribution (F-18); generalization open |
| H-8 | A dynamic, geometry-aware, recurrent stage 3 beats static pooling. | contradicted for the v1 brain implementation (F-17, F-19, F-20); research task T-19 |
| H-9 | Stage 2 on stage-1 latents detects stimuli moving across the skin. | supported (F-17, one seed; E-13b to confirm) |
| H-10 | Self-supervised objectives and stage-wise training improve data efficiency and generalization, more for the structured model. | open (E-14, E-15) |
| H-11 | Physical property estimation is a better generalization target than Earth class labels, and structure helps it. | open (E-5, E-7) |
| H-12 | Protocol v4's budget, max(3,000 steps, 15 epochs), trains every model adequately. | open: in E-2 most full-data best steps were late (F-22) (E-1b) |
| H-13 | The tactile information is also useful on granular, shifting and uneven terrain. | open (E-7, E-16) |
| H-14 | On terrain, most of the benefit comes from stage 1: one receptor encoder shared by all sensors of a modality, plus pooling. | **supported** (F-22: receptor_only within 1–2 points of somato_pool, 7–8 above the baselines) |

## 2. Experiment queue (ordered)

Order (user, 2026-10-08): signs of life → comparisons with baselines → generalization (sim-to-sim as a proxy). Avoid
very long runs until earlier steps are sound.

| # | ID | experiment | design | config | cost | status |
|---|---|---|---|---|---|---|
| 1 | E-5 | Property estimation (H-11) | Friction + measured sinkage regression with classification, the E-2 models at their learning rates, 10 % / 100 %, 3 seeds, checkpoint by lowest validation property error; then their property estimates on the OOD set (`ood_eval.py`, T-21) | `configs/experiments/v1_properties.yaml` | ~5.5 h | **running** (jobq `e5_properties`, from 11:50) |
| 2 | E-21 | Sensor-model randomization (H-2a, H-3) | Train with per-episode random FSR parameters (hysteresis 0–30 %, unloading 20–300 ms, gain spread) and re-run E-6. Does stage 1 stop overfitting the simulated hysteresis (F-23)? | to write: a sensor-config option for ranges drawn per sample | ~6 h | queued |
| 3 | E-13b | Slide proxy with the E-2 models (H-9, 3 seeds) | somato_pool, receptor_only, flat_gru, transformer at 84 and 840 episodes | adapt `configs/experiments/proxy_slide.yaml` | ~3 h | queued |
| 4 | E-1b | Budget adequacy (H-12) | somato_pool and transformer at 25 % and 100 % with twice the v4 budget | to write | ~3 h | queued |
| 4 | E-9 | Held-out parameter ranges (H-2b) | Train on part of the friction/compliance ranges, test on the rest | to write | ~3 h | queued |
| 5 | E-8 | Sim-to-sim (H-2c) | Train mjlab, test Isaac (and the reverse), paired episodes | to write | ~3 h | blocked (T-10) |
| 6 | E-12 | Sensor technology (H-5) | Ideal + shear and capacitive v1 vs FSR v1, for somato_pool / receptor_only / baselines | to write | ~4 h | queued |
| 7 | E-10 | Layout generalization (H-2d) | Dead-taxel patterns; other link counts / taxel densities (structured models only: the baselines are tied to one layout) | to write | < 1 h | queued |
| 8 | E-14 | SSL pretraining (H-10) | Masked sensor prediction / reconstruction / cross-modal, then fine-tune at small data | to write | ~6 h | blocked (T-11) |
| 9 | E-15 | Stage-wise vs end-to-end (H-10) | Stages trained on their own objectives and frozen, only stage 3 task-specific | to write | ~4 h | blocked (T-11) |
| 10 | E-11 | Larger data (H-4) | A 9,600-episode dataset; curves to 4× the current maximum | to write | ~10 h | queued |
| 11 | E-16 | Uneven / granular proxies (H-13) | Heightfield terrain, sand-like compliance | to write | — | blocked (T-13) |
| 12 | E-18 | Long-horizon streaming | Accuracy vs time since reset; TBPTT vs sliding window for the v1 models | — | ~2 h | blocked (T-17) |
| 13 | E-17 | Bio-inspired variants | Architectures from the somatosensory literature review | — | — | blocked (T-15) |

**Done:**
* E-6, sensor robustness (F-23).
* E-7, OOD terrains, class posteriors (F-24).
* E-1, signs of life (F-14, F-16).
* E-3 / E-4, structure ablations (F-18).
* E-13, slide proxy (F-17).
* E-19 / E-19b, brain diagnostics (F-19, F-20).
* E-1c, learning-rate sweep (F-21).
* E-2, main curves (F-22).

## 3. Tasks (code and setup)

| ID | task | for | status |
|---|---|---|---|
| T-7b | Add felt roughness (texture amplitude × skin attenuation) as a property-regression target. | E-5 | queued |
| T-21 | Extend `scripts/analysis/ood_eval.py` to property estimates (error vs the OOD set's true friction and measured sinkage) for models with a property head. | E-5 / E-7 | done (2026-10-10; tested on a smoke checkpoint) |
| T-22 | Sensor-model randomization: let sensor params be ranges drawn per sample/episode during training (the FSR already draws per taxel); a randomized `configs/sensors/fsr_v1_random.yaml`. | E-21 | queued |
| T-10 | Isaac parity for sim-to-sim: PhysX compliant contact per env from the same catalog, an Isaac v1 collection, a paired comparison. First check that Isaac runs on driver 580. | E-8 | queued |
| T-11 | SSL objectives (masked sensor prediction, reconstruction, next-step, cross-modal touch ↔ proprioception); trainer support for pretraining, freezing stages, task-specific last stage only. | E-14, E-15 | queued |
| T-13 | Uneven terrain: mjlab heightfields per env; the taxel contact model must query the terrain height under each taxel (it assumes a plane). | E-16 | queued |
| T-14 | Realism, by expected impact: (a) snow plasticity and ploughing/berm drag; (b) static friction and stick-slip; (c) temperature effects on the sensor; (d) taxel crosstalk and load sharing through the skin. | H-3 | queued |
| T-15 | Literature review of the somatosensory system (SA/RA receptors, dorsal-column pathways, somatotopy, cortical integration) for architecture ideas. | E-17 | queued |
| T-17 | TBPTT failure with 6 chunks per sequence (from v0): test 3 and 4 chunks, more distinct sequences per epoch. | E-18 | queued |
| T-18 | Disk hygiene: delete v0 datasets and caches, and `datasets/mjlab_terrain_600_dc/cache_v1` (4 GB), when no experiment needs them. | — | queued |
| T-19 | Stage-3 research: a dynamic, geometry-aware, temporal stage 3 that trains reliably. The v1 brain stalls or memorizes on sparse moving stimuli (F-17, F-19, F-20). Ideas: initialize as the pool head (gated residual, gate at 0); tokens without LayerNorm; gradient diagnosis at initialization; Perceiver-style latents. Long term stage 3 feeds a locomotion controller. | H-8 | queued |
| T-20 | Europa OOD set (catalog `ood_europa_v1.yaml`, config `collect_mjlab_europa.yaml`, gravity option): motivation only, not in the near-term queue (user, 2026-10-08). | — | parked |

Done tasks are in the commits and in `docs/findings.md`:
* v0 archive;
* dataset cache (F-5);
* simulation v1 (F-6 to F-12);
* model v1 and baselines;
* FSR v1;
* code review (F-15);
* property regression for friction and sinkage;
* robustness suite;
* OOD catalogs and scripts;
* slide proxy;
* loader prefetch;
* job queue `tools/jobq.py` (2026-10-10).

## 4. Other tracked items

* **Disk** (95 GB free on 2026-10-10):
  * `datasets/mjlab_v1_2400`: 1.4 GB plus a 16 GB cache;
  * `proxy_slide_1200`: plus its cache;
  * `mjlab_v1_ood_earth_600`;
  * v0 datasets: ~10 GB;
  * runs under `runs/v1/` (checkpoints in `main_curves`).
* **Main-study models and learning rates** (E-2): somato_pool 1e-3, receptor_only 1e-3, transformer 1e-3, flat_gru 3e-4;
  widths matched to somato_pool (~428k). Compute per step is not matched (the user accepts that).
* **The baselines are tied to one sensor layout**: layout transfer (E-10) runs for the structured models only; report it
  as a capability.
* **Open questions for the user:** none.

## 5. Decisions

| date | decision | why |
|---|---|---|
| 2026-10-08 | mjlab (MuJoCo-Warp) is the primary simulator; Isaac kept for sim-to-sim. | Agrees with Isaac, 6× faster, tunable contact (`docs/mjlab.md`). |
| 2026-10-08 | Version 0 archived; its results are not evidence for the hypothesis. | Mixed-modality stage 2, confounded baselines, synthetic sinkage (`docs/archive/README.md`). |
| 2026-10-08 | Model v1 follows the stated intent: per-modality receptors (stage 1), spatial interaction within each modality (stage 2: 3D kernel with a physical support radius; graph as alternative), modalities first meet in stage 3. | User's design intent (receptors → spinal cord → brain). |
| 2026-10-08 | Baselines get the same kinematics and dropout augmentation; a transformer baseline is added (dead sensors read zero, no attention mask). | Test the inductive bias, not the inputs; a dense mask over 2,000 tokens ran out of memory. |
| 2026-10-08 | Simulation v1: physical terrain compliance (explicit contact pairs, stiff friction), no virtual sinkage, literature friction and roughness, skin-filtered texture; validation thresholds unchanged. | Remove the synthetic cue; realism against H-3. |
| 2026-10-08 | Protocol v4: max(3,000 steps, 15 epochs), validation every 250 steps, best-val checkpoint, bf16, widths matched. | v3 under-trained full-data runs (F0-11). |
| 2026-10-08 | Tasks: classification and property estimation are both proxies (stage 3 will eventually feed a locomotion controller); Europa is motivation only; touch-indistinguishable classes stay (no thermal sensors). | User answers, 2026-10-08. |
| 2026-10-09 | Main studies use the static pool head as stage 3; the brain head is research (T-19). | It stalls or memorizes on moving stimuli and ties on terrain (F-17 to F-20). |
| 2026-10-10 | GPU work runs through `tools/jobq.py` (sequential, memory-capped, file-based waiting). | Ad-hoc queue scripts with `pgrep -f` waits deadlocked or killed themselves (2026-10-09/10). |
