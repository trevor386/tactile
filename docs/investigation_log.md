# Investigation log (version 1)

The working plan: what is running, what we believe and want to test, what comes next, and what has to be built.
**Keep it current**: update statuses when they change, add items as they come up, and record every experiment's outcome
in `docs/findings.md` (why it was run, result, evidence, caveats, implication). Design: `docs/architecture.md`.
Simulators: `docs/mjlab.md`, `docs/isaac_sim.md`. Literature: `docs/references/`. Version 0 and its caveats:
`docs/archive/`. How to run things on this machine: `CLAUDE.md` and `tools/README.md`.

IDs are stable: H = hypothesis, E = experiment, T = task (code or setup). Status words: *running*, *next*, *queued*,
*blocked (by X)*, *done (date, finding)*, *dropped (why)*.

## RESUME HERE (state as of 2026-10-12 18:15)

* **SSL so far:** the structured model gains +6.5 points at 170 episodes with a 100 ms forecasting pretext (F-43,
  F-44); the flat GRU gains nothing from the same pretext (F-45). Next: the transformer's counterpart (E-14f).
* **Done:** v5 3-seed reference (F-41): somato_pool 0.615 / 0.773, receptor_only 0.592 / 0.749, transformer
  0.537 / 0.720, flat_gru 0.525 / 0.703 at 10 % / 100 %.
* **Awaiting the user:**
  * priority after E-14: T-10/E-8 (sim-to-sim via Isaac) or E-11 (more data; needs old datasets deleted, T-18);
  * Isaac now stops at the Omniverse EULA prompt (`isaac_check`, 2026-10-12). Accepting it (or setting
    `OMNI_KIT_ACCEPT_EULA=YES`) needs the user's OK.
* **Deferred:** E-11 (4× data) needs ~64 GB of dataset cache; 79 GB free (2026-10-11). Ask the user before deleting
  old datasets (T-18).
* **Done recently:** E-5 (F-25), E-21 (F-26), E-13b (F-27), E-1b (F-28), E-9 (F-29), E-22 (F-30, fairness), T-23
  runtime (F-31), E-10 layout (F-32), E-23 compute matching (F-33), E-12 sensor technology (F-34), E-18a streaming
  (F-35), E-9b compliance holdout (F-36), E-10b 12-link transfer (F-37), E-10c fine-tuning recipes (F-38), E-24 stage probes (F-39), E-25 random stage 1 (F-40).
* **User direction (2026-10-11):** keep running experiments on stages 1–2 and their capabilities; no stage-3 or
  locomotion work until much more is learned about the other stages. Real-time concerns are noted (section 4), not
  pursued.
* **New defaults:** new studies use `configs/experiments/terrain_v1b.yaml` (randomized FSR, protocol v5 =
  max(6,000 steps, 15 epochs)). Check baseline learning rates on each new task (F-30).
* **Next after the queue:** 12-link fine-tuning (E-10 follow-up), E-12 seeds if its differences matter, T-11 (SSL)
  for E-14/E-15, E-11 once disk allows.

  Submit each with `python3 tools/jobq.py add NAME -- python -u scripts/data_efficiency.py --config ...` and wait in
  the background with `tools/jobq.py wait NAME`.

## 1. Hypotheses and conjectures

Central hypothesis (user, 2026-10-08): an intentionally structured hierarchy inspired by the somatosensory system is
(H-1) more data-efficient than unstructured models and (H-2) generalizes better. Context: short term, arctic monitoring
and robots on icy terrain in high-latitude cities; long term (motivation only), icy worlds like Europa with no training
data. The structure is a starting point, changed only on evidence.

| ID | hypothesis / conjecture | status |
|---|---|---|
| H-1 | The structured hierarchy is more data-efficient than the flat GRU and the transformer (equal parameters, same inputs incl. kinematics). | **supported in sim v1** (F-22: 2–3× fewer episodes, gaps ≥ 2 seed std at all sizes; also at matched online compute, F-33: a 37M-parameter flat GRU stays 5–7 points behind; fair inputs, F-30) |
| H-2 | It generalizes better under shift: (a) sensor non-idealities, (b) terrain outside the training range, (c) another simulator, (d) layout changes. | (a) **supported with sensor randomization** (F-26): trained on a fixed FSR, stage 1 overfit the simulated hysteresis (F-23); trained on randomized FSRs that shortcut is gone, the combined-shift drop shrinks from ~14 to ~8.5 points, and the structured lead under it grows to +8.2 points (larger hysteresis than trained still costs 5–7). (b) class posteriors give no OOD awareness for any model (F-24); property estimates keep the right order on unseen terrain with 1–3× the in-distribution error and a conservative friction bias, structure helps sinkage but not friction there (F-25). Held-out friction range: every model loses ~20 classification points and doubles its friction error; structure keeps its absolute lead but does not degrade less (F-29). Held-out compliance range: classification −1 point, sinkage error doubles for every model (F-36). (c) E-8. (d) **supported** (F-32): 4 dead links cost the structured model 1 point vs 4–6 for the baselines; zero-shot to 2× taxel density −1 point, to a 12-link robot −4.5; the baselines cannot run on another layout. Transfer is worth 170–840 episodes of new-robot data (F-37); safe fine-tuning in E-10c. |
| H-3 | A simple simulator lets models exploit sim shortcuts that fail in reality. | open. v1 removed the synthetic sinkage shortcut; F-23 found a sensor-model shortcut (hysteresis signature) used by every model that reads raw dynamics; sensor-model randomization removes it (F-26). |
| H-4 | Structure trades expressiveness for efficiency: the advantage shrinks with data and may cross over. | partial (F-22: the transformer gap halves over 40× data; no crossover; the flat gap does not shrink). E-11 for more data. |
| H-5 | Stage-2 spatial interaction matters when single taxels are ambiguous; its value depends on sensor physics and task. | task-dependent: +1–2 points on terrain (F-18, F-22), decisive for moving stimuli (F-17, F-27). Sensor physics changes it only modestly: +1–4 points at full data (most for FSR and ideal shear, least for capacitive), ≤ 1 at 10 % (F-34, one seed). |
| H-6 | The continuous 3D kernel beats feature-dependent graph message passing. | weakly supported (F-18, one seed) |
| H-7 | Keeping modalities segregated until stage 3 is at least as good and generalizes better. | no difference in-distribution (F-18); generalization open |
| H-8 | A dynamic, geometry-aware, recurrent stage 3 beats static pooling. | contradicted for the v1 brain implementation (F-17, F-19, F-20); research task T-19 |
| H-9 | Stage 2 on stage-1 latents detects stimuli moving across the skin. | **supported** (F-27, 3 seeds: 0.96–0.97 with stage 2, 0.35–0.40 without; F-30, fair baselines with full poses and tuned lr: flat GRU 0.57, transformer at chance, both fit the training set) |
| H-10 | Self-supervised objectives and stage-wise training improve data efficiency and generalization, more for the structured model. | **supported for SSL vs the flat GRU** (F-42–F-45): forecasting each taxel's FSR change 100 ms ahead (no labels) gives the structured model +6.5 / +3.5 / +1.5 points at 170 / 420 / 1,680 episodes (control passed); the same pretext gives the flat GRU nothing (it solves it through a gait shortcut). Transformer: E-14f. Stage-wise training (E-15) open; freezing trained stages during transfer gave no benefit (F-38). |
| H-11 | Physical property estimation is a better generalization target than Earth class labels, and structure helps it. | first half supported (F-25: estimates beat class lookup and transfer sensibly to unseen terrain). Structure: best in distribution (clearly for sinkage), small margin for friction, no help out of distribution (F-25; F-29 and F-36: friction and sinkage errors double beyond the trained range for every model). |
| H-12 | Protocol v4's budget, max(3,000 steps, 15 epochs), trains every model adequately. | **contradicted at intermediate data** (F-28: 2× budget gains 2–5 points at 25 % for all but flat, ≤ 2 at full data; ranking unchanged). Protocol v5 (2× floor, same epochs) adopted for new studies. |
| H-13 | The tactile information is also useful on granular, shifting and uneven terrain. | open (E-7, E-16) |
| H-14 | On terrain, most of the benefit comes from stage 1: one receptor encoder shared by all sensors of a modality, plus pooling. | **supported** (F-22: receptor_only within 1–2 points of somato_pool, 7–8 above the baselines). F-39: even an untrained stage 1 linearizes the FSR (true pressure R² 0.65 vs 0.52 raw); but a frozen random stage 1 costs 8–11 terrain points (F-40): the shared receptor's benefit is structure *and* learning. |

## 2. Experiment queue (ordered)

Order (user, 2026-10-08): signs of life → comparisons with baselines → generalization (sim-to-sim as a proxy). Avoid
very long runs until earlier steps are sound.

| # | ID | experiment | design | config | cost | status |
|---|---|---|---|---|---|---|
| 5 | E-8 | Sim-to-sim (H-2c) | Train mjlab, test Isaac (and the reverse), paired episodes | to write | ~3 h | blocked (T-10) |
| 1 | E-14f | SSL for the transformer (completes H-10's comparison) | A time-causal option for the transformer (patch tokens attend only to earlier or same patches) so it can forecast without seeing the future; per-token node head forecasting each sensor's 100 ms change; then fine-tune the standard transformer from it vs scratch at 170 / 420 episodes, 3 seeds | to write | ~3 h | next |
| 9 | E-15 | Stage-wise vs end-to-end (H-10) | Stages trained on their own objectives and frozen, only stage 3 task-specific | to write | ~4 h | blocked (T-11) |
| 10 | E-11 | Larger data (H-4) | A 9,600-episode dataset; curves to 4× the current maximum | to write | ~10 h | queued |
| 11 | E-16 | Uneven / granular proxies (H-13) | Heightfield terrain, sand-like compliance | to write | — | blocked (T-13) |
| 12 | E-18 | Long-horizon streaming | Part (a) done (F-35: streaming works to 3 s without retraining). Remaining: longer episodes with terrain switches mid-episode; whether TBPTT training (T-17) removes the structured models' small late dip | — | ~2 h | queued (TBPTT part blocked by T-17) |
| 13 | E-17 | Bio-inspired variants | Architectures from the somatosensory literature review | — | — | blocked (T-15) |

**Done:**
* E-14e, SSL for the flat GRU (F-45).
* E-14c/d, SSL checks and curve (F-44).
* E-14b, SSL with a 100 ms horizon (F-43).
* E-14, SSL next-step pretraining (F-42).
* E-25, random frozen stage 1 (F-40).
* E-24, stage probes (F-39).
* E-10c, safe fine-tuning recipes (F-38).
* E-10b, transfer to a 12-link robot (F-37).
* E-9b, held-out compliance range (F-36).
* E-18a, streaming without retraining (F-35).
* E-12, sensor technology (F-34); seeds if the per-skin stage-2 values matter.
* E-23, compute-matched flat GRU (F-33).
* E-10, layout generalization (F-32); fine-tuning on the 12-link robot is a possible follow-up.
* E-22, baseline fairness on the slide test (F-30).
* E-9, held-out friction range (F-29); a compliance holdout is a possible follow-up.
* E-1b, training budget (F-28).
* E-13b, slide proxy with 3 seeds (F-27).
* E-21, sensor-model randomization (F-26).
* E-5, property estimation, with T-21 on the OOD set (F-25).
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
| T-22 | Sensor-model randomization: let sensor params be ranges drawn per sample/episode during training (the FSR already draws per taxel); a randomized `configs/sensors/fsr_v1_random.yaml`. | E-21 | done (2026-10-10; FSR `randomize` option, per-window draws, per-sample sub-ranges for the per-taxel params; unit test) |
| T-10 | Isaac parity for sim-to-sim: PhysX compliant contact per env from the same catalog, an Isaac v1 collection, a paired comparison. First check that Isaac runs on driver 580. | E-8 | queued |
| T-11 | SSL objectives (masked sensor prediction, reconstruction, next-step, cross-modal touch ↔ proprioception); trainer support for pretraining, freezing stages, task-specific last stage only. | E-14, E-15 | partly done (2026-10-12): `node` head, `next_step` task (target = change from the last seen sample; the raw next step was 95 % predictable by persistence), loading across different heads, `train.freeze`; masked reconstruction and cross-modal still to do |
| T-13 | Uneven terrain: mjlab heightfields per env; the taxel contact model must query the terrain height under each taxel (it assumes a plane). | E-16 | queued |
| T-14 | Realism, by expected impact: (a) snow plasticity and ploughing/berm drag; (b) static friction and stick-slip; (c) temperature effects on the sensor; (d) taxel crosstalk and load sharing through the skin. | H-3 | queued |
| T-15 | Literature review of the somatosensory system (SA/RA receptors, dorsal-column pathways, somatotopy, cortical integration) for architecture ideas. | E-17 | queued |
| T-17 | TBPTT failure with 6 chunks per sequence (from v0): test 3 and 4 chunks, more distinct sequences per epoch. | E-18 | queued |
| T-18 | Disk hygiene: delete v0 datasets and caches, and `datasets/mjlab_terrain_600_dc/cache_v1` (4 GB), when no experiment needs them. | — | queued |
| T-23 | Runtime measurement per model (latency per output at batch 1, GPU and CPU, window and stream modes, FLOPs) and simulator real-time factor: `scripts/analysis/runtime.py`. | E-23 | done (2026-10-11, F-31) |
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
  widths matched to somato_pool (~428k). Compute per step is not matched; the user asked (2026-10-11) for matching by
  online cost too, so E-23 tests it.
* **Real-time concerns, noted, not pursued now** (user, 2026-10-11; F-31):
  * on a CPU, window-mode somato_pool (34 ms) and the transformer (129 ms) miss the 20 ms budget; streaming somato
    (2.7 ms) fits and works without retraining up to 3 s, more accurately than window mode (F-35); longer horizons and
    terrain switches untested (E-18);
  * the robot's actual compute is unknown;
  * the simulator runs at 0.34× real time for one robot (an online demo or hardware-in-the-loop would need a faster
    single-env path).
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
| 2026-10-10 | The measured-sinkage target is the penetration of each link capsule's lowest point (`data/derived.py`), regressed as log(sinkage + 0.1 mm); E-5 restarted with it. | The frame-origin depth read link tilt as sinkage: 0.8 ± 0.5 mm on rigid ground vs 0.05 mm true (packed snow 0.47, fresh snow 6.4 mm), so it was mostly pose noise outside fresh snow. No finding had used it. |
| 2026-10-10 | E-5 keeps the checkpoint with the lowest validation property error (`select_metric: -props/zmae`). | E-5 is about property estimates; terrain accuracy is reported at that checkpoint. |
| 2026-10-10 | E-21's randomization ranges contain the nominal sensor and "no hysteresis" but stop short of most E-6 perturbations. | Tests both interpolation (the F-23 failure) and extrapolation beyond the trained range. |
| 2026-10-11 | New studies train on `configs/sensors/fsr_v1_random.yaml` (approved by the user). E-13b, E-1b, E-9 and E-22 stay on fixed fsr_v1 to compare with E-13, E-2 and E-5. Ranges to be refit from bench data when the real skin exists. | F-26: removes the hysteresis shortcut, −0.8 points nominal for the structured models, +4–7 points under the combined shift. |
| 2026-10-11 | Protocol v5 = max(6,000 steps, 15 epochs), in the new base `configs/experiments/terrain_v1b.yaml`. The user asked for a balance instead of the proposed 2× budget (30 epochs). v4 results stand and are not re-run unless a conclusion depends on absolute accuracy. | F-28: the large under-training (2–5 points) is at intermediate data, where runs are cheap. Full-data runs (most of the compute) gain ≤ 2 points with the ranking unchanged. Cost: about +25 % for a 10 % / 100 % study, +45 % for a four-point curve, 0 % for full-data-only studies. Remaining caveat: full-data accuracies may be ~1–2 points low, the transformer's most (+1.9 at 2×). |
| 2026-10-11 | Focus on stages 1–2 and their capabilities; stage 3 (T-19) and locomotion wait. Real-time feasibility is tracked (F-31) but not optimized now. Baselines are also compared at matched online cost where a gap is large (E-23). | User direction, 2026-10-11. |
