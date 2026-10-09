# Investigation log (version 1)

The working plan of the project: what is running, what we believe and want to test, what comes next, and what has
to be built. **Keep it current**: update the status of every item when it changes, add new items as they come up, and
record every experiment's outcome in `docs/findings.md` (why it was run, what was found, the supporting evidence and the
caveats). Design: `docs/architecture.md`. Simulators: `docs/mjlab.md`, `docs/isaac_sim.md`. Literature:
`docs/references/`. Version-0 experiments and their caveats: `docs/archive/`.

Item IDs are stable: H = hypothesis, E = experiment, T = task (code or setup), Q = question for the user, D = decision.
Status words: *running*, *next*, *queued*, *blocked (by X)*, *done (date)*, *dropped (why)*.

## RESUME HERE (state as of 2026-10-08 ~21:20)

* **Running:** E-1, signs of life (log `runs/logs/v1_signs_of_life.log`, output `runs/v1/signs_of_life/`), started
  21:09. Order: the 10 % fraction (somato_v1, flat_gru, transformer), then 100 %. About 2 h (done ~23:00).
* **Next:** while E-1 runs, T-2 to T-5 (docs, independent code review). When E-1 lands, analyse it (findings entry):
  * accuracy vs the Bayes ceilings (F-13: 0.92 noisy, 0.98 exact);
  * per-class recall (expect glare ice ↔ concrete confusions);
  * calibration (NLL);
  * learning curves (log.jsonl);
  * best_step (budget adequacy).

  Then E-1b and E-3.
* **Status commands:**
  ```bash
  tail -3 runs/logs/collect_mjlab_v1_2400.log
  pgrep -af "data_efficiency|collect_mjlab|train.py"
  cat runs/v1/signs_of_life/summary.md
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
  free -g
  ```
* **Environment:**
  * `source ~/miniconda3/etc/profile.d/conda.sh; conda activate mjlab; unset PYTHONPATH`.
  * Strip `/opt/ros` from `LD_LIBRARY_PATH` (ROS Jazzy is sourced in `~/.bashrc`).
  * Run every Python process that loads a dataset or uses the GPU under a memory cap, so a mistake cannot freeze the
    machine again: `systemd-run --user --scope -q -p MemoryMax=24G python ...` for training on the 2,400-episode set,
    8–12G for analysis.
  * Caps should not throttle: mapped dataset pages count against the cap, and the training set touches ~13.5 GB. A
    running job's cap can be raised in place with
    `systemctl --user set-property --runtime <run-*.scope> MemoryMax=24G`.
  * Datasets load through the memory-mapped cache (`cache_v1/`), so several processes may share one dataset. Still
    run only one GPU training job at a time (8 GB card; somato_v1 peaks at 4.8 GB).
* **Git:** commit after every change and push when possible. Pushing currently fails (Q-4).

## 1. Ongoing experiments

| ID | what | started | config / output | status |
|---|---|---|---|---|
| E-1 | Signs of life: somato_v1 / flat_gru / transformer at 10 % and 100 %, seed 0 | 2026-10-08 21:09 | `configs/experiments/v1_signs_of_life.yaml` → `runs/v1/signs_of_life/`, log `runs/logs/v1_signs_of_life.log` | running (first eval: somato_v1 val 0.49 at 250 steps, no plateau) |

## 2. Hypotheses and conjectures

The central hypothesis (user, 2026-10-08): an intentionally structured hierarchy inspired by the somatosensory system
is (H-1) more data-efficient than unstructured models and (H-2) generalizes better. The deployment context:
* short term, arctic monitoring and robots crossing icy terrain in high-latitude cities;
* long term, icy worlds such as Europa, where there will be no training data.

Generalization is therefore the goal, and data efficiency is the first test. The exact structure is a starting
point, to be changed only on evidence.

| ID | hypothesis / conjecture | prediction (what would support it) | tested by | status |
|---|---|---|---|---|
| H-1 | The somato hierarchy is more data-efficient than the flat GRU and the transformer, at equal parameters and with the same inputs including kinematics. | Higher test accuracy at small training sets; the baselines need ≥ 2× the episodes for the same accuracy. | E-1, E-2 | open (v0 supportive but confounded, see findings F0-9/F0-10) |
| H-2 | The somato hierarchy generalizes better under distribution shift. | A smaller accuracy drop than the baselines under (a) sensor non-idealities unseen in training, (b) terrain parameters outside the training range (cold ice, Europa-like, sand, gravel), (c) another simulator, (d) layout changes (dead taxels, other robots). | E-6 to E-10 | open |
| H-3 | A simple simulator lets unstructured models exploit simulator shortcuts: they look good in sim and fall off in the real world. | Any baseline advantage shrinks as the simulation becomes more realistic (v1 vs v0) and reverses under shift; baselines rely on cues a real skin cannot see. | E-1/E-2 vs archived v0, E-6 to E-9, cue-reliance probes | open |
| H-4 | Structure trades expressiveness for efficiency: its advantage shrinks with data and may cross over. | The curves converge or cross at large data. | E-2, E-11 | open (v0: no crossover at 1,680 episodes with enough steps, F0-10) |
| H-5 | Stage-2 spatial interaction matters when single taxels are ambiguous, and how much it matters depends on the sensor physics. | no_spatial loses most on FSR (normal-only) and least with shear-sensing taxels. | E-3, E-12 | open (v0 supportive, F0-7) |
| H-6 | The continuous 3D kernel (weights a function of relative position only) is more data-efficient than feature-dependent graph message passing. | somato_v1 ≥ somato_v1_graph at small data. | E-4 | open |
| H-7 | Keeping modalities segregated until stage 3 is at least as data-efficient as mixing them in stage 2, and generalizes better: each receptor type keeps its own code, as in the nervous system. | somato_v1 ≥ somato_v1_mixed, especially under shift. | E-3, E-6 | open |
| H-8 | A dynamic stage 3 (attention at the current body geometry, memory over time) beats static pooling, because a fixed attention pattern does not fit a moving body. | somato_v1 > somato_v1_pool, especially on tasks needing body-wide or temporal integration. | E-3, E-13 | open |
| H-9 | Stage 2 acting on stage-1 latents, which encode each taxel's recent history, can detect stimuli moving across the skin (e.g. a force sliding along the body in a direction) without an explicit spatio-temporal kernel. | High accuracy on a slide-direction proxy task; it drops without stage 2. | E-13 | open |
| H-10 | Self-supervised objectives (masked sensor prediction, reconstruction, cross-modal prediction) improve data efficiency, more for the structured model, whose stages have natural local objectives. Training stages on their own, with only the last stage task-specific, may generalize better than end-to-end training. | Pretrained + fine-tuned > supervised-only at small data; layer-wise ≥ end-to-end under shift. | E-14, E-15 | open |
| H-11 | Physical property estimation (friction, compliance/sinkage, roughness) is a better target than Earth class labels for generalization to unknown worlds, and structure helps it too. | Properties transfer to OOD terrains where class labels do not exist; the structured model estimates them better. | E-5, E-7 | open |
| H-12 | The protocol-v4 budget, max(3,000 steps, 15 epochs), trains every model adequately at every data fraction. | Doubling the budget changes test accuracy by < 1 point. | E-1b | open (v0: 3,000 steps was not enough at full data, F0-11) |
| H-13 | Whether the tactile information learned for ice and snow is also useful on granular, shifting and uneven terrain. | Reasonable transfer or few-shot adaptation to sand, gravel and uneven heightfield terrain. | E-7, E-16 | open |

## 3. Experiment queue (ordered)

Order rule (user, 2026-10-08):
1. Correctness and signs of life first.
2. Then initial comparisons against the baselines.
3. Then generalization, which is harder; sim-to-sim is a proxy for it.

Avoid very long runs until the earlier steps are sound.

| # | ID | experiment | purpose / design | config | cost | depends on | status |
|---|---|---|---|---|---|---|---|
| 1 | E-1 | **Signs of life** | Does everything train on simulation v1? First comparison (H-1). somato_v1, flat_gru and transformer at 10 % and 100 % of the training episodes, seed 0, widths matched (~636k), protocol v4. Check learning curves, per-class recall and calibration against the Bayes ceilings (F-13). | `configs/experiments/v1_signs_of_life.yaml` | ~2.5 h | T-1, T-6 | running |
| 2 | E-1b | Budget adequacy (H-12) | somato_v1 and transformer at 25 % and 100 % with twice the v4 budget. | to write | ~2 h | E-1 | queued |
| 3 | E-3 | Structure ablations within the family (H-5, H-7, H-8) | somato_v1 vs `_no_spatial`, `_mixed`, `_pool` at 10 % and 100 %, seed 0 (then seeds). | to write | ~3 h | E-1 | queued |
| 4 | E-4 | 3D kernel vs graph (H-6) | somato_v1 vs somato_v1_graph, same fractions. | to write | ~1.5 h | E-1 | queued |
| 5 | E-2 | **Main learning curves** (H-1, H-4) | Fractions 0.025, 0.1, 0.25 and 1.0; seeds 0–2; somato_v1, flat_gru, flat_gru_raw, transformer. | to write | ~20 h | E-1, E-1b | queued |
| 6 | E-5 | Property estimation (H-11) | Regress friction, measured sinkage and roughness (plus classification) for the main models. | `configs/experiments/v1_properties.yaml` | ~3 h | T-7 | queued (ready to run) |
| 7 | E-6 | Sensor robustness (H-2a, H-3) | Evaluate E-2 checkpoints under single-factor perturbations of the FSR v1 model (slower or faster unloading, more hysteresis, gain spread, noise, dead taxels). | T-8 suite | < 1 h | E-2, T-8 | queued |
| 8 | E-8 | Sim-to-sim (H-2c) | Train on mjlab, test on Isaac (and the reverse) on paired episodes. | to write | ~3 h | T-10 | queued |
| 9 | E-9 | Held-out parameter ranges (H-2b) | Train on part of the friction and compliance ranges, test on the rest (extrapolation). | to write | ~3 h | E-1 | queued |
| 10 | E-7 | OOD terrains (H-2b, H-11, H-13) | Cold arctic ice, Europa-like ice (low g, low pressure), dry sand, gravel. Evaluate property estimates and class posteriors (open-set). | to write | ~2 h | T-9, E-5 | queued |
| 11 | E-10 | Layout generalization (H-2d) | Dead-taxel patterns at test time; a robot with different link count or taxel density (structured models only, since the baselines are tied to one layout; report this as a qualitative advantage). | to write | < 1 h | E-2 | queued |
| 12 | E-13 | Slide-direction proxy (H-9, H-8) | A force sliding along the body in a direction; classify the direction and speed. | to write | ~2 h | T-12 | queued |
| 13 | E-14 | SSL pretraining (H-10) | Masked sensor prediction / reconstruction / cross-modal prediction pretraining, then fine-tune at small data. | to write | ~6 h | T-11 | queued |
| 14 | E-15 | Layer-wise vs end-to-end (H-10) | Stages trained on their own objectives, frozen, only stage 3 task-specific; vs end-to-end. | to write | ~4 h | T-11 | queued |
| 15 | E-12 | Sensor technology (H-5) | Ideal + shear, capacitive v1 vs FSR v1 for somato_v1 vs no_spatial vs baselines. | to write | ~3 h | E-1 | queued |
| 16 | E-11 | Larger data (H-4) | A 9,600-episode dataset; curves to 4× the current maximum. | to write | ~10 h | E-2 | queued |
| 17 | E-16 | Uneven / granular proxy environments (H-13) | Heightfield terrain, sand-like compliance. | to write | — | T-13 | queued |
| 18 | E-17 | Bio-inspired variants | Architectures from the somatosensory literature review (T-15). | — | — | E-2, T-15 | queued |
| 19 | E-18 | Long-horizon streaming | The version-0 accuracy decay with time since reset; TBPTT vs sliding window for the v1 models. | — | ~2 h | E-1, T-17 | queued |

## 4. Tasks (code and setup)

**Ongoing**

| ID | task | status |
|---|---|---|
| T-1 | Collect `datasets/mjlab_v1_2400` with the revised catalog v1 and build its cache (16 GB). | done 2026-10-08 (validation 15/15, F-12) |
| T-2 | Write `docs/architecture.md` for version 1. Include: stage mapping receptor / spinal cord / brain; why stage 2 uses the current poses every step (that is how a 3D kernel follows a moving body); the kernel3d vs graph distinction; segregation; brain; baselines. | next |
| T-3 | Update `docs/mjlab.md` (compliant contact pairs, calibration), `README.md` (v1 configs and quickstart), `docs/isaac_sim.md` (driver 580 since 2026-10-08). | next |
| T-4 | Update the persistent memory notes (driver change, memory-cap rule, log structure). | done 2026-10-08 |
| T-5 | Independent code review of the v1 changes (sim pairs, segregation, brain, stride/streaming, baselines, FSR v1, cache). | pending: a three-reviewer workflow was interrupted on 2026-10-08 before it ran; re-run once the user agrees |
| T-6 | `scripts/analysis/task_ceiling.py` for catalog v1: Bayes ceilings from the sampled parameters (friction only, + compliance, + roughness), to interpret E-1. | done 2026-10-08 (F-13) |

**Planned**

| ID | task | for | status |
|---|---|---|---|
| T-7 | Property-regression task: friction, measured sinkage (from body poses: capsule depth when in contact), roughness amplitude; normalized targets, Huber loss; multi-task with classification. | E-5 | done 2026-10-08 for friction + measured sinkage (`property_regression` task, `data/derived.py`, heads created automatically, stats saved in the checkpoint; study config `configs/experiments/v1_properties.yaml`). Felt roughness as a target still to add. |
| T-8 | Robustness suite v1: perturbations relative to `fsr_v1.yaml`, plus dead-taxel patterns at test time (node mask in evaluation). | E-6 | done 2026-10-08 (`configs/sensors/robustness_fsr_v1.yaml`, `AugmentConfig.eval_sensor_dropout`) |
| T-9 | OOD terrain catalogs: cold arctic ice, Europa-like (g = 1.315 m/s² needs a per-scene gravity change in mjlab; skin assumption Q-2), dry sand, gravel; open-set evaluation path for classes absent in training. | E-7 | queued |
| T-10 | Isaac parity for sim-to-sim: PhysX compliant-contact stiffness and damping per env from the same catalog, an Isaac v1 collection, a paired comparison. First check that Isaac still runs on driver 580. | E-8 | queued |
| T-11 | SSL objectives: masked-sensor prediction (stage-1/2 targets), reconstruction, next-step prediction, cross-modal prediction (touch ↔ proprioception); trainer support for pretraining, freezing stages and a task-specific last stage only. | E-14, E-15 | queued |
| T-12 | Slide-direction proxy: generate sliding contacts along the body (scripted forces in mjlab or a stimulus-level generator) with direction and speed labels. | E-13 | queued |
| T-13 | Uneven terrain: mjlab heightfields per env; the taxel contact model must query the terrain height under each taxel (it assumes a plane today). | E-16 | queued |
| T-14 | Further realism, in order of expected impact: (a) snow plasticity (sinkage that does not recover) and ploughing/berm drag on laterally moving links; (b) static friction and stick-slip; (c) temperature effects on the sensor; (d) taxel crosstalk and load sharing through the skin. | H-3 | queued |
| T-15 | Literature review of the somatosensory system (receptor types SA/RA, dorsal-column pathways, somatotopy, cortical integration) for architecture ideas; only after E-2 gives evidence. | E-17 | queued |
| T-16 | Speed: data-loader prefetch (workers need per-worker reseeding of the crop jitter); optionally a fused FSR kernel. | throughput | queued |
| T-17 | TBPTT failure with 6 chunks per sequence (from v0): test 3 and 4 chunks and more distinct sequences per epoch. | E-18 | queued |
| T-18 | Disk hygiene: delete v0 datasets, their caches and the 600_dc cache (4 GB) once no experiment needs them. | — | queued |

**Done this session (2026-10-08)**, details in `docs/findings.md` and the commits:
* Archive of v0 docs and configs.
* Memory-mapped dataset cache.
* Simulation v1: compliant contact pairs, catalog v1, compliance validation check.
* Model v1:
  * segregated stage 2;
  * kernel3d with a support radius;
  * brain head with stride and exact streaming;
  * baselines with kinematics and dropout masking;
  * transformer baseline.
* FSR v1 (per-taxel unloading time constant, rate-independent hysteresis).
* Calibration, profiling and robustness scripts.
* Distance and sensor-loop speed-ups.
* Second literature review (`docs/references/terrain_physics_v2.md`).

## 5. Other tracked items

* **Disk** (117 GB free on 2026-10-08):
  * `datasets/mjlab_v1_2400`: ~1.4 GB of episodes plus a ~16 GB cache.
  * `datasets/mjlab_terrain_600_dc/cache_v1`: 4 GB, from testing the cache; delete if unused.
  * v0 datasets: ~10 GB in total.
  * v0 runs: `runs/v3*`, `runs/v2*`, `runs/isaac600*`.
* **Memory rule.** An uncached dataset of 2,400 episodes costs 13.7 GB per process. Always load datasets through the
  cache and cap every process. The freeze on 2026-10-08 came from four uncached copies.
* **GPU / OS.** Kernel 7.0.0-38 with NVIDIA 580.178 since 2026-10-08 evening (was 595 on 7.0.0-28). An unattended
  kernel update without the matching NVIDIA module broke the GPU once; after kernel updates check that
  `linux-modules-nvidia-*-$(uname -r)` is installed. Isaac Sim is not yet re-tested on 580 (T-10).
* **Parameter budget.** Models are width-matched to somato_v1 (~636k parameters). Compute is not matched: somato_v1 takes
  270 ms per step, the transformer 168 ms, flat_gru 50 ms. The user accepts compute differences unless extreme.
* **The baselines are tied to one sensor layout** (flat: fixed input vector; transformer: fixed group set, optional
  sensor IDs). Layout transfer (E-10) can only be run for the structured models; report that as a capability.

**Open questions for the user**

| ID | question | default until answered |
|---|---|---|
| Q-1 | Should the primary task stay 5-class terrain classification, or move to physical property estimation (friction, compliance/sinkage, roughness), which transfers to unknown worlds (H-11)? | Keep classification as primary for E-1/E-2; add property estimation (E-5). |
| Q-2 | Europa-like OOD set: compliant silicone skin, or glassy skin (silicone is glassy at ~100 K, which changes contact stiffness by orders of magnitude)? | Model both as two OOD sets. |
| Q-3 | With literature friction ranges (glare ice 0.05–1.0 across temperature), glare ice and dry concrete are nearly indistinguishable by touch (same skin-dominated stiffness, invisible fine texture). That is physically honest, but it caps the classification ceiling. Keep it? | Keep; T-6 quantifies the ceiling. |
| Q-4 | `git push` to github.com/trevor386/tactile fails with 403: this machine's gh/git login is user `schannap` (git identity "Suchitha", which also authors the commits). Grant that account write access, or run `gh auth login` as trevor386 (and set `git config user.name/email` if commits should carry your identity)? | Keep committing locally; push as soon as it works. |

## 6. Decisions

| date | decision | why |
|---|---|---|
| 2026-10-08 | mjlab (MuJoCo-Warp) is the primary simulator; Isaac is kept for sim-to-sim. | Agrees with Isaac on the quantities that matter, 6× faster, physically tunable contact (`docs/mjlab.md`). |
| 2026-10-08 | Version 0 archived; its results are not used as evidence for the hypothesis. | Mixed-modality stage 2, confounded baselines, synthetic sinkage cue (`docs/archive/README.md`). |
| 2026-10-08 | Model v1 follows the stated intent: per-modality receptors (stage 1); spatial interaction within each modality (stage 2, a 3D kernel with a physical support radius; the graph approach kept as the alternative); modalities first meet in a dynamic, geometry-aware, recurrent stage 3. | User's design intent (receptors → spinal cord → brain); v0 mixed modalities in stage 2 and pooled statically. |
| 2026-10-08 | The baselines get the same kinematics (body poses / sensor positions in the robot frame) and the same dropout augmentation. | Test the inductive bias, not the inputs. |
| 2026-10-08 | Transformer baseline: (sensor, 100 ms) tokens, global attention; dead sensors read zero rather than being masked out of the attention. | Least-structured standard model; a dense mask over 2,000 tokens ran out of memory and blocks flash attention. |
| 2026-10-08 | Simulation v1: physical terrain compliance (explicit contact pairs, stiff friction), no virtual sinkage, literature friction and roughness ranges, skin-filtered texture. | Remove the synthetic cue that favoured local operators by construction; realism against H-3. |
| 2026-10-08 | Validation: rigid-support static checks run on the stiffest class of a compliant catalog; new check soft_terrain_sinks_deeper; no threshold changed. | On soft terrain the robot sinks physically, so "taxels at ground level" no longer applies there. |
| 2026-10-08 | Protocol v4: budget max(3,000 steps, 15 epochs), validation every 250 steps, best-val checkpoint, bf16, widths matched. | v3's fixed 3,000 steps under-trained full-data runs (F0-11). |
| 2026-10-08 | Every process that loads data or uses the GPU runs under a systemd memory cap; datasets load through the mmap cache. | RAM freeze (F-1). |
| 2026-10-08 | Analysis is done in the main session; subagents only for bulky low-judgment work (literature, mechanical code) or easy Sonnet tasks. | User preference. |
