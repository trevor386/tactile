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

**F-47. E-15, stage-wise training (stages 1–2 trained only by SSL, then frozen; 7 % of the weights learn terrain)
matches scratch training at 170 episodes but falls well behind end-to-end fine-tuning from the same weights.** *Why:*
H-10's stage-wise half (the user's "layer-wise training" idea). *Setup:* `configs/experiments/v1_e15_stagewise.yaml`:
the 100 ms SSL weights (F-43) with stages 1 and 2 frozen (`freeze: [temporal, spatial]`); only the projection, group
embedding and stage-3 head train (29.8k of 428k parameters); 170 / 420 episodes, 3 seeds, `terrain_v1b`.

*Result (test acc, mean ± std over seeds):*

| training | 170 episodes | 420 |
|---|---|---|
| stages 1–2 SSL-only, frozen | 0.602 ± 0.009 | 0.631 ± 0.011 |
| stage 1 SSL-only, frozen (F-43) | 0.606 ± 0.015 | 0.665 ± 0.004 |
| SSL init, all fine-tuned (F-43) | **0.665 ± 0.007** | **0.724 ± 0.016** |
| from scratch, end-to-end | 0.600 ± 0.018 | 0.689 ± 0.016 |
| (F-40) stage 1 random, frozen | 0.536 | — |

*Evidence:*
* Frozen self-supervised stages 1–2 plus a small trainable head reach the scratch model at 170 episodes. Label-free
  stages already carry the terrain-relevant features.
* With more labels the frozen variants fall behind (−5.8 vs scratch at 420), and end-to-end fine-tuning from the SSL
  weights is best everywhere (+6.3 / +9.3 over frozen stages 1–2).
* Freezing stage 2 as well costs nothing at 170 episodes and 3.4 points at 420.

*Caveats:* one SSL objective; stages trained jointly by SSL (not one at a time); 3 seeds.
*Implication:*
* H-10, stage-wise half: stage-wise (frozen) training is viable and label-cheap but not better than end-to-end. The
  recipe is SSL pretraining followed by end-to-end fine-tuning.
* A deployment that must keep stages fixed (e.g. receptor firmware) could use frozen SSL stages at a cost of 0–6
  points.

**F-46. E-14f, the same SSL hurts the transformer (−1.7 / −3.7 points): across all three architectures, label-free
contact forecasting helps only the structured one.** *Why:* complete H-10's comparison (F-45 covered the flat GRU).
*Setup:*
* `configs/experiments/ssl_pretrain_transformer_v1b.yaml`: the transformer (width 108) with time-causal per-token
  outputs (new option `causal`: each 100 ms patch encoded together with earlier patches only);
* each (patch, sensor) token forecasts its sensor's FSR change 100 ms past the end of its patch;
* same data and pretext as F-43;
* fine-tuning `v1_e14f_ssl_transformer.yaml`: the standard transformer from these weights vs from scratch, 170 / 420
  episodes, 3 seeds.

*Result:* pretext test NMSE 0.311 (flat GRU 0.412, structured 0.726). Test acc, mean ± std over seeds:

| model | scratch → SSL, 170 episodes | scratch → SSL, 420 episodes |
|---|---|---|
| somato_pool (F-43) | 0.600 → **0.665 (+6.5)** | 0.689 → **0.724 (+3.5)** |
| flat GRU (F-45) | 0.525 → 0.529 (+0.4) | 0.594 → 0.582 (−1.2) |
| transformer | 0.530 → 0.513 (−1.7) | 0.630 → 0.593 (−3.7) |

*Evidence:*
* The ordering on the pretext is the reverse of the downstream benefit. The models that see every modality (transformer,
  flat GRU) predict the skin's future best, through the robot's own motion (joints, IMU). That leaves terrain-
  irrelevant features, and for the transformer a worse initialization.
* The structured model sees only touch at the per-taxel head (modalities segregated until stage 3). It has to model
  local contact dynamics in the shared receptor, which is exactly what terrain recognition uses (F-40, F-44).

*Caveats:*
* The transformer is pretrained with causal (prefix) attention and fine-tuned with full attention, a context mismatch
  that may cost it something.
* One pretext and one pretraining run per model.
* A touch-only pretext for the baselines (inputs restricted to touch) would remove the gait shortcut; untested.

*Implication:*
* H-10's second half is supported: SSL improves the structured model (+6.5 points at 170 episodes) and not the
  unstructured ones. With SSL the structured lead at 170 episodes is 13.6 points over the flat GRU and 15.2 over the
  transformer (the v5 reference lead without SSL: 7.8).
* The architecture's modality segregation and shared per-sensor receptor make a simple self-supervised objective
  useful. Structure turns cheap unlabelled touch data into better terrain perception.

**F-45. E-14e, the same SSL does nothing for the flat GRU (+0.4 / −1.2 points), although it solves the pretext far
better: SSL helps the structured model specifically.** *Why:* H-10's second half (SSL helps the structured model
*more*). *Setup:*
* `configs/experiments/ssl_pretrain_flat_v1b.yaml`: the flat GRU (width 156) forecasts every taxel's FSR change
  100 ms ahead from its global state at the last 20 steps, through a node head (the same pretext and data as F-43);
* fine-tuning `v1_e14e_ssl_flat.yaml` vs the flat GRU from scratch, 170 / 420 episodes, 3 seeds.

*Result:* pretext test NMSE 0.412 (structured model: 0.726). Test acc, mean ± std over seeds:

| start | 170 episodes | 420 |
|---|---|---|
| flat GRU from scratch | 0.525 ± 0.015 | 0.594 ± 0.006 |
| flat GRU, SSL 100 ms | 0.529 ± 0.010 | 0.582 ± 0.012 |
| (F-43) somato_pool, scratch → SSL | 0.600 → **0.665** | 0.689 → **0.724** |

*Evidence:*
* No gain for the flat GRU at either size, while the structured model gains +6.5 / +3.5 points from the same pretext
  and data.
* The flat GRU predicts the pretext much better (NMSE 0.41 vs 0.73). Its global state sees the joints and IMU, so it
  can forecast contact changes from the gait, a cross-modal shortcut. The structured model's per-taxel head sees only
  touch (modalities segregated until stage 3) and must model local contact dynamics in the shared receptor. Those
  features transfer to terrain; the gait-driven global ones do not.

*Caveats:*
* The transformer has no SSL counterpart yet: it is not causal in time, so it needs a time-causal mask or a
  masked-token pretext.
* One pretraining run per model.
* The flat GRU's pretext differs in what it can see; this is the natural form for its architecture.

*Implication:*
* H-10 supported for the structured model and against the flat baseline. The receptor architecture (shared,
  per-taxel, touch-only) turns a label-free forecasting task into terrain-relevant features; an unstructured model
  takes the easiest predictive shortcut.
* Extends the data-efficiency result: with SSL the structured model's lead over the flat GRU at 170 episodes grows from
  7.5 to 13.6 points.

**F-44. E-14c/d, the SSL gain is genuine (not input statistics), largest at 170 labelled episodes, and the horizon can
go to 200 ms.** *Why:* check F-43's caveat, try a longer horizon, and fill the curve. *Setup:*
* `configs/experiments/v1_e14c_ssl_checks.yaml`, 170 episodes, 3 seeds:
  * the 100 ms SSL weights with input statistics refitted on the labelled subset;
  * SSL with a 200 ms horizon (`horizon: 10`, all 5 brain steps as outputs);
* `v1_e14b_ssl_h5.yaml --set fractions=[0.025,1.0]`: the 100 ms SSL model at 42 and 1,680 episodes.

*Result (test acc, mean ± std over seeds; scratch from E-14 and F-41):*

| start | 42 episodes | 170 | 420 | 1,680 |
|---|---|---|---|---|
| from scratch | 0.491 ± 0.015 | 0.600 ± 0.018 | 0.689 ± 0.016 | 0.773 ± 0.005 |
| SSL 100 ms | 0.506 ± 0.018 | 0.665 ± 0.007 | 0.724 ± 0.016 | **0.788 ± 0.004** |
| SSL 100 ms, statistics refitted | — | 0.666 ± 0.002 | — | — |
| SSL 200 ms | — | **0.673 ± 0.006** | — | — |
| SSL 100 ms, stage 1 frozen | 0.488 ± 0.011 | 0.606 ± 0.015 | 0.665 ± 0.004 | 0.724 ± 0.006 |

Pretext test NMSE: 20 ms 0.816, 100 ms 0.726, 200 ms 0.661.

*Evidence:*
* With statistics refitted on the labelled subset, as in scratch training, the gain is unchanged (0.666 vs 0.665). It
  comes from the pretrained weights.
* The gain is largest at 170 episodes (+6.5 points), still clear at 420 (+3.5) and at full data (+1.5, 3 seed stds),
  and within noise at 42 (+1.5). With ~8 labelled episodes per class the head cannot be fitted well whatever the
  features.
* Longer horizons are more predictable and slightly better: 200 ms 0.673 vs 100 ms 0.665, within ~1 std.
* Frozen SSL receptors trail full fine-tuning more as the labels grow (−6 at 420, −6.4 at 1,680). Labels still refine
  the receptors.

*Caveats:*
* One pretraining run per horizon.
* The unlabelled pool is the 1,680 training episodes.
* The baselines have no SSL yet (E-14e).

*Implication:*
* SSL on contact dynamics improves the structured model at every data size above a few episodes per class. That
  supports H-10 for the structured model; whether it helps it *more* than the baselines needs E-14e.
* Default SSL horizon: 100–200 ms.
* A larger unlabelled pool is the obvious next lever: unlabelled touch data is cheap to log on a robot.

**F-43. E-14b, SSL forecasting 100 ms ahead gives a real data-efficiency gain: +6.5 points at 170 labelled episodes,
+3.5 at 420 (20 ms: +2.2 / +1.6).** *Why:* F-42's 20 ms target was mostly noise. *Setup:*
* as E-14 but `next_step` with `horizon: 5` (the FSR change 100 ms ahead): `ssl_pretrain_v1b.yaml --set
  tasks.ssl.horizon=5`;
* fine-tuning `configs/experiments/v1_e14b_ssl_h5.yaml`, 3 seeds.

*Result:* pretext test NMSE 0.726 (27 % of the 100 ms change predictable; 20 ms: 18 %). Test acc, mean ± std over
seeds:

| start | 170 episodes | 420 |
|---|---|---|
| from scratch (E-14) | 0.600 ± 0.018 | 0.689 ± 0.016 |
| SSL 20 ms (E-14) | 0.622 ± 0.016 | 0.705 ± 0.008 |
| **SSL 100 ms** | **0.665 ± 0.007** | **0.724 ± 0.016** |
| SSL 100 ms, stage 1 frozen | 0.606 ± 0.015 | 0.665 ± 0.004 |

*Evidence:*
* +6.5 points at 170 episodes (> 3 seed stds) and +3.5 at 420. 170 labelled episodes with SSL get within 2.4 points
  of 420 without it.
* The longer horizon is more predictable and gives better receptors: frozen, 0.606 vs 0.597 (20 ms) vs 0.536 (random).
  It is also a better initialization: +4.3 points over 20 ms.
* For context (v5 reference, F-41): the transformer at 170 episodes reaches 0.537, so somato_pool with SSL leads it
  by 12.8 points.

*Caveats:*
* The SSL runs keep input statistics fitted on the 1,680 unlabelled episodes, while scratch fits them on the labelled
  subset. **Control (F-44): with refitted statistics the gain is unchanged.**
* One pretraining run.
* The unlabelled pool is the training split itself.
* The baselines have no SSL counterpart yet, so "SSL helps the structured model more" (H-10) is untested.

*Implication:*
* Self-supervised forecasting of contact dynamics is a real data-efficiency lever for the receptor architecture, and
  the horizon matters.
* E-14c checks the statistics control and a 200 ms horizon; E-14d fills the curve at 42 and 1,680 episodes.

**F-42. E-14, self-supervised next-step pretraining: receptors learned without labels are almost as good as
supervised ones (frozen: 0.597 vs 0.600 at 10 %, random 0.536), but as an initialization they add only ~2 points.**
*Why:* H-10. F-40 showed that trained receptors are worth 8–11 points; can they be trained without labels? *Setup:*
* `configs/experiments/ssl_pretrain_v1b.yaml`: somato_pool with a per-node head;
* task `next_step`: forecast every taxel's FSR *change* over the next 20 ms (10 samples) from its stage-2 features at a
  predicted step;
* the 1,680 training episodes without labels, randomized FSR, protocol v5, best checkpoint by validation NMSE;
* fine-tuning: `configs/experiments/v1_e14_ssl.yaml`, 42 / 170 / 420 labelled episodes, 3 seeds:
  * from scratch;
  * from the SSL weights (input statistics kept);
  * SSL weights with stage 1 frozen.

*Result:* pretext test NMSE 0.816 (the next 20 ms change is 18 % predictable; it was 95 % predictable without the
delta). Test acc, mean ± std over seeds:

| start | 42 episodes | 170 | 420 |
|---|---|---|---|
| from scratch | 0.491 ± 0.015 | 0.600 ± 0.018 | 0.689 ± 0.016 |
| SSL, fine-tune all | 0.485 ± 0.018 | **0.622 ± 0.016** | **0.705 ± 0.008** |
| SSL, stage 1 frozen | 0.462 ± 0.008 | 0.597 ± 0.016 | 0.646 ± 0.011 |
| (F-40) random stage 1, frozen | — | 0.536 | — |

*Evidence:*
* Frozen SSL receptors match fully trained ones at 170 episodes (0.597 vs 0.600) and beat frozen random receptors by
  6 points. Forecasting contact dynamics teaches the receptor stage most of what terrain needs, without labels.
* At 420 episodes frozen SSL receptors fall 4 points behind (supervised refinement matters with more labels), and at
  42 they are 3 behind.
* As an initialization, SSL gains +2.2 / +1.6 points at 170 / 420 episodes (about 1 seed std), nothing at 42.

*Caveats:*
* One pretext objective with a short horizon, whose target is mostly noise (R² 0.18).
* The unlabelled pool is the same 1,680 episodes, only 4–40× the labelled sets.
* Same simulator distribution.

*Implication:*
* H-10: partly supported. SSL produces useful receptors (the stage-1 capability is learnable without labels), but the
  data-efficiency gain over supervised training is small with this objective and pool.
* Next: a longer forecast horizon (100 ms, more dynamics, less noise: E-14b). If that helps, a larger unlabelled pool
  (more simulation, cheap on a real robot too) and spatial masked reconstruction for stage 2.

**F-41. v5 reference (randomized FSR, protocol v5, 3 seeds): the structured lead holds and slightly grows over E-2.**
*Why:* a 3-seed reference for studies on the new defaults (`terrain_v1b`). *Setup:* seed 0 from E-12
(`runs/v1/e12_fsr_random`), seeds 1–2 from `configs/experiments/v1b_reference_seeds.yaml` (receptor_only, flat_gru,
transformer at E-12's matched widths), somato_pool's 3 seeds from E-25.

*Result (test acc, mean ± std; E-2 = fixed FSR, protocol v4):*

| model | 170 episodes | 1,680 episodes |
|---|---|---|
| somato_pool | **0.615 ± 0.015** (E-2 0.606) | **0.773 ± 0.005** (0.778) |
| receptor_only | 0.592 ± 0.009 (0.597) | 0.749 ± 0.003 (0.759) |
| transformer | 0.537 ± 0.021 (0.548) | 0.720 ± 0.009 (0.734) |
| flat_gru | 0.525 ± 0.015 (0.531) | 0.703 ± 0.007 (0.684) |

*Evidence:*
* Lead of somato_pool over the best baseline: +7.8 / +5.3 points (E-2: +5.8 / +4.4).
* Randomized-sensor training costs the transformer 1–1.4 points and the structured models ≤ 1. The flat GRU gains
  2 at full data; it relied least on the hysteresis signature (F-23).
* v5's doubled step floor changes nothing at 10 %: v4 already trained ≥ 43 epochs there. E-1b's gains were at 25 %.

*Caveats:* nominal accuracy on randomized test sensors (each study's test split); the E-2 comparison mixes sensor
model and protocol.
*Implication:* the main comparison (F-22) holds under the realistic defaults. Use these numbers as the reference for
v5 studies.

**F-40. E-25, stage 1 must be trained: frozen at random weights it costs 8–11 points; the architecture alone (random
receptors + trained stages 2–3) still beats the baselines at 10 % but not at full data.** *Why:* F-39 found that an
untrained stage 1 already makes physical quantities linearly readable. Is the receptor stage's benefit architectural?
*Setup:* `configs/experiments/v1_e25_random_stage1.yaml`, `terrain_v1b` (randomized FSR, protocol v5), 3 seeds; new
`train.freeze: [temporal]` keeps stage 1 at its random initialization (input statistics still fitted).

*Result (test acc, mean ± std over seeds; baselines from E-12, same protocol, seed 0):*

| model | trainable params | 170 episodes | 1,680 episodes |
|---|---|---|---|
| somato_pool, all trained | 428k | **0.615 ± 0.015** | **0.773 ± 0.005** |
| somato_pool, stage 1 random and frozen | 311k | 0.536 ± 0.007 | 0.663 ± 0.008 |
| receptor_only, stage 1 random and frozen | 311k | 0.491 ± 0.011 | 0.610 ± 0.004 |
| (E-12) transformer / flat GRU | 439k / 430k | 0.507 / 0.504 | 0.731 / 0.700 |

*Evidence:*
* Training stage 1 is worth 8 points at 10 % and 11 at full data for somato_pool; for receptor_only (no stage 2)
  about 11 / 14 (trained receptor_only on this protocol: 0.605 / 0.748, E-12 seed 0).
* With random receptors, the 3D kernel still adds 4.5–5 points (somato 0.536 / 0.663 vs receptor 0.491 / 0.610).
  Stage 2 can partly compensate for untrained receptors.
* Random receptors plus trained stages 2–3 still beat both baselines at 10 % (0.536 vs 0.507), but fall 4–7 points
  behind them at full data.

*Caveats:*
* One random initialization scheme (PyTorch defaults).
* Frozen models have ~27 % fewer trainable parameters, though the same compute.
* Baseline numbers are single-seed.

*Implication:*
* The receptor stage's advantage is the combination of architecture and learning. Its structure (one encoder shared by
  all taxels, own history only) makes learning efficient, but the learned temporal features matter (8–11 points).
  F-39's probes showed that physical linearity alone is not what terrain recognition needs.
* H-14 refined: stage 1 carries most of the benefit *because* a shared, trained receptor learns task-relevant temporal
  features from every taxel's data at once.
* For SSL (T-11): a pretext task should train stage 1's temporal features, not just keep it random. This makes
  stage-1 pretraining a plausible data-efficiency lever.

**F-39. E-24, what the stages encode: stage 1's per-taxel features make the true pressure more linearly readable than
the raw FSR history (R² 0.63 vs 0.52 in contact), and even untrained (random-weight) stage 1 does so (0.65); training
for terrain adds a little link-level contact and slip information and gives up some per-taxel detail. Link slip is
barely encoded anywhere.** *Why:* the user's direction (2026-10-11) to learn what the stages can do. *Setup:*
* `scripts/analysis/stage_probes.py`: linear probes (ridge, strength chosen on a hold-out) fitted on half of the test
  episodes, R² on the other half;
* fixed FSR v1 readings; features at the last step of each window;
* E-21 somato_pool and receptor_only, 3 seeds, plus the same models with re-initialized weights (input statistics
  kept; seed 0);
* per taxel: true normal pressure, true shear (an FSR does not sense it), link slip;
* per link (mean and max of the features over its taxels): contact, slip speed, normal force.

*Result (R², mean over seeds; std ≤ 0.03):*

| features | pressure, all | pressure, in contact | shear, in contact | link contact | link slip | link force |
|---|---|---|---|---|---|---|
| raw FSR, last step (10) | 0.61 | 0.52 | 0.43 | 0.39 | 0.00 | 0.65 |
| raw FSR, whole window (250) | 0.62 | 0.52 | 0.42 | 0.39 | 0.00 | 0.65 |
| somato_pool stage 1, trained | 0.69 | 0.63 | 0.47 | 0.75 | 0.10 | 0.71 |
| somato_pool stage 2, trained | 0.60 | 0.56 | 0.42 | 0.71 | **0.12** | 0.67 |
| receptor_only stage 1, trained | 0.71 | 0.63 | 0.48 | **0.76** | 0.09 | 0.71 |
| somato_pool stage 1, untrained | 0.72 | 0.65 | 0.49 | 0.73 | 0.04 | 0.72 |
| receptor_only stage 1, untrained | **0.73** | **0.65** | **0.49** | 0.74 | 0.04 | **0.72** |

*Evidence:*
* A linear filter of a taxel's raw history (250 samples) is no better than its last 10 samples. The FSR's hysteresis
  and saturation are nonlinear, so linear filtering cannot undo them.
* Stage 1's nonlinear temporal encoder (shared by all taxels) recovers the true pressure much better, trained or not.
  With random weights it is a random nonlinear feature bank over each taxel's history (a reservoir), which already
  linearizes the sensor.
* Training for terrain classification slightly *reduces* per-taxel pressure and shear information (0.65 → 0.63;
  stage 2 to 0.56). It slightly increases link contact (+0.02) and link slip (0.04 → 0.10–0.12), the most of any
  features in stage 2.
* Link slip speed is barely linearly decodable from any features (≤ 0.12). An FSR measures normal force only; slip is
  better sensed through shear (E-12: the shear skin is +15 points).
* Shear is readable to R² ~0.4–0.5 from the normal pressure alone (in sliding contact shear ≈ friction × pressure),
  and stage 1 adds little.

*Caveats:*
* Linear probes only.
* Feature dimension differs (10 / 250 / 64 / 104).
* One evaluation sensor (fixed FSR); E-21 models trained with randomized FSRs.

*Implication:*
* Stage 1's value is partly structural: a shared nonlinear temporal filter over each taxel's history turns a
  hysteretic, saturating sensor into features from which physical quantities are linearly readable, before any
  training. This fits F-22: stage 1 carries most of the data-efficiency advantage.
* Natural follow-up, **E-25**: train with stage 1 frozen at random weights. If the terrain accuracy stays close, the
  receptor stage's benefit is architectural (random features suffice).
* For SSL (T-11, E-14), a pretext task that rewards physical quantities (e.g. reconstructing neighbours' pressure)
  could keep the per-taxel detail that terrain training discards.

**F-38. E-10c, safe fine-tuning: validating the starting weights before the first update prevents the F-37 collapse;
fine-tuning gains only from ~170 episodes; freezing stages does not help.** *Why:* F-37 found naive fine-tuning on
10–40 episodes fell below zero-shot. *Setup:*
* `configs/experiments/v1_e10c_finetune_recipes.yaml`: the E-21 16-link somato_pool, 12-link data, ~10 / 40 / 170
  episodes, 3 seeds (different subsets per seed), 1,500-step budget, input statistics kept;
* the trainer now validates the starting weights when `init_checkpoint` is set, so they can be kept;
* new `train.freeze` (parameter-name prefixes).

*Result (test acc, mean ± std over seeds; zero-shot 0.754, E-10b):*

| recipe | 10 episodes | 40 | 170 |
|---|---|---|---|
| all weights, lr 1e-4 | 0.760 ± 0.003 (initial weights kept) | 0.753 ± 0.002 | **0.781 ± 0.006** |
| stage 1 frozen, lr 3e-4 | 0.760 ± 0.003 (initial kept) | 0.745 ± 0.011 | 0.777 ± 0.009 |
| stages 1–2 frozen, lr 3e-4 | 0.760 ± 0.003 (initial kept) | 0.749 ± 0.009 | 0.772 ± 0.007 |
| (E-10b, seed 0) naive fine-tuning / from scratch | 0.646 / 0.383 | 0.700 / 0.510 | 0.777 / 0.700 |

*Evidence:*
* With 10 episodes every recipe keeps the starting weights (best step 0 in all 9 runs), so the collapse is gone.
* With 40 episodes fine-tuning cannot beat zero-shot (validation on a small held-out set picks a slightly worse
  model).
* With 170 episodes all recipes gain 2–3 points over zero-shot and 7–8 over scratch.
* Freezing stage 1 (receptors) or stages 1–2 is no better than fine-tuning everything at a low lr, slightly worse at
  170. The new robot's differences are not confined to stage 3.

*Caveats:*
* One source model (seed 0) and one target robot.
* The 12-link robot differs in body dynamics as well as skin layout.
* The 40-episode differences are within noise.

*Implication:*
* Recipe for a new robot: use the structured model as is; fine-tune all weights at lr 1e-4, validating the starting
  weights, once ~150+ episodes exist.
* H-10 (stage-wise training): freezing trained stages is not a free win for transfer. SSL-style stage-wise training is
  still untested (T-11).

**F-37. E-10b, transfer to a new robot: the 16-link model used as is on the 12-link robot (0.754) beats training from
scratch until ~170+ episodes; naive fine-tuning on 10–40 episodes makes it worse; with 840 episodes fine-tuning edges
out scratch training.** *Why:* how much data does a new robot need when starting from the structured model of another
robot (F-32 follow-up)? *Setup:*
* `configs/experiments/v1_e10b_finetune_12links.yaml`: 12-link training set `mjlab_v1_links12_1200` (collection seed
  2, disjoint from E-10's test set);
* protocol v5, randomized FSR, seed 0;
* the E-21 16-link somato_pool (seed 0):
  * zero-shot;
  * fine-tuned at lr 3e-4 with its input statistics kept (new `train.init_checkpoint`, `train.fit_normalizers`);
* from scratch: somato_pool (lr 1e-3) and flat GRU (lr 3e-4).

*Result (test acc on the 12-link test split):*

| training episodes (12-link) | 10 | 40 | 170 | 840 |
|---|---|---|---|---|
| 16-link model, zero-shot | 0.754 | 0.754 | 0.754 | 0.754 |
| 16-link model, fine-tuned | 0.646 | 0.700 | **0.777** | **0.825** |
| somato_pool from scratch | 0.383 | 0.510 | 0.700 | 0.813 |
| flat GRU from scratch | 0.345 | 0.489 | 0.656 | 0.743 |

*Evidence:*
* Transfer is worth a lot of data: the 16-link model with no 12-link data matches scratch training with somewhere
  between 170 and 840 episodes.
* Fine-tuning with 10–40 episodes lowers accuracy below zero-shot (−11 / −5 points). The first validation comes after
  250 steps, by which time the pretrained solution is gone, and the protocol cannot fall back to the initial weights.
* With 170 episodes fine-tuning gains 2.3 points over zero-shot and leads scratch training by 7.7. With 840 it leads
  scratch by 1.2.
* The structured model from scratch beats the flat GRU from scratch at every size (+3.8 to +7.0), consistent with
  F-22 on the 16-link robot.

*Caveats:*
* One seed; the 10- and 40-episode points are noisy.
* The fine-tuning recipe was not tuned (lr 3e-4, all weights, protocol v5's 6,000 steps on 10 episodes).

*Implication:*
* A structured encoder trained on one robot is a strong starting point for another (H-2d): days of data collection
  saved.
* Fine-tuning needs a safer recipe:
  * validate before the first update, so the pretrained weights are a candidate;
  * a lower lr or fewer steps;
  * freezing stage 1 (the receptors should transfer unchanged; stage-wise training also bears on H-10).
* E-10c tests this.

**F-36. E-9b, held-out compliance range: classification barely suffers (−1 point), but the sinkage error doubles for
every model; structure does not extrapolate better here either.** *Why:* the compliance counterpart of E-9 (H-2b,
H-11): does the sinkage estimate extrapolate to softer ground than trained? *Setup:* E-9 (`v1_e9_holdout.yaml`, E-5
tasks and models, 3 seeds) with `holdout.param = contact_width`. The softest 30 % of each class is held out: fresh snow
38–50 mm transition width, packed snow 5.9–8 mm, ice and concrete ~0.9–2 mm.

*Result (mean ± std over seeds):*

| model | acc in range → held out | friction MAE in range → held out | sinkage MAE in range → held out (mm) |
|---|---|---|---|
| somato_pool | **0.787 → 0.780** | 0.070 → 0.067 | **0.25** → 0.50 |
| receptor_only | 0.775 → 0.769 | 0.072 → 0.069 | 0.36 → 0.60 |
| transformer | 0.721 → 0.704 | 0.075 → 0.071 | 0.29 → 0.52 |
| flat_gru | 0.674 → 0.664 | 0.072 → 0.071 | 0.28 → **0.47** |

*Evidence:*
* Softer-than-trained variants are still classified correctly (−0.6 to −1.7 points). Softness extremes stay within
  their class's character, unlike friction extremes (F-29: −20 points).
* The sinkage error doubles for every model on the held-out range, with no structural advantage in extrapolation:
  somato_pool is best in range, the flat GRU marginally best held out.
* Friction error is unaffected, as expected (the held-out episodes differ in compliance only).

*Caveats:* one holdout fraction; no per-episode bias analysis (presumably underestimation of the deepest sinkage).

*Implication:* together with F-25 and F-29, physical property estimates degrade about 2× beyond the trained range for
every architecture. Structure helps estimation in distribution (sinkage especially), not extrapolation. To widen the
range, train on wider ranges (domain randomization of terrain, as for the sensor in F-26); no architecture here
extrapolates.

**F-35. E-18a, streaming without retraining works: with the state carried over whole episodes, window-trained models
keep improving past their 0.5 s training window, by 3–5 points over window mode after 1 s; the structured models lose
1–2 points again at 2–3 s.** *Why:* streaming is the cheap online mode (somato_pool 2.7 ms per step on a CPU vs 34 ms
in window mode, F-31). Is it usable for models trained only on independent 0.5 s windows? *Setup:*
* `scripts/analysis/streaming_eval.py`: E-21 full-data checkpoints (randomized-FSR training), 3 seeds;
* every test episode (150 latent steps, 3 s) in one pass from a fresh state, equal to step-by-step streaming;
* accuracy per time-since-reset bin, fixed FSR v1;
* window-mode reference: E-21's nominal robustness accuracy (independent 0.5 s windows).

*Result (mean of 3 seeds):*

| model | window mode | stream 0–0.5 s | 0.5–1 s | 1–2 s | 2–3 s |
|---|---|---|---|---|---|
| somato_pool | 0.776 | 0.728 | 0.799 | **0.820** | 0.807 |
| receptor_only | 0.752 | 0.684 | 0.766 | 0.784 | 0.774 |
| flat_gru | 0.695 | 0.617 | 0.711 | 0.741 | 0.742 |

*Evidence:*
* All models are stable beyond their training horizon and better than in window mode: they integrate evidence over
  more time (+3–5 points at 1–2 s).
* The first 0.5 s bin is lower than window mode because it averages over steps with less context (from step 5 on),
  whereas window mode always predicts after 25 steps.
* somato_pool and receptor_only drop 1–1.3 points from the 1–2 s to the 2–3 s bin (seed 1 most: −2); the flat GRU does
  not. This is a hint of state drift beyond the trained horizon for the structured models, small at 3 s.
* The ranking is unchanged in every bin.

*Caveats:*
* Episodes are only 3 s, so longer horizons (minutes on a robot) are untested and the 2–3 s dip could grow.
* Terrain is constant within an episode, so streaming here never has to *switch* its estimate. A terrain change
  mid-episode would test that.

*Implication:*
* The CPU real-time path (streaming, F-31) is usable for these models without retraining, and is more accurate than
  window mode. This partly resolves the real-time concern.
* For long horizons or terrain switches, remaining E-18 work: longer episodes with terrain changes, and whether TBPTT
  training (T-17) removes the late dip.

**F-34. E-12, sensor technology: the structured lead holds on every skin (5–10 points over the best baseline); the
skin itself matters far more than the model (an ideal shear-sensing skin is +15 points over the FSR); stage 2 adds 1–4
points at full data, depending on the skin.** *Why:* H-5 (stage 2's value depends on sensor physics), and whether the
model comparison depends on the FSR. *Setup:*
* `configs/experiments/v1_e12_sensor_tech.yaml` on the new base (`terrain_v1b`: protocol v5);
* the E-2 models and learning rates (tuned on the FSR), widths matched per skin, seed 0, 10 % / 100 %;
* three skins, with the same joint and IMU models:
  * randomized FSR v1 (normal force only, hysteresis);
  * `ideal_shear_v1.yaml`: normal and shear pressure, no noise or dynamics, an upper bound;
  * `capacitive.yaml`: viscoelastic dielectric, drift, normal only.

*Result (test acc, 170 / 1,680 training episodes):*

| skin | somato_pool | receptor_only | transformer | flat_gru | lead over best baseline | stage 2 (somato − receptor) |
|---|---|---|---|---|---|---|
| FSR, randomized | **0.607 / 0.785** | 0.605 / 0.748 | 0.507 / 0.731 | 0.504 / 0.700 | +10.0 / +5.4 | +0.2 / +3.7 |
| ideal + shear | **0.786 / 0.932** | 0.776 / 0.900 | 0.731 / 0.838 | 0.616 / 0.784 | +5.5 / +9.4 | +1.0 / +3.2 |
| capacitive | 0.697 / **0.831** | **0.699** / 0.819 | 0.619 / 0.772 | 0.617 / 0.778 | +7.8 / +5.3 | −0.2 / +1.2 |

*Evidence:*
* The ranking (structured > transformer ≈ flat) holds on all three skins at both data sizes. The structured models
  lead by 5–10 points, so F-22's conclusion does not hinge on the FSR model.
* The skin matters most: an ideal skin with shear lifts somato_pool from 0.785 to 0.932 at full data, and a
  capacitive skin to 0.831. The FSR's dynamics and lack of shear are the main limit on what touch can tell about
  terrain here.
* With shear, the transformer gains most among the baselines (+10.7 points at full data); the flat GRU gains less
  (+8.4).
* Stage 2's value at full data is largest for the FSR and the ideal shear skin (3–4 points) and small for the
  capacitive one (1.2). At 10 % it is ≤ 1 point for all. That fits F-22 (on terrain, stage 1 carries most of the
  benefit) and F-27 (stage 2 is decisive only for moving stimuli).

*Caveats:*
* One seed. Differences of 1–2 points (e.g. the per-skin stage-2 values) are within seed noise (std 0.005–0.02).
* Learning rates were tuned on the FSR.
* The capacitive model is the v0-era model without per-taxel variability or randomization.
* Ideal + shear is an upper bound, not a real sensor.

*Implication:*
* H-5: stage 2's value depends on the skin only modestly (1–4 points at full data). The task matters much more (F-27).
* Robustness of the main conclusion across sensor technologies: supported.
* For hardware, shear sensing is worth far more than any architecture choice here (+15 points). Worth telling whoever
  designs the skin.

**F-33. E-23, compute matching: given somato_pool's online compute (a 13× wider, 37M-parameter flat GRU), the flat
baseline gains ~2 points and stays 5–7 points behind; the ranking is unchanged.** *Why:* the user (2026-10-11) asked to
match models by online cost, since that matters for deployment. At matched parameters the flat GRU is 7–19× cheaper per
output than somato_pool (F-31); the transformer and receptor_only are within 1.3–2× (on the GPU), so only the flat GRU
had headroom worth testing. *Setup:*
* `configs/experiments/v1_e23_compute_matched.yaml` (widths 512 / 1,024 / 2,048 at lr 3e-4 / 1e-4, seed 0) and
  `v1_e23_seeds.yaml` (the two lr-1e-4 models, seeds 1–2);
* E-2 settings (protocol v4, fixed FSR v1), so E-2 is the comparison.
* Width 2,048 runs a window in 4.0 ms on the GPU, the same as somato_pool's 4.1 ms; on the CPU the matched width is
  ~1,300–1,750, between the two tested sizes.

*Result (test acc, mean ± std over 3 seeds):*

| model | params | GPU ms / window | 170 episodes | 1,680 episodes |
|---|---|---|---|---|
| flat_gru, E-2 (width 156, lr 3e-4) | 0.43 M | 0.6 | 0.531 ± 0.013 | 0.684 ± 0.008 |
| flat_gru, width 1,024, lr 1e-4 | 9.9 M | 1.5 | 0.544 ± 0.011 | **0.708 ± 0.004** |
| flat_gru, width 2,048, lr 1e-4 (latency-matched) | 36.6 M | 4.0 | **0.553 ± 0.005** | 0.706 ± 0.008 |
| transformer, E-2 | 0.44 M | 5.3 | 0.548 ± 0.022 | 0.734 ± 0.008 |
| somato_pool, E-2 | 0.43 M | 4.1 | 0.606 ± 0.016 | 0.778 ± 0.004 |

At lr 3e-4 the wide models were worse (0.497–0.509 / 0.659–0.675, seed 0) and over-confident (test NLL up to 3.9).

*Evidence:*
* Matching compute instead of parameters closes about a quarter of the flat GRU's gap: 7.5 → 5.3 points at 10 %,
  9.4 → 7.2 at full data. The remaining gap is > 3 seed stds at both sizes.
* The wide flat GRU reaches the transformer at 10 % (0.553 vs 0.548) but not at full data (0.706 vs 0.734).
* Going from width 1,024 to 2,048 adds nothing at full data: capacity is not the flat GRU's bottleneck.

*Caveats:*
* lr 1e-4 was not tried at width 156, so part of the gain may come from the learning rate rather than the width.
* Protocol v4 (F-28: up to 2–5 points of under-training at intermediate data, similar across models).
* Latency matching uses this GPU; on a CPU the matched width is somewhat smaller.

*Implication:*
* The structured models' advantage is not an artifact of matching parameters instead of compute. With the same
  online budget, the extra capacity does not substitute for the inductive bias.
* No further compute-matching work is needed for the flat GRU. The transformer is already at similar (GPU) or higher
  (CPU) cost than somato_pool, so it cannot gain from matching.

**F-32. E-10, layout generalization: the structured models shrug off contiguous skin loss and transfer zero-shot to a
denser skin (−1 point) and a shorter robot (−4.5 points); the baselines lose 4–6 points to dead links and cannot run on
another layout at all.** *Why:* H-2d. A deployed skin loses whole segments (connectors, tears), and the same encoder
should serve other robots and skins. *Setup:*
* E-21 full-data checkpoints (randomized-FSR training), 3 seeds, no retraining; evaluation on the fixed fsr_v1;
* `scripts/analysis/robustness.py` with `configs/sensors/robustness_layout_v1.yaml`:
  * new option `eval_dead_links`: all taxels of k random links are dead (masked; the baselines read zeros);
* new-layout test sets, 400 mjlab episodes each, all episodes evaluated, collection seed 0:
  * `mjlab_v1_dense_400`: 4 × 12 taxels per link, 768 taxels;
  * `mjlab_v1_links12_400`: 12 links, 288 taxels;
  * `mjlab_v1_control_400`: the original layout. Same seed and environment count, so it is a paired control with the
    same terrain and gait draws (training data used seed 1).

*Result (a), training layout (test split), change from nominal in points:*

| model | nominal | 1 dead link | 2 | 4 (25 % of the skin) | 25 % random dead taxels |
|---|---|---|---|---|---|
| somato_pool | 0.776 | 0.0 | −0.4 | **−1.1** | −2.6 |
| receptor_only | 0.752 | −0.6 | −0.9 | −1.4 | −1.0 |
| transformer | 0.709 | −0.4 | −1.4 | −4.0 | −2.3 |
| flat_gru | 0.695 | −2.2 | −3.7 | −6.3 | −5.4 |

*Result (b), zero-shot on new layouts (accuracy, mean of 3 seeds, seed std ≤ 0.011):*

| model | paired control (training layout) | 2× taxel density | 12-link robot |
|---|---|---|---|
| somato_pool | 0.805 | 0.794 (−1.1) | 0.760 (−4.5) |
| receptor_only | 0.771 | 0.763 (−0.8) | 0.708 (−6.3) |
| flat_gru, transformer | — | cannot run | cannot run |

*Evidence:*
* With 4 of 16 links' skin dead, somato_pool loses 1.1 points; the baselines lose 4.0 (transformer) and 6.3 (flat).
  The structured models pool over whichever sensors are alive, and stage 1 is shared, so no weight depends on one
  link's skin. The flat GRU's per-input weights do.
* For somato_pool, contiguous loss hurts less than the same number of scattered dead taxels (−1.1 vs −2.6).
  Scattered loss thins every neighbourhood of the spatial stage; contiguous loss leaves the remaining neighbourhoods
  intact.
* A skin with twice the taxels (each sensor's neighbourhood in the 45 mm kernel twice as dense) costs ~1 point
  without retraining.
* A 12-link robot (different dynamics as well as a different skin) costs 4.5 points for somato_pool, 6.3 for
  receptor_only. Part of this is physics: the shorter, lighter body moves differently on the same terrain.

*Caveats:*
* Single-terrain-distribution test sets of 400 episodes; the paired control removes most of the sampling noise.
* The 12-link set's gait draws may differ from the control's (fewer joints).
* Taxel area and sensor parameters were kept as trained, so a denser skin with smaller real taxels would differ.
* The baselines could only be tested in (a).

*Implication:*
* H-2d is supported. Layout independence is a real capability of the structured design: robust to contiguous damage,
  transferable to other skins and robots, while the baselines are tied to one layout by construction.
* A fine-tuning test on the 12-link robot (how many episodes recover the gap) would quantify transfer cost; possible
  follow-up.

**F-31. Runtime (T-23): on this GPU every model runs well inside the 20 ms budget; on a CPU only the flat GRU (window
mode) and the recurrent models in streaming mode do. At matched parameters the flat GRU is 7–19× cheaper than
somato_pool, so it has room to grow (E-23). The simulator runs at 0.34× real time for one robot.** *Why:* the user
(2026-10-11): the models will run online on a robot. How fast is each, at what rate, is the sim real-time, and should
models be matched by online cost instead of parameters? *Setup:*
* `scripts/analysis/runtime.py`: batch 1, sensor-model simulation excluded (a robot reads real sensors), median of 100
  repetitions;
* trained E-2 models (`runs/v1/runtime_e2.csv`) and width sweeps (`runs/v1/runtime_sweep.csv`);
* RTX 3060 Ti (fp32 and bf16) and CPU with 4 threads (16-core desktop);
* FLOPs per window from the PyTorch FLOP counter (fused attention disabled for counting) plus an analytic GRU count.
* Two online modes:
  * *window*: re-run the trained 0.5 s window at every 20 ms step from a fresh state. This is the training regime and
    the only option for the transformer.
  * *stream*: one new step with the carried state, for recurrent models. Its accuracy beyond the window is untested
    (E-18).

*Result (E-2 widths, ~430k parameters; ms per output):*

| model | GFLOP / window | GPU fp32 window | GPU bf16 window | CPU window | CPU stream mean / p95 |
|---|---|---|---|---|---|
| somato_pool (dim 64) | 1.8 | 4.1 | 5.5 | 34 | 2.7 / 7.7 |
| receptor_only (104) | 1.5 | 2.1 | 3.3 | 28 | 1.6 / 2.3 |
| flat_gru (156) | 0.02 | **0.6** | 0.9 | **1.8** | 0.4 / 0.4 |
| transformer (108) | 8.5 | 5.3 | 2.4 | **129** | — |

Width sweeps:
* The structured models' latency hardly depends on width (somato dim 64 → 96: 4.1 → 4.1 ms GPU; receptor 104 → 208:
  2.1 → 2.2 ms). Their cost is stage 1 running over 384 taxels × 250 samples.
* The flat GRU reaches somato_pool's latency at width ~2,048 on the GPU (window 4.0 ms, 36.6 M parameters), ~1,750 on
  the CPU window and ~1,300 in CPU streaming.
* The transformer's floor is set by attention over ~1,940 tokens: 4.5 ms GPU and 118 ms CPU even at dim 64.

Output rates:
* All models can emit one estimate per latent step (50 Hz) in window mode.
* In streaming mode the hierarchical models emit at 10 Hz (`brain_stride` 5); the flat GRU emits at 50 Hz.
* Sensors are simulated at 500 Hz (touch) and 250 Hz (joints, IMU).

Simulator: mjlab with the full stimulus pipeline (`collect_mjlab.py`) needs 11.7 s of wall time per 4 s of simulated
time with one environment, i.e. 0.34× real time. With 400 environments it reaches 40× real time in aggregate (0.1× per
environment).

*Evidence:* the numbers above.

*Caveats:*
* One machine. A robot's computer (e.g. Jetson-class GPU or ARM CPU) will be slower, and batch-1 GPU latency here is
  dominated by kernel-launch overhead, not FLOPs.
* No optimization (TorchScript, TensorRT, fp16 export).
* The somato streaming mode needs E-18 before it can be relied on.

*Implication:*
* Not a blocker on a GPU. On a CPU, window-mode somato_pool (34 ms) and especially the transformer (129 ms) miss
  50 Hz. Streaming fixes it for somato_pool (2.7 ms mean) once validated (E-18); the transformer would need fewer
  tokens. Noted, not pursued now (user).
* The sim is not real time for a single robot (MuJoCo-Warp is built for many parallel environments). Hardware-in-the-
  loop or online demos would need CPU MuJoCo or a faster single-env path. Noted.
* Cost matching: the transformer is already within 1.3× of somato_pool on the GPU (slower on the CPU) and receptor_only
  within 2×. The flat GRU has 7–19× headroom, so E-23 tests it at widths 512–2,048.

**F-30. E-22, baseline fairness on the slide test: with the full sensor pose and a suitable learning rate, the
baselines fit the training set perfectly but still do not generalize (flat GRU 0.57, transformer at chance); the
structured model's advantage comes from the architecture, not from having the poses.** *Why:* the user questioned
F-27's near-chance baselines (2026-10-11). Do the baselines receive the sensor poses, so that the benefit is from
building the poses into the architecture rather than from having them?
*Setup:*
* Code check of the inputs:
  * structured stage 2: each neighbour's offset in the sensor's own frame, whose x axis runs along the link;
  * flat GRU: every body's pose in the robot frame plus taxel identity by input position, so complete;
  * E-13b transformer: position with a coarse Fourier encoding (finest wavelength 12.5 cm vs 2–3 cm taxel spacing) and
    the normal, but not the along-link axis.
* New option `kin_axes: full` (`configs/models/transformer_fullpose.yaml`): position, x axis and normal, Fourier bands
  down to ~3 cm.
* `configs/experiments/v1_e22_slide_fairness.yaml`: 840 episodes, seed 0, widths matched, protocol v4.
* `v1_e22_terrain_fullpose.yaml`: the full-pose transformer on terrain with E-2's settings.

*Result, slide proxy (test acc, chance 0.25; final training loss):*

| model | test acc | training loss at the end |
|---|---|---|
| somato_pool | **0.976** | — |
| flat_gru, lr 1e-3 (E-13b: lr 3e-4, 0.30) | 0.568 | 0.006 |
| flat_gru, lr 3e-3 | 0.525 | — |
| transformer (E-13b inputs), lr 3e-4 | 0.260 | 0.000 |
| transformer, full pose, lr 1e-3 / 3e-4 | 0.281 / 0.256 | 0.003 |
| transformer, full pose + learned sensor IDs | 0.271 | 0.000 |

*Result, terrain (seed 0; E-2 transformer in brackets):* full-pose transformer 0.492 (0.517) at 10 %, 0.680 (0.741) at
100 %.

*Evidence:*
* Every baseline fits its 840 training episodes almost perfectly, so information and capacity are not the limit. The
  transformers then memorize: validation stays at chance while validation loss climbs to 6–7. The flat GRU memorizes
  too, but partly generalizes (validation 0.58).
* Giving the transformer exactly the pose information the structured kernel uses, even plus absolute sensor identity,
  does not help.
* The flat GRU's E-13b result was handicapped by its terrain-tuned learning rate (0.30 → 0.57; E-13 also found 0.54 at
  lr 1e-3).
* Within the structured family, building the poses in is worth +0.58: stage 1 with pooling and no poses scores
  0.35–0.40 (F-27), plus the local-frame kernel 0.96–0.98.
* On terrain the full-pose transformer is worse than E-2's (−6 points at full data, beyond E-2's seed std of 0.008),
  plausibly because fine position features invite overfitting. E-2's transformer was not handicapped.

*Caveats:*
* One seed per variant. The flat GRU's 0.57 is supported by two independent runs (E-13, E-22).
* Protocol v4.
* More data might eventually teach the transformer the rule; that is the data-efficiency question itself.

*Implication:*
* The user's criterion holds: the baselines have the poses and the capacity, and the generalizing solution comes from
  the architecture (a shared, local operator in each sensor's frame). The corrected slide gap is 0.976 vs 0.57 (flat)
  and chance (transformer).
* Keep `transformer.yaml` as the terrain baseline. Use the full-pose transformer whenever a task needs local
  orientation, and report both.
* Learning rates must be checked per task, not carried over from terrain: F-27's flat GRU number was too low.

**F-29. E-9, held-out friction range: every model degrades about equally (−20 points in classification, 2× friction
error); structure keeps its absolute lead but does not extrapolate better.** *Why:* H-2b (terrain outside the
training range) and H-11 (property estimates under extrapolation). *Setup:*
* `configs/experiments/v1_e9_holdout.yaml`: within each class the 30 % highest-friction episodes are held out of
  training, validation and test (`holdout` option) and evaluated separately. The held-out ranges are glare ice
  0.73–1.00, rough ice 0.58–0.80, packed snow 0.48–0.60, fresh snow 0.33–0.40 and concrete 0.88–1.00; overall training
  friction ends at 0.88.
* E-5 tasks (classification + friction and sinkage), models and learning rates; 1,176 training episodes; 3 seeds.

*Result (mean ± std over seeds):*

| model | acc in range | acc held out | friction MAE in range → held out | sinkage MAE in range → held out |
|---|---|---|---|---|
| somato_pool | **0.828 ± 0.003** | **0.625 ± 0.002** | 0.055 → 0.118 | **0.31 → 0.31 mm** |
| receptor_only | 0.815 ± 0.006 | 0.610 ± 0.006 | 0.060 → 0.127 | 0.40 → 0.41 |
| transformer | 0.758 ± 0.014 | 0.568 ± 0.017 | 0.058 → 0.115 | 0.37 → 0.36 |
| flat_gru | 0.744 ± 0.005 | 0.532 ± 0.002 | 0.055 → **0.114** | 0.32 → 0.32 |

*Evidence:*
* Classification falls by 19–21 points for every model on the unseen top of each class's friction range. The structured
  models stay ahead in absolute terms (+5.7 points over the transformer, +9.3 over flat; in range +7.0 / +8.4).
* Friction error doubles for every model (0.055–0.060 → 0.114–0.127). The baselines are marginally best, as on the E-7
  terrains (F-25). The held-out error is above E-5's in-distribution error at similar friction (0.07–0.10), so this is
  extrapolation error, not only range-edge shrinkage.
* Sinkage error is unchanged (the held-out episodes differ in friction, not compliance). somato_pool and flat_gru are
  best on it.
* In-range accuracy is higher than in E-5 (0.828 vs 0.793) despite 30 % fewer episodes: without the high-friction tail,
  glare ice overlaps less with concrete.

*Caveats:*
* One holdout axis (friction).
* The held-out episodes are the extreme of each class, so classification also suffers from class overlap (e.g.
  high-friction glare ice looks like concrete).
* No per-episode bias analysis (no checkpoints saved).

*Implication:*
* H-2b, friction extrapolation: the structured models generalize *as well as*, not better than, the baselines. Their
  in-distribution advantage carries over to unseen parameter ranges, but the degradation is no smaller.
* H-11: friction estimation extrapolates poorly for every architecture (2× error), consistent with F-25. Physical
  friction beyond the trained range is not something any of these models infers.
* Possible follow-ups:
  * a compliance holdout (does sinkage extrapolate?);
  * whether randomized-sensor training (F-26) or SSL (E-14) changes extrapolation.

**F-28. E-1b, training budget: protocol v4 under-trains at 25 % data (2–5 points), is nearly adequate at full data;
the model ranking is unchanged.** *Why:* H-12; in E-2 most best checkpoints were near the end of training (F-22).
*Setup:*
* `configs/experiments/v1_e1b_budget.yaml`: the E-2 models and learning rates with twice the v4 budget (6,000 steps /
  30 epochs, cosine schedule stretched accordingly);
* 420 / 1,680 training episodes (25 % / 100 %), seed 0;
* compared with E-2 (same split, 3 seeds).

*Result (test acc; E-2 mean ± std over 3 seeds → 2× budget, seed 0):*

| model | 25 % | 100 % |
|---|---|---|
| somato_pool | 0.676 ± 0.014 → **0.722** (+4.7) | 0.778 ± 0.004 → **0.788** (+1.0) |
| receptor_only | 0.665 ± 0.012 → 0.689 (+2.3) | 0.759 ± 0.009 → 0.766 (+0.7) |
| transformer | 0.608 ± 0.022 → 0.662 (+5.5) | 0.734 ± 0.008 → 0.753 (+1.9) |
| flat_gru | 0.592 ± 0.015 → 0.603 (+1.1) | 0.684 ± 0.008 → 0.678 (−0.7) |

*Evidence:*
* At 25 % the gains for somato_pool and the transformer exceed three and two E-2 seed stds. Their best checkpoints are
  still late (step 5,500 and 6,000 of ~6,000).
* At full data the gains are ≤ 2 points. Both somato_pool and the transformer gain about 2.5 E-2 stds.
* The ranking is identical at both sizes. The somato–transformer gap at 25 % stays about the same (6.8 → 6.0 points);
  the somato–flat gap grows (8.4 → 11.9), because the flat GRU gains least.

*Caveats:*
* One seed with the doubled budget.
* Only 25 % and 100 % were tested. At 2.5–10 % the v4 floor of 3,000 steps already means ≥ 45 epochs.
* The longer cosine schedule also keeps the learning rate higher for longer, so this is "budget plus schedule".

*Implication:*
* H-12 is contradicted at intermediate data. The absolute E-2 accuracies at 25 % are 2–5 points low, but F-22's
  comparisons hold: the structured and transformer models gain about equally, and the flat GRU gains least.
* Proposed protocol v5 for new studies: max(6,000 steps, 30 epochs), about 2× compute. Existing v4 results are not
  re-run unless a conclusion depends on absolute accuracy.

**F-27. E-13b, slide proxy with 3 seeds: the spatial stage decides it; nothing without it gets far beyond chance.**
*Why:* confirm F-17 (H-9; one seed) with the E-2 models and learning rates. *Setup:*
* `configs/experiments/v1_e13b_slide.yaml`: 4 slide directions (chance 0.25);
* 84 / 840 training episodes, 3 seeds;
* widths matched, fixed FSR v1;
* learning rates as in E-2, tuned on terrain, not on this task.

*Result (test acc, mean ± std over seeds):*

| model | 84 eps | 840 eps |
|---|---|---|
| somato_pool (stage 1 + 3D kernel + pooling) | **0.958 ± 0.009** | **0.974 ± 0.010** |
| receptor_only (stage 1 + pooling) | 0.345 ± 0.096 | 0.395 ± 0.101 |
| flat_gru | 0.285 ± 0.019 | 0.295 ± 0.025 |
| transformer | 0.285 ± 0.021 | 0.273 ± 0.012 |

*Evidence:*
* somato_pool solves the task from 84 episodes in every seed (≥ 0.945).
* The only difference between somato_pool and receptor_only is the spatial stage, and removing it drops accuracy to
  0.35–0.40, with large seed variation. Reading motion needs interaction between neighbouring taxels' temporal
  features.
* Both baselines see the same raw data and kinematics and stay within 5 points of chance, even with 10× the data.

*Caveats:*
* A synthetic stimulus-level proxy.
* The baselines' learning rates were tuned on terrain. In E-13 the flat GRU at lr 1e-3 (one seed) reached 0.54 at 840
  episodes, so its number here is lr-sensitive, but far below 0.97 either way.
* **Update (F-30):** with lr 1e-3 the flat GRU reaches 0.57. The E-13b transformer lacked the along-link sensor axis,
  but with the full pose (and even sensor IDs) it stays at chance. The corrected gap is 0.976 vs 0.57 / chance.

*Implication:*
* H-9 is supported with seeds. This is the clearest case so far of structure enabling something the unstructured
  models do not learn at these data sizes.
* It complements F-22, where on terrain most of the advantage came from stage 1: which stage matters depends on
  whether the information is in single-taxel dynamics (terrain) or in how patterns move across the skin (slip,
  sliding contacts). A locomotion controller needs both.

**F-26. E-21, sensor-model randomization removes the hysteresis shortcut at almost no cost, and widens the structured
lead under sensor shift.** *Why:* F-23 found that every model reading raw sensor dynamics had learned the simulated
FSR hysteresis (H-2a, H-3). *Setup:*
* `configs/experiments/v1_e21_sensor_random.yaml`: the E-2 models and learning rates, 3 seeds, 10 % / 100 % of the data;
* training sensor `configs/sensors/fsr_v1_random.yaml`, one FSR drawn per window: hysteresis 0–20 % and unloading
  10–150 ms (per-sample sub-ranges, then per taxel), gain spread 0.05–0.25, creep 0–10 %, noise 0.001–0.005;
* evaluation exactly as E-6 (`robustness.py`, fixed fsr_v1 plus perturbations, full-data checkpoints).
  "No hysteresis" lies inside the training ranges; more hysteresis, slow unloading, noise, gain spread and creep lie
  partly beyond them.

*Result (full data, mean over 3 seeds; change vs the nominal sensor in points, E-6 → E-21):*

| perturbation | somato_pool | receptor_only | transformer | flat_gru |
|---|---|---|---|---|
| nominal accuracy | 0.784 → 0.776 | 0.760 → 0.752 | 0.738 → 0.709 | 0.686 → 0.695 |
| no hysteresis (inside range) | −8.6 → **+0.3** | −12.3 → **−0.3** | −16.6 → **+0.3** | −2.6 → +1.8 |
| slow unloading (beyond) | −3.6 → −2.0 | −4.9 → −2.3 | −12.4 → −4.3 | −7.6 → −5.5 |
| more hysteresis (beyond) | −7.8 → −5.3 | −10.6 → −6.2 | −8.8 → −6.7 | −1.2 → −1.5 |
| 3× noise (beyond) | −1.0 → −0.6 | −0.2 → −0.3 | −0.2 → +0.1 | −4.2 → −0.5 |
| 30 % dead taxels (not randomized) | −4.3 → −3.3 | −1.8 → −1.2 | −9.1 → −3.6 | −6.1 → −6.6 |
| all combined | −14.7 → −8.2 | −13.3 → −8.5 | −14.9 → −9.6 | −14.8 → −8.5 |
| accuracy under the combined shift | 0.637 → **0.694** | 0.627 → 0.667 | 0.589 → 0.612 | 0.538 → 0.610 |

Test accuracy on randomized sensors (this study's `results.csv`), at 10 % / 100 %:
* somato_pool 0.605 / 0.776;
* receptor_only 0.590 / 0.749;
* transformer 0.539 / 0.708;
* flat_gru 0.525 / 0.703.

*Evidence:*
* The hysteresis shortcut is gone: the no-hysteresis drop is within ±2 points for every model, from up to −17.
* Robustness also improves beyond the trained ranges: slow unloading, gain spread and noise drops shrink. Larger
  hysteresis still costs the stage-1 and transformer models 5–7 points, as expected when extrapolating from 0–20 % to
  17–30 %.
* The nominal cost is small for the structured models (−0.8 points) and larger for the transformer (−2.9).
* The structured lead under the combined shift grows from +4.8 to +8.2 points over the best baseline. somato_pool keeps
  its lead on every single perturbation.
* Data efficiency under randomized training is preserved: at 10 % somato_pool leads the transformer by 6.6 points (5.8
  in E-2). Seed std ≤ 0.012 throughout.

*Caveats:*
* The perturbation magnitudes and the randomization ranges are both assumptions.
* Simulation only.
* Robustness was measured on the full-data checkpoints only.
* Property estimation under randomized sensors is untested.

*Implication:*
* Sensor-model randomization fixes the sim-to-real risk found in F-23 for the shifts it covers, and helps somewhat
  beyond them. Larger hysteresis than trained remains a risk, so the ranges should be set from bench measurements of
  the real skin.
* H-2a becomes **supported** with randomized training: the structured models are both the most accurate and, under the
  combined shift, the most robust.
* Proposed default for future studies: train on `fsr_v1_random` (after the current queue, whose experiments compare
  against fixed-sensor results).

**F-25. E-5 + T-21, property estimation: every model reads friction and sinkage beyond the terrain class; the
structured model is best in distribution (clearly for sinkage); on unseen terrain the estimates keep the right order
and err on the conservative side.** *Why:* H-11 (are physical properties a better generalization target than class
labels, and does structure help?), after F-24 found class posteriors give no novelty signal. *Setup:*
* `configs/experiments/v1_properties.yaml`: friction and measured sinkage (penetration of the lowest capsule point, after
  the label fix of 2026-10-10; regressed as log(sinkage + 0.1 mm)) plus the 5-class label;
* the E-2 models and learning rates, 170 / 1,680 training episodes, 3 seeds;
* the kept checkpoint has the lowest validation property error;
* then `ood_eval.py` (T-21) on the full-data checkpoints with the E-7 OOD set.

*Result, in distribution (test MAE, mean ± std over seeds):*

| model | friction 10 % | friction 100 % | sinkage 10 % | sinkage 100 % | terrain acc 10 % / 100 % |
|---|---|---|---|---|---|
| somato_pool | **0.101 ± 0.004** | **0.062 ± 0.001** | **0.39 ± 0.04 mm** | **0.27 ± 0.01 mm** | 0.610 / 0.793 |
| receptor_only | 0.104 ± 0.003 | 0.066 ± 0.000 | 0.50 ± 0.02 | 0.35 ± 0.02 | 0.598 / 0.770 |
| flat_gru | 0.108 ± 0.006 | 0.064 ± 0.001 | 0.46 ± 0.02 | 0.28 ± 0.00 | 0.538 / 0.704 |
| transformer | 0.115 ± 0.009 | 0.068 ± 0.001 | 0.46 ± 0.01 | 0.31 ± 0.01 | 0.566 / 0.736 |
| *global mean* | 0.209 | | 1.82 | | |
| *perfect class + class mean* | 0.135 | | 0.55 | | |

*Result, unseen terrains (full data; estimate vs truth, mean over models; MAE range over models):*

| OOD terrain | friction: true → estimated | friction MAE | sinkage: true → estimated | sinkage MAE |
|---|---|---|---|---|
| cold arctic ice | 0.99 → 0.78–0.80 | 0.22–0.25 | 0.045 → 0.06–0.07 mm | 0.03 |
| gravel | 0.76 → 0.63–0.68 | 0.11–0.14 (somato 0.14) | 0.33 → 0.15–0.19 mm | 0.18–0.20 (somato 0.18) |
| dry sand | 0.59 → 0.41–0.47 | 0.13–0.18 (somato 0.17) | 2.11 → 1.4–1.7 mm | 0.73–1.06 (somato **0.73**) |

*Evidence:*
* All models beat the "perfect classifier + class mean" reference on both properties (friction 0.06–0.07 vs 0.135 at
  full data). They read the within-class variation (e.g. ice friction across temperature), which classification
  cannot express.
* somato_pool has the lowest error on every in-distribution metric. The margin is large for sinkage at 10 % (0.39 vs
  ≥ 0.46 mm), small for friction (flat_gru is within 0.002 at full data).
* receptor_only is the worst model on sinkage (0.35 vs 0.27 mm at full data). The 3D kernel (stage 2) matters for
  sinkage, plausibly because sinkage shows in how far contact wraps around the body, a spatial pattern.
* Multitask training did not hurt classification (somato_pool 0.610 / 0.793 vs 0.606 / 0.778 in E-2).
* On the unseen terrains, every model ranks the three correctly on both properties (friction: cold ice > gravel >
  sand; sinkage: sand > gravel > cold ice).
* Every model underestimates friction on all three. Part of this is shrinkage at the top of the training range: in
  distribution, test episodes with friction ≥ 0.9 (true 0.95) are estimated at 0.85 (MAE 0.10, flat_gru and
  somato_pool alike). On cold ice the error is about twice that.
* The OOD error relative to in-distribution episodes with similar values: friction 2–3× (gravel, sand), sinkage about
  1× (gravel) to 1.1–1.5× (sand).
* Structure helps the sinkage estimate on unseen terrain (sand 0.73 vs 0.87–1.06 mm) but not friction (the transformer
  is best on sand and gravel, 0.125 / 0.111 vs somato 0.168 / 0.136).

*Caveats:*
* One OOD set of three terrains, simulated.
* Cold-ice friction (0.99) is at the edge of the training range (glare ice reaches 0.999), so that row mostly tests the
  range edge.
* The in-distribution binning used seed 0 of two models (scratch analysis on the CPU).
* Checkpoints were selected on property error, so terrain accuracy is not exactly comparable with E-2.

*Implication:*
* Property estimation works (signs of life) and transfers to unseen terrain far more usefully than class posteriors:
  correct order, moderate degradation, and a conservative bias in friction (safer for a controller). This supports the
  first half of H-11.
* Structure helps property estimation in distribution and for sinkage out of distribution; it does not help friction
  out of distribution. The data-efficiency advantage is much smaller than for classification: at 10 % somato_pool's
  standardized error is 8 % below the best baseline, vs 2–3× fewer episodes for classification (F-22).
* Next:
  * E-9 should test extrapolation along the friction range directly (holdout option ready), with property
    estimation as a task.
  * E-21's randomized-sensor models should also be checked on properties later.

**F-24. E-7, unseen terrains: physically sensible mappings, but no model knows what it doesn't know.** *Why:* H-2b
and H-13 (generalization to terrain never seen in training). *Setup:*
* `scripts/analysis/ood_eval.py` on the E-2 full-data checkpoints (3 seeds);
* OOD set `datasets/mjlab_v1_ood_earth_600`: cold arctic ice, dry sand, gravel (`configs/terrains/ood_earth_v1.yaml`);
* AUROC = P(confidence on in-distribution test windows > confidence on OOD windows); 0.5 = indistinguishable.

*Result (means over seeds):*

| OOD terrain | predicted as | AUROC: somato / receptor / transformer / flat |
|---|---|---|
| cold arctic ice | concrete 0.63–0.77, rough ice 0.09–0.31 | 0.69 / 0.67 / 0.66 / 0.58 |
| dry sand | packed and fresh snow, confidence 0.87–0.93 | 0.44 / 0.44 / 0.33 / 0.31 |
| gravel | structured: rough ice ~0.50; baselines: spread over rough, packed, concrete | 0.61 / 0.62 / 0.66 / 0.60 |

*Evidence:*
* The class mappings follow the physics: hard, grippy ice → concrete; compliant sand → snow (also hard to tell apart
  in the literature); bumpy gravel → rough ice, most consistently for the structured models.
* Confidence separates in- from out-of-distribution poorly for every model. All are more confident on sand than on
  their own test set.

*Caveats:* AUROC on window-level max-softmax only (no calibration methods or ensembles); one OOD set.
*Implication:*
* Class posteriors are not a usable novelty signal, for structured and unstructured models alike.
* For generalization to unseen terrain (the Europa motivation), physical property estimates are the meaningful output
  (H-11). Next: E-5 (property estimation), then evaluate the property estimates on this OOD set (extend `ood_eval.py`,
  T-21).

**F-23. E-6, sensor robustness: structured models keep their lead, but every model that reads raw sensor dynamics has
learned the simulated FSR hysteresis.** *Why:* H-2a and H-3 (sim-to-real proxy). *Setup:*
* `scripts/analysis/robustness.py` with `configs/sensors/robustness_fsr_v1.yaml`;
* E-2 checkpoints, 3 seeds × {10 %, 100 %};
* the same noise draws for all models.

*Result (full data, mean accuracy change over seeds, points):*

| perturbation | somato_pool | receptor_only | transformer | flat_gru |
|---|---|---|---|---|
| slow unloading (100–300 ms) | −3.6 | −4.9 | −12.4 | −7.6 |
| more hysteresis (17–30 %) | −7.8 | −10.6 | −8.8 | −1.1 |
| no hysteresis (cleaner sensor) | −8.6 | −12.3 | −16.6 | −2.6 |
| 3× noise | −1.0 | −0.2 | −0.1 | −4.2 |
| gain spread 0.35 | −1.9 | −2.2 | −2.9 | −0.3 |
| 10 % / 30 % dead taxels | −1.3 / −4.3 | −0.6 / −1.8 | −2.0 / −9.1 | −1.8 / −6.1 |
| all combined | −14.7 | −13.3 | −14.9 | −14.8 |
| accuracy under the combined shift | **0.637** | 0.627 | 0.589 | 0.537 |

At 10 % data all drops are smaller (combined −3 to −6).

*Evidence:*
* The combined shift costs every model ~14 points, so the ranking is preserved; the structured models stay best in
  absolute terms.
* Per factor, the structured models are much more robust than the transformer to slower unloading and to dead taxels.
* Yet changing the FSR hysteresis in *either* direction costs the models that read raw per-sample dynamics (stage 1,
  transformer) 8–17 points. The flat GRU's coarse per-step statistics lose only 1–3 points there, but more under noise.

*Caveats:* the perturbation magnitudes are assumptions (pessimistic end of the literature); simulation only.
*Implication:*
* Data efficiency does not automatically buy sensor robustness: stage 1 overfits the simulated sensor model's
  dynamics, a concrete sim-to-real risk (H-3).
* Natural next test, **E-21**: randomize sensor-model parameters (hysteresis, unloading time constants, gain) during
  training and re-run E-6. A receptor stage trained across sensor variability is also closer to the biological idea.

**F-22. E-2, main learning curves (3 seeds): the structured models are 2–3× more data-efficient than both tuned
baselines; most of it comes from stage 1.** *Why:* the main test of H-1, with H-4 and H-14.
*Setup:*
* `configs/experiments/v1_main_curves.yaml`: simulation v1, fixed FSR v1, protocol v4;
* widths matched (~428k), learning rate per model from E-1c (F-21);
* seeds 0–2, 360-episode test set.

*Result (test acc, mean ± std over seeds):*

| model | 42 eps | 168 | 420 | 1,680 | NLL (1,680) |
|---|---|---|---|---|---|
| somato_pool (receptors + 3D kernel + pool) | **0.490 ± 0.018** | **0.606 ± 0.016** | **0.676 ± 0.014** | **0.778 ± 0.004** | 0.50 |
| receptor_only (receptors + pool) | 0.481 ± 0.006 | 0.597 ± 0.008 | 0.665 ± 0.012 | 0.759 ± 0.009 | 0.55 |
| transformer | 0.405 ± 0.023 | 0.548 ± 0.022 | 0.608 ± 0.022 | 0.734 ± 0.008 | 0.63 |
| flat_gru | 0.407 ± 0.017 | 0.531 ± 0.013 | 0.592 ± 0.015 | 0.684 ± 0.008 | 0.82 |

*Evidence:*
* somato_pool beats the transformer by +8.5 / +5.8 / +6.8 / +4.4 points and flat by +8.3 / +7.5 / +8.4 / +9.4. Every
  gap is ≥ 2× the larger seed std.
* Log-linear interpolation of the curves: the transformer needs ~2.1–2.5× as many episodes to match somato_pool, flat
  ~3×.
* The transformer gap halves over 40× data (8.5 → 4.4); the flat gap does not shrink. No crossover in range (H-4
  partially: converging against the transformer).
* receptor_only is consistently 1–2 points below somato_pool (~1 seed std) and 7–8 points above both baselines at
  ≤ 420 episodes. Shared receptor encoders + pooling carry most of the benefit (H-14 supported); the 3D kernel adds a
  small, consistent increment.
* Per class (mean recall), full data:

  | class | somato | transformer | flat |
  |---|---|---|---|
  | rough ice | **0.80** | 0.76 | 0.57 |
  | packed snow | **0.76** | 0.65 | 0.67 |
  | concrete | 0.83 | 0.83 | 0.76 |
  | glare ice | 0.51 | 0.45 | 0.45 |
  | fresh snow | 0.99 | 0.99 | 0.97 |

  At 42 episodes the structured lead is fresh snow (0.95 vs 0.84 / 0.77) and concrete (0.59 vs 0.37 / 0.40).
  Glare ice stays ~0.5 everywhere: the touch-ambiguous pair with concrete (Q-3).
* Calibration: the structured models have the lowest NLL; flat is overconfident at small data (NLL 5.0 at 42
  episodes).

*Caveats:*
* At full data most runs' best checkpoint is late in the budget (8,000–10,245 of 10,245 steps), so all models may gain
  a little from more steps (E-1b; affects all similarly).
* Simulation-only, one sensor technology (FSR), one task.

*Implication:* H-1 supported in sim v1 (data efficiency, 2–3×). The useful prior on terrain is mainly "every taxel is
the same kind of receptor" (stage 1); spatial structure matters more for moving stimuli (F-17). Next: robustness and
OOD generalization (E-6, E-7), then property estimation (E-5).

**F-21. E-1c, learning-rate sweep: learning rate barely matters; with tuned baselines the structured model leads by
8–10 points at 10 % and 4.5–11 points at full data.** *Why:* baseline fairness before E-2. *Setup:*
`configs/experiments/v1_e1c.yaml` (fixed FSR v1, pool head, widths matched ~428k), lr 3e-4 / 1e-3 / 3e-3, seed 0.
The lr is chosen per model by validation accuracy.
*Result:* test acc at the chosen lr (validation-best):

| model | lr | 170 eps | 1,680 eps |
|---|---|---|---|
| somato_pool (stage 1 + 3D kernel + pool) | 1e-3 | **0.616** | **0.784** |
| receptor_only (stage 1 + pool) | 1e-3 | 0.605 | 0.750 |
| transformer | 1e-3 | 0.531 | 0.739 |
| flat_gru | 3e-4 | 0.513 | 0.678 |

*Evidence:*
* Across the three learning rates, each model's test accuracy varies by ≤ 3 points (≤ 5 for flat at full data), so
  the comparisons are not an artifact of one learning rate.
* The tuned transformer is clearly stronger than in E-1 (0.739 vs 0.635 at full data; fixed FSR, width 108 vs 132).

*Caveats:* one seed; the lr is chosen on validation and reported on test.
*Implication:*
* E-2 uses lr 1e-3 for the somato models and the transformer, 3e-4 for flat.
* The somato–transformer gap shrinks from 8.5 to 4.5 points with 10× data: data efficiency (H-1) with a possible
  convergence trend (H-4), to be checked with seeds.
* The 3D kernel adds +1 / +3.4 points over receptor-only (H-14: stage 1 carries most of the benefit on terrain).

**F-20. E-19b: a residual memory does not rescue the brain head; the main studies use the pool head.**
*Setup:* `configs/experiments/e19b_brain_residual.yaml`, as E-19.
*Result:*

| stage 3 | proxy (84 eps) | terrain (170 eps) |
|---|---|---|
| brain + residual memory | 0.562 (stall to step ~250, then memorizes) | 0.621 |
| brain + residual memory + node-pool path | 0.247 (ln 4 for the whole run) | 0.625 |
| pool, from F-19 for reference | 0.985 | 0.612 |

*Evidence:* adding the node-pool path (the pool head's exact input) to the brain makes the stall *worse*. So the
problem is not only the GRU; something in how the brain's fused representation trains on sparse moving stimuli. Not
resolved.
*Implication:*
* E-1c and E-2 use the **static pool head**: robust on both tasks and tied on terrain.
* The brain (dynamic, geometry-aware, recurrent stage 3) becomes a research task (T-19). The user expects stage 3 to
  grow into a locomotion controller anyway.
* The structure spectrum for the main study: receptor-only (stage 1 + pool), somato_pool (+ 3D kernel), flat GRU,
  transformer.

**F-19. E-19: the brain head's stall comes from its GRU.** *Why:* F-17 (brain stalls, then memorizes on the slide
proxy). *Setup:* `configs/experiments/e19_brain_diagnostics.yaml`; stages 1–2 identical (dim 64, no width matching),
only stage 3 varies; slide proxy at 84 episodes and terrain at 170; seed 0.
*Result (test acc):*

| stage 3 | proxy | terrain |
|---|---|---|
| pool (static mean+max) | **0.985** | 0.612 |
| brain (attention + GRU) | 0.241 | 0.601 |
| brain without GRU | **0.906** | 0.588 |
| brain without attention | 0.250 | 0.555 |
| brain + direct node-pool path | 0.250 | 0.573 |
| brain, smaller (GRU 64, 1 layer) | 0.692 | 0.643 |

*Evidence:*
* Every variant with the full-size GRU in the output path holds the training loss at exactly ln 4 for 1,000 to more
  than 3,000 steps. That includes the variant with a direct path from the pooled sensor features: the GRU sits between
  it and the output.
* Removing the GRU removes the stall (val 0.92 by step 1,000).
* On terrain all variants are within noise (SE ±2.6).

*Caveats:* one seed. The mechanism (GRU saturation by sample-independent features at initialization, F0-8) is
plausible but not shown.
*Implication:*
* Keep memory in stage 3 (the stated intent), but as a residual correction: z + GRU(LayerNorm(z)) (`residual_memory`).
  The head then works without the memory from the start.
* Tested in E-19b before E-1c.

**F-18. E-3/E-4, structure ablations on terrain: stage-2 choices barely matter here; the family's lead over the
baselines comes from stage 1 and pooling.**
*Why:* what does each imposed structure contribute (H-5, H-6, H-7, H-8)?
*Setup:* `configs/experiments/v1_structure_ablations.yaml`, fixed FSR model, widths matched (~636k), seed 0,
360-episode test (SE ≈ ±2.6 points at 0.6, ±2.1 at 0.8). Validation re-run after the review fixes: 15/15.
*Result (test acc):*

| model | 170 eps | 1,680 eps | NLL (full) |
|---|---|---|---|
| somato_v1 (kernel3d, segregated, brain) | 0.615 | 0.798 | 0.48 |
| somato_v1_mixed (modalities mixed in stage 2) | 0.604 | **0.800** | 0.47 |
| somato_v1_pool (static mean+max head) | 0.617 | 0.776 | 0.50 |
| somato_v1_no_spatial (no stage 2) | **0.618** | 0.775 | 0.51 |
| somato_v1_graph (KNN attention instead of the kernel) | 0.600 | 0.748 | 0.58 |

*Evidence:*
* At 170 episodes all five are within ±1 SE.
* At full data the kernel models lead no_spatial and pool by ~2 points (< 1 SE). The graph variant is ~5 points lower
  (~2.4 SE; one seed).
* Every family member, including no_spatial, beats E-1's baselines by a similar margin (~0.61 vs 0.49–0.51 at 10 %;
  0.75–0.80 vs 0.64–0.67 at full).
* The brain trains normally on terrain (no stall), unlike on the slide proxy (F-17).

*Caveats:*
* One seed.
* E-1's baselines ran with the FSR bug (the family's E-3 numbers are within 1–2 points of its E-1 numbers, so the
  comparison is roughly fair).
* Baselines not yet tuned (E-1c).

*Implication:*
* On terrain classification in sim v1, the decisive bias is **stage 1**: one shared receptor encoder per modality,
  followed by permutation-invariant pooling, i.e. "every taxel is the same kind of receptor". Spatial interaction
  (H-5), segregation (H-7) and the dynamic brain (H-8) add ≤ 2 points here; the 3D kernel ≥ the graph alternative
  (H-6, weak).
* With F-17, the value of stage 2 is task-dependent. It is decisive for stimuli moving across the skin and marginal
  for terrain identity, whose cues (pressure level, sinkage, roughness statistics) survive per-taxel encoding plus
  pooling.
* For E-2, include the receptor-only end of the spectrum (no_spatial) next to the full model and the baselines.

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
