# somato: working in this repo

Research code for a somatosensory-inspired tactile encoder (snake robot, ice and snow terrain). Start every session
with the **RESUME HERE** section of `docs/investigation_log.md`.

## Where things go
* `docs/investigation_log.md`: the plan. Running experiments, hypotheses (H-*), the ordered experiment queue (E-*),
  tasks (T-*), decisions. Every request from the user becomes an H/E/T item; keep statuses current.
* `docs/findings.md`: one entry per experiment (F-*): why, setup, result with numbers, evidence, caveats, implication.
* `docs/architecture.md` (design), `docs/mjlab.md` / `docs/isaac_sim.md` (simulators), `docs/references/`
  (literature), `docs/archive/` (version 0, with caveats; not evidence).
* Configs: `configs/experiments/` (studies and collections), `configs/models/`, `configs/sensors/`,
  `configs/terrains/`.
* Reusable research scripts: `scripts/` (`scripts/analysis/` for evaluation). Operational helpers: `tools/`. Run
  outputs, datasets and logs live under `runs/` and `datasets/` (gitignored).

## Running things
* Environment: `source tools/env.sh` (mjlab conda env, ROS paths stripped, repo root). Never import mjlab in an Isaac
  process (Isaac uses the `isaaclab23` env).
* GPU work goes through the job queue: `python3 tools/jobq.py add NAME [--mem 24G] -- CMD`, then `start` (idempotent),
  `status`, and `wait NAME` (run that in the background to get notified). One GPU job at a time (8 GB card).
  Never wait on processes with `pgrep -f` / `pkill -f` patterns: they match the waiter's own command line.
* Memory: every data/GPU process under a cap (`jobq` does this; by hand:
  `systemd-run --user --scope -q -p MemoryMax=12G ...`). Datasets load through the memory-mapped cache, so several
  processes can share one; an uncached 2,400-episode store costs 13.7 GB of RAM. Caps should be generous (24G for
  training), not throttling.
* Studies: `python -u scripts/data_efficiency.py --config configs/experiments/<study>.yaml [--set key=value ...]`.
  Per-model training overrides go in the study's `models` entry as `{model: <path>, train: {lr: 0.001}}`.
* Analysis:
  * `scripts/analysis/summarize_study.py <run dir>`: accuracy, NLL, best step, val curve, per-class recall;
  * `robustness.py`: sensor perturbations;
  * `ood_eval.py`: unseen terrains;
  * `task_ceiling.py`: Bayes ceilings;
  * `profile_step.py`: speed.
* Simulator changes: `python scripts/mjlab/validate_mjlab.py --num_envs 10` must pass. Never loosen a threshold
  without the user's approval.

## Gotchas
* YAML reads `1e-3` as a string: write learning rates as decimals (`0.001`).
* Windows must be a multiple of the model's `brain_stride` (5 in the v1 configs).
* Main-study models and their learning rates: `configs/experiments/v1_main_curves.yaml`.

## Code and git
* Edit repo files with the Edit/Write tools (the user reviews diffs), not sed/heredocs.
* Tests: `CUDA_VISIBLE_DEVICES="" python -m pytest -q` (~30 s, CPU); keep them passing.
* Branch `claude/hierarchical-tactile-architecture-vwlzjv`: commit after each change and push to origin. Commits must
  carry the user's identity, trevor386 <trevorjohst@proton.me>; check `git config user.email` before pushing.
