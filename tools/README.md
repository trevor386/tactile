# tools/

Operational helpers for running experiments on this machine (not research code; that lives in `scripts/` and
`src/`).

| tool | use |
|---|---|
| `source tools/env.sh` | Before any manual GPU, simulator or dataset command: activates the `mjlab` conda env, strips the ROS paths, changes to the repo root. |
| `tools/jobq.py` | Sequential GPU job queue: one job at a time, each under a systemd memory cap, state in `runs/jobs/*.json`, log in `runs/logs/<name>.log`. |

Typical use:

```bash
python3 tools/jobq.py start                                   # once (and after a reboot); safe to repeat
python3 tools/jobq.py add e2_main -- python -u scripts/data_efficiency.py --config configs/experiments/v1_main_curves.yaml
python3 tools/jobq.py add e6_robust --mem 16G -- python -u scripts/analysis/robustness.py --runs "..." --dataset ...
python3 tools/jobq.py status                                  # jobs, worker, GPU/RAM, tail of the running log
python3 tools/jobq.py wait e2_main e6_robust                  # blocks until both finish (run it in the background)
python3 tools/jobq.py cancel e6_robust
```

Rules it encodes:
* one GPU job at a time (8 GB card);
* a memory cap per job (default 24G: enough for a training run on the 2,400-episode cached dataset without
  throttling, small enough that a runaway process cannot freeze the 31 GB machine);
* waiting reads job files only, never process names. `pgrep -f` loops in waiters matched their own command lines and
  deadlocked or killed themselves twice.

Jobs run `bash -c "source tools/env.sh && <command>"` from the repo root, so commands use repo-relative paths. The
worker keeps running after the Claude session ends. After a reboot, `start` resumes the queued jobs; a job that was
running at the time shows as `running?` and has to be added again.
