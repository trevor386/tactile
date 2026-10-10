#!/usr/bin/env python3
"""Sequential GPU job queue: one job at a time, each under a memory cap, state in plain files.

    tools/jobq.py add NAME [--mem 24G] -- CMD ...   # queue a shell command (run from the repo root after tools/env.sh)
    tools/jobq.py start                             # start the worker if it is not running (safe to repeat)
    tools/jobq.py status [-n 15]                    # jobs, worker, GPU/RAM, tail of the running job's log
    tools/jobq.py wait NAME|ID ... | --all          # block until those jobs (or the whole queue) finish
    tools/jobq.py cancel NAME|ID                    # cancel a queued job, or kill a running one

Each job is ``runs/jobs/<id>_<name>.json``: command, memory cap, state (queued, running, done, failed, cancelled),
exit code, times and log file (``runs/logs/<name>.log``). ``wait`` only reads these files, so it never matches its own
command line, unlike ``pgrep -f`` loops, which deadlocked and killed waiters before. Use ``wait`` with
``run_in_background`` to be notified when work finishes. The worker survives the Claude session; after a reboot run
``start`` again (queued jobs resume; a job that was running is shown as ``running?`` and must be re-added).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "runs" / "jobs"
LOGS = ROOT / "runs" / "logs"
PIDFILE = JOBS / "worker.pid"
FINAL = ("done", "failed", "cancelled")


def _jobs() -> list[tuple[Path, dict]]:
    JOBS.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(JOBS.glob("*.json"), key=lambda p: int(p.name.split("_", 1)[0])):
        try:
            out.append((p, json.loads(p.read_text())))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _save(path: Path, job: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=1))
    os.replace(tmp, path)


def _alive(pid: int | None, needle: str | None = None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    if needle:
        try:
            return needle in Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            return False
    return True


def _worker_pid() -> int | None:
    try:
        pid = int(PIDFILE.read_text())
    except (OSError, ValueError):
        return None
    return pid if _alive(pid, "jobq.py worker") else None


def _find(key: str) -> list[tuple[Path, dict]]:
    return [(p, j) for p, j in _jobs() if str(j["id"]) == key or j["name"] == key]


def cmd_add(args) -> None:
    jobs = _jobs()
    if any(j["name"] == args.name and j["state"] not in FINAL for _, j in jobs):
        sys.exit(f"a job named {args.name!r} is already queued or running")
    jid = max((j["id"] for _, j in jobs), default=0) + 1
    cmd = " ".join(args.cmd)
    if not cmd:
        sys.exit("no command given (tools/jobq.py add NAME [--mem 24G] -- CMD ...)")
    job = {"id": jid, "name": args.name, "cmd": cmd, "mem": args.mem, "state": "queued", "exit": None,
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "started": None, "ended": None, "pid": None,
           "log": str((LOGS / f"{args.name}.log").relative_to(ROOT))}
    _save(JOBS / f"{jid:04d}_{args.name}.json", job)
    print(f"queued #{jid} {args.name}: {cmd}")
    if _worker_pid() is None:
        print("note: the worker is not running (tools/jobq.py start)")


def cmd_worker(_args) -> None:
    PIDFILE.write_text(str(os.getpid()))
    LOGS.mkdir(parents=True, exist_ok=True)
    while True:
        queued = [(p, j) for p, j in _jobs() if j["state"] == "queued"]
        if not queued:
            time.sleep(15)
            continue
        path, job = queued[0]
        log = ROOT / job["log"]
        shell = f"source tools/env.sh && {job['cmd']}"
        argv = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={job['mem']}", "bash", "-c", shell]
        with open(log, "w") as fh:
            proc = subprocess.Popen(argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            job.update(state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"), pid=proc.pid)
            _save(path, job)
            code = proc.wait()
        job = json.loads(path.read_text())  # cancel may have changed it
        if job["state"] == "running":
            job["state"] = "done" if code == 0 else "failed"
        job.update(exit=code, ended=time.strftime("%Y-%m-%d %H:%M:%S"))
        _save(path, job)


def cmd_start(_args) -> None:
    pid = _worker_pid()
    if pid:
        print(f"worker already running (pid {pid})")
        return
    JOBS.mkdir(parents=True, exist_ok=True)
    out = open(JOBS / "worker.out", "a")
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker"], cwd=ROOT, stdout=out,
                            stderr=out, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"worker started (pid {proc.pid})")


def _elapsed(job: dict) -> str:
    if not job["started"]:
        return ""
    end = time.mktime(time.strptime(job["ended"], "%Y-%m-%d %H:%M:%S")) if job["ended"] else time.time()
    m = int((end - time.mktime(time.strptime(job["started"], "%Y-%m-%d %H:%M:%S"))) // 60)
    return f"{m // 60}h{m % 60:02d}m"


def cmd_status(args) -> None:
    pid = _worker_pid()
    print(f"worker: {'running (pid %d)' % pid if pid else 'NOT running (tools/jobq.py start)'}")
    jobs = _jobs()[-args.n:]
    for _, j in jobs:
        state = j["state"]
        if state == "running" and not _alive(j["pid"]):
            state = "running?"  # the process is gone (reboot or worker killed)
        print(f"  #{j['id']:<4d} {j['name']:28s} {state:10s} exit={j['exit']!s:5s} {j['started'] or j['created']}  "
              f"{_elapsed(j):>7s}  {j['log']}")
    for _, j in jobs:
        if j["state"] == "running":
            try:
                lines = (ROOT / j["log"]).read_text(errors="replace").splitlines()
                print(f"--- {j['name']} (last lines)")
                print("\n".join(line[:160] for line in lines[-3:]))
            except OSError:
                pass
    for q in (["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
              ["free", "-g"]):
        try:
            print(subprocess.run(q, capture_output=True, text=True, timeout=10).stdout.strip())
        except (OSError, subprocess.TimeoutExpired):
            pass


def cmd_wait(args) -> None:
    def pending():
        if args.all:
            return [j for _, j in _jobs() if j["state"] not in FINAL]
        found = [j for k in args.keys for _, j in _find(k)]
        if not found:
            sys.exit(f"no job matches {args.keys}")
        return [j for j in found if j["state"] not in FINAL]

    while pending():
        time.sleep(args.poll)
    keys = args.keys if not args.all else []
    for k in keys:
        for _, j in _find(k):
            print(f"#{j['id']} {j['name']}: {j['state']} (exit {j['exit']}) {j['started']} -> {j['ended']}  log {j['log']}")
    if args.all:
        print("queue empty")


def cmd_cancel(args) -> None:
    for path, j in _find(args.key):
        if j["state"] == "queued":
            j["state"] = "cancelled"
        elif j["state"] == "running" and _alive(j["pid"]):
            j["state"] = "cancelled"
            os.killpg(j["pid"], signal.SIGTERM)
        else:
            continue
        _save(path, j)
        print(f"cancelled #{j['id']} {j['name']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("add")
    a.add_argument("name")
    a.add_argument("--mem", default="24G")
    sub.add_parser("worker")
    sub.add_parser("start")
    s = sub.add_parser("status")
    s.add_argument("-n", type=int, default=15)
    w = sub.add_parser("wait")
    w.add_argument("keys", nargs="*")
    w.add_argument("--all", action="store_true")
    w.add_argument("--poll", type=float, default=30.0)
    c = sub.add_parser("cancel")
    c.add_argument("key")
    argv = sys.argv[1:]
    cmd = []
    if "--" in argv:  # everything after "--" is the job's command
        cmd, argv = argv[argv.index("--") + 1:], argv[:argv.index("--")]
    args = p.parse_args(argv)
    args.cmd = cmd
    {"add": cmd_add, "worker": cmd_worker, "start": cmd_start, "status": cmd_status, "wait": cmd_wait,
     "cancel": cmd_cancel}[args.command](args)


if __name__ == "__main__":
    main()
