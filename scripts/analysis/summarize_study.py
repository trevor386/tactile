"""Summarize a data-efficiency study directory: accuracy, calibration, budget use and per-class recall per run.

    python scripts/analysis/summarize_study.py runs/v1/signs_of_life [--markdown]

Reads every ``<model>/<run>/result.json`` (test metrics, best_step, confusion) and ``log.jsonl`` (validation curve).
Columns:
* acc: test acc_last;
* nll: test NLL at the last step;
* best/last: best-validation step and total steps (a best step at the end suggests under-training);
* val@25/50/100 %: validation accuracy at those fractions of the budget (how fast it learns);
* recall per class from the confusion matrix (rows: true class).
"""

import argparse
import json
from pathlib import Path


def per_class_recall(confusion: list[list[int]]) -> list[float]:
    return [row[i] / max(1, sum(row)) for i, row in enumerate(confusion)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("study")
    p.add_argument("--classes", nargs="*", default=["glare", "rough", "packed", "fresh", "concrete"])
    args = p.parse_args()
    rows = []
    for res in sorted(Path(args.study).glob("*/*/result.json")):
        r = json.loads(res.read_text())
        log = [json.loads(line) for line in (res.parent / "log.jsonl").read_text().splitlines()]
        val = [(x["step"], x.get("val/terrain/acc_last")) for x in log if "val/terrain/acc_last" in x]
        last = val[-1][0] if val else 0

        def at(frac):
            target = frac * last
            return next((a for s, a in val if s >= target), float("nan"))

        rec = per_class_recall(r["confusion"]) if "confusion" in r else []
        rows.append((res.parent.parent.name, r.get("train_episodes"), r["test/terrain/acc_last"],
                     r.get("test/terrain/nll_last", float("nan")), r.get("best_step"), last, at(0.25), at(0.5),
                     at(1.0), rec, {k: v for k, v in r.items() if k.startswith("test/") and "mae" in k}))
    rows.sort(key=lambda x: (x[1], x[0]))
    head = (f"{'model':22s} {'eps':>5s} {'acc':>6s} {'nll':>6s} {'best/last':>11s} {'val@25/50/100%':>17s}  "
            + " ".join(f"{c[:7]:>7s}" for c in args.classes))
    print(head)
    for m, eps, acc, nll, best, last, v25, v50, v100, rec, extra in rows:
        print(f"{m:22s} {eps:5d} {acc:6.3f} {nll:6.3f} {best:5d}/{last:<5d} {v25:5.3f}/{v50:5.3f}/{v100:5.3f}  "
              + " ".join(f"{x:7.3f}" for x in rec)
              + ("  " + " ".join(f"{k[5:]}={v:.3g}" for k, v in extra.items()) if extra else ""))


if __name__ == "__main__":
    main()
