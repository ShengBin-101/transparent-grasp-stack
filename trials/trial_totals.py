#!/usr/bin/env python3
"""Held rates per view and pooled over the views, from trials/trial_runs.csv: the report's results tables.

    python3 trials/trial_totals.py [--out FILE]

Counted attempts only (status starting with "counted", re-runs included). Per view and pooled: held / attempts and the percentage for each condition
and object kind (percentages with their counts), the failure codes, and the held count per object.
Prints the tables, and also writes them to FILE when --out is given.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDS = ("RAW", "BASELINE", "ADAPTED")      # as trial_runs.csv records them
CODES = {"N": "no executable grasp", "P": "planner refused", "C": "collision", "S": "slip", "W": "wrong depth",
         "E": "execution fault", "X": "other / operator stop"}


def kind(r):
    return "transparent" if r["object"].startswith("T") else "opaque"


def table(rows, title):
    L = [f"## {title} ({len(rows)} counted attempts)", "",
         "| condition | transparent held | opaque held | all held | failures (code: count) |", "|---|---|---|---|---|"]
    held = {}
    for c in CONDS:
        cells = []
        for kd in ("transparent", "opaque", "all"):
            sel = [r for r in rows if r["condition"] == c and (kd == "all" or kind(r) == kd)]
            k, n = sum(r["held"] == "Y" for r in sel), len(sel)
            held[(c, kd)] = (k, n)
            cells.append(f"{k}/{n} ({100 * k / n:.0f} %)" if n else "-")
        fails = Counter(r["fail"] for r in rows if r["condition"] == c and r["held"] == "N")
        cells.append(", ".join(f"{f}: {v}" for f, v in sorted(fails.items(), key=lambda t: -t[1])) or "-")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    by_obj = Counter((r["object"], r["condition"]) for r in rows if r["held"] == "Y")
    objs = sorted({r["object"] for r in rows})
    L += ["", "| object | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    for o in objs:
        n = {c: sum(1 for r in rows if r["object"] == o and r["condition"] == c) for c in CONDS}
        L.append(f"| {o} | " + " | ".join(f"{by_obj[(o, c)]}/{n[c]}" for c in CONDS) + " |")
    return L + [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, help="also write the tables to this file")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(ROOT / "trials" / "trial_runs.csv")) if r["status"].startswith("counted")]
    views = sorted({r["view"] for r in rows})
    L = ["# Robot trials: held rates per view and pooled", "",
         "From `trials/trial_runs.csv` (counted attempts), by `trials/trial_totals.py`. "
         "Failure codes: " + ", ".join(f"{k} {v}" for k, v in CODES.items()) + ".", ""]
    for v in views:
        L += table([r for r in rows if r["view"] == v], f"View {v}")
    if len(views) > 1:
        L += table(rows, "All views pooled (" + " + ".join(views) + ")")
    if a.out:
        a.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
