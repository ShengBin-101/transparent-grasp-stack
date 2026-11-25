#!/usr/bin/env python3
"""Rows of trials/trial_runs.csv from the schedule, the tracker and the run folders.

    python3 trials/build_trial_runs.py --tracker data/tracker_export_<date> --views V2           # print, write nothing
    python3 trials/build_trial_runs.py --tracker data/tracker_export_<date> --views V2 --write   # replace the V2 rows
    python3 trials/build_trial_runs.py --tracker data/tracker_export_<date> --check V1           # rebuild V1, compare

The tracker export is one JSON file per attempt, <tracker>/attempts/a<n>.json, with date, run_time, held, fail and
notes. It is not published: the tracker was a hosted web page used during the sessions, and trial_runs.csv is the
record (trials/README.md). For each scheduled attempt of a view, the run folder is the one named
<date>_<run_time>_* under runs/trial/ or runs/, and its pick records are its pick_*.json.

One row per attempt, status "counted" when the tracker has held Y or N, "excluded" when its notes start with EXCLUDED,
otherwise "not run". Rows the tracker does not describe (warm-ups, superseded first captures and their technical
re-runs; V1 has seven) are written by hand and kept: --write replaces only the rows of the given views that carry an attempt number and a counted,
excluded or not-run status, and appends the rest. Nothing under runs/ is touched.
object_name comes from the schedule, and trial_runs.csv shortens two of them ("thick tape roll" to "tape roll",
"plastic bottle, lying (water)" to "plastic bottle"). --check does not compare it.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "trials" / "trial_runs.csv"
FIELDS = ["attempt", "block", "view", "condition", "object", "object_name", "object_attempt", "status", "held", "fail",
          "first_capture_outcome", "run", "folder", "pick_records"]


def find_run(date, run_time):
    """(run name, folder relative to ROOT) for <date>_<run_time>_*, runs/trial/ first; warnings."""
    hits, warn = [], []
    for base in (ROOT / "runs" / "trial", ROOT / "runs"):
        for d in sorted(base.glob(f"{date}_{run_time}_*")):
            if d.is_dir() and d.name not in [h.name for h in hits]:
                hits.append(d)
    if not hits:
        return "", "", [f"no run folder {date}_{run_time}_* under runs/trial or runs"]
    if len(hits) > 1:
        warn.append("several folders: " + ", ".join(h.name for h in hits) + "; took the first")
    d = hits[0]
    return d.name, str(d.relative_to(ROOT)), warn


def build(tracker, views):
    rows, warnings = [], []
    for s in csv.DictReader(open(ROOT / "trials" / "trial_schedule.csv")):
        if s["view"] not in views:
            continue
        a = f"a{s['n']}"
        f = Path(tracker) / "attempts" / f"{a}.json"
        doc = json.load(open(f)) if f.exists() else {}
        doc = doc.get("data", doc)
        held, notes = (doc.get("held") or "").strip(), (doc.get("notes") or "").strip()
        row = {"attempt": a, "block": s["block"], "view": s["view"], "condition": s["condition"],
               "object": s["object"], "object_name": s["object_name"], "object_attempt": s["attempt"],
               "held": held, "fail": (doc.get("fail") or "").strip(), "first_capture_outcome": "",
               "run": "", "folder": "", "pick_records": ""}
        if notes.upper().startswith("EXCLUDED"):
            row["status"] = "excluded: " + notes[:120]
        elif held in ("Y", "N"):
            row["status"] = "counted"
        else:
            row["status"] = "not run"
        if doc.get("run_time") and doc.get("date"):
            run, folder, w = find_run(doc["date"], doc["run_time"])
            row["run"], row["folder"] = run, folder
            warnings += [f"{a}: {x}" for x in w]
            if folder:
                row["pick_records"] = " ".join(p.name for p in sorted((ROOT / folder).glob("pick_*.json")))
        if row["status"] == "counted" and not row["folder"]:
            warnings.append(f"{a}: counted but no run folder")
        if row["status"] == "counted" and held == "N" and not row["fail"]:
            warnings.append(f"{a}: not held but no failure code")
        if row["status"] == "counted" and held == "Y" and row["fail"]:
            warnings.append(f"{a}: held but a failure code {row['fail']!r}")
        rows.append(row)
    return rows, warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracker", required=True, help="the tracker export (holds attempts/a<n>.json)")
    ap.add_argument("--views", default="", help="views to build, comma-separated (V1,V2)")
    ap.add_argument("--check", default="", help="rebuild these views and compare with the CSV; write nothing")
    ap.add_argument("--write", action="store_true", help="replace the built views' tracker rows in the CSV")
    a = ap.parse_args()
    old = list(csv.DictReader(open(CSV)))
    if a.check:
        views = a.check.split(",")
        rows, warnings = build(a.tracker, views)
        by = {r["attempt"]: r for r in old if r["attempt"] and r["status"].startswith("counted")}
        bad = 0
        for r in rows:
            o = by.get(r["attempt"])
            if r["status"] != "counted":
                continue
            keys = ("run", "folder", "held", "fail", "pick_records", "condition", "object")
            diff = {k: (o.get(k) if o else None, r[k]) for k in keys if not o or (o[k] or "") != (r[k] or "")}
            if diff:
                bad += 1
                print(f"{r['attempt']}: CSV vs rebuilt {diff}")
        n = sum(r["status"] == "counted" for r in rows)
        print(f"{n} counted attempts rebuilt for {a.check}; {bad} differ from the CSV")
        for w in warnings:
            print("warning:", w)
        return
    views = [v for v in a.views.split(",") if v]
    rows, warnings = build(a.tracker, views)
    from collections import Counter
    print("statuses:", dict(Counter(r["status"].split(":")[0] for r in rows)))
    for r in rows:
        print(f"{r['attempt']:5s} {r['view']} {r['condition']:9s} {r['object']} {r['status'][:9]:9s} held={r['held'] or '-'} "
              f"fail={r['fail'] or '-'} {r['run'] or '-'} {r['pick_records'] or ''}")
    for w in warnings:
        print("warning:", w)
    if not a.write:
        print("(nothing written; --write to update", CSV.relative_to(ROOT), ")")
        return
    built = {r["attempt"] for r in rows}
    keep = [r for r in old if not (r["view"] in views and r["attempt"] in built
                                   and r["status"].split(":")[0] in ("counted", "excluded", "not run"))]
    out = keep + rows
    out.sort(key=lambda r: (int(r["attempt"][1:]) if r["attempt"] else 10 ** 6, r["run"]))
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} rows ({len(rows)} for {','.join(views)}) to {CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
