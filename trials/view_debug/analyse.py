#!/usr/bin/env python3
"""Tables for the side-view analysis, which led to the object-only fill. Reads files only.

    python3 trials/view_debug/analyse.py PAIRED_DIR PLAN_DIR

PAIRED_DIR holds the paired replays views_20261027_side, views_20261027_v1 and views_20261027_v1_blank02 (the V1
frames with 0.2% of their read pixels blanked at random: SBG's noise floor). PLAN_DIR holds the planner replays of
each (plan_side.json, plan_v1.json, plan_blank.json: nuc/tools/replay_plan.py --voxel-mm 10 --json) and
conds_side.json / conds_v1.json (conds.py: tilt and phantom points per run and condition).
"""
import collections
import csv
import json
import sys
from pathlib import Path

PAIRED, PLAN = Path(sys.argv[1]), Path(sys.argv[2])
ORDER = ["RAW", "BASELINE", "ADAPTED", "BASELINE-OBJECT-FILL", "ADAPTED-OBJECT-FILL"]


def group(tilt):
    return "V1 (~5)" if tilt < 12 else "13-21" if tilt < 25 else "V2 (36)" if tilt < 40 else "V3 (47)"


def load(batch, plan):
    rows = {(r["run"], r["condition"]): r for r in csv.DictReader(open(PAIRED / batch / "summary.csv"))}
    feas = {}
    for path, v in json.load(open(PLAN / plan)).items():
        run, cond = Path(path).parts[-2:]
        ok = sum(1 for c in v.get("candidates", []) if c.get("ok"))
        feas[(run, cond)] = {"feasible": ok, "summary": v.get("summary", {}),
                             "chosen": next((c["id"] for c in v.get("candidates", []) if c.get("ok")), None)}
    return rows, feas


def num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


tilt = {}
phantom = {}
for f in ("conds_side.json", "conds_v1.json"):
    for r in json.load(open(PLAN / f)):
        tilt[r["run"]] = r["tilt"]
        phantom[(r["run"], r["cond"])] = r["phantom_pts"]

out = {}
print("## Grasp funnel and planner, per view and condition\n")
print("| view | condition | frames | phantom points (median) | SBG raw | collision-free | executable | frames with an executable grasp | frames with a feasible grasp | planner refusals |")
print("|---|---|---|---|---|---|---|---|---|---|")
for batch, plan in (("views_20261027_v1", "plan_v1.json"), ("views_20261027_side", "plan_side.json")):
    rows, feas = load(batch, plan)
    for g in ("V1 (~5)", "13-21", "V2 (36)", "V3 (47)"):
        for c in ORDER:
            keys = [k for k in rows if k[1] == c and k[0] in tilt and group(tilt[k[0]]) == g and not k[0].startswith("20261023")]
            if not keys:
                continue
            rs = [rows[k] for k in keys]
            ph = sorted(phantom.get(k, 0) for k in keys)
            refusals = collections.Counter()
            for k in keys:
                refusals.update(feas.get(k, {}).get("summary", {}))
            top = ", ".join(f"{n} {cat}" for cat, n in refusals.most_common(3))
            cell = dict(frames=len(keys), phantom=ph[len(ph) // 2], raw=sum(num(r["sbg_raw"]) for r in rs),
                        cfree=sum(num(r["sbg_collision_free"]) for r in rs), exe=sum(num(r["executable"]) for r in rs),
                        with_exe=sum(num(r["executable"]) > 0 for r in rs),
                        with_feasible=sum(feas.get(k, {}).get("feasible", 0) > 0 for k in keys))
            out[f"{g} {c}"] = cell
            print(f"| {g} | {c} | {cell['frames']} | {cell['phantom']} | {cell['raw']} | {cell['cfree']} | {cell['exe']} | "
                  f"{cell['with_exe']} | {cell['with_feasible']} | {top} |")

print("\n## V1: does object-only fill change more than SBG's noise floor?\n")
v1, f1 = load("views_20261027_v1", "plan_v1.json")
bl, fb = load("views_20261027_v1_blank02", "plan_blank.json")


def compare(a_rows, a_feas, a_cond, b_rows, b_feas, b_cond):
    runs = sorted({k[0] for k in a_rows if k[1] == a_cond} & {k[0] for k in b_rows if k[1] == b_cond})
    best = exe = flip = 0
    for r in runs:
        A, B = a_rows[(r, a_cond)], b_rows[(r, b_cond)]
        best += A["best"] != B["best"]
        exe += abs(num(A["executable"]) - num(B["executable"]))
        flip += (a_feas.get((r, a_cond), {}).get("feasible", 0) > 0) != (b_feas.get((r, b_cond), {}).get("feasible", 0) > 0)
    n = len(runs)
    return dict(frames=n, best_changed=best, mean_abs_exe_change=round(exe / max(1, n), 2), feasible_flipped=flip)


tests = {
    "full -> object-only fill, BASELINE": compare(v1, f1, "BASELINE", v1, f1, "BASELINE-OBJECT-FILL"),
    "full -> object-only fill, ADAPTED": compare(v1, f1, "ADAPTED", v1, f1, "ADAPTED-OBJECT-FILL"),
    "noise floor: 0.2% blanked, RAW": compare(v1, f1, "RAW", bl, fb, "RAW"),
    "noise floor: 0.2% blanked, BASELINE-OBJECT-FILL": compare(v1, f1, "BASELINE-OBJECT-FILL", bl, fb, "BASELINE-OBJECT-FILL"),
    "noise floor: 0.2% blanked, ADAPTED-OBJECT-FILL": compare(v1, f1, "ADAPTED-OBJECT-FILL", bl, fb, "ADAPTED-OBJECT-FILL"),
}
print("| comparison | frames | best grasp changed | mean abs. change in executable grasps | feasible yes/no flipped |")
print("|---|---|---|---|---|")
for name, t in tests.items():
    print(f"| {name} | {t['frames']} | {t['best_changed']} | {t['mean_abs_exe_change']} | {t['feasible_flipped']} |")
out["v1_tests"] = tests
json.dump(out, open(PLAN / "view_debug_tables.json", "w"), indent=1)
