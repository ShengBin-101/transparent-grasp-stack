"""Do the training changes help on the robot's own frames? L_combo_both against A_ft, the same six-epoch TransCG
fine-tune with nothing changed, on each mask; the released weights and RAW for reference. Offline only.

    python3 trials/view_debug/aft_compare.py runs/paired/aft_20261029 trials/view_debug/results/plan_aft.json OUT.md

Per frame and condition: SBG's executable grasps on the target (paired_replay's summary) and the grasps the NUC's
planner accepts from home (replay_plan.py). Pairs are the same frame under two conditions, so the comparison is
paired: it counts the frames where only one of the two has a feasible grasp, each way.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PAIRED, PLAN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ROOT = Path(__file__).resolve().parents[2]
CONDS = ["RAW", "BASELINE", "MASK-ONLY", "AFT-SAM3", "WEIGHTS-ONLY", "ADAPTED-AFT", "ADAPTED"]
WHAT = {"RAW": "raw depth, no mask", "BASELINE": "released + SAM 3", "MASK-ONLY": "released + segmenter",
        "AFT-SAM3": "A_ft + SAM 3", "WEIGHTS-ONLY": "L_combo_both + SAM 3",
        "ADAPTED-AFT": "A_ft + segmenter", "ADAPTED": "L_combo_both + segmenter"}
PAIRS = [("ADAPTED", "ADAPTED-AFT", "training changes, segmenter mask"),
         ("WEIGHTS-ONLY", "AFT-SAM3", "training changes, SAM 3 mask"),
         ("ADAPTED-AFT", "MASK-ONLY", "TransCG fine-tune alone, segmenter mask"),
         ("AFT-SAM3", "BASELINE", "TransCG fine-tune alone, SAM 3 mask"),
         ("ADAPTED", "BASELINE", "the whole package (the trials' comparison)")]
# side-view frames: (view, object) -- the V3 warm-up and the five uncounted V3 attempts (not in trial_runs.csv), and
# the two V2 captures of the side-view analysis, from the trial viewer's export (a development tool, not in this release)
SIDE = {"20261027_065835_auto": ("V3", "O1"), "20261027_065941_auto": ("V3", "T3"), "20261027_070036_auto": ("V3", "T2"),
        "20261027_070138_auto": ("V3", "T4"), "20261027_070417_auto": ("V3", "O2"), "20261027_070758_auto": ("V3", "T2"),
        "20261027_071607_auto": ("V2", "O1"), "20261027_073304_auto": ("V2", "T1")}

frames = {}
for r in csv.DictReader(open(ROOT / "trials" / "trial_runs.csv")):
    if r["run"]:
        frames[r["run"]] = (r["view"], r["object"])
frames.update(SIDE)

execu = defaultdict(dict)
for r in csv.DictReader(open(PAIRED / "summary.csv")):
    execu[r["run"]][r["condition"]] = int(r["executable"] or 0)
feas = defaultdict(dict)
for path, rec in json.load(open(PLAN)).items():
    run, cond = Path(path).parent.name, Path(path).name
    m = re.match(r"(\d+) of (\d+) feasible", rec["summary_text"])
    feas[run][cond] = int(m.group(1)) if m else 0


def group(run):
    view, obj = frames[run]
    return ("V1" if view == "V1" else "side (V2, V3)"), ("transparent" if obj.startswith("T") else "opaque")


runs = sorted(r for r in execu if r in frames)
groups = defaultdict(list)
for r in runs:
    groups[group(r)].append(r)

lines = ["# L_combo_both against A_ft on the robot's frames (offline)", "",
         "Every frame replayed under each condition, object-only fill throughout, the planner from home",
         "(`replay_plan.py --voxel-mm 10`). \"Feasible frames\": at least one grasp the planner accepts. RAW proposes",
         "grasps on anything in view (no target filter), so its counts are not on the target alone.", ""]
for key in sorted(groups):
    rs = groups[key]
    lines += ["## %s, %s objects (%d frames)" % (key[0], key[1], len(rs)), "",
              "| condition | weights + mask | frames with an executable target grasp | executable grasps | feasible frames | feasible grasps |",
              "|---|---|---|---|---|---|"]
    for c in CONDS:
        ex = [execu[r].get(c, 0) for r in rs]
        fe = [feas[r].get(c, 0) for r in rs]
        lines.append("| %s | %s | %d | %d | %d | %d |" % (c, WHAT[c], sum(v > 0 for v in ex), sum(ex),
                                                        sum(v > 0 for v in fe), sum(fe)))
    lines += ["", "| paired comparison | feasible in first only | in second only | executable grasps, first − second (frames up / down) |",
              "|---|---|---|---|"]
    for a, b, what in PAIRS:
        fa = [feas[r].get(a, 0) > 0 for r in rs]
        fb = [feas[r].get(b, 0) > 0 for r in rs]
        only_a = sum(x and not y for x, y in zip(fa, fb))
        only_b = sum(y and not x for x, y in zip(fa, fb))
        d = [execu[r].get(a, 0) - execu[r].get(b, 0) for r in rs]
        lines.append("| %s vs %s (%s) | %d | %d | %+d (%d / %d) |" % (
            a, b, what, only_a, only_b, sum(d), sum(x > 0 for x in d), sum(x < 0 for x in d)))
    lines.append("")
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
