#!/usr/bin/env python3
"""Predicted vs measured end posture, from the leg log lines written after a time.

    tools/compare_legs.py logs/legs-YYYYMMDD.jsonl SINCE_UNIX_TIME TOPIC_FILTER

TOPIC_FILTER picks the controller: "target_pose" (position) or "equilibrium_pose" (impedance).
Read-only. The prediction is at the target, so under impedance part of the error is tracking,
not the posture model; re-predict at the reached pose (planning.check_leg) to separate them.
"""
import json
import sys

import numpy as np

path, since, topic_filter = sys.argv[1], float(sys.argv[2]), sys.argv[3]
rows = []
for line in open(path):
    e = json.loads(line)
    if e["start"]["t"] is None or e["start"]["t"] < since or topic_filter not in e["command_topic"]:
        continue
    pl = e.get("planner") or {}
    q_start, q_end = np.array(e["start"]["q"]), np.array(e["end"]["q"])
    preds = pl.get("q_end") or {}
    diffs = {k: np.abs(np.array(v) - q_end) for k, v in preds.items()}
    rows.append((e, q_start, q_end, diffs, pl))
print("%-5s %-7s %-6s %-9s %-26s %-24s %s" % ("kind", "success", "plan", "dq3 meas", "elbow_c[0] start->end",
                                             "max|pred-meas| (joint)", "per prediction"))
for e, qs, qe, diffs, pl in rows:
    ec_s, ec_e = e["start"]["elbow_c"], e["end"]["elbow_c"]
    best = min(diffs.items(), key=lambda kv: kv[1].max()) if diffs else None
    per = "  ".join("%s %.3f" % (k, v.max()) for k, v in sorted(diffs.items()))
    print("%-5s %-7s %-6s %+8.4f  %+.4f -> %+.4f %s  %s  %s" % (
        e["kind"], e["success"], pl.get("ok"), qe[2] - qs[2], ec_s[0], ec_e[0],
        "" if ec_s[1] == ec_e[1] else "SIGN FLIP",
        "%.3f rad (j%d)" % (best[1].max(), 1 + int(np.argmax(best[1]))) if best else "n/a", per))
if rows:
    allmax = {}
    for _, _, _, diffs, _ in rows:
        for k, v in diffs.items():
            allmax.setdefault(k, []).append(v.max())
    for k, v in sorted(allmax.items()):
        print("prediction %-12s over %d legs: median max-joint error %.3f rad, worst %.3f rad"
              % (k, len(v), float(np.median(v)), float(np.max(v))))
