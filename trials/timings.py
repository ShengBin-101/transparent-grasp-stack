#!/usr/bin/env python3
"""Run times from the trial records: the medians the README and the dissertation quote. Reads files only.

    python3 trials/timings.py [--view V1] [--runs runs]

For the counted attempts of one view (trial_runs.csv), from each archived run (the release asset
trial_runs_v1.tar.zst unpacked into runs/, so runs/trial/<run>):

  capture to grasp pose   the headline time: wall clock from the run request (rg-run, the orchestrator's
                          /live_pipeline/run_chain service) to the grasp pose being written, before any
                          planning or arm motion. It is run_config.json's run_chain wall_seconds minus the
                          chain's last stage, the point cloud drawn for display. Complete runs only (*_auto).
  per stage               each node's own time for its stage (run_config.json seconds): capture, the
                          segmenter, SAM 3, ReMake, SBG, the grasp pose. They do not add up to the headline
                          time: the orchestrator's service calls between the stages add the rest ("overhead").
  planning                the NUC planner's time for one request (pick_*.json plan_reply.elapsed_s).
  executed pick           pick_*.json finished - started for the picks the planner accepted, so the arm moved:
                          planning, the approach, the grasp and the lift.
"""
import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = [("capture", "capture"), ("segmenter", "segment_trans"), ("SAM 3", "segment_sam3"),
          ("ReMake", "depth_remake"), ("SBG", "sbg"), ("grasp pose", "grasp_poses")]


def med(xs):
    return "%.2f s (n=%d)" % (statistics.median(xs), len(xs)) if xs else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--view", default="V1")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs", help="the folder the trial asset was unpacked into")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(ROOT / "trials" / "trial_runs.csv"))
            if r["status"].startswith("counted") and r["view"] == a.view]
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    to_pose, overhead, stage, plan, pick = {}, {}, {}, [], []
    for r in rows:
        run = a.runs / r["folder"].split("runs/", 1)[1]
        c = r["condition"]
        st = json.load(open(run / "run_config.json"))["stages"]
        node = {name: st[key]["seconds"] for name, key in STAGES
                if isinstance(st.get(key), dict) and st[key].get("seconds") is not None}
        for name, s in node.items():
            stage.setdefault((c, name), []).append(s)
        chain = st.get("run_chain") or {}
        if r["run"].endswith("_auto") and chain.get("wall_seconds") is not None:
            t = chain["wall_seconds"] - (chain.get("stage_seconds") or {}).get("pointcloud", 0.0)
            to_pose.setdefault(c, []).append(t)
            overhead.setdefault(c, []).append(t - sum(node.values()))
        for p in sorted(run.glob("pick_*.json")):
            d = json.load(open(p))
            e = (d.get("plan_reply") or {}).get("elapsed_s")
            if e is not None:
                plan.append(e)
            # executed: the planner replied and accepted a grasp, so the arm moved (a pick refused before or by
            # the planner has no plan_reply, or a summary "0 of N feasible")
            moved = d.get("plan_reply") is not None and not str(d.get("plan_summary", "")).startswith("0 of")
            if moved and d.get("started") and d.get("finished"):
                t = datetime.fromisoformat(d["finished"]) - datetime.fromisoformat(d["started"])
                pick.append(t.total_seconds())
    print(f"View {a.view}, {len(rows)} counted attempts. Medians:")
    print("| condition | capture to grasp pose (wall clock) | " + " | ".join(n for n, _ in STAGES) + " | overhead |")
    print("|---|---|" + "---|" * (len(STAGES) + 1))
    for c in conds:
        print(f"| {c} | {med(to_pose.get(c, []))} | " + " | ".join(med(stage.get((c, n), [])) for n, _ in STAGES)
              + f" | {med(overhead.get(c, []))} |")
    print(f"\nplanning, one request: {med(plan)}")
    print(f"executed pick, start to finish: {med(pick)}")


if __name__ == "__main__":
    main()
