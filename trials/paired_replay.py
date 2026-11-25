#!/usr/bin/env python3
"""Paired offline replay: every condition on the same archived frames. MOVES NOTHING.

    docker exec remake_grasp bash /opt/remake_grasp/trials/paired_replay.sh \\
        /opt/remake_grasp/runs/trial/20261027_*_auto --out /opt/remake_grasp/runs/paired/20261027
    ... --conditions RAW,BASELINE,ADAPTED           (default: those three plus the ablations)
    ... --summary-only                              (re-read the outputs, run nothing)

WHY. On the robot each attempt runs one condition, so two conditions never see the same frame. Every
capture archives what all of them start from (rgb.png, the raw depth, its coverage, the intrinsics
and the arm's pose), so offline each frame can go through every condition. Lighting, placement and
view are then identical, and the per-stage counts show where each condition loses grasps.

HOW. A condition is a preset plus overrides, configured by the live launch files themselves
(remake.launch.py, pipeline.launch.py, sbg.launch.py), so the settings cannot drift from the robot's:
the orchestrator's chain gives the stages, each stage runs the same node class with the same
parameters, called through its service handler on the condition's own directory
(<out>/<run>/<condition>/). Work is shared: two conditions with the same stages so far (BASELINE and
WEIGHTS-ONLY share SAM 3's mask) compute them once. Each model is loaded once per batch, in a worker
process on a private ROS domain, so nothing reaches a live pipeline -- but it needs the GPU memory a
live session uses: run it with the pipeline down (rg-down).

OUTPUT. <out>/summary.csv and summary.json, one row per run x condition: SBG's counts (raw, after NMS,
collision-free, empty, accepted), the pose node's (executable, off target, unseen side), the best
grasp, and -- for every condition alike -- how many executable grasps land on a transparent object
as the segmenter sees it (post hoc: RAW's own filter is off, so this is the only way to say where
its grasps go). BASELINE and ADAPTED fill the object only, the trials' rule after the first side-view attempts; *-FULL-FILL replays
the compositing blocks 1-3 ran under, so a V1 run captured under BASELINE reproduces as
BASELINE-FULL-FILL. Arm-side reachability is the planner's replay: run nuc/tools/replay_plan.py
on the host, from nuc/ (trials/README.md), on each condition directory (they hold grasp_poses.json, the depth SBG used,
the intrinsics and capture_tool.json).

NAMES. BASELINE's trial runs carry the pipeline label "baseline (targets=depth)" (trials/README.md);
LIVE_LABELS maps it, so a recorded run is still recognised as the condition it ran.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(os.environ.get("RG_ROOT", Path(__file__).resolve().parents[1]))
LAUNCH = ROOT / "ros/remake_grasp/launch"
INPUTS = ("rgb.png", "depth_aligned_mm.npy", "depth_coverage.npy", "color_intrinsics.npy", "capture_tool.json")
# An older run without capture_tool.json gets paired_axis.json: the approach cone's "down" its own SBG
# used, so the replay's cone matches the robot's.
# condition -> (pipeline:=, overrides). The first three are the trials' conditions;
# the rest separate their ingredients and run offline only.
CONDITIONS = {
    "RAW": ("raw", {}),
    "BASELINE": ("baseline", {}),
    "ADAPTED": ("adapted", {}),
    "ADAPTED-NO-REMAKE": ("adapted", {"remake": "none"}),         # the trials' earlier RAW, before RAW dropped the mask
    "MASK-ONLY": ("adapted", {"remake": "official"}),              # our mask, the authors' weights
    "WEIGHTS-ONLY": ("adapted", {"mask": "sam3"}),                 # SAM 3's mask, our weights (the other pairing)
    # the control fine-tune (released checkpoint, six epochs on TransCG, nothing changed): with L_combo_both, the
    # same-schedule comparison of the training changes, on each mask (offline only; not in the default run;
    # A_ft trained on all 34,191 frames, L_combo_both on 30,963: training/README.md)
    "ADAPTED-AFT": ("adapted", {"remake": "A_ft"}),
    "AFT-SAM3": ("adapted", {"mask": "sam3", "remake": "A_ft"}),
    # ReMake's depth in every sensor hole too, not only the mask: the compositing all V1 attempts ran under, before
    # the trials' BASELINE and ADAPTED were set to fill the object only
    "BASELINE-FULL-FILL": ("baseline", {"fill_holes": "true"}),
    "ADAPTED-FULL-FILL": ("adapted", {"fill_holes": "true"}),
    # a narrower approach cone about "down" than the trials' 30 deg (offline only): steeper grasps, which the planner
    # accepted more often in replays
    "ADAPTED-CONE20": ("adapted", {"sbg": {"approach_cone_deg": 20.0}}),
    "ADAPTED-CONE15": ("adapted", {"sbg": {"approach_cone_deg": 15.0}}),
    "BASELINE-CONE15": ("baseline", {"sbg": {"approach_cone_deg": 15.0}}),
}
# The pipeline label BASELINE's trial runs recorded -> today's (see NAMES above).
LIVE_LABELS = {"baseline (targets=depth)": "baseline"}
# chain stage -> (node executable, its module and class, the Trigger handler)
STAGES = {
    "segment_trans": ("seg_node", "seg_node", "SegNode", "_handle"),
    "segment_sam3": ("sam3_node", "sam3_node", "Sam3Node", "_handle"),
    "depth_remake": ("remake_node", "remake_node", "RemakeNode", "_handle"),
    "grasps_sbg": ("sbg_node", "sbg_node", "SbgNode", "_handle"),
    "grasp_poses": ("grasp_pose_node", "grasp_pose_node", "GraspPoseNode", "_handle"),
}
SKIPPED = {"pointcloud"}          # visual only
DOMAIN = "97"                     # the workers' private ROS domain


# ------------------------------------------------------------------ plans, from the launch files

def _launch(name):
    spec = importlib.util.spec_from_file_location(name, LAUNCH / f"{name}.launch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _evaluate(mod, values):
    """mod._nodes() with every declared default, then `values`: [(executable, params)]."""
    from launch import LaunchContext
    from launch_ros.utilities import evaluate_parameters
    ctx = LaunchContext()
    for a in mod.generate_launch_description().entities:
        if hasattr(a, "default_value") and a.default_value is not None:
            ctx.launch_configurations[a.name] = "".join(s.perform(ctx) for s in a.default_value)
    ctx.launch_configurations.update(values)
    out = []
    for n in mod._nodes(ctx):
        params = {}
        for d in evaluate_parameters(ctx, getattr(n, "_Node__parameters", [])):
            if isinstance(d, dict):
                params.update(d)
        out.append((n.node_executable, params))
    return out


def plan(pipeline, over):
    """[(stage, executable, params, optional)] for one condition, in the orchestrator's order, and its run label."""
    # every preset-controlled argument given explicitly, so the container's $MASK / $REMAKE / $TARGETS never leak in
    values = {"pipeline": pipeline, "mask": over.get("mask", ""), "remake": over.get("remake", ""),
              "targets": over.get("targets", "")}
    nodes = dict(_evaluate(_launch("remake"), {**values, **({"fill_holes": over["fill_holes"]}
                                                            if "fill_holes" in over else {})}))
    nodes.update(_evaluate(_launch("pipeline"), {**values, "detector": "sbg", "archive": "off"}))
    nodes.update(_evaluate(_launch("sbg"), {"pipeline": pipeline, "remake": values["remake"], "depth_name": ""}))
    if "sbg" in over:                                   # SBG settings beyond the launch file's profile
        nodes["sbg_node"] = {**nodes["sbg_node"], **over["sbg"]}
    orch = nodes.pop("orchestrator")
    steps = []
    for raw in orch["chain"].split(","):
        stage, optional = raw.rstrip("?"), raw.endswith("?")
        if stage in SKIPPED:
            continue
        if stage not in STAGES:
            raise RuntimeError(f"stage {stage!r} ({pipeline} {over}) is not replayable here")
        exe = STAGES[stage][0]
        steps.append((stage, exe, nodes[exe], optional))
    return steps, orch["pipeline"]


# ------------------------------------------------------------------ the worker: one model, many directories

def worker(job_path):
    """Run one stage's node on each directory of a job file; print one JSON line per directory."""
    job = json.loads(Path(job_path).read_text())
    exe, module, cls, handler = STAGES[job["stage"]]
    sys.path.insert(0, str(ROOT / "ros/remake_grasp"))
    import yaml
    import rclpy
    from std_srvs.srv import Trigger
    params = Path(tempfile.mkdtemp()) / "params.yaml"
    params.write_text(yaml.safe_dump({"/**": {"ros__parameters": {**job["params"], "live_dir": job["dirs"][0]}}}))
    rclpy.init(args=["--ros-args", "--params-file", str(params)])
    mod = importlib.import_module(f"remake_grasp.{module}")
    node = getattr(mod, cls)()
    for d in job["dirs"]:
        d = Path(d)
        node.live_dir = d
        tool = None
        if job["stage"] == "grasps_sbg":
            node._tool = None
            rec = json.loads((d / "capture_tool.json").read_text()) if (d / "capture_tool.json").exists() else {}
            if rec.get("live"):
                # the approach cone's "down": the arm's pose at the capture, as the live node had it
                node._tool, node._tool_at = (tuple(rec["position"]), tuple(rec["orientation_xyzw"])), time.monotonic()
                tool = True
            elif (d / "paired_axis.json").exists():
                # a run from before captures saved the arm's pose: the axis its own SBG used
                import numpy as np
                axis = np.array(json.loads((d / "paired_axis.json").read_text())["axis_camera"])
                node._down_in_camera = lambda axis=axis: (axis, "recorded")
                tool = "recorded"
            else:
                node._down_in_camera = type(node)._down_in_camera.__get__(node)
        t0 = time.time()
        res = getattr(node, handler)(None, Trigger.Response())
        print(json.dumps({"dir": str(d), "success": bool(res.success), "message": str(res.message)[:400],
                          "seconds": round(time.time() - t0, 2), "tool_pose": tool}), flush=True)
    node.destroy_node()
    rclpy.shutdown()


def run_worker(stage, params, dirs):
    """{dir: result} from a worker process for these directories."""
    job = Path(tempfile.mkdtemp()) / "job.json"
    job.write_text(json.dumps({"stage": stage, "params": params, "dirs": [str(d) for d in dirs]}))
    env = dict(os.environ, ROS_DOMAIN_ID=DOMAIN, ROS_LOCALHOST_ONLY="1")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-W", "ignore", __file__, "--worker", str(job)], env=env,
                       capture_output=True, text=True)
    out = {}
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            out[r["dir"]] = r
    for d in dirs:
        out.setdefault(str(d), {"success": False, "message": f"worker exited {p.returncode}: "
                                                             f"{p.stderr.strip().splitlines()[-1:] or ''}"})
    print(f"  {stage:<14} {len(dirs):3d} dir(s) in {time.time() - t0:5.0f}s, "
          f"{sum(not r['success'] for r in out.values())} failed", flush=True)
    return out


# ------------------------------------------------------------------ the driver

def _key(prev, stage, params):
    return hashlib.sha1(json.dumps([prev, stage, params], sort_keys=True, default=str).encode()).hexdigest()[:16]


def _link_tree(src, dst):
    """dst := src's files, hard-linked (each stage writes only files of its own, never in place), except that
    dst's run record keeps its own paired_replay entry (which condition the directory is)."""
    sys.path.insert(0, str(ROOT / "ros/remake_grasp"))
    from remake_grasp.common import write_json
    own = json.loads((dst / "run_config.json").read_text()).get("paired_replay")
    for f in dst.iterdir():
        if f.is_file():
            f.unlink()
    for f in src.iterdir():
        if f.is_file():
            os.link(f, dst / f.name)
    rc = json.loads((dst / "run_config.json").read_text())
    (dst / "run_config.json").unlink()
    write_json(dst / "run_config.json", dict(rc, paired_replay=own, live_dir=str(dst)))


def prepare(run, cdir, cond, label):
    """A fresh condition directory with the run's capture inputs and a run record naming its source."""
    sys.path.insert(0, str(ROOT / "ros/remake_grasp"))
    from remake_grasp.common import write_json
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True)
    inputs = cdir.parent / "_inputs"
    if not inputs.exists():
        inputs.mkdir()
        for name in INPUTS:
            if (run / name).exists():
                shutil.copy2(run / name, inputs / name)
        cone = {}
        try:
            cone = json.loads((run / "grasps_sbg.json").read_text()).get("approach_cone") or {}
        except (OSError, ValueError):
            pass
        if not (run / "capture_tool.json").exists() and cone.get("axis_source") == "tool_pose":
            (inputs / "paired_axis.json").write_text(json.dumps({"axis_camera": cone["axis_camera"],
                                                                 "from": "the run's own grasps_sbg.json"}))
    for f in inputs.iterdir():
        os.link(f, cdir / f.name)
    capture = {}
    try:
        capture = json.loads((run / "run_config.json").read_text())["stages"].get("capture", {})
    except (OSError, ValueError, KeyError):
        pass
    write_json(cdir / "run_config.json", {"schema": 1, "host": "paired_replay", "live_dir": str(cdir),
                                          "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                          "stages": {"capture": capture},
                                          "paired_replay": {"source": str(run), "condition": cond,
                                                            "pipeline": label}})


def replay(runs, out, names):
    plans = {c: plan(*CONDITIONS[c]) for c in names}
    for c, (steps, label) in plans.items():
        print(f"{c:<19} label '{label}': {' -> '.join(s for s, *_ in steps)}")
    todo = []                       # (run, condition, its directory)
    for run in runs:
        if not all((run / n).exists() for n in INPUTS[:4]):
            print(f"{run.name}: missing capture inputs -- skipped")
            continue
        for c in names:
            cdir = out / run.name / c
            if (cdir / "paired_done.json").exists():
                continue
            prepare(run, cdir, c, plans[c][1])
            todo.append((run, c, cdir))
    state = {(r.name, c): {"key": "", "failed": None} for r, c, _ in todo}
    for wave in range(max((len(s) for s, _ in plans.values()), default=0)):
        groups, dupes = {}, {}      # (stage, params) -> [dirs]; key -> [dirs to copy into]
        firsts = {}                 # key -> the directory that computes it
        for run, c, cdir in todo:
            st, steps = state[(run.name, c)], plans[c][0]
            if st["failed"] or wave >= len(steps):
                continue
            stage, exe, params, optional = steps[wave]
            key = _key(f"{run.name}:{st['key']}", stage, params)
            st["key"], st["step"] = key, steps[wave]
            if key in firsts:
                dupes.setdefault(key, []).append(cdir)
                continue
            firsts[key] = cdir
            groups.setdefault((stage, json.dumps(params, sort_keys=True)), []).append(cdir)
        if not groups:
            break
        results = {}
        for (stage, pjson), dirs in groups.items():
            results.update(run_worker(stage, json.loads(pjson), dirs))
        for key, dirs in dupes.items():
            for d in dirs:
                _link_tree(firsts[key], d)
                results[str(d)] = results[str(firsts[key])]
        for run, c, cdir in todo:
            st = state[(run.name, c)]
            r = results.get(str(cdir))
            if r is None or st["failed"]:
                continue
            stage, _, _, optional = st["step"]
            st.setdefault("stages", {})[stage] = {k: r.get(k) for k in ("success", "message", "seconds", "tool_pose")}
            if not r["success"] and not optional:
                st["failed"] = stage
    for run, c, cdir in todo:
        st = state[(run.name, c)]
        (cdir / "paired_done.json").write_text(json.dumps({"failed_stage": st["failed"], "stages": st.get("stages", {})},
                                                          indent=1))


# ------------------------------------------------------------------ summary

def _glass_mask(run_out):
    """The segmenter's transparent objects for this run, dilated 15 px as the pose node dilates targets, from
    whichever condition directory ran it (None if none did)."""
    sys.path.insert(0, str(ROOT / "ros/remake_grasp"))
    from remake_grasp.grasp_pose_node import mask_labels, target_masks
    for d in sorted(run_out.iterdir()):
        lab = mask_labels(d / "seg_masks.npy") if d.is_dir() else None
        if d.is_dir() and (d / "seg_masks.npy").exists():
            return (target_masks(d, 15, objects=(lab, {1: True})) if lab is not None
                    else False)               # the segmenter ran and found nothing transparent
    return None


def _reproduces_live(d):
    """When the run was captured under this very condition: does the replay give the grasps the robot got
    (SBG's array, to 1e-4)? None when the run ran another condition or has no grasps to compare."""
    import numpy as np
    try:
        rp = json.loads((d / "run_config.json").read_text())["paired_replay"]
        src = Path(rp["source"])
        live_rc = json.loads((src / "run_config.json").read_text())["stages"]
        mine = json.loads((d / "run_config.json").read_text())["stages"]
        live = live_rc["capture"]["settings"].get("pipeline")
        live = LIVE_LABELS.get(live, live)
        if live != rp["pipeline"] or not (src / "grasps_sbg.npy").exists() or not (d / "grasps_sbg.npy").exists():
            return None
        # the label does not name the compositing rule: compare only runs composited alike (a recorded run with the
        # keep-nearer rule on matches no condition here)
        rule = lambda st: ((st.get("depth_remake") or {}).get("settings", {}).get("keep_nearer_sensor_mm", 0.0),  # noqa: E731
                           (st.get("depth_remake") or {}).get("settings", {}).get("fill_holes", True))
        if rule(live_rc) != rule(mine):
            return None
        a, b = np.load(src / "grasps_sbg.npy"), np.load(d / "grasps_sbg.npy")
        return bool(a.shape == b.shape and np.allclose(a, b, atol=1e-4))
    except (OSError, ValueError, KeyError):
        return None


def summarise(out, names):
    import numpy as np
    sys.path.insert(0, str(ROOT / "ros/remake_grasp"))
    from remake_grasp.grasp_pose_node import on_target
    rows = []
    for run_out in sorted(p for p in out.iterdir() if p.is_dir() and not p.name.startswith("_")):
        glass = _glass_mask(run_out)
        for c in names:
            d = run_out / c
            if not (d / "paired_done.json").exists():
                continue
            done = json.loads((d / "paired_done.json").read_text())
            row = {"run": run_out.name, "condition": c, "failed_stage": done["failed_stage"]}
            row["reproduces_live"] = _reproduces_live(d)
            try:
                s = json.loads((d / "grasps_sbg.json").read_text())
                row.update({f"sbg_{k[2:]}": s.get(k) for k in ("n_raw", "n_after_nms", "n_collision_free",
                                                                "n_empty", "n_accepted")})
                row["sbg_depth"] = s.get("depth_source")
                row["cone_axis"] = (s.get("approach_cone") or {}).get("axis_source")
            except (OSError, ValueError):
                pass
            try:
                gp = json.loads((d / "grasp_poses.json").read_text())
                g = gp["grasps"]
                ex = [x for x in g if x["executable"]]
                row.update(executable=len(ex),
                           not_on_target=sum(x["recommendation"] == "not_on_target" for x in g),
                           unseen_side=sum(x["recommendation"] == "approach_from_unseen_side" for x in g),
                           objects=len(gp.get("objects") or []),
                           best=gp.get("best_index"))
                if glass is not None:
                    K = np.load(d / "color_intrinsics.npy")

                    def lands(x):
                        if glass is False:
                            return False
                        c0 = np.array(x["grasp_pose"]["position"])
                        y, h = np.array(x["jaw_axis"]), 0.5 * (x["width_m"] or 0.0)
                        return bool(on_target([c0, c0 + h * y, c0 - h * y], K, glass))
                    row["executable_on_glass"] = sum(lands(x) for x in ex)
                    row["best_on_glass"] = None if gp.get("best") is None else lands(gp["best"])
            except (OSError, ValueError, KeyError):
                pass
            rows.append(row)
    cols = ["run", "condition", "failed_stage", "reproduces_live", "sbg_raw", "sbg_after_nms", "sbg_collision_free", "sbg_empty",
            "sbg_accepted", "executable", "not_on_target", "unseen_side", "objects", "best",
            "executable_on_glass", "best_on_glass", "sbg_depth", "cone_axis"]
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    print(f"\n{'condition':<19} {'runs':>4} {'failed':>6} {'SBG acc':>7} {'exec':>5} {'on transparent':>14} "
          f"{'runs w/ exec':>12} {'best on transparent':>19}")
    for c in names:
        rs = [r for r in rows if r["condition"] == c]
        if not rs:
            continue
        tot = lambda k: sum(r.get(k) or 0 for r in rs)  # noqa: E731
        print(f"{c:<19} {len(rs):>4} {sum(bool(r['failed_stage']) for r in rs):>6} {tot('sbg_accepted'):>7} "
              f"{tot('executable'):>5} {tot('executable_on_glass'):>14} {sum((r.get('executable') or 0) > 0 for r in rs):>12} "
              f"{sum(bool(r.get('best_on_glass')) for r in rs):>19}")
    live = [r for r in rows if r.get("reproduces_live") is not None]
    if live:
        bad = [f"{r['run']} {r['condition']}" for r in live if not r["reproduces_live"]]
        print(f"\nreplay vs the robot's own record, where a run was captured under that condition: "
              f"{len(live) - len(bad)} of {len(live)} identical" + (f"; DIFFERENT: {', '.join(bad[:8])}" if bad else ""))
    print(f"\n{out / 'summary.csv'}  (one row per run x condition)")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        return worker(sys.argv[2])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conditions", default=",".join(c for c in CONDITIONS if "AFT" not in c),
                    help="default: all but the A_ft control")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    names = [c.strip() for c in a.conditions.split(",") if c.strip()]
    bad = [c for c in names if c not in CONDITIONS]
    if bad:
        ap.error(f"unknown condition(s) {bad}; use {', '.join(CONDITIONS)}")
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        replay([r for r in a.runs if r.is_dir()], a.out, names)
    summarise(a.out, names)


if __name__ == "__main__":
    main()
