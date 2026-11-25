#!/usr/bin/env python3
"""Replay an archived capture through the reachability planner. MOVES NOTHING.

    python3 tools/replay_plan.py RUN_DIR                      tool pose from RUN_DIR/capture_tool.json
    python3 tools/replay_plan.py RUN_DIR --tool-pose X Y Z QX QY QZ QW
    python3 tools/replay_plan.py RUN_DIR --start capture      plan from the capture pose, not home
    python3 tools/replay_plan.py RUN_DIR --json out.json      also write every verdict

Does what plan_pick does on a live pick, from files: every executable grasp in
RUN_DIR/grasp_poses.json, in both finger orders, is resolved into the base frame from the
capture tool pose, turned into standoff / grasp / retreat legs and walked by
planning.check_legs from home (or the capture pose), against the scene voxelised from the
detector's depth, return home included. plan_pick ranks the feasible candidates by utility
(confidence x comfort) and does not walk those that cannot win; this walks every candidate's full
chain, reports the choice under both ranks (utility, and confidence: the most confident feasible),
and says whether each scene refusal came near the table or an object, so a capture that yields no
pick is explained before the arm moves. Settings come from this workspace's config files.

The target, as the Orin marks it (--labels). "instances", the default: every voxel and every
candidate carries its object label, so each candidate's own object is its target, neighbours stay
obstacles and the fingers are checked (planning.candidate_scene). "flags": one target for every
candidate, and the keep-out box removes whatever is inside it. --target-px 0 marks no target.
The labels come from the run's target_objects.npy (the grasp-pose node's merged targets), else
seg_masks.npy (the segmenter), else sam3_masks.npy.

The capture pose is O_T_EE (fr3_link0) when the frame was taken, from RUN_DIR/capture_tool.json
or --tool-pose; a wrong one silently moves every grasp and the scene with it. The pose nodes'
scene_voxels and the sequencers' build_pick_candidates are copied here, not imported, so this
needs only franka_ros (setup_franka.sh); ee_to_grasp is identity, as there.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "src" / "fr3_pose_commander"
sys.path.insert(0, str(PKG / "src"))
from fr3_pose_commander import kinematics, planning  # noqa: E402
from fr3_pose_commander.pose_math import angle_between, quaternion_from_matrix  # noqa: E402
from fr3_pose_commander.scene import Scene  # noqa: E402

LIMITS = HERE.parent / "src" / "franka_ros" / "franka_description" / "robots" / "fr3" / "joint_limits.yaml"


# -- configuration, as the commander builds it -------------------------------------

def load_config(margin=None):
    """(PlanConfig, ee_to_camera, home joints) from the config files; margin overrides
    planner/scene_margin (m)."""
    y = yaml.safe_load(open(PKG / "config" / "pose_commander.yaml"))
    pl, lim, ramp = y.get("planner", {}), y.get("limits", {}), y.get("ramp", {})
    ee_to_camera = yaml.safe_load(open(PKG / "config" / "camera_extrinsics.yaml"))["ee_to_camera"]
    cfg = planning.PlanConfig(
        joint_margin=float(pl.get("joint_margin", 0.05)),
        max_step_dq=float(pl.get("max_step_dq", 0.1)),
        table_z=float(pl.get("table_z", planning.TABLE_Z)),
        clearance=float(pl.get("clearance", 0.03)),
        max_speed_frac=float(pl.get("max_speed_frac", 0.5)),
        min_sigma=float(pl.get("min_sigma", 0.0)),
        scene_margin=float(pl.get("scene_margin", 0.01)) if margin is None else margin,
        linear_speed=float(ramp.get("linear_speed", 0.012)),
        angular_speed=float(ramp.get("angular_speed", 0.40)),
        rotation_max=float(lim.get("rotation_max", 1.571)),
        z_min=float(lim["z_min"]) if "z_min" in lim else None,
        radius_max=float(lim["radius_max"]) if "radius_max" in lim else None,
        rank=str(pl.get("rank", "utility")),
        ee_to_camera=tuple(ee_to_camera[:3]))
    for key in ("table_mode", "table_margin", "table_band"):   # the table-plane settings, if present
        if key in pl and hasattr(cfg, key):
            setattr(cfg, key, pl[key])
    home_q = [float(v) for v in yaml.safe_load(open(PKG / "config" / "home_pose.yaml"))["home_joints"]]
    return cfg, ee_to_camera, home_q


# -- geometry (numpy; the sequencers' eye_in_hand, restated) -------------------------

def mat(pose):
    """4x4 matrix from ((x, y, z), (qx, qy, qz, qw))."""
    return planning.matrix_from_pose((tuple(pose[0]), tuple(pose[1])))


def pose_of(m):
    """((x, y, z), (qx, qy, qz, qw)) from a 4x4 matrix."""
    return (tuple(float(v) for v in m[:3, 3]), quaternion_from_matrix(m))


def along_approach(m, d):
    """m moved d metres along its own approach (z) axis."""
    out = m.copy()
    out[:3, 3] = m[:3, 3] + d * m[:3, 2]
    return out


def jaw_flip(m):
    """m turned 180 deg about its approach (z) axis: the other finger order."""
    out = m.copy()
    out[:3, :2] = -m[:3, :2]
    return out


def scene_voxels(depth_mm, K, voxel_m=0.02, min_points=3, z_min=0.15, z_max=1.5, stride=2,
                 max_voxels=8000):
    """Occupied voxel centres (camera frame, m) and the voxel size, as the pose nodes' scene_voxels.

    Depth (mm) is subsampled by stride and kept within (z_min, z_max) m; a voxel needs min_points
    returns, and the voxel size doubles until at most max_voxels remain.
    """
    z = np.asarray(depth_mm, np.float32)[::stride, ::stride] / 1000.0
    h, w = z.shape
    vv, uu = np.mgrid[0:h, 0:w]
    uu, vv = uu * stride, vv * stride
    ok = np.isfinite(z) & (z > z_min) & (z < z_max)
    z, uu, vv = z[ok], uu[ok], vv[ok]
    pts = np.stack([(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z], axis=1)
    while True:
        idx, counts = np.unique(np.floor(pts / voxel_m).astype(np.int64), axis=0, return_counts=True)
        idx = idx[counts >= min_points]
        if len(idx) <= max_voxels:
            return (idx + 0.5) * voxel_m, voxel_m
        voxel_m *= 2.0


# -- the run ------------------------------------------------------------------------

def load_run(run, tool_pose=None):
    """(grasps, capture tool pose, depth in mm or None, depth file name, K) from a run directory.

    The depth is the detector's depth_source if recorded, else ReMake's, else the aligned raw depth.
    """
    run = Path(run)
    gp = json.load(open(run / "grasp_poses.json"))
    if gp.get("frame_id") not in (None, "camera_color_optical_frame") and gp.get("transformed"):
        raise SystemExit("%s: grasps are in %s, not the camera frame" % (run, gp.get("frame_id")))
    if tool_pose is None:
        f = run / "capture_tool.json"
        rec = json.load(open(f)) if f.exists() else None
        if rec is not None and rec.get("live") is not False:
            tool_pose = list(rec["position"]) + list(rec["orientation_xyzw"])
        else:
            # a pick record carries the tool pose the sequencer resolved the grasps from
            for pf in sorted(run.glob("pick_*.json")):
                c = json.load(open(pf)).get("capture_tool")
                if c:
                    tool_pose = list(c[0]) + list(c[1])
                    break
        if tool_pose is None:
            raise SystemExit("%s: %s; give --tool-pose X Y Z QX QY QZ QW (the tool pose when the "
                             "frame was taken)" % (run, "capture_tool.json says the tool pose was not live (%s)"
                                                   % rec.get("why") if rec is not None else
                                                   "no capture_tool.json and no pick record with one"))
    capture = (tuple(tool_pose[:3]), tuple(tool_pose[3:7]))
    depth_name = None
    for name in ("grasps_sbg.json", "grasps_peggnet.json"):
        if (run / name).exists():
            depth_name = json.load(open(run / name)).get("depth_source") or depth_name
    depth_name = depth_name or next((n for n in ("remake_depth_mm.npy", "depth_aligned_mm.npy")
                                     if (run / n).exists()), None)
    depth = np.load(run / depth_name) if depth_name else None
    K = np.load(run / "color_intrinsics.npy")
    return gp["grasps"], capture, depth, depth_name, K


def label_image(run, dilate_px):
    """The run's SAM 3 masks as an int label image (1..N, 0 = none), or None without masks.

    Each object is grown dilate_px into the background (nearer object wins), as the pose node's
    target_labels.
    """
    # The pose node's merged targets (target_objects.npy), else the segmenter's or SAM 3's mask.
    # ADAPTED runs have no sam3_masks.npy.
    path = next((Path(run) / n for n in ("target_objects.npy", "seg_masks.npy", "sam3_masks.npy")
                 if (Path(run) / n).exists()), None)
    if path is None:
        return None
    m = np.load(path)
    if m.ndim == 3:
        lab = np.zeros(m.shape[1:], np.int32)
        for k in range(m.shape[0]):
            lab[(m[k] > 0) & (lab == 0)] = k + 1
    else:
        lab = m.astype(np.int32)
    if not lab.any():
        return None
    if dilate_px > 0:
        from scipy.ndimage import distance_transform_edt
        dist, (iy, ix) = distance_transform_edt(lab == 0, return_indices=True)
        lab = np.where(dist <= dilate_px, lab[iy, ix], 0).astype(np.int32)
    return lab


def _project(K, pts, shape):
    """Pixels (u, v) of camera-frame points, and a mask of those in front of and inside the image."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    z = pts[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.round(K[0, 0] * pts[:, 0] / z + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * pts[:, 1] / z + K[1, 2]).astype(int)
    ok = (z > 0) & (u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0])
    return u, v, ok


def grasp_instance(g, K, labels):
    """The pose node's grasp_instance: majority label of TCP and contacts, ties to the TCP."""
    if g.get("instance") is not None:
        return int(g["instance"])
    c = np.array(g["grasp_pose"]["position"], float)
    j = np.array(g.get("jaw_axis") or mat((c, g["grasp_pose"]["orientation_xyzw"]))[:3, 1], float)
    half = 0.5 * float(g.get("width_m") or 0.0)
    u, v, ok = _project(K, [c, c + half * j, c - half * j], labels.shape)
    hits = [int(labels[v[i], u[i]]) if ok[i] else 0 for i in range(3)]
    found = [k for k in hits if k]
    if not found:
        return 0
    counts = {k: found.count(k) for k in set(found)}
    best = [k for k, n in counts.items() if n == max(counts.values())]
    return hits[0] if hits[0] in best else min(best)


def build(grasps, capture, ee_to_camera, start_pose, standoff=0.10, retreat=0.10, max_width=0.080,
          K=None, labels=None):
    """Candidates as plan_legs takes them: most confident first, both finger orders (the one
    nearer start_pose's orientation first), legs along the grasp's approach axis. With labels
    (label_image), each carries the object it grasps as "instance"."""
    t_base_cam = mat(capture).dot(mat((ee_to_camera[:3], ee_to_camera[3:7])))
    out = []
    ranked = sorted(((j, g) for j, g in enumerate(grasps) if g.get("executable")),
                    key=lambda jg: -float(jg[1].get("score", jg[1].get("confidence", 0.0)) or 0.0))
    for j, g in ranked:
        width = g.get("width_m")
        if width is None or not 0.0 < width <= max_width:
            continue
        cam = mat((g["grasp_pose"]["position"], g["grasp_pose"]["orientation_xyzw"]))
        goal = t_base_cam.dot(cam)
        orders = sorted((("direct", goal), ("flipped", jaw_flip(goal))),
                        key=lambda o: angle_between(pose_of(o[1])[1], start_pose[1]))
        for jaw, m in orders:
            legs = [("standoff", pose_of(along_approach(m, -standoff))), ("grasp", pose_of(m)),
                    ("retreat", pose_of(along_approach(m, -retreat)))]
            out.append({"id": "%d:%s" % (j, jaw), "index": j, "jaw": jaw,
                        "confidence": float(g.get("score", g.get("confidence", 0.0)) or 0.0),
                        "width_m": float(width), "legs": legs, "grasp_matrix": m,
                        "instance": None if labels is None else grasp_instance(g, K, labels)})
    return out


class _Witness(Scene):
    """A Scene that remembers the voxel nearest the part that came too close."""

    def __init__(self, points, voxel_m, margin):
        Scene.__init__(self, points, voxel_m)
        self.margin, self.hits = margin, []

    def clearance(self, spheres):
        gap, part = Scene.clearance(self, spheres)
        if gap < self.margin and self._tree is not None:
            for name, centres, _ in spheres:
                if name == part:
                    d, i = self._tree.query(centres)
                    self.hits.append((gap, part, self.points[i[int(np.argmin(d))]]))
        return gap, part

    def without_box(self, pose_matrix, half_extents, z_range):
        s = Scene.without_box(self, pose_matrix, half_extents, z_range)
        w = _Witness(s.points, s.voxel_m, self.margin)
        w.target = s.target
        w.box_target_only = s.box_target_only
        w.hits = self.hits                       # one record for both scenes
        return w


def voxel_labels(run, cam_pts, K, dilate_px):
    """One SAM 3 object label per voxel (camera frame), 0 = none; None without masks."""
    labels = label_image(run, dilate_px)
    if labels is None:
        return None
    u, v, ok = _project(K, cam_pts, labels.shape)
    out = np.zeros(len(cam_pts), np.int32)
    out[ok] = labels[v[ok], u[ok]]
    return out


def target_flags(run, cam_pts, K, dilate_px):
    """One flag per voxel (camera frame): on the run's dilated SAM mask, as the pose node's
    target marking. None without a mask."""
    path = next((Path(run) / n for n in ("target_objects.npy", "seg_masks.npy", "sam3_masks.npy")
                 if (Path(run) / n).exists()), None)
    if path is None:
        return None
    m = np.load(path)
    m = (m.any(axis=0) if m.ndim == 3 else m).astype(bool)
    # scipy is required: without the dilation the target flags (and so the verdicts) would differ.
    from scipy.ndimage import distance_transform_edt
    m = distance_transform_edt(~m) <= dilate_px
    z = cam_pts[:, 2]
    u = np.round(K[0, 0] * cam_pts[:, 0] / z + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * cam_pts[:, 1] / z + K[1, 2]).astype(int)
    ok = (z > 0) & (u >= 0) & (u < m.shape[1]) & (v >= 0) & (v < m.shape[0])
    flags = np.zeros(len(cam_pts), bool)
    flags[ok] = m[v[ok], u[ok]]
    return flags


def evaluate(cands, q_start, start_pose, home, scene_pts, voxel, limits, cfg, model="elbow_locked",
             q_ns=None, return_home=True, target=None, instances=None):
    """Every candidate's verdict, checking the full chain for each (no early stop).

    target: one flag per scene point, true on the object being picked. instances: one SAM 3
    label per point instead, so each candidate's own object is its target and the rest are
    obstacles (planning.candidate_scene). A scene refusal is labelled table or object by the
    height of the nearest voxel. The detected table plane is left on evaluate.table_seen.
    """
    out = []
    table_seen = None
    tgt_pts = None
    labelled = None
    if scene_pts is not None and instances is not None:
        labelled = planning.objects_only(Scene.labelled(scene_pts, voxel, instances), cfg)
        table_seen = getattr(labelled, "table_seen", None)
        scene_pts = None
    if scene_pts is not None:
        whole = Scene.split(scene_pts, voxel, target) if target is not None else Scene(scene_pts, voxel)
        if cfg.table_mode == "plane":
            whole = planning.objects_only(whole, cfg)
            table_seen = whole.table_seen
        scene_pts = whole.points
        tgt_pts = None if whole.target is None else whole.target.points
    evaluate.table_seen = table_seen
    for c in cands:
        scene = _Witness(scene_pts, voxel, cfg.scene_margin) if scene_pts is not None else None
        if scene is not None:
            scene.table_removed = cfg.table_mode == "plane"   # already stripped; plan_legs must not redo it
            if tgt_pts is not None:
                scene.target = Scene(tgt_pts, voxel)
        if labelled is not None:
            cs = planning.candidate_scene(labelled, c)
            scene = _Witness(cs.points, voxel, cfg.scene_margin)
            scene.target, scene.box_target_only, scene.table_removed = cs.target, True, True
        grasp = c["legs"][1][1]
        near = planning.near_target(scene, grasp, c["width_m"], cfg) if scene is not None else None
        legs = list(c["legs"])
        if return_home:
            legs += planning.home_legs(legs[-1][1], home, cfg)
        refused = planning.prescreen(c["legs"], cfg)
        if refused:
            r = {"ok": False, "failed_leg": refused.split(":")[0], "reasons": [refused], "legs": []}
        else:
            r = planning.check_legs(q_start, start_pose, legs, model, limits, cfg, q_ns, None, scene, near)
        reason = (r["reasons"] or [""])[0]
        cat = None if r["ok"] else planning.category(reason)
        what = None
        if cat == "arm vs scene" and scene is not None and scene.hits:
            gap, part, p = min(scene.hits, key=lambda h: h[0])
            up = float(p[2] - cfg.table_z)
            what = {"part": part, "voxel": [round(float(v), 3) for v in p],
                    "voxel_above_table_mm": round(up * 1e3), "from_grasp_mm":
                    round(float(np.linalg.norm(p - np.asarray(grasp[0]))) * 1e3),
                    "is_table": bool(up < voxel * 1.25)}
            cat = "scene: table" if what["is_table"] else "scene: object"
        m = c["grasp_matrix"]
        util = None
        if r["ok"]:
            util = round(c["confidence"] * planning.comfort(r, cfg)[0], 4)
        out.append({"id": c["id"], "confidence": round(c["confidence"], 4), "ok": bool(r["ok"]),
                    "utility": util,
                    "failed_leg": r["failed_leg"], "category": cat, "reason": reason or "ok",
                    "scene_hit": what,
                    "tilt_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, -m[2, 2])))), 1),
                    "tcp_above_table_mm": round((m[2, 3] - cfg.table_z) * 1e3)})
    return out


def choice(rows):
    """(chosen by utility, chosen by confidence): what plan_pick would pick in each rank."""
    ok = [r for r in rows if r["ok"]]
    if not ok:
        return None, None
    return (max(ok, key=lambda r: r["utility"])["id"],
            max(ok, key=lambda r: r["confidence"])["id"])


def summary(rows):
    """Refusal counts by category, and a one-line "k of n feasible" summary."""
    counts = {}
    for r in rows:
        if not r["ok"]:
            counts[r["category"]] = counts.get(r["category"], 0) + 1
    ok = sum(r["ok"] for r in rows)
    parts = ", ".join("%d %s" % (n, k) for k, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    return counts, "%d of %d feasible%s" % (ok, len(rows), "; " + parts if parts else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", nargs="+", help="archived run directories")
    ap.add_argument("--tool-pose", type=float, nargs=7, metavar="V",
                    help="capture tool pose x y z qx qy qz qw (fr3_link0), for a single run")
    ap.add_argument("--start", choices=("home", "capture"), default="home",
                    help="where the pick starts (sequence.pick_from); default home")
    ap.add_argument("--start-q", type=float, nargs=7, metavar="Q",
                    help="joints at the capture pose, for --start capture (default: IK from home)")
    ap.add_argument("--model", choices=("elbow_locked", "nullspace"), default="elbow_locked",
                    help="position control (default) or impedance")
    ap.add_argument("--no-return-home", action="store_true")
    ap.add_argument("--margin", type=float, help="override planner/scene_margin, m")
    ap.add_argument("--voxel-mm", type=float, default=10.0,
                    help="scene voxel size (the pose node's scene_voxel_mm); default 10")
    ap.add_argument("--target-px", type=int, default=20,
                    help="voxels within this many pixels of an object's mask belong to it "
                         "(the pose node's target_voxel_px); 0 = no target split")
    ap.add_argument("--labels", choices=("instances", "flags"), default="instances",
                    help="instances (default): each candidate's own object is its target, other "
                         "objects are obstacles, fingers checked; flags: the older rule, one target "
                         "for all and a keep-out box cleared around the grasp")
    ap.add_argument("--baseline", action="store_true",
                    help="the older planner: table as scene voxels, link1 checked")
    ap.add_argument("--quiet", action="store_true", help="summary lines only")
    ap.add_argument("--json", help="write every verdict here")
    args = ap.parse_args()

    cfg, ee_to_camera, home_q = load_config(args.margin)
    if args.baseline:
        cfg.table_mode, cfg.static_parts = "voxels", ()
    limits = planning.load_limits(str(LIMITS))
    home = pose_of(kinematics.ee_matrix(home_q))
    report = {}
    for run in args.run:
        try:
            grasps, capture, depth, depth_name, K = load_run(run, args.tool_pose)
        except SystemExit as exc:
            if len(args.run) == 1:
                raise
            print("%s: skipped -- %s" % (Path(run).name, exc))
            continue
        if args.start == "capture":
            start_pose = capture
            if args.start_q:
                q0 = list(args.start_q)
            else:
                q0, ep, er, _ = planning.ik_dls(mat(capture), np.array(home_q), args.model,
                                                home_q if args.model == "nullspace" else None, cfg)
                if ep > 1e-3 or er > 1e-2:
                    raise SystemExit("no IK to the capture pose from home; give --start-q")
                q0 = list(q0)
        else:
            q0, start_pose = home_q, home
        labels = label_image(run, 15) if args.labels == "instances" else None
        cands = build(grasps, capture, ee_to_camera, start_pose, K=K, labels=labels)
        pts, vox, flags, inst = None, None, None, None
        if depth is not None:
            cam_pts, vox = scene_voxels(depth, K, voxel_m=args.voxel_mm / 1000.0,
                                        max_voxels=8000 if args.voxel_mm >= 20 else 200000)
            t = mat(capture).dot(mat((ee_to_camera[:3], ee_to_camera[3:7])))
            pts = cam_pts.dot(t[:3, :3].T) + t[:3, 3]
            flags = target_flags(run, cam_pts, K, args.target_px) if args.target_px > 0 else None
            inst = (voxel_labels(run, cam_pts, K, args.target_px)
                    if args.labels == "instances" and args.target_px > 0 else None)
        rows = evaluate(cands, q0, start_pose, home, pts, vox, limits, cfg, args.model,
                        home_q if args.model == "nullspace" else None, not args.no_return_home,
                        None if inst is not None else flags, inst)
        counts, line = summary(rows)
        by_u, by_c = choice(rows)
        if by_u:
            u = {r["id"]: r for r in rows}
            line += "; chooses %s (utility %.3f)%s" % (
                by_u, u[by_u]["utility"], "" if by_u == by_c else
                ", by confidence %s (utility %.3f)" % (by_c, u[by_c]["utility"]))
        seen = getattr(evaluate, "table_seen", None)
        if seen:
            line += "  [table seen %+.0f mm, %.1f deg]" % (seen["offset_mm"], seen["tilt_deg"])
        report[str(run)] = {"capture_tool": [list(capture[0]), list(capture[1])],
                            "start": args.start, "model": args.model,
                            "scene_voxels": 0 if pts is None else len(pts), "depth": depth_name,
                            "summary": counts, "summary_text": line, "candidates": rows}
        print("%s: %s" % (Path(run).name, line))
        if not args.quiet:
            for r in rows:
                extra = ""
                if r["scene_hit"]:
                    h = r["scene_hit"]
                    extra = "  [%s near a voxel %d mm above the table, %d mm from the grasp]" % (
                        h["part"], h["voxel_above_table_mm"], h["from_grasp_mm"])
                print("  %-10s %.3f  tilt %3.0f  tcp %4d mm  %-4s %-16s %s%s" % (
                    r["id"], r["confidence"], r["tilt_deg"], r["tcp_above_table_mm"],
                    "OK" if r["ok"] else "--", r["category"] or ("utility %.3f" % r["utility"]), "" if r["ok"] else
                    "%s: %s" % (r["failed_leg"], r["reason"][:70]), extra))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    from fr3_pose_commander import scene as scene_module
    if scene_module.MESH_ERROR:
        print("WARNING: franka_description's collision meshes could not be read (%s), so the arm was checked as "
              "capsules, which refuses more grasps than the robot did. Run ./setup_franka.sh --ros first."
              % scene_module.MESH_ERROR, file=sys.stderr)


if __name__ == "__main__":
    main()
