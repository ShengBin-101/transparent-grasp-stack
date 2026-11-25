"""Point clouds of an archived run, with its grasps drawn in, for any point-cloud viewer.

    ros2 run remake_grasp run_export RUN_DIR [RUN_DIR ...]   one or more runs
    ros2 run remake_grasp run_export --all                   every run in runs/ that is missing files
    ros2 run remake_grasp run_export RUN_DIR --ply           also write .ply copies (MeshLab, Blender)

Writes into the run directory:
  cloud_sensor.pcd     the L515's depth (depth_aligned_mm.npy), coloured from rgb.png
  cloud_completed.pcd  the completed depth (remake_depth_mm.npy), coloured from rgb.png
  grasps.pcd           the completed cloud plus every grasp candidate as a small gripper:
                       yellow executable, grey rejected, blue the pose node's best
  grasp_final.pcd      once a pick record (pick_*.json) exists: the completed cloud plus the grasp
                       the planner chose and the arm executed (red, full hand), its standoff (orange),
                       the approach between them (white), and where the arm stopped (magenta)
  export.json          frame, ee_to_camera used, point counts, legend

Every file is a plain coloured point cloud (the grippers are drawn as dense points, not meshes),
so CloudCompare, Open3D, pcl_viewer and MeshLab (via --ply) all show it the same way.

THE FRAME. fr3_link0 (the robot base, z up, table at z = -0.041) when the arm's pose at capture is
known -- capture_tool.json (the orchestrator writes it at every capture in newer runs; the
sequencer at every pick in older ones) or a pick record's capture_tool -- else the camera's optical
frame. In fr3_link0 the runs of a scene line up across viewpoints. The hand-eye calibration used
is the one in force when the run was captured (CALIBRATIONS), not today's.

Gripper drawing (grasp frame: z = approach, y = jaw, fingertips at z = 0): fingers 21 x 9 x 58 mm
at the grasp width; the full hand for the final grasp is the FR3 hand's box, 60 x 206 x 52 mm,
58-110 mm behind the fingertips.
"""
import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .common import RUNS_DIR  # noqa: E402
# ee_to_camera in force, by capture time ("YYYYmmdd_HHMMSS" prefix of the run name).
CALIBRATIONS = [
    ("00000000_000000", [0.105, 0.0, -0.048, 0.0, 0.0, 0.7071067811865476, 0.7071067811865476], "nominal"),
    ("20260930_035200", [0.109234, 0.008700, -0.030235, 0.002532, -0.015076, 0.715291, 0.698659],
     "AprilGrid calibration 2026-09-30"),
]
COLOURS = {"executable": (255, 210, 0), "rejected": (130, 130, 130), "best": (40, 120, 255),
           "final": (255, 30, 30), "standoff": (255, 140, 0), "approach": (255, 255, 255),
           "stopped": (255, 0, 255)}
Z_RANGE = (0.15, 1.5)          # camera depth kept, m (the pose node's scene_voxels range)


# ---------------------------------------------------------------- geometry

def quat_to_rot(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat(position, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(*quat_xyzw)
    T[:3, 3] = position
    return T


def calibration_for(run_name):
    """(ee_to_camera, label) in force when `run_name` was captured."""
    stamp = run_name[:15]
    chosen = CALIBRATIONS[0]
    for c in CALIBRATIONS:
        if stamp >= c[0]:
            chosen = c
    return chosen[1], chosen[2]


def backproject(depth_mm, K, rgb, stride=2):
    """-> (points N x 3 camera frame, colours N x 3 uint8) for valid depth within Z_RANGE."""
    d = np.asarray(depth_mm, np.float32)[::stride, ::stride] / 1000.0
    h, w = d.shape
    vv, uu = np.mgrid[0:h, 0:w]
    uu, vv = uu * stride, vv * stride
    ok = np.isfinite(d) & (d > Z_RANGE[0]) & (d < Z_RANGE[1])
    z, u, v = d[ok], uu[ok], vv[ok]
    pts = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    if rgb is not None and rgb.shape[0] > v.max(initial=0) and rgb.shape[1] > u.max(initial=0):
        col = rgb[v, u][:, ::-1]                          # BGR -> RGB
    else:
        col = np.full((len(pts), 3), 200, np.uint8)
    return pts.astype(np.float32), np.ascontiguousarray(col, dtype=np.uint8)


def box_points(lo, hi, step=0.002):
    """Points on the surface of the axis-aligned box [lo, hi] (3-vectors), about `step` apart."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    out = []
    axes = [np.linspace(lo[i], hi[i], max(2, int(math.ceil((hi[i] - lo[i]) / step)) + 1)) for i in range(3)]
    for i in range(3):
        j, k = [a for a in range(3) if a != i]
        A, B = np.meshgrid(axes[j], axes[k], indexing="ij")
        for face in (lo[i], hi[i]):
            p = np.empty((A.size, 3))
            p[:, i], p[:, j], p[:, k] = face, A.ravel(), B.ravel()
            out.append(p)
    return np.concatenate(out)


def gripper_points(width, full_hand=False):
    """A parallel gripper in its grasp frame: fingers ending at the contacts, a palm bar and a stem;
    with `full_hand`, the FR3 hand's body instead of the bar and stem."""
    h = 0.5 * float(width or 0.0)
    parts = [box_points((-0.0105, s * h if s > 0 else -h - 0.009, -0.058),
                        (0.0105, h + 0.009 if s > 0 else -h, 0.0)) for s in (1, -1)]
    if full_hand:
        parts.append(box_points((-0.030, -0.103, -0.110), (0.030, 0.103, -0.058), step=0.004))
    else:
        parts.append(box_points((-0.006, -h - 0.009, -0.066), (0.006, h + 0.009, -0.058)))
        parts.append(box_points((-0.004, -0.004, -0.126), (0.004, 0.004, -0.066)))
    return np.concatenate(parts)


def placed(T, local):
    return local.dot(T[:3, :3].T) + T[:3, 3]


def segment_points(a, b, step=0.002):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.linalg.norm(b - a) / step) + 1)
    return a + np.linspace(0, 1, n)[:, None] * (b - a)


# ---------------------------------------------------------------- writers

def write_pcd(path, xyz, rgb):
    """Binary PCD v0.7, FIELDS x y z rgb (rgb packed in a float, the PCL convention)."""
    xyz = np.ascontiguousarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, np.uint32).reshape(-1, 3)
    packed = ((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]).astype(np.uint32).view(np.float32)
    data = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    data["x"], data["y"], data["z"], data["rgb"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], packed
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\n"
              "TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\n"
              "DATA binary\n" % (len(xyz), len(xyz)))
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def write_ply(path, xyz, rgb):
    """Binary little-endian PLY of coloured points."""
    xyz = np.ascontiguousarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, np.uint8).reshape(-1, 3)
    data = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("r", "u1"), ("g", "u1"), ("b", "u1")])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["r"], data["g"], data["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
              "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
              % len(xyz))
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def read_pcd(path):
    """(xyz, rgb) from a file write_pcd wrote -- for tests and checks."""
    raw = Path(path).read_bytes()
    end = raw.index(b"DATA binary\n") + len(b"DATA binary\n")
    n = int(re.search(rb"POINTS (\d+)", raw[:end]).group(1))
    data = np.frombuffer(raw[end:], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")], count=n)
    packed = data["rgb"].view(np.uint32)
    rgb = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], 1).astype(np.uint8)
    return np.stack([data["x"], data["y"], data["z"]], 1), rgb


# ---------------------------------------------------------------- the run

def capture_pose(run):
    """((position, quat_xyzw), source) of the arm when this run was captured, or (None, why)."""
    ct = run / "capture_tool.json"
    if ct.exists():
        try:
            rec = json.loads(ct.read_text())
            if rec.get("live", True) and rec.get("position") is not None:
                return (rec["position"], rec["orientation_xyzw"]), "capture_tool.json (%s)" % rec.get("source", "?")
        except (OSError, ValueError, KeyError):
            pass
    for p in sorted(run.glob("pick_*.json")):
        try:
            c = json.loads(p.read_text()).get("capture_tool")
        except (OSError, ValueError):
            continue
        if c:
            return (c[0], c[1]), p.name
    return None, "no capture pose recorded"


def final_grasp(run):
    """The last pick's executed grasp: dict(id, legs {name: 4x4 base}, width, stopped_at, success,
    file), or None. The chosen id's legs come from what was SENT to the planner (base frame)."""
    picks = sorted(run.glob("pick_*.json"))
    for p in reversed(picks):
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        reply = rec.get("plan_reply") or {}
        chosen = reply.get("chosen") if isinstance(reply, dict) else None
        cand = next((c for c in rec.get("candidates_sent") or [] if c.get("id") == chosen), None)
        if cand is None:
            continue
        legs = {l["name"]: mat(l["position"], l["orientation_xyzw"]) for l in cand.get("legs", [])}
        m = re.search(r"grasp: at \[([-\d.]+) ([-\d.]+) ([-\d.]+)\]", rec.get("message") or "")
        return {"id": chosen, "legs": legs, "width": cand.get("width_m"), "file": p.name,
                "success": rec.get("success"),
                "stopped_at": [float(v) for v in m.groups()] if m else None}
    return None


def export(run, stride=2, ply=False, final_only=False, log=print):
    """Write the files listed in the module docstring into `run`; returns the export.json dict."""
    run = Path(run)
    import cv2
    rgb = cv2.imread(str(run / "rgb.png"))
    K = np.load(run / "color_intrinsics.npy")
    pose, pose_src = capture_pose(run)
    e2c, cal = calibration_for(run.name)
    if pose is not None:
        T = mat(*pose).dot(mat(e2c[:3], e2c[3:7]))
        frame = "fr3_link0"
    else:
        T, frame = np.eye(4), "camera_color_optical_frame"
    info = {"schema": 1, "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "frame": frame,
            "pose_source": pose_src, "ee_to_camera": e2c if pose is not None else None,
            "calibration": cal if pose is not None else None, "stride": stride, "files": {},
            "legend": {k: list(v) for k, v in COLOURS.items()}}
    old = run / "export.json"
    if final_only and old.exists():
        try:
            info["files"] = json.loads(old.read_text()).get("files", {})
        except ValueError:
            pass

    def save(name, xyz, col):
        write_pcd(run / (name + ".pcd"), xyz, col)
        if ply:
            write_ply(run / (name + ".ply"), xyz, col)
        info["files"][name + ".pcd"] = int(len(xyz))

    clouds = {}
    for name, src in (("cloud_sensor", "depth_aligned_mm.npy"), ("cloud_completed", "remake_depth_mm.npy")):
        if (run / src).exists():
            p, c = backproject(np.load(run / src), K, rgb, stride)
            clouds[name] = (placed(T, p), c)
            if not final_only:
                save(name, *clouds[name])
    base_cloud = clouds.get("cloud_completed") or clouds.get("cloud_sensor")
    if base_cloud is None:
        raise FileNotFoundError("%s has no depth to export" % run)

    gp_path = run / "grasp_poses.json"
    if gp_path.exists() and not final_only:
        gp = json.loads(gp_path.read_text())
        best = gp.get("best_index")
        xs, cs = [base_cloud[0]], [base_cloud[1]]
        for i, g in enumerate(gp.get("grasps", [])):
            G = T.dot(mat(g["grasp_pose"]["position"], g["grasp_pose"]["orientation_xyzw"])) \
                if not g.get("transformed") else mat(g["grasp_pose"]["position"], g["grasp_pose"]["orientation_xyzw"])
            kind = "best" if i == best and g.get("executable") else ("executable" if g.get("executable") else "rejected")
            pts = placed(G, gripper_points(g.get("width_m") or 0.04))
            xs.append(pts)
            cs.append(np.tile(np.array(COLOURS[kind], np.uint8), (len(pts), 1)))
        save("grasps", np.concatenate(xs), np.concatenate(cs))
        info["grasps"] = len(gp.get("grasps", []))

    fin = final_grasp(run) if pose is not None else None
    if fin is not None and "grasp" in fin["legs"]:
        xs, cs = [base_cloud[0]], [base_cloud[1]]

        def add(pts, kind):
            xs.append(pts)
            cs.append(np.tile(np.array(COLOURS[kind], np.uint8), (len(pts), 1)))
        G = fin["legs"]["grasp"]
        add(placed(G, gripper_points(fin["width"] or 0.04, full_hand=True)), "final")
        if "standoff" in fin["legs"]:
            S = fin["legs"]["standoff"]
            add(placed(S, gripper_points(fin["width"] or 0.04)), "standoff")
            add(segment_points(S[:3, 3], G[:3, 3]), "approach")
        if fin["stopped_at"]:
            add(box_points(np.array(fin["stopped_at"]) - 0.006, np.array(fin["stopped_at"]) + 0.006), "stopped")
        save("grasp_final", np.concatenate(xs), np.concatenate(cs))
        info["final"] = {k: fin[k] for k in ("id", "width", "file", "success", "stopped_at")}
    (run / "export.json").write_text(json.dumps(info, indent=1))
    log("%s: %s in %s%s" % (run.name, ", ".join("%s %d pts" % kv for kv in info["files"].items()), frame,
                           "" if pose is not None else " (no capture pose)"))
    return info


def needs_export(run):
    """True when a run lacks its clouds, or has a pick record newer than its final export."""
    if not (run / "rgb.png").exists() or not (run / "color_intrinsics.npy").exists():
        return False
    if not (run / "cloud_sensor.pcd").exists():
        return True
    picks = sorted(run.glob("pick_*.json"))
    fin = run / "grasp_final.pcd"
    return bool(picks) and (not fin.exists() or picks[-1].stat().st_mtime > fin.stat().st_mtime)


def spawn(run, final=False):
    """Export `run` in a separate, low-priority process (the caller never waits): output appended to
    run/export.log. Returns the Popen, or None if it could not start."""
    cmd = [sys.executable, "-m", "remake_grasp.run_export", str(run)] + (["--final"] if final else [])
    try:
        log = open(Path(run) / "export.log", "ab")
        return subprocess.Popen(["nice", "-n", "10"] + cmd, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*", help="run directories")
    ap.add_argument("--all", action="store_true", help="every run in --runs-dir that needs it")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--final", action="store_true", help="only (re)write grasp_final (after a pick)")
    ap.add_argument("--force", action="store_true", help="with --all, re-export every run")
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth pixel (default 2)")
    ap.add_argument("--ply", action="store_true", help="also write .ply copies")
    a = ap.parse_args(argv)
    runs = [Path(r) for r in a.runs]
    if a.all:
        runs += [d for d in sorted(Path(a.runs_dir).iterdir()) if d.is_dir() and (a.force or needs_export(d))]
    if not runs:
        print("nothing to export")
        return 0
    bad = 0
    for r in runs:
        try:
            export(r, a.stride, a.ply, final_only=a.final and (r / "cloud_sensor.pcd").exists())
        except Exception as exc:  # noqa: BLE001 -- one bad run must not stop a backfill
            bad += 1
            print("%s: not exported: %s" % (r.name, exc), file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
