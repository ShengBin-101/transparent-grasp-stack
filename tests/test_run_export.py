"""run_export: point clouds of an archived run, its grasps and the executed one.

    docker exec remake_grasp python3 /opt/remake_grasp/tests/test_run_export.py
"""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros/remake_grasp"))
from remake_grasp import run_export as X  # noqa: E402

fail = []


def ck(name, ok, detail=""):
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  -- %s" % (detail,)))
    if not ok:
        fail.append(name)


# --- PCD / PLY writers
d = Path(tempfile.mkdtemp())
xyz = np.array([[0.1, -0.2, 0.3], [1.0, 2.0, 3.0]], np.float32)
rgb = np.array([[255, 0, 10], [1, 2, 3]], np.uint8)
X.write_pcd(d / "a.pcd", xyz, rgb)
back, col = X.read_pcd(d / "a.pcd")
ck("PCD round trip: points and packed colours", np.allclose(back, xyz) and (col == rgb).all(), (back, col))
head = (d / "a.pcd").read_bytes()[:300].decode("ascii", "replace")
ck("PCD header: x y z rgb, binary, 2 points", "FIELDS x y z rgb" in head and "POINTS 2" in head and "DATA binary" in head)
X.write_ply(d / "a.ply", xyz, rgb)
raw = (d / "a.ply").read_bytes()
ck("PLY: 2 coloured vertices, 15 bytes each", b"element vertex 2" in raw and len(raw.split(b"end_header\n", 1)[1]) == 30)

# --- calibration by capture date
ck("calibration: before 2026-09-30 03:52 the nominal value", X.calibration_for("20260929_223844_auto")[1] == "nominal")
ck("calibration: after it, the AprilGrid value", X.calibration_for("20261023_005356_auto")[0][0] == 0.109234)

# --- gripper drawing in the grasp frame: fingertips at z = 0, fingers outside +-width/2
g = X.gripper_points(0.06)
ck("gripper: nothing between the fingers' inner faces below the palm",
   not ((np.abs(g[:, 1]) < 0.03 - 1e-6) & (g[:, 2] > -0.058 + 1e-6)).any())
tips = g[np.abs(g[:, 2]) < 1e-9]
ck("gripper: the fingertips are at the TCP (z 0), 30-39 mm either side, nothing past them",
   abs(g[:, 2].max()) < 1e-9 and len(tips) and (np.abs(tips[:, 1]) >= 0.03 - 1e-9).all()
   and (np.abs(tips[:, 1]) <= 0.039 + 1e-9).all())
full = X.gripper_points(0.06, full_hand=True)
ck("full hand: the FR3 hand's 206 mm body behind the fingers",
   abs(full[:, 1].max() - 0.103) < 1e-9 and full[:, 2].min() <= -0.110 + 1e-9)


# --- a synthetic run: a flat 'table' 0.5 m in front of the camera, two grasps, one executed
def make_run(path, pose=True, pick=True, live=True):
    path.mkdir(parents=True)
    K = np.array([[600.0, 0, 64], [0, 600.0, 48], [0, 0, 1]])
    np.save(path / "color_intrinsics.npy", K)
    cv2.imwrite(str(path / "rgb.png"), np.full((96, 128, 3), (10, 20, 30), np.uint8))
    np.save(path / "depth_aligned_mm.npy", np.full((96, 128), 500.0, np.float32))
    np.save(path / "remake_depth_mm.npy", np.full((96, 128), 480.0, np.float32))
    grasp = {"position": [0.0, 0.0, 0.45], "orientation_xyzw": [0, 0, 0, 1]}
    json.dump({"best_index": 0, "grasps": [
        {"grasp_pose": grasp, "width_m": 0.05, "executable": True, "transformed": False},
        {"grasp_pose": grasp, "width_m": 0.09, "executable": False, "transformed": False}]},
        open(path / "grasp_poses.json", "w"))
    tool = ([0.4, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0])            # TCP 0.4 m up, pointing down
    if pose:
        json.dump({"live": live, "position": tool[0], "orientation_xyzw": tool[1], "source": "test"},
                  open(path / "capture_tool.json", "w"))
    if pick:
        legs = [{"name": "standoff", "position": [0.42, 0.0, 0.15], "orientation_xyzw": [1, 0, 0, 0]},
                {"name": "grasp", "position": [0.42, 0.0, 0.05], "orientation_xyzw": [1, 0, 0, 0]}]
        json.dump({"success": True, "capture_tool": [tool[0], tool[1]], "plan_reply": {"chosen": "0:direct"},
                   "candidates_sent": [{"id": "0:flipped", "width_m": 0.05, "legs": legs},
                                       {"id": "0:direct", "width_m": 0.05, "legs": legs}],
                   "message": "standoff: at [0.420 0.000 0.150]; grasp: at [0.421 0.001 0.071] (CONTACT ..."},
                  open(path / "pick_010203.json", "w"))
    return path


root = Path(tempfile.mkdtemp())
r1 = make_run(root / "20261023_010000_auto")
ck("needs_export: a fresh run does", X.needs_export(r1))
info = X.export(r1, stride=4, log=lambda *a: None)
ck("export: all four clouds written", all((r1 / f).exists() for f in
                                           ("cloud_sensor.pcd", "cloud_completed.pcd", "grasps.pcd", "grasp_final.pcd")),
   sorted(p.name for p in r1.iterdir()))
ck("export: in fr3_link0, from capture_tool.json, AprilGrid calibration",
   info["frame"] == "fr3_link0" and "capture_tool.json" in info["pose_source"] and info["calibration"].startswith("AprilGrid"))
pts, _ = X.read_pcd(r1 / "cloud_sensor.pcd")
cam = X.mat([0.4, 0.0, 0.4], [1, 0, 0, 0]).dot(X.mat(info["ee_to_camera"][:3], info["ee_to_camera"][3:]))
expect_z = cam[2, 3] + 0.5 * cam[2, 2]                      # the plane 0.5 m along the optical axis
ck("export: the sensor plane lands where the capture pose puts it", abs(np.median(pts[:, 2]) - expect_z) < 0.01,
   (float(np.median(pts[:, 2])), expect_z))
xyz, col = X.read_pcd(r1 / "grasps.pcd")
n = lambda c: int((col == np.array(X.COLOURS[c], np.uint8)).all(1).sum())
ck("grasps: the best (blue) and a rejected one (grey) drawn, no plain-executable left", n("best") > 0 and n("rejected") > 0
   and n("executable") == 0, (n("best"), n("rejected"), n("executable")))
xyz, col = X.read_pcd(r1 / "grasp_final.pcd")
fin = xyz[(col == np.array(X.COLOURS["final"], np.uint8)).all(1)]
stop = xyz[(col == np.array(X.COLOURS["stopped"], np.uint8)).all(1)]
ck("final: the chosen 0:direct, drawn at its grasp leg (fingertips at z 0.05, hand above)",
   info["final"]["id"] == "0:direct" and abs(fin[:, 2].min() - 0.05) < 1e-4 and fin[:, 2].max() > 0.15,
   (info.get("final"), float(fin[:, 2].min()), float(fin[:, 2].max())))
ck("final: where the arm stopped, read from the pick message", info["final"]["stopped_at"] == [0.421, 0.001, 0.071]
   and np.allclose(stop.mean(0), [0.421, 0.001, 0.071], atol=1e-3))
ck("needs_export: not again once done", not X.needs_export(r1))

r2 = make_run(root / "20261023_020000_auto", pose=False, pick=False)
info = X.export(r2, stride=4, log=lambda *a: None)
ck("no capture pose: camera frame, no final", info["frame"] == "camera_color_optical_frame"
   and not (r2 / "grasp_final.pcd").exists())
r3 = make_run(root / "20261023_030000_auto", live=False)
info = X.export(r3, stride=4, log=lambda *a: None)
ck("capture pose not live: falls back to the pick record's", info["frame"] == "fr3_link0" and info["pose_source"].startswith("pick_"))
r4 = make_run(root / "20261023_040000_auto")
json.dump({"success": False, "plan_reply": {"chosen": None}, "candidates_sent": [], "message": "no grasp the arm can do"},
          open(r4 / "pick_050000.json", "w"))
(r4 / "pick_010203.json").unlink()
info = X.export(r4, stride=4, log=lambda *a: None)
ck("a refused pick: no final grasp to draw", "final" not in info and not (r4 / "grasp_final.pcd").exists())

print("%d failed" % len(fail) if fail else "all passed")
sys.exit(1 if fail else 0)
