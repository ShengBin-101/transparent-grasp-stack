"""pins_node's pure parts: pins.yaml parsing, gripper geometry, captures resolved into base.

    docker exec remake_grasp python3 /opt/remake_grasp/tests/test_pins.py
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros/remake_grasp"))
from remake_grasp import eye_in_hand as eh  # noqa: E402
from remake_grasp.pins_node import FINGER, camera_link_in_ee, capture_to_base, pin_gripper_segments, load_pins, pose_from_entry  # noqa: E402

fail = []


def ck(name, ok, detail=""):
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  -- %s" % (detail,)))
    if not ok:
        fail.append(name)


def close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol)


# --- pose entries
ck("no orientation -> identity", pose_from_entry({"position": [1, 2, 3]}) == ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0)))
_, q = pose_from_entry({"rpy_deg": [180, 0, 0]})
ck("roll 180 flips z", close(eh.rotate(q, (0, 0, 1)), (0, 0, -1)), q)
_, q = pose_from_entry({"rpy_deg": [0, 0, 90]})
ck("yaw 90 turns x to y", close(eh.rotate(q, (1, 0, 0)), (0, 1, 0)), q)
_, q = pose_from_entry({"rpy_deg": [10, 20, 30]})
r, p, y = (math.radians(v) for v in (10, 20, 30))
Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
ck("rpy is fixed-axis roll, pitch, yaw (ROS)", close(eh.matrix_from_quat(q), Rz @ Ry @ Rx), q)
_, q = pose_from_entry({"orientation_xyzw": [0, 0, 2, 2]})
ck("quaternion normalised", close(q, (0, 0, math.sqrt(0.5), math.sqrt(0.5))))

# --- pins.yaml
d = Path(tempfile.mkdtemp())
(d / "p.yaml").write_text("pins:\n  - {name: a, position: [0.4, 0, 0]}\n"
                          "  - {name: b, frame: fr3_EE, shape: gripper, width: 0.06, rpy_deg: [180, 0, 0]}\n")
pins = load_pins(d / "p.yaml")
ck("two pins", [p["name"] for p in pins] == ["a", "b"])
ck("default frame fr3_link0, shape axes", pins[0]["frame"] == "fr3_link0" and pins[0]["shape"] == "axes")
ck("frame and width kept", pins[1]["frame"] == "fr3_EE" and pins[1]["width"] == 0.06)
(d / "e.yaml").write_text("")
ck("empty file -> no pins", load_pins(d / "e.yaml") == [])
(d / "bad.yaml").write_text("pins:\n  - {name: x, shape: cone}\n")
try:
    load_pins(d / "bad.yaml")
    ck("unknown shape rejected", False)
except ValueError:
    ck("unknown shape rejected", True)

# --- gripper geometry: fingertips on the contacts, fingers back along -approach
pose = ((0.5, 0.0, 0.1), pose_from_entry({"rpy_deg": [180, 0, 0]})[1])     # approach straight down
seg = pin_gripper_segments(pose, 0.06)
ck("fingertips at TCP +- width/2 along the jaw", close(seg[0][0], (0.5, 0.03, 0.1)) and close(seg[1][0], (0.5, -0.03, 0.1)),
   seg[:2])
ck("fingers rise above the TCP for a top-down grasp", close(seg[0][1], (0.5, 0.03, 0.1 + FINGER)))

# --- a capture into base: same resolution as the sequencer's goal
tool = ((0.4, 0.0, 0.4), pose_from_entry({"rpy_deg": [180, 0, 0]})[1])
e2c = eh.seven([0.109234, 0.0087, -0.030235, 0.002532, -0.015076, 0.715291, 0.698659], "e2c")
g_cam = {"position": [0.01, -0.02, 0.45], "orientation_xyzw": [0, 0, 0, 1]}
payload = {"frame_id": "camera_color_optical_frame", "stamp": 1.0, "best_index": 3,
           "candidates": [{"index": 3, "width_m": 0.05, "grasp_pose": g_cam},
                          {"index": 5, "width_m": None, "grasp_pose": g_cam}],
           "obstacles": {"frame_id": "camera_color_optical_frame", "voxel_m": 0.01,
                         "points": [[0, 0, 0.5], [0.1, 0, 0.5]], "target": [1, 0]}}
cap = capture_to_base(payload, tool, e2c, "camera_color_optical_frame")
seq = eh.camera_pose_to_base(tool, e2c, ((0.01, -0.02, 0.45), (0, 0, 0, 1)))
ck("grasp in base equals the sequencer's goal (ee_to_grasp identity)",
   close(cap["grasps"][0][1][0], seq[0]) and close(cap["grasps"][0][1][1], seq[1]))
ck("best flagged, index kept", cap["grasps"][0][0] == 3 and cap["grasps"][0][3] and not cap["grasps"][1][3])
ck("missing width -> 0", cap["grasps"][1][2] == 0.0)
cam = eh.compose(tool, e2c)
ck("only target voxels, in base", len(cap["target"]) == 1 and close(cap["target"][0], eh.compose(cam, ((0, 0, 0.5), (0, 0, 0, 1)))[0]))
try:
    capture_to_base({**payload, "frame_id": "fr3_link0"}, tool, e2c, "camera_color_optical_frame")
    ck("wrong frame rejected", False)
except ValueError:
    ck("wrong frame rejected", True)
ck("no obstacles -> no target", capture_to_base({**payload, "obstacles": None}, tool, e2c,
                                                "camera_color_optical_frame")["target"] == [])

# --- the camera mount: driver chain (camera_link -> optical, from the L515's tf_static) composes to ee_to_camera
link_to_opt = eh.compose(((0.004327, 0.000191, 0.013908), eh.normalize((-0.003782, -0.008142, -0.001911, 0.999958))),
                         ((0, 0, 0), (-0.5, 0.5, -0.5, 0.5)))
mount = camera_link_in_ee(e2c, link_to_opt)
got = eh.compose(mount, link_to_opt)
ck("fr3_EE -> camera_link -> optical == ee_to_camera", close(got[0], e2c[0], 1e-9) and
   eh.quat_angle(got[1], e2c[1]) < 1e-9, got)

# --- the shipped pins.yaml parses
ck("config/pins.yaml parses", len(load_pins(ROOT / "ros/remake_grasp/config/pins.yaml")) >= 1)

print("%d failed" % len(fail) if fail else "all passed")
sys.exit(1 if fail else 0)
