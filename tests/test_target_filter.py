"""Tests the grasp-pose node's target association (grasps must land on a target mask) and object labels.

    docker exec remake_grasp python3 /opt/remake_grasp/tests/test_target_filter.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros/remake_grasp"))
from remake_grasp.grasp_pose_node import on_target, target_masks  # noqa: E402

fail = []


def ck(name, ok, detail=""):
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  -- %s" % (detail,)))
    if not ok:
        fail.append(name)


K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
mask = np.zeros((480, 640), bool)
mask[200:280, 300:340] = True                                   # a thin upright object at the centre
d = Path(tempfile.mkdtemp())
ck("no sam3_masks.npy: no filter", target_masks(d, 15) is None)
np.save(d / "sam3_masks.npy", np.zeros((2, 480, 640), np.uint8))
ck("an empty mask stack: no filter", target_masks(d, 15) is None)
np.save(d / "sam3_masks.npy", mask[None].astype(np.uint8))
m = target_masks(d, 15)
centre = np.array([0.0, 0.0, 0.5])                              # projects to (320, 240)
ck("a grasp on the object is on target", on_target([centre], K, m))
beside = np.array([0.0 + 0.030 * 0.5 / 0.6 * 1.0, 0.0, 0.5])    # ~ (350, 240): 10 px off, inside the dilation
ck("10 px beside a thin mask still counts (dilated 15 px)", on_target([beside], K, m))
far = np.array([0.2, 0.0, 0.5])                                  # (560, 240)
ck("a grasp far from any mask is not", not on_target([far], K, m))
ck("... unless one of its contact points lands on it",
   on_target([far, far + np.array([-0.2, 0, 0])], K, m))
ck("points behind the camera are ignored", not on_target([np.array([0, 0, -0.5])], K, m))

run = Path(os.environ.get("RG_RUNS", ROOT / "runs")) / "20260929_223844_auto"
if not (run / "sam3_masks.npy").exists():
    print("  skip  %s: not under RG_RUNS (a rig run of Sep 2026, not in this repository)" % run.name)
if (run / "sam3_masks.npy").exists():
    mr, Kr = target_masks(run, 15), np.load(run / "color_intrinsics.npy")
    off = []
    for j, x in enumerate(json.load(open(run / "grasp_poses.json"))["grasps"]):
        if not x["executable"]:
            continue
        c, y, h = np.array(x["grasp_pose"]["position"]), np.array(x["jaw_axis"]), 0.5 * (x["width_m"] or 0)
        if not on_target([c, c + h * y, c - h * y], Kr, mr):
            off.append(j)
    ck("archived 20260929_223844: the four grasps on the other object are dropped, the mug's kept",
       off == [4, 5, 6, 10], off)

# scene voxels on the target (target_voxel_px) -- the marks the NUC's target_margin reads
from remake_grasp.grasp_pose_node import GraspPoseNode  # noqa: E402
from remake_grasp import sequencer as S  # noqa: E402
run = Path(os.environ.get("RG_RUNS", ROOT / "runs")) / "20260930_020037_auto"
if not (run / "sam3_masks.npy").exists():
    print("  skip  %s: not under RG_RUNS (a rig run of Sep 2026, not in this repository)" % run.name)
if (run / "sam3_masks.npy").exists():
    n = GraspPoseNode.__new__(GraspPoseNode)
    n.live_dir, n.source, n.scene_voxel_m, n.target_voxel_px = run, "sbg", 0.01, 20
    n.graspable_targets = True

    class _L:
        def warning(self, *a, **k): pass
    n.get_logger = lambda: _L()
    ob = n._obstacles("camera_color_optical_frame", None)
    t = ob.get("target") if ob else None
    # replay_plan and this node both dilate by exact distance (scipy); their counts can still differ by a
    # voxel on the boundary (247 against 248 here).
    ck("obstacles: one target mark per voxel, ~247 of 3693 on 20260930_020037 (as replay_plan counts)",
       t is not None and len(t) == len(ob["points"]) and abs(sum(t) - 247) <= 2, (len(ob["points"]) if ob else None, sum(t) if t else None))
    base = S.obstacles_to_base(ob, ((0, 0, 0), (0, 0, 0, 1)), ((0, 0, 0), (0, 0, 0, 1)),
                               "camera_color_optical_frame", "fr3_link0")
    ck("sequencer: the marks travel with their points into the base frame", base["target"] == t)
    inst = ob.get("instance") if ob else None
    ck("obstacles: one object label per voxel too, and its union is the target marks",
       inst is not None and len(inst) == len(t) and [int(v > 0) for v in inst] == list(t), None if inst is None else len(inst))
    ck("sequencer: the labels travel with their points into the base frame", base.get("instance") == inst)
    n.target_voxel_px = 0
    ck("target_voxel_px 0: no marks sent", "target" not in n._obstacles("camera_color_optical_frame", None))

# --- per-object labels: the NUC's keep-out box is the grasped object's only
from remake_grasp.grasp_pose_node import grasp_instance, target_labels  # noqa: E402
d3 = Path(tempfile.mkdtemp())
lab = np.zeros((480, 640), np.int32)
lab[200:280, 300:340] = 1                                       # bottle
lab[200:280, 350:400] = 2                                       # mug, 10 px to its right
np.save(d3 / "sam3_masks.npy", lab)
L = target_labels(d3, 3)
ck("labels: each object keeps its own label", L[240, 320] == 1 and L[240, 375] == 2)
ck("labels: grown into the background, the gap split to the nearer object",
   L[240, 342] == 1 and L[240, 348] == 2 and L[240, 330] == 1, (L[240, 342], L[240, 348]))
ck("labels: nothing beyond the growth", L[240, 420] == 0 and L[100, 100] == 0)
stack = np.zeros((2, 480, 640), np.uint8); stack[0][lab == 1] = 1; stack[1][lab == 2] = 1
np.save(d3 / "sam3_masks.npy", stack)
ck("labels: a (N, H, W) mask stack gives the same labels", (target_labels(d3, 3) == L).all())
z = 0.5
def cam(u, v):
    return [(u - 320) * z / 600.0, (v - 240) * z / 600.0, z]
ck("grasp_instance: TCP and both contacts on the bottle -> 1", grasp_instance([cam(320, 240), cam(305, 240), cam(335, 240)], K, L) == 1)
ck("grasp_instance: a contact on the mug is outvoted by TCP + the other contact",
   grasp_instance([cam(335, 240), cam(320, 240), cam(360, 240)], K, L) == 1)
ck("grasp_instance: a 1-1 split with the TCP on neither goes to the lower label (deterministic)",
   grasp_instance([cam(500, 240), cam(320, 240), cam(375, 240)], K, L) == 1)
ck("grasp_instance: nothing on an object -> 0", grasp_instance([cam(600, 50), cam(610, 50), cam(590, 50)], K, L) == 0)
ck("target_labels: no masks -> None", target_labels(Path(tempfile.mkdtemp()), 3) is None)
wire, _ = S.build_pick_candidates({"frame_id": "camera_color_optical_frame", "stamp": 1.0, "candidates": [
    {"index": 0, "confidence": 0.5, "width_m": 0.05, "instance": 2,
     "grasp_pose": {"position": [0, 0, 0.5], "orientation_xyzw": [0, 0, 0, 1]}}]}, 1.0,
    ((0, 0, 0), (0, 0, 0, 1)), ((0, 0, 0), (0, 0, 0, 1)), ((0, 0, 0), (0, 0, 0, 1)), ((0, 0, 0), (0, 0, 0, 1)),
    0.1, 0.1, True, 0.08, "camera_color_optical_frame")
ck("sequencer: both finger orders of a candidate carry its object", [w["instance"] for w in wire] == [2, 2])

print("\nFAILED: " + ", ".join(fail) if fail else "\nall passed")
sys.exit(1 if fail else 0)
