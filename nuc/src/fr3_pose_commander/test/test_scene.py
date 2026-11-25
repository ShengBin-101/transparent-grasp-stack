#!/usr/bin/env python3
"""Tests scene.py (arm model, scene index, labelled scenes) and the planner's arm-vs-scene check.

    python3 src/fr3_pose_commander/test/test_scene.py

No ROS, no robot. scene.CAPSULES, a hand-copied table of the FR3's self-collision capsules,
is re-derived from test/data/fr3_hand.urdf, and the spheres arm_spheres() places are compared
with the URDF's own sc-link frames at random postures.
"""
import math
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
WS_SRC = os.path.dirname(PKG)
sys.path.insert(0, os.path.join(PKG, "src"))
from fr3_pose_commander import kinematics as k  # noqa: E402
from fr3_pose_commander import planning as P  # noqa: E402
from fr3_pose_commander import scene as S  # noqa: E402
from fr3_pose_commander.pose_math import quaternion_from_matrix  # noqa: E402

FAILED = []


def check(label, condition, detail=""):
    print("%-4s %s%s" % ("ok" if condition else "FAIL", label,
                         "  (%s)" % detail if detail else ""))
    if not condition:
        FAILED.append(label)


LIMITS = P.load_limits(os.path.join(
    WS_SRC, "franka_ros", "franka_description", "robots", "fr3", "joint_limits.yaml"))
HOME = [0.164563, -0.222906, -0.123741, -2.13951, -0.0294044, 1.91649, 0.839196]
CAMERA = (0.105, 0.0, -0.048)
CFG = P.PlanConfig(ee_to_camera=CAMERA)

# -- the capsule table against the URDF ---------------------------------------------

root = ET.parse(os.path.join(HERE, "data", "fr3_hand.urdf")).getroot()
BY_CHILD = {j.find("child").get("link"): j for j in root.findall("joint")}


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p),
                              math.cos(y), math.sin(y))
    return np.array(((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                     (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                     (-sp, cp * sr, cp * cr)))


def urdf_fk(q, tip):
    chain, link = [], tip
    while link in BY_CHILD:
        chain.append(BY_CHILD[link])
        link = BY_CHILD[link].find("parent").get("link")
    angles = {"fr3_joint%d" % (i + 1): q[i] for i in range(7)}
    t = np.eye(4)
    for joint in reversed(chain):
        o = joint.find("origin")
        step = np.eye(4)
        step[:3, :3] = rpy_matrix(*[float(v) for v in (o.get("rpy") or "0 0 0").split()])
        step[:3, 3] = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
        t = t.dot(step)
        if joint.get("type") == "revolute":
            c, s = math.cos(angles[joint.get("name")]), math.sin(angles[joint.get("name")])
            spin = np.eye(4)
            spin[:2, :2] = ((c, -s), (s, c))
            t = t.dot(spin)
    return t


def urdf_capsules():
    """{sc link: [(radius, end A, end B)]}: each cylinder's two end spheres."""
    out = {}
    for link in root.findall("link"):
        name = link.get("name")
        if not name.endswith("_sc"):
            continue
        spheres = []
        for c in link.findall("collision"):
            g = c.find("geometry")[0]
            if g.tag == "sphere":
                xyz = tuple(float(v) for v in c.find("origin").get("xyz").split())
                spheres.append((float(g.get("radius")), xyz))
        out[name] = [(spheres[i][0], spheres[i][1], spheres[i + 1][1])
                     for i in range(0, len(spheres), 2)]
    return out


SC_OF = {1: "fr3_link1_sc", 2: "fr3_link2_sc", 3: "fr3_link3_sc", 4: "fr3_link4_sc",
         5: "fr3_link5_sc", 6: "fr3_link6_sc", "7sc": "fr3_link7_sc", "hand": "fr3_hand_sc"}
urdf = urdf_capsules()
table_ok = True
for name, frame, radius, a, b in S.CAPSULES:
    match = [c for c in urdf[SC_OF[frame]]
             if abs(c[0] - radius) < 1e-9 and np.allclose(c[1], a) and np.allclose(c[2], b)]
    table_ok &= len(match) == 1
check("every capsule in scene.CAPSULES is one of the URDF's sc capsules", table_ok)
skipped = [(n, c) for n, cs in urdf.items() for c in cs
           if not any(abs(c[0] - r) < 1e-9 and np.allclose(c[1], a)
                      for _, _, r, a, _ in S.CAPSULES)]
check("only link0 and the hand's finger-region capsule are left out",
      sorted(n for n, _ in skipped) == ["fr3_hand_sc", "fr3_link0_sc"], skipped)

rng = np.random.RandomState(11)
worst = 0.0
LINK_OF = {6: "fr3_link6", 7: "fr3_link7", "hand": "fr3_hand"}
mesh_worst = 0.0
for _ in range(100):
    q = rng.uniform(LIMITS.lower + 0.1, LIMITS.upper - 0.1)
    parts = {n: c for n, c, _ in S.arm_spheres(q)}
    meshed = {n for n, _, _ in S.MESH_PARTS}
    for name, frame, radius, a, b in S.CAPSULES:
        if name.rstrip("b") in meshed or name not in parts:
            continue
        t = urdf_fk(q, SC_OF[frame])
        for end, centre in ((a, parts[name][0]), (b, parts[name][-1])):
            worst = max(worst, float(np.linalg.norm(t[:3, :3].dot(end) + t[:3, 3] - centre)))
    for name, frame, _ in S.MESH_PARTS:
        t = urdf_fk(q, LINK_OF[frame])
        local = S._mesh_parts()[name]
        mesh_worst = max(mesh_worst, float(np.abs(local.dot(t[:3, :3].T) + t[:3, 3] - parts[name]).max()))
check("capsule sphere chains sit on the URDF's own sc-link frames (100 postures)",
      worst < 1e-9, "worst %.1e m" % worst)
check("mesh samples sit on the URDF's own link / hand frames (100 postures)",
      mesh_worst < 1e-9, "worst %.1e m" % mesh_worst)
gaps = max(float(np.linalg.norm(np.diff(c, axis=0), axis=1).max()) / r
           for n, c, r in S.arm_spheres(HOME) if n in ("link1", "link2", "link3", "link4", "link5", "link5b"))
check("spheres along a capsule are no further apart than a third of its radius",
      gaps <= 1.0 / 3.0 + 1e-9, "%.3f" % gaps)

# -- the mesh model: links 6, 7 and the hand ----------------------------------------
check("the collision meshes load (else links 6, 7, hand fall back to capsules)",
      S._mesh_parts() and S.MESH_ERROR is None, S.MESH_ERROR)
names = [n for n, _, _ in S.arm_spheres(HOME)]
check("links 6, 7 and the hand come from their meshes, not their capsules",
      "link6" in names and "link7b" not in names and names.count("hand") == 1)
cover = True
for name, frame, fname in S.MESH_PARTS:
    tris = S._read_stl(os.path.join(S._mesh_dir(), fname))
    w = rng.dirichlet((1, 1, 1), size=(len(tris), 4))
    probes = np.einsum("tki,tij->tkj", w, tris).reshape(-1, 3)
    samples = S._mesh_parts()[name]
    d = np.sqrt(((probes[:, None, :] - samples[None, :, :]) ** 2).sum(-1)).min(1)
    cover &= bool(d.max() <= S.MESH_RADIUS + 1e-9)
check("every point on each mesh surface is within one sample radius (random probes)", cover)
hand_local = S._mesh_parts()["hand"]
check("the hand model ends where the mesh does, 66 mm along the approach (+ the 3 mm "
      "merge shift) -- not the capsule's 110 mm",
      hand_local[:, 2].max() <= 0.066 + S.MESH_SPACING / 8.0 * math.sqrt(3.0),
      "%.4f" % hand_local[:, 2].max())
cam7 = (0.105, 0.0, -0.048, 0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
check("the camera is entirely behind its lens (nothing in front of the optical frame)",
      float((S.CAMERA_LOCAL[:, 2] + S.CAMERA_SPHERE_R).max()) <= 1e-9)

# -- the scene index ----------------------------------------------------------------

pts = rng.uniform(-0.5, 0.5, (3000, 3))
tree = S.Scene(pts, 0.02)
brute = S.Scene(pts, 0.02)
brute._tree = None
queries = rng.uniform(-0.6, 0.6, (500, 3))
check("KD-tree and brute-force nearest distances agree",
      np.allclose(tree.nearest(queries), brute.nearest(queries)))
one = S.Scene([[1.0, 0.0, 0.0]], 0.02)
gap, part = one.clearance([("probe", np.array([[0.0, 0.0, 0.0]]), 0.1)])
check("clearance subtracts the radius and the voxel half-diagonal",
      abs(gap - (1.0 - 0.1 - 0.02 * math.sqrt(3) / 2)) < 1e-12 and part == "probe", gap)
check("an empty scene is infinitely clear",
      S.Scene(np.zeros((0, 3)), 0.02).clearance(S.arm_spheres(HOME))[0] == np.inf)
box = S.Scene([[0, 0, 0], [0.05, 0, 0], [0, 0.5, 0], [0, 0, -0.2]], 0.02)
kept = box.without_box(np.eye(4), (0.06, 0.06), (-0.1, 0.1)).points
check("without_box removes the voxels inside the box and keeps the rest",
      len(kept) == 2 and any(np.allclose(p, [0, 0.5, 0]) for p in kept))

# -- the planner's scene check --------------------------------------------------------


def pose_of(m):
    return (tuple(m[:3, 3]), quaternion_from_matrix(m))


def cube(centre, half, step=0.02):
    g = np.arange(-half, half + 1e-9, step)
    return np.array([(centre[0] + x, centre[1] + y, centre[2] + z)
                     for x in g for y in g for z in g])


T0 = k.ee_matrix(HOME)
START = pose_of(T0)
down = T0.copy()
down[2, 3] -= 0.15
DOWN = pose_of(down)
# Where the camera ends up after the descent: a 6 cm cube of obstacle right there.
_, _, _, _, ee_end = k.forward(P.ik_dls(down, HOME, "elbow_locked", cfg=CFG)[0])
camera_end = ee_end[:3, :3].dot(CAMERA) + ee_end[:3, 3]
blocked = S.Scene(cube(camera_end, 0.03), 0.02)
r = P.check_legs(HOME, START, [("standoff", DOWN)], "elbow_locked", LIMITS, CFG, scene=blocked)
check("a leg driving the camera into an obstacle is refused, naming the part",
      r["ok"] is False and "scene" in r["reasons"][0] and r["reasons"][0].startswith("camera"),
      r["reasons"])
clear = S.Scene(cube(camera_end + np.array([0.0, 0.4, 0.0]), 0.03), 0.02)
r = P.check_legs(HOME, START, [("standoff", DOWN)], "elbow_locked", LIMITS, CFG, scene=clear)
check("the same obstacle 40 cm to the side lets the leg through", r["ok"], r["reasons"])
check("... and its clearance is recorded",
      r["legs"][0]["min_scene_clearance"] is not None and r["legs"][0]["min_scene_clearance"] > 0.1)

# A target: a 4 cm object right between the fingers at the grasp pose.
target = S.Scene(cube(down[:3, 3], 0.02), 0.02)
standoff = T0.copy()
standoff[2, 3] = down[2, 3] + 0.10
legs = [("standoff", pose_of(standoff)), ("grasp", DOWN), ("retreat", pose_of(standoff))]
r = P.check_legs(HOME, START, legs, "elbow_locked", LIMITS, CFG, scene=target)
check("without the target exclusion the hand body is refused on its own target",
      r["ok"] is False and r["failed_leg"] == "grasp", r["reasons"])
near = P.near_target(target, DOWN, 0.04, CFG)
r = P.check_legs(HOME, START, legs, "elbow_locked", LIMITS, CFG, scene=target,
                 scene_near_target=near)
check("with it the grasp goes through: the gripper's own volume is the target's",
      r["ok"], r["reasons"])
cands = [{"id": "into-obstacle", "confidence": 0.9,
          "legs": [("standoff", DOWN), ("grasp", DOWN)], "width_m": 0.04},
         {"id": "clear", "confidence": 0.8, "legs": legs, "width_m": 0.04}]
plan = P.plan_legs(HOME, START, cands, "elbow_locked", LIMITS, CFG, scene=blocked)
check("plan_legs refuses a colliding candidate and says why in its entry",
      plan["candidates"][0]["ok"] is False
      and "of the scene" in plan["candidates"][0]["reasons"][0]
      and plan["scene_voxels"] == len(blocked), plan["candidates"][0]["reasons"])

# -- ground truth: a grasp the arm executed without touching anything ---------------
import json  # noqa: E402
gt = json.load(open(os.path.join(HERE, "data", "gt_no_contact_20260925_214205.json")))
gt_scene = S.Scene(np.asarray(gt["obstacles"]["points"]), gt["obstacles"]["voxel_m"])
gt_start = (tuple(gt["start_tool"][0]), tuple(gt["start_tool"][1]))
gt_legs = [(leg["name"], (tuple(leg["position"]), tuple(leg["orientation_xyzw"])))
           for leg in gt["candidate"]["legs"]]
gt_cfg = P.PlanConfig(ee_to_camera=cam7)
gt_near = P.near_target(gt_scene, dict(gt_legs)["grasp"], gt["candidate"]["width_m"], gt_cfg)
for model in ("nullspace", "elbow_locked"):
    r = P.check_legs(HOME, gt_start, gt_legs, model, LIMITS, gt_cfg, q_ns=HOME,
                     scene=gt_scene, scene_near_target=gt_near)
    check("%s: the pick executed with no contact on 2026-09-26 passes the scene check" % model,
          r["ok"], r["reasons"])

# -- runtime --------------------------------------------------------------------------

scene_big = S.Scene(np.vstack([cube(camera_end + np.array([0.0, 0.4, 0.0]), 0.1),
                               rng.uniform((0.2, -0.4, -0.04), (0.8, 0.4, -0.03), (1500, 3))]),
                    0.02)
t0 = time.time()
for model in ("nullspace", "elbow_locked"):
    for _ in range(8):
        P.check_legs(HOME, START, legs + [("home", START)], model, LIMITS, CFG, q_ns=HOME,
                     scene=scene_big, scene_near_target=P.near_target(scene_big, DOWN, 0.04, CFG))
elapsed = time.time() - t0
check("8 full picks per model against a %d-voxel scene in under 10 s here" % len(scene_big),
      elapsed < 10.0, "%.2f s for 16 picks" % elapsed)

# -- the table as a plane ---------------------------------------------------------------
# A grasp 45 deg off vertical, TCP 20 mm above the table: the hand's heel goes below it.
tilt = math.radians(45)
Rx = np.array(((1, 0, 0), (0, math.cos(-tilt), -math.sin(-tilt)), (0, math.sin(-tilt), math.cos(-tilt))))
low = np.eye(4)
low[:3, :3] = T0[:3, :3].dot(Rx)
low[:3, 3] = (0.50, 0.0, CFG.table_z + 0.020)
plane = P.PlanConfig(ee_to_camera=CAMERA)
r = P.check_legs(HOME, START, [("grasp", pose_of(low))], "elbow_locked", LIMITS, plane)
check("plane: a tilted grasp whose hand goes into the table is refused, naming the table",
      r["ok"] is False and "table" in r["reasons"][0] and P.category(r["reasons"][0]) == "arm vs table",
      str(r["reasons"]))
old = P.PlanConfig(ee_to_camera=CAMERA, table_mode="voxels")
r_old = P.check_legs(HOME, START, [("grasp", pose_of(low))], "elbow_locked", LIMITS, old)
check("... which the old model, with no scene, let through (the hand was never checked "
      "against the table)", r_old["ok"] is True, str(r_old["reasons"]))
high = low.copy()
high[:3, :3] = T0[:3, :3]
high[:3, 3] = (0.50, 0.0, CFG.table_z + 0.020)
r = P.check_legs(HOME, START, [("grasp", pose_of(high))], "elbow_locked", LIMITS, plane)
check("plane: the same point from straight above passes (fingers are SBG's, not the plane's)",
      r["ok"] is True, str(r["reasons"]))

# link1 sits on the table: a table voxel beside the base must not refuse anything.
base_ring = np.array([[0.12 * math.cos(a), 0.12 * math.sin(a), CFG.table_z + 0.03]
                      for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)])
ring = S.Scene(base_ring, 0.02)
ring.table_removed = True                      # keep them: this tests static_parts, not the band
r = P.check_legs(HOME, START, [("down", pose_of(high))], "elbow_locked", LIMITS, plane, scene=ring)
check("static parts: voxels hugging the base do not refuse (link1 is not checked)",
      r["ok"] is True, str(r["reasons"]))
with_link1 = P.PlanConfig(ee_to_camera=CAMERA, static_parts=())
r = P.check_legs(HOME, START, [("down", pose_of(high))], "elbow_locked", LIMITS, with_link1, scene=ring)
check("... and with link1 checked, the same scene refuses every leg (runs/20260929_211103_auto)",
      r["ok"] is False and "link1" in r["reasons"][0], str(r["reasons"]))

# objects_only: a table seen 8 mm high, tilted 1.5 deg, with a 6 cm box standing on it.
rng = np.random.RandomState(3)
xy = rng.uniform((0.3, -0.3), (0.8, 0.3), (1500, 2))
slope = math.tan(math.radians(1.5))
tab = np.c_[xy, CFG.table_z + 0.008 + slope * (xy[:, 0] - 0.55) + rng.normal(0, 0.003, len(xy))]
box = np.array([[x, y, CFG.table_z + 0.008 + zz] for x in np.arange(0.50, 0.56, 0.02)
                for y in np.arange(-0.03, 0.03, 0.02) for zz in np.arange(0.01, 0.07, 0.02)])
sc = S.Scene(np.r_[tab, box], 0.02)
only = P.objects_only(sc, plane)
seen = only.table_seen
check("objects_only: the table is found where the depth put it, not where table_z says",
      seen is not None and abs(seen["offset_mm"] - 8.0) < 3.0 and abs(seen["tilt_deg"] - 1.5) < 0.5,
      seen)
kept_box = sum(1 for p in only.points if 0.49 < p[0] < 0.57 and abs(p[1]) < 0.04)
check("... the table goes, the box above the band stays",
      len(only) < len(sc) and not any(abs(p[2] - (CFG.table_z + 0.008)) < 0.004 and not
      (0.49 < p[0] < 0.57 and abs(p[1]) < 0.04) for p in only.points) and kept_box >= 9,
      str((len(only), kept_box)))
check("... and a scene already stripped is left alone", P.objects_only(only, plane) is only)
check("objects_only: table_mode voxels keeps the table in the scene",
      P.objects_only(sc, old) is sc)

# the reply's summary
cs, text = P.summarise([{"ok": True}, {"ok": False, "reasons": ["hand goes 3 mm into the table at 9%"]},
                        {"ok": False, "reasons": ["the leg turns the wrist 120 deg, past the 90 deg per-command limit"]},
                        {"ok": False, "reasons": ["the leg turns the wrist 99 deg, past the 90 deg per-command limit"]}])
check("summary: one line a person can read", text == "1 of 4 feasible; 2 wrist turn, 1 arm vs table", text)

# -- the target's own voxels -------------------------------------------------------------
g = np.eye(4); g[:3, 3] = (0.5, 0.0, 0.05)
inside = np.array([[0.5, 0.0, 0.05], [0.5, 0.01, 0.06]])           # in the gripper's own volume
beyond = np.array([[0.5, 0.0, 0.20]])                              # 15 cm behind the TCP
only_target = S.Scene.split(np.r_[inside, beyond], 0.01, [True, True, True])
near = P.near_target(only_target, pose_of(g), 0.04, CFG)
check("near_target: a scene that is ALL target still loses the gripper's own volume",
      len(near) == 0 and near.target is not None and len(near.target) == 1, len(near.target) if near.target else None)
mixed = S.Scene.split(np.r_[inside, beyond], 0.01, [True, False, True])
near = P.near_target(mixed, pose_of(g), 0.04, CFG)
check("near_target: ... and so do both parts of a mixed one", len(near) == 0 and len(near.target) == 1)
check("split: the flags divide the points", len(mixed) == 1 and len(mixed.target) == 2)
t_only = P.objects_only(S.Scene.split(np.r_[tab[:300], box], 0.02, [False] * 300 + [True] * len(box)), plane)
check("objects_only: the table goes from both parts, the target box stays",
      len(t_only) == 0 and t_only.target is not None and len(t_only.target) >= 9, str((len(t_only), t_only.target and len(t_only.target))))
far_box = S.Scene(np.array([[0.0, 0.0, 5.0]]), 0.02, target=S.Scene(np.array([[0.0, 0.0, 5.0]]), 0.02))
r = P.check_legs(HOME, START, [("down", pose_of(high))], "elbow_locked", LIMITS, plane, scene=far_box)
check("check_leg: far target and scene pass, and the target clearance is recorded",
      r["ok"] is True and r["legs"][0]["min_target_clearance"] is not None, str(r["reasons"]))
hug = S.Scene(np.zeros((0, 3)), 0.02, target=S.Scene(np.array([k.ee_matrix(HOME)[:3, 3] + [0, 0, 0.12]]), 0.02))
hug.table_removed = True
r = P.check_legs(HOME, START, [("down", pose_of(high))], "elbow_locked", LIMITS, plane, scene=hug)
check("check_leg: a target voxel inside the hand is refused as 'arm vs target', even with no other voxels",
      r["ok"] is False and P.category(r["reasons"][0]) == "arm vs target", str(r["reasons"]))

# -- labelled scenes: the keep-out box is the grasped object's only -------------------------
# A top-down grasp 60 mm wide on object 1, with object 2 -- a handle, say -- standing 54 mm
# along the jaw from the TCP: inside the old keep-out box (60/2 + 30 mm), 3 mm outside the
# open finger's outer face (78/2 + 12 mm).
G = T0.copy()
G[:3, 3] = (0.50, 0.0, CFG.table_z + 0.060)
jaw, app = G[:3, 1], G[:3, 2]
# object 1 reaches back toward the palm (so it matters whether it is the target); object 2
# stands at fingertip height, 0-20 mm above the TCP, where only a finger reaches it
obj1 = np.array([G[:3, 3] + a * jaw + b * G[:3, 0] + c * app
                 for a in (-0.01, 0.0, 0.01) for b in (-0.01, 0.0, 0.01) for c in (-0.05, -0.03, -0.01, 0.01)])
obj2 = np.array([G[:3, 3] + 0.054 * jaw + dz * np.array([0, 0, 1.0]) for dz in (0.0, 0.01, 0.02)])
above = G.copy()
above[:3, 3] -= 0.10 * app
legs = [("standoff", pose_of(above)), ("grasp", pose_of(G))]
lab_cfg = P.PlanConfig(ee_to_camera=CAMERA)
pts12 = np.r_[obj1, obj2]
labels12 = np.r_[np.ones(len(obj1), int), 2 * np.ones(len(obj2), int)]

def cand(instance, width=0.06):
    return {"id": "g", "confidence": 0.5, "legs": legs, "width_m": width, "instance": instance}

flags = S.Scene.split(pts12, 0.01, np.ones(len(pts12), bool))
flags.table_removed = True
r_old = P.plan_legs(HOME, START, [cand(None)], "elbow_locked", LIMITS, lab_cfg, scene=flags)
check("labels: with one union target (before), the neighbour inside the keep-out box is removed "
      "and the grasp passes -- the bug", r_old["chosen"] == "g", str(r_old["candidates"][0].get("reasons")))
lab = S.Scene.labelled(pts12, 0.01, labels12)
lab.table_removed = True
r = P.plan_legs(HOME, START, [cand(1)], "elbow_locked", LIMITS, lab_cfg, scene=lab)
check("labels: per object, the same grasp is refused by the FINGER against the neighbour",
      r["chosen"] is None and "finger" in r["candidates"][0]["reasons"][0]
      and r["scene_labels"] == "instances", str(r["candidates"][0].get("reasons")))
alone = S.Scene.labelled(obj1, 0.01, np.ones(len(obj1), int))
alone.table_removed = True
r = P.plan_legs(HOME, START, [cand(1)], "elbow_locked", LIMITS, lab_cfg, scene=alone)
check("labels: without the neighbour it passes -- the grasped object between the fingers is not an obstacle",
      r["chosen"] == "g", str(r["candidates"][0].get("reasons")))
far = S.Scene.labelled(np.r_[obj1, obj2 + 0.25 * jaw], 0.01, labels12)
far.table_removed = True
r = P.plan_legs(HOME, START, [cand(1)], "elbow_locked", LIMITS, lab_cfg, scene=far)
check("labels: with the neighbour 25 cm further along the jaw it passes", r["chosen"] == "g",
      str(r["candidates"][0].get("reasons")))
r = P.plan_legs(HOME, START, [cand(None)], "elbow_locked", LIMITS, lab_cfg, scene=alone)
check("labels: a candidate naming no object has no target -- its object stays an obstacle, refused",
      r["chosen"] is None, str(r["candidates"][0].get("reasons")))
nofing = P.PlanConfig(ee_to_camera=CAMERA, finger_check=False)
r = P.plan_legs(HOME, START, [cand(1)], "elbow_locked", LIMITS, nofing, scene=lab)
# The hand mesh reaches to 20 mm behind the fingertips, +-100 mm along the jaw, so with the
# fingers off it still refuses the neighbour; the finger spheres add the last 20 mm.
check("labels: finger_check off, the hand mesh still refuses the neighbour once the box keeps it",
      r["chosen"] is None and "hand" in r["candidates"][0]["reasons"][0], str(r["candidates"][0].get("reasons")))
fi = lab.for_instance(2)
check("labels: for_instance splits target / scene by label, and marks the box target-only",
      len(fi.target) == len(obj2) and len(fi) == len(obj1) and fi.box_target_only, "%d / %d" % (len(fi), len(fi.target)))
# A handle from fingertip height up past the palm: the hand alone refuses it once the box keeps it.
tall = np.array([G[:3, 3] + 0.054 * jaw + dz * np.array([0, 0, 1.0]) for dz in (0.0, 0.02, 0.04, 0.06, 0.08)])
tl = S.Scene.labelled(np.r_[obj1, tall], 0.01, np.r_[np.ones(len(obj1), int), 2 * np.ones(len(tall), int)])
tl.table_removed = True
r = P.plan_legs(HOME, START, [cand(1)], "elbow_locked", LIMITS, nofing, scene=tl)
check("labels: a handle reaching up past the palm is refused by the hand alone once the box keeps it",
      r["chosen"] is None and "hand" in r["candidates"][0]["reasons"][0], str(r["candidates"][0].get("reasons")))
boxed = P.near_target(lab.for_instance(1), pose_of(G), 0.06, lab_cfg)
check("labels: the keep-out box removes the grasped object's voxels and keeps the neighbour's",
      (boxed.target is None or len(boxed.target) == 0) and len(boxed) == len(obj2),
      "target %s, scene %d" % (None if boxed.target is None else len(boxed.target), len(boxed)))
fl = S.finger_spheres(HOME, 0.078)[0][1]
local = (fl - T0[:3, 3]).dot(T0[:3, :3])
check("finger_spheres: two fingers straddling the TCP along its y, behind it along z",
      abs(abs(local[:, 1]).min() - 0.044) < 1e-6 and local[:, 2].max() < 0 and local[:, 2].min() > -0.06
      and (local[:, 1] > 0).sum() == (local[:, 1] < 0).sum(), "y %.3f..%.3f z %.3f..%.3f" % (
          local[:, 1].min(), local[:, 1].max(), local[:, 2].min(), local[:, 2].max()))
tab = S.Scene.labelled(np.r_[pts12, [[0.3, 0.2, CFG.table_z + 0.001]]], 0.01, np.r_[labels12, [0]])
kept = P.objects_only(tab, lab_cfg)
check("labels: objects_only drops the table and keeps every remaining point's label",
      kept.instance is not None and len(kept.instance) == len(kept) and 0 not in set(kept.instance.tolist()),
      str(None if kept.instance is None else sorted(set(kept.instance.tolist()))))
import json as _json
msg = _json.dumps({"obstacles": {"frame_id": "fr3_link0", "voxel_m": 0.01, "points": pts12.tolist(),
                                 "target": [1] * len(pts12), "instance": labels12.tolist()}})
sc = P.parse_obstacles(msg, "fr3_link0")
check("parse_obstacles: 'instance' wins over 'target' and gives a labelled scene",
      sc.instance is not None and sorted(set(sc.instance.tolist())) == [1, 2] and sc.target is None)
try:
    P.parse_obstacles(_json.dumps({"obstacles": {"frame_id": "fr3_link0", "voxel_m": 0.01,
                                                 "points": pts12.tolist(), "instance": [1]}}), "fr3_link0")
    check("parse_obstacles: a label count that does not match the points is refused", False)
except ValueError:
    check("parse_obstacles: a label count that does not match the points is refused", True)

if FAILED:
    print("\n%d FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("\nall scene checks passed")
