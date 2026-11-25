#!/usr/bin/env python3
"""Tests planning.py (IK, leg checks, candidate choice) and the forward kinematics under it.

    python3 src/fr3_pose_commander/test/test_planning.py

No ROS, no robot. The forward kinematics is checked against test/data/fr3_hand.urdf (from
franka_description: xacro fr3.urdf.xacro hand:=true, franka_ros 0.10.1), an independent
xyz/rpy parameterisation of the modified-DH chain; every other check builds on it.
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
from fr3_pose_commander.pose_math import quaternion_from_matrix  # noqa: E402

FAILED = []


def check(label, condition, detail=""):
    print("%-4s %s%s" % ("ok" if condition else "FAIL", label,
                         "  (%s)" % detail if detail else ""))
    if not condition:
        FAILED.append(label)


LIMITS = P.load_limits(os.path.join(
    WS_SRC, "franka_ros", "franka_description", "robots", "fr3", "joint_limits.yaml"))
# The capture pose, from config/home_pose.yaml.
HOME = [0.164563, -0.222906, -0.123741, -2.13951, -0.0294044, 1.91649, 0.839196]
CAMERA = (0.105, 0.0, -0.048)      # config/camera_extrinsics.yaml, EE frame
CFG = P.PlanConfig(ee_to_camera=CAMERA)


def pose_of(m):
    return (tuple(m[:3, 3]), quaternion_from_matrix(m))


def moved(pose, dx=0.0, dy=0.0, dz=0.0, roll=0.0):
    """`pose` translated in the base frame and rolled about its own z."""
    m = P.matrix_from_pose(pose)
    c, s = math.cos(roll), math.sin(roll)
    m[:3, :3] = m[:3, :3].dot(np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))))
    m[:3, 3] += (dx, dy, dz)
    return pose_of(m)


HOME_POSE = pose_of(k.ee_matrix(HOME))
rng = np.random.RandomState(7)


def random_q():
    return rng.uniform(LIMITS.lower + 0.1, LIMITS.upper - 0.1)


# -- limits and model selection ---------------------------------------------------

check("joint limits come from franka_description",
      abs(LIMITS.upper[3] + 0.4461) < 1e-9 and abs(LIMITS.lower[5] - 0.8521) < 1e-9
      and abs(LIMITS.velocity[1] - 1.0) < 1e-9)
check("position controller topic -> elbow_locked",
      P.model_for_topic("/cartesian_pose_tracking_controller/target_pose") == "elbow_locked")
check("impedance controller topic -> nullspace",
      P.model_for_topic("/cartesian_impedance_example_controller/equilibrium_pose")
      == "nullspace")

# -- forward kinematics against the URDF --------------------------------------------

root = ET.parse(os.path.join(HERE, "data", "fr3_hand.urdf")).getroot()
BY_CHILD = {j.find("child").get("link"): j for j in root.findall("joint")}


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p),
                              math.cos(y), math.sin(y))
    return np.array(((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                     (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                     (-sp, cp * sr, cp * cr)))


def urdf_fk(q, tip="fr3_hand_tcp"):
    chain, link = [], tip
    while link in BY_CHILD:
        joint = BY_CHILD[link]
        chain.append(joint)
        link = joint.find("parent").get("link")
    angles = {"fr3_joint%d" % (i + 1): q[i] for i in range(7)}
    t = np.eye(4)
    for joint in reversed(chain):
        origin = joint.find("origin")
        step = np.eye(4)
        step[:3, :3] = rpy_matrix(*[float(v) for v in (origin.get("rpy") or "0 0 0").split()])
        step[:3, 3] = [float(v) for v in (origin.get("xyz") or "0 0 0").split()]
        t = t.dot(step)
        if joint.get("type") == "revolute":
            c, s = math.cos(angles[joint.get("name")]), math.sin(angles[joint.get("name")])
            spin = np.eye(4)
            spin[:2, :2] = ((c, -s), (s, c))
            t = t.dot(spin)
    return t


worst_p = worst_r = 0.0
for _ in range(500):
    q = random_q()
    a, b = k.ee_matrix(q), urdf_fk(q)
    worst_p = max(worst_p, float(np.linalg.norm(a[:3, 3] - b[:3, 3])))
    worst_r = max(worst_r, math.degrees(math.acos(
        max(-1.0, min(1.0, (np.trace(a[:3, :3].T.dot(b[:3, :3])) - 1) / 2)))))
check("ee_matrix matches the URDF's fr3_hand_tcp over 500 postures",
      worst_p < 5e-4 and worst_r < 0.1, "worst %.1e m, %.1e deg" % (worst_p, worst_r))
flange = k.forward(HOME)[3]
check("flange matches the URDF's fr3_link8",
      np.allclose(flange, urdf_fk(HOME, "fr3_link8"), atol=1e-9))
cam = k.link_points(HOME, CAMERA)["camera"]
ee = k.ee_matrix(HOME)
check("camera point is ee_to_camera applied in the EE frame",
      np.allclose(cam, ee[:3, 3] + ee[:3, :3].dot(CAMERA)))

# -- inverse kinematics ----------------------------------------------------------------

counts = {"elbow_locked": 0, "nullspace": 0}
q3_exact = True
worst = 0.0
N = 500
for _ in range(N):
    q_true = random_q()
    target = k.ee_matrix(q_true)
    seed = q_true + rng.uniform(-0.05, 0.05, 7)
    seed[2] = q_true[2]                  # the locked joint must admit q_true
    for model in counts:
        q, ep, er, _ = P.ik_dls(target, seed, model, cfg=CFG)
        if ep < CFG.converge_m and er < CFG.converge_rad:
            counts[model] += 1
            worst = max(worst, ep)
        if model == "elbow_locked" and q[2] != seed[2]:
            q3_exact = False
check("elbow_locked IK converges from a nearby seed (>= 98 percent)",
      counts["elbow_locked"] >= 0.98 * N, "%d/%d" % (counts["elbow_locked"], N))
check("nullspace IK converges from a nearby seed (>= 98 percent)",
      counts["nullspace"] >= 0.98 * N, "%d/%d" % (counts["nullspace"], N))
check("elbow_locked leaves joint 3 EXACTLY where it was", q3_exact)

target = k.ee_matrix(HOME)
off = np.array(HOME) + np.array((0.3, 0, -0.3, 0, 0.3, 0, 0))
free, _, _, _ = P.ik_dls(target, off, "nullspace", cfg=CFG)
pulled, ep, er, _ = P.ik_dls(target, off, "nullspace", q_ns=HOME, cfg=CFG)
check("the nullspace pull moves the posture toward q_ns",
      np.linalg.norm(pulled - HOME) < 0.5 * np.linalg.norm(free - HOME) and ep < 1e-4,
      "no pull %.3f rad, pulled %.3f rad from q_ns"
      % (np.linalg.norm(free - HOME), np.linalg.norm(pulled - HOME)))

# -- single legs ----------------------------------------------------------------------

for model in ("nullspace", "elbow_locked"):
    r = P.check_legs(HOME, HOME_POSE, [("down", moved(HOME_POSE, dz=-0.2)),
                                       ("home", HOME_POSE)], model, LIMITS, CFG, q_ns=HOME)
    check("%s: 20 cm straight down and back passes" % model, r["ok"], r["reasons"])

beyond = moved(HOME_POSE, dx=0.5)
r = P.check_legs(HOME, HOME_POSE, [("reach", beyond)], "nullspace", LIMITS, CFG, q_ns=HOME)
# Near full reach joint 4 passes max_step_dq before the pose is unreachable; either reason must say reach.
check("a goal beyond reach is refused, and the reason says reach",
      r["ok"] is False and ("no joint solution" in r["reasons"][0]
                            or "edge of reach" in r["reasons"][0]), r["reasons"])

# Joint 7 at home is 0.839 of a +/-2.690 range: a +120 deg tool roll needs
# ~2.93, and the same grasp with the fingers swapped is a -60 deg roll.
plus = P.check_legs(HOME, HOME_POSE, [("roll", moved(HOME_POSE, roll=math.radians(120)))],
                    "elbow_locked", LIMITS, CFG)
minus = P.check_legs(HOME, HOME_POSE, [("roll", moved(HOME_POSE, roll=math.radians(-60)))],
                     "elbow_locked", LIMITS, CFG)
check("a 120 deg roll runs joint 7 into its limit",
      plus["ok"] is False and plus["reasons"][0].startswith("joint 7"), plus["reasons"])
check("... the same grasp with the fingers swapped (-60 deg) passes",
      minus["ok"] is True, minus["reasons"])

# Pointing down, the camera sits 48 mm above the TCP: a TCP on the table clears a 30 mm
# margin, and a wider margin is refused naming the camera.
tcp_z = HOME_POSE[0][2]
to_table = moved(HOME_POSE, dz=CFG.table_z - tcp_z)
r = P.check_legs(HOME, HOME_POSE, [("table", to_table)], "nullspace", LIMITS, CFG, q_ns=HOME)
check("TCP on the table: camera 48 mm up clears a 30 mm margin", r["ok"], r["reasons"])
wide = P.PlanConfig(ee_to_camera=CAMERA, clearance=0.06)
r = P.check_legs(HOME, HOME_POSE, [("table", to_table)], "nullspace", LIMITS, wide, q_ns=HOME)
check("... and a 60 mm margin refuses it, naming the camera",
      r["ok"] is False and r["reasons"][0].startswith("camera comes within"), r["reasons"])
r = P.check_legs(HOME, HOME_POSE, [("above", moved(HOME_POSE, dz=CFG.table_z - tcp_z + 0.12))],
                 "nullspace", LIMITS, CFG, q_ns=HOME)
check("TCP 12 cm above the table clears everything", r["ok"], r["reasons"])

fast = P.PlanConfig(ee_to_camera=CAMERA, linear_speed=5.0, angular_speed=20.0)
leg = [("down", moved(HOME_POSE, dz=-0.2))]
r_pos = P.check_legs(HOME, HOME_POSE, leg, "elbow_locked", LIMITS, fast)
r_imp = P.check_legs(HOME, HOME_POSE, leg, "nullspace", LIMITS, fast, q_ns=HOME)
check("joint speed over the limit refuses under position control",
      r_pos["ok"] is False and "joint speed" in r_pos["reasons"][0], r_pos["reasons"])
check("... and is only recorded under impedance control",
      r_imp["ok"] is True and r_imp["legs"][0]["max_speed_frac"] > fast.max_speed_frac)

jumpy = P.PlanConfig(ee_to_camera=CAMERA, max_step_dq=1e-4)
r = P.check_legs(HOME, HOME_POSE, leg, "elbow_locked", LIMITS, jumpy)
check("a joint step over max_step_dq is reported as a posture flip",
      r["ok"] is False and "posture flip" in r["reasons"][0], r["reasons"])

r = P.check_legs(HOME, HOME_POSE, leg, "nullspace", LIMITS, CFG, q_ns=HOME)
check("impedance legs carry both predictions",
      set(r["q_end"]) == {"no_pull", "pulled"})

# A start far from q_ns (franka's "ready" posture, the NUC mock's) must not read as a flip:
# the pulled prediction settles at the start pose before it walks.
READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]
ready_pose = pose_of(k.ee_matrix(READY))
r = P.check_legs(READY, ready_pose, [("down", moved(ready_pose, dz=-0.05))], "nullspace",
                 LIMITS, CFG, q_ns=HOME)
check("a start far from q_ns settles first, then walks without a false flip",
      r["ok"] and not any("posture flip" in s for s in r["reasons"]), r["reasons"])

# -- rotation_max is per ramp: one ramp past it is refused before any IK; a chain
# cuts the leg into pieces under it (cut_over_cap) and joint 7 decides.
turn_cfg = P.PlanConfig(ee_to_camera=CAMERA, rotation_max=math.radians(60))
one = P.check_leg(HOME, HOME_POSE, moved(HOME_POSE, roll=math.radians(80)), "elbow_locked",
                  LIMITS, turn_cfg)
check("one 80 deg ramp is refused at a 60 deg rotation_max, before any sampling",
      one["ok"] is False and "per-command limit" in one["reasons"][0]
      and one["stats"]["samples"] == 0, one["reasons"])
r = P.check_legs(HOME, HOME_POSE, [("roll", moved(HOME_POSE, roll=math.radians(80)))],
                 "elbow_locked", LIMITS, turn_cfg)
check("... the same leg in a chain is cut into pieces under the cap and passes",
      r["ok"] and [st["leg"] for st in r["legs"]] == ["roll 1/2", "roll 2/2"],
      str([st["leg"] for st in r["legs"]]))
past7 = P.check_legs(HOME, HOME_POSE, [("roll", moved(HOME_POSE, roll=math.radians(170)))],
                     "elbow_locked", LIMITS, turn_cfg)
check("... and a turn joint 7 cannot make is still refused, naming the joint",
      past7["ok"] is False and "joint 7" in past7["reasons"][0], past7["reasons"])
r = P.check_legs(HOME, HOME_POSE, [("roll", moved(HOME_POSE, roll=math.radians(45)))],
                 "elbow_locked", LIMITS, turn_cfg)
check("a 45 deg turn passes it", r["ok"], r["reasons"])
wide_turn = [{"id": "big:direct", "confidence": 0.9,
              "legs": [("roll", moved(HOME_POSE, roll=math.radians(130)))]},
             {"id": "big:flipped", "confidence": 0.9,
              "legs": [("roll", moved(HOME_POSE, roll=math.radians(130 - 180)))]}]
plan = P.plan_legs(HOME, HOME_POSE, wide_turn, "elbow_locked", LIMITS, turn_cfg)
got = {c["id"]: c for c in plan["candidates"]}
check("plan_pick: of the two finger orders (130 deg / -50 deg) the one joint 7 allows is chosen",
      plan["chosen"] == "big:flipped" and got["big:direct"]["ok"] is False
      and "joint 7" in got["big:direct"]["reasons"][0], str(plan["chosen"]))
check("no rotation_max (None) keeps the old behaviour",
      P.check_legs(HOME, HOME_POSE, [("roll", moved(HOME_POSE, roll=math.radians(80)))],
                   "elbow_locked", LIMITS, CFG)["ok"])

# -- choosing among candidates ------------------------------------------------------------


def pick(grasp):
    """standoff 10 cm above, the grasp, a 10 cm retreat -- the sequencer's legs."""
    return [("standoff", moved(grasp, dz=0.10)), ("grasp", grasp),
            ("retreat", moved(grasp, dz=0.10))]


table_grasp = moved(HOME_POSE, dx=0.05, dz=CFG.table_z - tcp_z + 0.10)
candidates = [
    {"id": "wrist-limit", "confidence": 0.95,
     "legs": pick(moved(table_grasp, roll=math.radians(120)))},
    {"id": "out-of-reach", "confidence": 0.90, "legs": pick(moved(table_grasp, dx=0.5))},
    {"id": "good", "confidence": 0.80, "legs": pick(table_grasp)},
    {"id": "never-reached", "confidence": 0.70, "legs": pick(table_grasp)},
]
CFG_CONF = P.PlanConfig(ee_to_camera=CAMERA, rank="confidence")
for model in ("nullspace", "elbow_locked"):
    plan = P.plan_legs(HOME, HOME_POSE, candidates, model, LIMITS, CFG_CONF, q_ns=HOME,
                       home_pose=HOME_POSE)
    ids = [c["id"] for c in plan["candidates"]]
    check("%s: the most confident FEASIBLE candidate is chosen" % model,
          plan["chosen"] == "good", plan["chosen"])
    check("%s: more confident ones are evaluated first, with reasons" % model,
          ids == ["wrist-limit", "out-of-reach", "good"]
          and all(c["reasons"] for c in plan["candidates"][:2]), ids)
    check("%s: nothing after the choice is evaluated" % model, "never-reached" not in ids)

# The return home is cut as go_home cuts it (home_legs), so a pick whose last leg
# ends 80 deg of wrist turn from home is judged on the legs that would run.
CFG90 = P.PlanConfig(ee_to_camera=CAMERA, rotation_max=math.radians(90))
turned = moved(table_grasp, roll=math.radians(80))
plan = P.plan_legs(HOME, HOME_POSE, [{"id": "turned", "confidence": 1.0,
                                      "legs": [("grasp", turned)]}],
                   "elbow_locked", LIMITS, CFG90, home_pose=HOME_POSE)
names = [leg["leg"] for leg in plan["candidates"][0]["legs"]]
check("plan_legs: the return home is checked in go_home's legs",
      plan["chosen"] == "turned" and names[-2:] == ["home 1/2", "home 2/2"],
      "%s %s" % (names, plan["candidates"][0]["reasons"]))
check("home_legs: 45 deg a leg, or 5 under rotation_max",
      P.leg_split(CFG90) == (45.0, 0.20)
      and abs(P.leg_split(P.PlanConfig(rotation_max=math.radians(30)))[0] - 25.0) < 1e-9)

# -- rank utility: comfort breaks near-ties; what cannot win is not walked --
for model in ("nullspace", "elbow_locked"):
    plan = P.plan_legs(HOME, HOME_POSE, candidates, model, LIMITS, CFG, q_ns=HOME,
                       home_pose=HOME_POSE)
    got = {c["id"]: c for c in plan["candidates"]}
    check("%s utility: the feasible one is chosen and carries its utility" % model,
          plan["chosen"] == "good" and 0 < got["good"].get("utility", 0) <= 0.80, plan["chosen"])
    check("%s utility: a candidate whose confidence cannot beat it is listed, not walked" % model,
          got.get("never-reached", {}).get("ok") is None
          and "cannot beat" in got["never-reached"]["reasons"][0]
          or got["good"]["utility"] < 0.70, got.get("never-reached"))

# A marginal grasp -- rolled 1.75 rad, which leaves joint 7 ~0.13 rad from its limit --
# a little more confident than the same grasp unrolled, which is comfortable.
tight = moved(HOME_POSE, dx=0.05, dz=-0.25, roll=1.75)
roomy = moved(HOME_POSE, dx=0.05, dz=-0.25)
pair = [{"id": "marginal", "confidence": 0.90, "legs": pick(tight)},
        {"id": "comfortable", "confidence": 0.85, "legs": pick(roomy)}]
plan = P.plan_legs(HOME, HOME_POSE, pair, "elbow_locked", LIMITS, CFG, home_pose=HOME_POSE)
u = {c["id"]: c.get("utility") for c in plan["candidates"]}
check("utility: the comfortable grasp beats the marginal, more confident one",
      plan["chosen"] == "comfortable" and plan["by_confidence"] == "marginal", str(u))
plan_c = P.plan_legs(HOME, HOME_POSE, pair, "elbow_locked", LIMITS, CFG_CONF, home_pose=HOME_POSE)
check("confidence: the same pair reproduces the old choice", plan_c["chosen"] == "marginal", plan_c["chosen"])

# prescreen: the commander's floor refuses before any path is walked.
floor = P.PlanConfig(ee_to_camera=CAMERA, z_min=-0.036)
below = moved(HOME_POSE, dz=-0.036 - HOME_POSE[0][2] - 0.01)
plan = P.plan_legs(HOME, HOME_POSE, [{"id": "under", "confidence": 1.0, "legs": pick(below)}],
                   "elbow_locked", LIMITS, floor)
c0 = plan["candidates"][0]
check("prescreen: a leg below the commander's floor is refused without a walk",
      c0["ok"] is False and c0["legs"] == [] and P.category(c0["reasons"][0]) == "commander limit",
      c0["reasons"])

spent = P.plan_legs(HOME, HOME_POSE, candidates, "nullspace", LIMITS, CFG, q_ns=HOME,
                    budget_s=0.0)
check("a spent budget chooses nothing and says so",
      spent["chosen"] is None and all(c["ok"] is None for c in spent["candidates"]))
top = P.plan_legs(HOME, HOME_POSE, candidates, "nullspace", LIMITS, CFG, q_ns=HOME, top_k=2)
check("top_k caps the candidates considered",
      top["chosen"] is None and top["not_considered"] == 2)

# -- runtime -------------------------------------------------------------------------------

CFG_TOP_K = 8          # plan_legs' default top_k

# Worst case: top_k candidates each walked through every leg and home (plan_legs would
# stop at the first feasible one, so they are walked directly).
slow = []
for i in range(CFG_TOP_K):
    grasp = moved(table_grasp, dy=0.02 * i, roll=math.radians(10 * i))
    slow.append(pick(grasp) + [("home", HOME_POSE)])
for model in ("nullspace", "elbow_locked"):
    t0 = time.time()
    for legs in slow:
        P.check_legs(HOME, HOME_POSE, legs, model, LIMITS, CFG, q_ns=HOME)
    every = time.time() - t0
    check("%s: the worst case, %d full picks, in under 5 s" % (model, CFG_TOP_K),
          every < 5.0, "%.2f s" % every)

if FAILED:
    print("\n%d FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("\nall passed")
