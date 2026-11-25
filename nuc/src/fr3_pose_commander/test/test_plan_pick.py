#!/usr/bin/env python3
"""Tests plan_pick, the per-leg planner re-check, go_home and piecewise moves in the commander.

    python3 src/fr3_pose_commander/test/test_plan_pick.py

Needs rospy and rospkg importable (the node imports them), so run it on the NUC; no robot or
master, just a bare PoseCommander with its state set by hand. plan_pick takes only a fresh
list and chooses the most confident feasible candidate; report mode never refuses, enforce
refuses when nothing passes, and --no-checks still overrules a leg re-check.
"""
import importlib.util
import json
import math
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
WS_SRC = os.path.dirname(PKG)
sys.path.insert(0, os.path.join(PKG, "src"))
spec = importlib.util.spec_from_file_location(
    "pose_commander_node", os.path.join(PKG, "scripts", "pose_commander_node.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from fr3_pose_commander import kinematics as k  # noqa: E402
from fr3_pose_commander.pose_math import quaternion_from_matrix  # noqa: E402
import numpy as np  # noqa: E402

FAILED = []


def check(label, condition, detail=""):
    print("%-4s %s%s" % ("ok" if condition else "FAIL", label,
                         "  (%s)" % detail if detail else ""))
    if not condition:
        FAILED.append(label)


HOME = [0.164563, -0.222906, -0.123741, -2.13951, -0.0294044, 1.91649, 0.839196]
T = k.ee_matrix(HOME)
HOME_POSE = (T[:3, 3].copy(), quaternion_from_matrix(T))
LIMITS = os.path.join(WS_SRC, "franka_ros", "franka_description", "robots", "fr3",
                      "joint_limits.yaml")
HOME_FILE = os.path.join(PKG, "config", "home_pose.yaml")


def commander(mode="report", topic="/cartesian_impedance_example_controller/equilibrium_pose",
              enforce_checks=True, rotation_max=math.pi):
    c = mod.PoseCommander.__new__(mod.PoseCommander)   # no __init__: no ROS
    c._lock = threading.Lock()
    c._busy = False
    c.base_frame = "fr3_link0"
    c.enforce_checks = enforce_checks
    c.linear_speed, c.angular_speed = 0.012, 0.40
    c.rotation_max = rotation_max
    params = {"~planner/mode": mode, "~planner/joint_limits": LIMITS,
              "~planner/home_pose_file": HOME_FILE,
              "~ee_to_camera": [0.105, 0.0, -0.048, 0, 0, 0.7071, 0.7071]}
    c._init_planner(lambda name, default=None: params.get(name, default), topic)
    c._state = (HOME_POSE[0], HOME_POSE[1], None)
    c._q = list(HOME)
    c._home = HOME_POSE
    c._candidates = c._approved = c._plan_check = None
    return c


def leg(name, dx=0.0, dz=0.0, roll=0.0):
    m = T.copy()
    cr, sr = math.cos(roll), math.sin(roll)
    m[:3, :3] = m[:3, :3].dot(np.array(((cr, -sr, 0), (sr, cr, 0), (0, 0, 1))))
    m[:3, 3] += (dx, 0.0, dz)
    return {"name": name, "position": list(m[:3, 3]),
            "orientation_xyzw": list(quaternion_from_matrix(m))}


def pick(dx=0.05, roll=0.0):
    return [leg("standoff", dx, -0.15, roll), leg("grasp", dx, -0.25, roll),
            leg("retreat", dx, -0.15, roll)]


def payload(candidates, plan_id="p1", frame="fr3_link0", obstacles=None):
    out = {"plan_id": plan_id, "frame_id": frame, "capture_stamp": 1.0,
           "candidates": candidates}
    if obstacles is not None:
        out["obstacles"] = obstacles
    return json.dumps(out)


GOOD = {"id": "good", "confidence": 0.8, "legs": pick()}
WRIST = {"id": "wrist", "confidence": 0.95, "legs": pick(roll=math.radians(120))}

c = commander()
check("planner loaded: limits, home joints, camera", c.planner_limits is not None
      and c.home_joints is not None and c.planner_cfg.ee_to_camera is not None,
      c.planner_error)
check("impedance topic -> nullspace model", c.planner_model == "nullspace")

r = c.on_plan_pick(None)
check("no list -> refused, and says so",
      not r.success and "no /live_pipeline/pick_candidates" in r.message)

c._candidates = (payload([WRIST, GOOD]), time.time() - 10.0)
r = c.on_plan_pick(None)
check("a list older than candidates_max_age is refused as a previous pick's",
      not r.success and "earlier pick" in r.message, r.message)

c._candidates = (payload([WRIST, GOOD]), time.time())
r = c.on_plan_pick(None)
reply = json.loads(r.message)
check("the most confident feasible candidate is chosen",
      r.success and reply["chosen"] == "good", reply.get("chosen"))
check("plan_id is echoed", reply["plan_id"] == "p1")
check("the more confident failure is reported with its reason",
      reply["candidates"][0]["id"] == "wrist" and reply["candidates"][0]["reasons"]
      and reply["candidates"][0]["reasons"][0].startswith("joint 7"),
      reply["candidates"][0]["reasons"])
check("the chosen legs are remembered as approved",
      c._approved is not None and c._approved["id"] == "good")

c._candidates = (payload([GOOD], frame="camera_color_optical_frame"), time.time())
r = c.on_plan_pick(None)
check("camera-frame legs are refused", not r.success and "not 'fr3_link0'" in r.message)

c._candidates = (payload([WRIST]), time.time())
r = c.on_plan_pick(None)
check("report mode, nothing feasible: success, chosen null",
      r.success and json.loads(r.message)["chosen"] is None)
e = commander("enforce")
e._candidates = (payload([WRIST]), time.time())
r = e.on_plan_pick(None)
check("enforce mode, nothing feasible: refused",
      not r.success and json.loads(r.message)["chosen"] is None)

off = commander("off")
off._candidates = (payload([WRIST]), time.time())
r = off.on_plan_pick(None)
check("off: nothing is evaluated", r.success and json.loads(r.message)["note"] == "planner is off")

c._busy = True
check("refused while a leg is moving", c.on_plan_pick(None).message == "already moving")
c._busy = False

lim = commander(rotation_max=math.radians(60))
check("the commander hands its rotation_max to the planner",
      abs(lim.planner_cfg.rotation_max - math.radians(60)) < 1e-12)
lim._candidates = (payload([WRIST, GOOD]), time.time())
reply = json.loads(lim.on_plan_pick(None).message)
check("... which cuts the 120 deg candidate into pieces, so it is refused by joint 7 -- "
      "the real limit -- not by the per-ramp cap (since 2026-09-30)",
      reply["chosen"] == "good"
      and "joint 7" in reply["candidates"][0]["reasons"][0],
      reply["candidates"][0]["reasons"])

# -- the scene (arm vs the Orin's voxels) -------------------------------------------
standoff_tcp = np.array(GOOD["legs"][0]["position"])
g = np.arange(-0.06, 0.0601, 0.02)
cube = [[float(standoff_tcp[0] + x), float(standoff_tcp[1] + y), float(standoff_tcp[2] + z)]
        for x in g for y in g for z in g]
sc = commander()
sc._candidates = (payload([GOOD], obstacles={"frame_id": "fr3_link0", "voxel_m": 0.02,
                                             "points": cube}), time.time())
reply = json.loads(sc.on_plan_pick(None).message)
check("obstacles in pick_candidates reach the planner (scene_voxels in the reply)",
      reply.get("scene_voxels") == len(cube), reply.get("scene_voxels"))
check("a candidate whose standoff drives the hand into them is refused for the scene",
      reply["chosen"] is None and "of the scene" in reply["candidates"][0]["reasons"][0],
      reply["candidates"][0]["reasons"])
sc._candidates = (payload([GOOD], obstacles={"frame_id": "camera_color_optical_frame",
                                             "voxel_m": 0.02, "points": cube}), time.time())
r = sc.on_plan_pick(None)
check("obstacles in the camera frame are refused, not checked",
      not r.success and "obstacles are in" in r.message, r.message)
sc._candidates = (payload([GOOD]), time.time())
reply = json.loads(sc.on_plan_pick(None).message)
check("no obstacles sent: the scene check is skipped and the reply says 0 voxels",
      reply["chosen"] == "good" and reply.get("scene_voxels") == 0)

# -- a labelled scene: the per-leg re-checks get the CANDIDATE's scene --------------
# plan_legs checks each candidate against candidate_scene() (its own object the target,
# the keep-out box on it only); the re-checks must use that scene, not the one as sent.
grasp_tcp = np.array(GOOD["legs"][1]["position"])
mine = [[float(grasp_tcp[0] + x), float(grasp_tcp[1] + y), float(grasp_tcp[2] + z)]
        for x in (-0.01, 0.0, 0.01) for y in (-0.01, 0.0, 0.01) for z in (-0.03, -0.01)]
other = [[float(grasp_tcp[0] + 0.20), float(grasp_tcp[1] + y), float(grasp_tcp[2] - 0.02)]
         for y in (-0.01, 0.0, 0.01)]
lab = commander()
lab._candidates = (payload([dict(GOOD, instance=1)], obstacles={
    "frame_id": "fr3_link0", "voxel_m": 0.01, "points": mine + other,
    "instance": [1] * len(mine) + [2] * len(other)}), time.time())
reply = json.loads(lab.on_plan_pick(None).message)
ap = lab._approved
check("labelled scene: the candidate on object 1 is chosen",
      reply["chosen"] == "good" and ap is not None, reply.get("candidates"))
check("... and its re-checks get its own scene: object 1 the target, object 2 the scene",
      ap is not None and getattr(ap["scene"], "box_target_only", False)
      and ap["scene"].target is not None and len(ap["scene"].target) == len(mine)
      and len(ap["scene"]) == len(other),
      "" if ap is None else "box_target_only %s, %d scene voxels"
      % (getattr(ap["scene"], "box_target_only", None), len(ap["scene"])))
check("... with the keep-out box on the target only (the fingers are checked on it)",
      ap is not None and getattr(ap["near"], "box_target_only", False)
      and len(ap["near"]) == len(other))
lab._candidates = (payload([WRIST], plan_id="p2"), time.time())
reply = json.loads(lab.on_plan_pick(None).message)
check("a later plan that chooses nothing clears the last approval (2026-10-12)",
      reply["chosen"] is None and lab._approved is None, reply.get("chosen"))

pos = commander(topic="/cartesian_pose_tracking_controller/target_pose")
check("position topic -> elbow_locked model", pos.planner_model == "elbow_locked")

# -- the per-leg re-check ---------------------------------------------------------


def as_goal(entry):
    return (np.asarray(entry["position"]), tuple(entry["orientation_xyzw"]))


bad_leg = as_goal(leg("roll", roll=math.radians(120)))
good_leg = as_goal(leg("down", dz=-0.1))
start = (HOME_POSE[0], HOME_POSE[1])

c._approved = None
c._check_leg(good_leg, start)
check("a feasible leg passes and carries the predicted end posture",
      c._plan_check["ok"] is True and "pulled" in c._plan_check["q_end"])

c._check_leg(bad_leg, start)
check("report mode: an infeasible leg is logged, not refused",
      c._plan_check["ok"] is False)

try:
    e._check_leg(bad_leg, start)
    refused = False
except mod.Refused as exc:
    refused = "joint 7" in str(exc)
check("enforce mode: an infeasible leg is refused, naming the joint", refused)

lax = commander("enforce", enforce_checks=False)
try:
    lax._check_leg(bad_leg, start)
    overruled = True
except mod.Refused:
    overruled = False
check("--no-checks overrules the planner like any other check", overruled)

broken = commander()
broken.planner_limits = None
broken._check_leg(good_leg, start)
check("an unavailable planner says so instead of approving",
      broken._plan_check == {"ok": None, "error": "planner unavailable"})

# -- home is home_pose.yaml, reached in legs ------------------------------------------
import yaml  # noqa: E402
from fr3_pose_commander.pose_math import angle_between  # noqa: E402

NEW_HOME = [float(v) for v in yaml.safe_load(open(HOME_FILE))["home_joints"]]


class _Pub(object):
    def __init__(self):
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


def homer(source="home_pose", q=None, home_file=HOME_FILE):
    c = commander(mode="enforce", topic="/cartesian_pose_tracking_controller/target_pose",
                  rotation_max=math.radians(90))
    c.home_joints = None
    params = {"~planner/home_pose_file": home_file, "~planner/joint_limits": LIMITS}
    c._init_planner(lambda name, default=None: params.get(name, default),
                    "/cartesian_pose_tracking_controller/target_pose")
    c._home = c._home_source = None
    c._init_home(lambda name, default=None: {"~home/source": source}.get(name, default))
    c.dry_run, c.status_pub = True, _Pub()
    c.z_min, c.radius_max, c.step_max = -0.036, 0.85, 0.60
    q = list(q or NEW_HOME)
    m = k.ee_matrix(q)
    c._q, c._state = q, (m[:3, 3].copy(), quaternion_from_matrix(m), None)
    c._current = lambda: (c._state[0], c._state[1])
    c.ramps = []

    def ramp(p0, q0, p1, q1, guard=False):
        c.ramps.append((np.asarray(p1), q1))
        with c._lock:                           # the arm arrives where it was sent
            c._state = (np.asarray(p1), q1, None)
        return "dry run"
    c._ramp = ramp
    return c


c = homer()
fk = k.ee_matrix(NEW_HOME)
check("home = forward kinematics of home_pose.yaml's joints",
      c._home_source == "home_pose.yaml" and np.allclose(c._home[0], fk[:3, 3]))
check("home/source startup leaves it to the first state, as before",
      homer(source="startup")._home is None)
check("an unreadable home_pose.yaml falls back to startup, and says why",
      homer(home_file="/nonexistent.yaml")._home is None
      and homer(home_file="/nonexistent.yaml").home_error)

# From the near viewpoint -- 180 deg of wrist turn from home -- go_home must come back.
from fr3_pose_commander import viewpoints as V  # noqa: E402
near = V.plan_viewpoint(2, NEW_HOME, V.home_pose(NEW_HOME), c.planner_limits, c.planner_cfg,
                        [0.105, 0.0, -0.048, 0, 0, 0.7071067811865476, 0.7071067811865476])
c = homer(q=near["q_end"])
r = c._home_leg(c._home)
turns = [math.degrees(angle_between(a[1], b[1]))
         for a, b in zip([(None, c._state[1])] + c.ramps[:-1], c.ramps)]
check("go_home from the near view succeeds, in legs", r.success and len(c.ramps) >= 4,
      "%s; %d ramps" % (r.message[:80], len(c.ramps)))
check("... every leg under rotation_max, and the last ends at home",
      c.ramps and np.allclose(c.ramps[-1][0], c._home[0])
      and angle_between(c.ramps[-1][1], c._home[1]) < 1e-6)
check("... and the whole chain was checked by the planner first",
      c._plan_check and c._plan_check["ok"] is True, c._plan_check)

c = homer(q=near["q_end"])
c.planner_cfg.rotation_max = None                 # a single leg, as before
c.rotation_max = math.radians(90)
old_split = mod.planning.leg_split
mod.planning.leg_split = lambda cfg: (360.0, 10.0)
r = c._home_leg(c._home)
mod.planning.leg_split = old_split
check("as ONE leg the same move is refused (the pre-2026-09-29 go_home)",
      not r.success and "reorientation" in r.message and not c.ramps, r.message[:120])

c = homer()
r = c._home_leg(c._home)
check("go_home at home: one leg, and no refusal", r.success and len(c.ramps) == 1, r.message[:80])

# -- the retreat is matched as the retreat, not the standoff -------------------------
A, B = np.array([0.674, 0.268, 0.131]), np.array([0.645, 0.332, 0.059])
check("approved_leg: standoff and retreat share a point -- the order decides",
      mod.approved_leg(A, [A, B, A], 0, 0.002) == 0
      and mod.approved_leg(A, [A, B, A], 2, 0.002) == 2)
check("approved_leg: nothing within tolerance is None",
      mod.approved_leg(A + 0.01, [A, B, A], 0, 0.002) is None)

c = commander(mode="enforce", topic="/cartesian_pose_tracking_controller/target_pose")
FULL, NEAR = object(), object()
c._approved = {"plan_id": "p", "id": "5:direct", "positions": [A, B, A],
               "names": ["standoff", "grasp", "retreat"], "next": 0,
               "scene": FULL, "near": NEAR}
seen = []
real = mod.planning.check_legs


def fake(q, start, legs, *a, **kw):
    seen.append((legs[0][0], kw.get("scene")))
    return {"ok": True, "reasons": [], "prediction": None, "q_end": {}}


mod.planning.check_legs = fake
for pos in (A, B, A):
    c._check_leg((pos, HOME_POSE[1]), HOME_POSE)
mod.planning.check_legs = real
check("standoff, grasp, retreat are named in order, not nearest-first",
      [n for n, _ in seen] == ["standoff", "grasp", "retreat"], [n for n, _ in seen])
check("... and the retreat is checked with the target excluded (the 21:15 refusal)",
      seen[2][1] is NEAR and seen[0][1] is FULL)

c._approved["next"] = 0
seen = []


def refuse_first(q, start, legs, *a, **kw):
    seen.append(legs[0][0])
    return {"ok": len(seen) > 1, "reasons": ["x"], "prediction": None, "q_end": {}}


mod.planning.check_legs = refuse_first
try:
    c._check_leg((A, HOME_POSE[1]), HOME_POSE)
except mod.Refused:
    pass
c._check_leg((A, HOME_POSE[1]), HOME_POSE)
mod.planning.check_legs = real
check("a REFUSED standoff does not advance: its retry is still the standoff",
      seen == ["standoff", "standoff"], seen)

# -- a leg that does not happen fails (leg_outcome) ---------------------------------------
class _Listened(object):
    def get_num_connections(self):
        return 1
    resolved_name = "/cartesian_pose_tracking_controller/target_pose"


def live_homer(mode, arm_moves=True, q=None):
    c = homer(q=q)
    c.dry_run, c.goal_pub = False, _Listened()
    c.fail_distance = 0.010
    c._mode = mode
    if not arm_moves:                               # a dead controller: the ramp changes nothing
        c._ramp = lambda *a, **k: (c.ramps.append(a[2]), "stopped closing at 75.4 mm")[1]
    return c


c = live_homer(4, q=near["q_end"])
r = c._home_leg(c._home)
check("go_home refuses while the robot is in Reflex, before anything is ramped",
      not r.success and "Reflex" in r.message and not c.ramps, r.message[:100])
c = live_homer(2, arm_moves=False, q=near["q_end"])
r = c._home_leg(c._home)
check("go_home FAILS when the arm does not follow (the 2026-09-30 case), at the first leg",
      not r.success and "did not happen" in r.message and len(c.ramps) == 1, r.message[:120])
c = live_homer(2, q=near["q_end"])
r = c._home_leg(c._home)
check("go_home with a live controller that follows still succeeds", r.success, r.message[:100])

# -- an unguarded move past the per-ramp cap is ramped in pieces --------------------------
from geometry_msgs.msg import PoseStamped as _PS  # noqa: E402


def target_at(pose):
    msg = _PS()
    msg.header.frame_id = "fr3_link0"
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[0]
    (msg.pose.orientation.x, msg.pose.orientation.y,
     msg.pose.orientation.z, msg.pose.orientation.w) = pose[1]
    return msg


def mover(turn_deg, guarded=False):
    c = live_homer(2)
    c.rotation_max = math.radians(90)
    c.planner_cfg.rotation_max = c.rotation_max
    m = k.ee_matrix(NEW_HOME)
    r = math.radians(turn_deg)
    m[:3, :3] = m[:3, :3].dot(np.array(((math.cos(r), -math.sin(r), 0), (math.sin(r), math.cos(r), 0), (0, 0, 1))))
    m[:3, 3] += (0.03, 0.0, -0.10)
    c._target, c._width = target_at((tuple(m[:3, 3]), quaternion_from_matrix(m))), -1.0
    c._gripper = lambda width: "left alone"
    c._last_goal = None
    return c


c = mover(100)
r = c._execute()
check("a 100 deg unguarded move runs, in pieces under the 90 deg cap",
      r.success and len(c.ramps) == 3, "%s; %d ramps" % (r.message[:90], len(c.ramps)))
check("... each piece turns at most 45 deg",
      all(math.degrees(angle_between(a[1], b[1])) <= 45.0 + 1e-6
          for a, b in zip([(None, quaternion_from_matrix(k.ee_matrix(NEW_HOME)))] + c.ramps[:-1], c.ramps)))
c = mover(60)
r = c._execute()
check("a 60 deg move is one ramp, as before", r.success and len(c.ramps) == 1, r.message[:90])
c = mover(100)
r = c._execute(guarded=True)
check("a GUARDED move past the cap is still refused, not cut", not r.success and not c.ramps
      and "reorientation" in r.message, r.message[:100])
c = mover(100)
calls = {"n": 0}


def dies_after_first(*a, **kw):
    calls["n"] += 1
    c.ramps.append(a[2])
    if calls["n"] == 1:
        c._mode = 4                                   # a reflex mid-move
    return "stopped closing"
c._ramp = dies_after_first
r = c._execute()
check("a reflex after the first piece stops the move there, fingers untouched",
      not r.success and calls["n"] == 1 and "piece 1/3" in r.message and "Reflex" in r.message, r.message[:120])

# -- measure_load relaxes the impedance spring first --------------------------------------
_sleep = mod.rospy.sleep
mod.rospy.sleep = lambda t: None
imp = commander(topic="/cartesian_impedance_example_controller/equilibrium_pose")
imp.command_topic, imp.dry_run = "/cartesian_impedance_example_controller/equilibrium_pose", False
arm = np.array([0.502, -0.097, 0.1225])
imp._state = (arm, (1.0, 0.0, 0.0, 0.0), None)
imp._last_setpoint = arm - np.array([0.0, 0.0, 0.0044])          # stalled 4.4 mm above it
sent = []
imp._publish = lambda p, q: sent.append(np.asarray(p))
gap = imp._relax_setpoint()
check("impedance: the setpoint is moved onto the measured arm before the load is read",
      len(sent) == 1 and np.allclose(sent[0], arm) and gap == 4.4, str((gap, sent)))
pos = commander(topic="/cartesian_pose_tracking_controller/target_pose")
pos.command_topic, pos.dry_run = "/cartesian_pose_tracking_controller/target_pose", False
pos._state = (arm, (1.0, 0.0, 0.0, 0.0), None)
sent2 = []
pos._publish = lambda p, q: sent2.append(p)
check("position control: nothing to relax, nothing published", pos._relax_setpoint() is None and not sent2)
mod.rospy.sleep = _sleep

if FAILED:
    print("\n%d FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("\nall plan_pick checks passed")
