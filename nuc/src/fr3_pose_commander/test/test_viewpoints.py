#!/usr/bin/env python3
"""Tests viewpoints.py: the five capture viewpoints and the way home. No ROS, no robot.

    python3 src/fr3_pose_commander/test/test_viewpoints.py

From the capture pose, under the position controller's model (joint 3 held), every default
viewpoint must be reachable with margin, put the camera `height` above the table looking at
the centre, and split into legs under the wrist-turn limit; the far view at a 30 deg tilt
must stay refused.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
WS_SRC = os.path.dirname(PKG)
sys.path.insert(0, os.path.join(PKG, "src"))
from fr3_pose_commander import kinematics as k  # noqa: E402
from fr3_pose_commander import planning as P  # noqa: E402
from fr3_pose_commander import viewpoints as V  # noqa: E402
from fr3_pose_commander.pose_math import angle_between, quaternion_from_matrix  # noqa: E402

FAILED = []


def check(label, condition, detail=""):
    print("%-4s %s%s" % ("ok" if condition else "FAIL", label,
                         "  (%s)" % detail if detail else ""))
    if not condition:
        FAILED.append(label)


LIMITS = P.load_limits(os.path.join(
    WS_SRC, "franka_ros", "franka_description", "robots", "fr3", "joint_limits.yaml"))
HOME = [0.164563, -0.222906, -0.123741, -2.13951, -0.0294044, 1.91649, 0.839196]
EE_CAM = (0.105, 0.0, -0.048, 0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
T = k.ee_matrix(HOME)
START = (tuple(T[:3, 3]), quaternion_from_matrix(T))
CFG = P.PlanConfig(ee_to_camera=EE_CAM[:3], rotation_max=math.radians(90))
TABLE = -0.041

plans = {n: V.plan_viewpoint(n, HOME, START, LIMITS, CFG, EE_CAM, table_z=TABLE)
         for n in range(1, 6)}
for n, plan in plans.items():
    check("viewpoint %d (%s) is reachable from home" % (n, plan["name"]), plan["ok"],
          plan.get("reasons"))
for n, plan in plans.items():
    if not plan["ok"]:
        continue
    check("viewpoint %d keeps >= 0.25 rad from every joint limit" % n,
          plan["joint_margin"] >= 0.25, "%.2f" % plan["joint_margin"])
    cam = np.array(plan["camera_position"])
    check("viewpoint %d: camera exactly 0.50 m above the table" % n,
          abs(cam[2] - (TABLE + 0.50)) < 1e-9)
    # Where the camera actually ends, from the planned joints, and where it looks.
    ee = k.ee_matrix(plan["q_end"])
    c = ee.dot(V._matrix(EE_CAM))
    want = np.array(plan["target"]) - c[:3, 3]
    off = math.degrees(math.acos(np.clip(c[:3, 2].dot(want) / np.linalg.norm(want), -1, 1)))
    check("viewpoint %d: the optical axis passes through the centre (%.2f deg)" % (n, off),
          off < 0.6 and np.linalg.norm(c[:3, 3] - cam) < 1.5e-3)
    turns = [math.degrees(angle_between(a[1], b[1])) for a, b in
             zip([START] + [g for _, g in plan["legs"]][:-1], [g for _, g in plan["legs"]])]
    check("viewpoint %d: every leg turns the wrist at most 45 deg" % n, max(turns) <= 45.0 + 1e-6,
          "%.1f" % max(turns))

check("viewpoint 1, almost where home is, needs almost no wrist turn",
      plans[1]["ok"] and plans[1]["turn_deg"] < 10.0, "%.1f deg" % plans[1].get("turn_deg", -1))

far30 = V.plan_viewpoint(3, HOME, START, LIMITS, CFG, EE_CAM, table_z=TABLE, tilt_deg=30.0)
check("the far view at a 30 deg tilt is refused before moving, as measured",
      far30["ok"] is False and far30["reasons"], far30.get("reasons"))
d = V.camera_position(4, tilt_deg=20.0, table_z=TABLE)[0][1] - V.DEFAULT_CENTRE[1]
check("side cameras sit height * tan(tilt) off centre", abs(d - 0.50 * math.tan(math.radians(20))) < 1e-12)
try:
    V.camera_position(6)
    raised = False
except ValueError:
    raised = True
check("a viewpoint outside 1-5 is refused", raised)

# -- viewpoint 1 is home ---------------------------------------------------------
import yaml  # noqa: E402
NEW_HOME = [float(v) for v in yaml.safe_load(open(os.path.join(
    PKG, "config", "home_pose.yaml")))["home_joints"]]
home = V.home_pose(NEW_HOME)
cam = k.ee_matrix(NEW_HOME).dot(P.matrix_from_pose((EE_CAM[:3], EE_CAM[3:])))
want = V.camera_position(1, table_z=TABLE)[0]
check("home_pose.yaml puts the camera at viewpoint 1, 0.50 m above the centre",
      np.linalg.norm(cam[:3, 3] - want) < 0.001, "%.1f mm" % (1e3 * np.linalg.norm(cam[:3, 3] - want)))
check("... looking straight down", cam[2, 2] < -0.9999, "%.5f" % cam[2, 2])
at_home = V.plan_home(NEW_HOME, home, home, LIMITS, CFG)
check("home from home: one leg, no turn", at_home["ok"] and len(at_home["legs"]) == 1
      and at_home["turn_deg"] < 0.01, at_home.get("reasons"))
near = V.plan_viewpoint(2, NEW_HOME, home, LIMITS, CFG, EE_CAM, table_z=TABLE)
check("the near view is reachable from the NEW home", near["ok"], near.get("reasons"))
if near["ok"]:
    q2 = near["q_end"]
    m2 = k.ee_matrix(q2)
    at2 = (tuple(m2[:3, 3]), quaternion_from_matrix(m2))
    back = V.plan_home(q2, at2, home, LIMITS, CFG)
    check("home from the near view (%.0f deg of wrist turn) is reachable, in legs"
          % back["turn_deg"], back["ok"] and len(back["legs"]) >= 2
          and back["turn_deg"] > 90, str((back.get("reasons"), len(back["legs"]))))
    check("... each leg under rotation_max",
          all(math.degrees(angle_between(a[1][1], b[1][1])) <= 45.0 + 1e-6
              for a, b in zip([(None, at2)] + back["legs"][:-1], back["legs"])))
    check("... and it ends at the posture home_pose.yaml holds (joint 3 held)",
          back["ok"] and max(abs(a - b) for a, b in zip(back["q_end"], NEW_HOME)) < 0.01,
          back.get("q_end"))
    one = P.check_leg(q2, at2, home, "elbow_locked", LIMITS, CFG)
    check("... where ONE ramp home is refused by rotation_max (a chain is cut instead)",
          one["ok"] is False and "per-command limit" in one["reasons"][0], str(one["reasons"]))
legs = P.split_move(START, home, 45.0, 0.20, label="home")
check("split_move: one leg is named plainly, and ends exactly at the goal",
      [n for n, _ in legs] == ["home"] and np.allclose(legs[-1][1][0], home[0]))

# -- under the impedance controller ------------------------------------------------------
imp = V.plan_viewpoint(5, NEW_HOME, home, LIMITS, CFG, EE_CAM, table_z=TABLE, model="nullspace",
                       q_ns=NEW_HOME)
check("viewpoint 5 under impedance (both nullspace predictions) is reachable from home",
      imp["ok"] and imp["joint_margin"] >= 0.25, str(imp.get("reasons") or imp["joint_margin"]))
back = V.plan_home(imp["q_end"], (tuple(k.ee_matrix(imp["q_end"])[:3, 3]),
                                  quaternion_from_matrix(k.ee_matrix(imp["q_end"]))),
                   home, LIMITS, CFG, "nullspace", NEW_HOME) if imp["ok"] else {"ok": False}
check("... and home from there", back["ok"], str(back.get("reasons")))

if FAILED:
    print("\n%d FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("\nall viewpoint checks passed")
