#!/usr/bin/env python3
"""Tests contact_hold: where the setpoint goes when the contact guard fires.

    python3 src/fr3_pose_commander/test/test_contact_hold.py

Needs rospy importable (the node imports it); no robot or master. The hold must be the
measured pose pulled back along the approach, never the ramp setpoint, which is past the
arm and would push into the object.
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "src"))
spec = importlib.util.spec_from_file_location(
    "pose_commander_node", os.path.join(PKG, "scripts", "pose_commander_node.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
hold = mod.contact_hold
FAILED = []


def check(label, condition, detail=""):
    print("%-4s %s%s" % ("ok" if condition else "FAIL", label, "  (%s)" % detail if detail else ""))
    if not condition:
        FAILED.append(label)


start = np.array([0.538, -0.012, 0.092])      # standoff
goal = np.array([0.538, -0.012, -0.021])      # grasp, 113 mm straight down
measured = np.array([0.5381, -0.0119, 0.010])  # arm met something 82 mm in

h = hold(measured, start, goal, 0.003)
check("holds at the MEASURED pose, pulled back 3 mm UP the approach",
      np.allclose(h, measured + [0, 0, 0.003]), str(np.round(h, 4)))
check("never past the arm along the approach",
      np.dot(h - measured, goal - start) <= 0.0)

h0 = hold(measured, start, goal, 0.0)
check("backoff 0 holds exactly at the measured pose", np.allclose(h0, measured))

tilted_goal = np.array([0.560, -0.012, -0.021])
ht = hold(measured, start, tilted_goal, 0.003)
d = (tilted_goal - start) / np.linalg.norm(tilted_goal - start)
check("backs off along a TILTED approach, not just up",
      np.allclose(ht, measured - 0.003 * d), str(np.round(ht - measured, 5)))

hz = hold(measured, start, start, 0.003)
check("no approach direction (start == goal): measured pose, no NaN",
      np.allclose(hz, measured) and np.all(np.isfinite(hz)))

check("returns a copy, never the caller's array", hold(measured, start, goal, 0.0) is not measured)

if FAILED:
    print("\n%d FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("\nall contact-hold checks passed")
