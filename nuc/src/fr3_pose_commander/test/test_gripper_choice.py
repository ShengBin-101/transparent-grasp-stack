#!/usr/bin/env python3
"""Tests choose_gripper_action: move, grasp, hold or leave for each leg of a pick.

    python3 src/fr3_pose_commander/test/test_gripper_choice.py

Needs rospy importable (the node imports it). franka_gripper's `grasp` closes onto an object
and squeezes, while `move` drives the fingers to a width; a `grasp` wider than the current
opening drives the jaws shut and fails silently, so every opening must be a `move`.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "src"))

spec = importlib.util.spec_from_file_location(
    "pose_commander_node", os.path.join(PKG, "scripts", "pose_commander_node.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
choose = mod.choose_gripper_action

FAILED = []


def check(label, condition, detail=""):
    print("%-4s %s%s" % ("ok" if condition else "FAIL", label,
                         "  (%s)" % detail if detail else ""))
    if not condition:
        FAILED.append(label)


TOL = 0.004

# -- the three legs of a real pick, in order ---------------------------------
# Fingers start shut from the previous attempt; the object is 16 mm.

check("standoff opens rather than grasping (0 mm -> 78 mm)",
      choose(0.078, 0.0002, TOL) == "move", choose(0.078, 0.0002, TOL))
check("grasp closes onto the object (78 mm -> 16 mm)",
      choose(0.016, 0.078, TOL) == "grasp", choose(0.016, 0.078, TOL))
check("retreat holds what the grasp took, it does not re-squeeze",
      choose(0.016, 0.0161, TOL) == "hold", choose(0.016, 0.0161, TOL))

# -- a grasp that caught nothing: the retreat opens with a move, not another grasp --
check("after a grasp that caught nothing, the retreat moves instead",
      choose(0.016, 0.0002, TOL) == "move", choose(0.016, 0.0002, TOL))

# -- the boundary ------------------------------------------------------------
check("exactly at the tolerance counts as already there",
      choose(0.020, 0.020 + TOL, TOL) == "hold")
check("just past it does not",
      choose(0.020, 0.020 + TOL * 1.01, TOL) == "grasp")
check("the tolerance is symmetric",
      choose(0.020, 0.020 - TOL, TOL) == "hold")

# -- the retreat leaves the fingers alone: a negative width means "do not touch them" --
check("a negative width leaves the fingers alone",
      choose(-1.0, 0.055, TOL) == "leave", choose(-1.0, 0.055, TOL))
check("and does so whatever the fingers are doing",
      all(choose(-1.0, c, TOL) == "leave" for c in (0.0, 0.016, 0.055, 0.078)))
check("it wins even when the fingers are unknown",
      choose(-1.0, None, TOL) == "leave",
      "otherwise the unknown-width fallback would fire a grasp")
# Zero is a real width (close all the way), not the leave sentinel.
check("zero still means close all the way, it is not 'leave'",
      choose(0.0, 0.05, TOL) == "grasp")

# -- why the retreat names no width: 38 mm repeated on a 55 mm object reads as "close" --
check("the old retreat width would have re-gripped a wider-than-predicted object",
      choose(0.038, 0.055, TOL) == "grasp",
      "which is why the retreat must not name a width at all")

# -- unknown width (no /franka_gripper/joint_states): fall back to grasp ------
check("unknown current width falls back to grasp",
      choose(0.078, None, TOL) == "grasp")

# -- it must never open when asked to close ----------------------------------
for commanded, current in ((0.01, 0.08), (0.0, 0.05), (0.03, 0.078)):
    if choose(commanded, current, TOL) != "grasp":
        check("closing %.0f->%.0f mm must be a grasp"
              % (current * 1e3, commanded * 1e3), False)
        break
else:
    check("every narrowing is a grasp, never a move", True)

# -- and never close when asked to open --------------------------------------
for commanded, current in ((0.08, 0.01), (0.05, 0.0), (0.078, 0.03)):
    if choose(commanded, current, TOL) != "move":
        check("opening %.0f->%.0f mm must be a move"
              % (current * 1e3, commanded * 1e3), False)
        break
else:
    check("every widening is a move, never a grasp", True)

print()
if FAILED:
    print("%d check(s) FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("all checks passed")
