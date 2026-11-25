#!/usr/bin/env python3
"""Tests how the grasp is commanded (grasp_goal_params) and reported (grasp_verdict).

    python3 src/fr3_pose_commander/test/test_grasp_mode.py

Needs rospy importable (the node imports it); no robot or master. The FR3 hand reopens when
its fingers close well past the commanded width and holds when they stop before it, so
"under" mode commands a width narrower than any object, with a window up to the full opening.
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
params, verdict = mod.grasp_goal_params, mod.grasp_verdict
FAILED = []
UNDER, UNDER_OUT, EPS_IN, EPS_OUT, MISS = 0.005, 0.075, 0.055, 0.02, 0.003


def check(label, condition, detail=""):
    print("%-4s %s%s" % ("ok" if condition else "FAIL", label, "  (%s)" % detail if detail else ""))
    if not condition:
        FAILED.append(label)


# -- under mode: the width commanded never depends on the detector -------------------
for predicted in (0.020, 0.057, 0.069, 0.078):
    w, ei, eo = params("under", predicted, UNDER, UNDER_OUT, EPS_IN, EPS_OUT)
    check("under: commands 5 mm whatever SBG predicted (%.0f mm)" % (predicted * 1e3),
          w == UNDER, "%.3f" % w)
check("under: window starts at 0 (a narrow object is never 'too narrow')", abs(w - ei) < 1e-12)
check("under: window reaches the full 80 mm opening", abs((w + eo) - 0.080) < 1e-9, "%.3f" % (w + eo))
for obj in (0.0258, 0.0281, 0.040, 0.065):   # foam core, rigid block, glass, wide bottle
    check("under: a %.1f mm object stops the fingers BEFORE the commanded width, inside the window"
          % (obj * 1e3), obj >= w and w - ei <= obj <= w + eo)

# -- predicted mode keeps the old behaviour ---------------------------------------------
w, ei, eo = params("predicted", 0.057, UNDER, UNDER_OUT, EPS_IN, EPS_OUT)
check("predicted: the detector's width and the configured window", (w, ei, eo) == (0.057, EPS_IN, EPS_OUT))

# -- the verdict ------------------------------------------------------------------------
check("HELD when the hand says success and the fingers are on something",
      verdict(True, 0.0258, MISS).startswith("HELD at 26 mm"), verdict(True, 0.0258, MISS))
check("MISSED when the fingers closed to nothing -- even if the hand says success",
      verdict(True, 0.0003, MISS).startswith("MISSED"), verdict(True, 0.0003, MISS))
check("MISSED on a failed close to nothing", verdict(False, 0.0006, MISS).startswith("MISSED"))
check("FAILED when the hand says no and the fingers are apart",
      verdict(False, 0.080, MISS).startswith("FAILED"), verdict(False, 0.080, MISS))
check("no result is said as such", verdict(None, 0.030, MISS).startswith("no result"))
check("unknown finger width does not crash", "unknown" in verdict(None, None, MISS))

if FAILED:
    print("\n%d FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("\nall grasp-mode checks passed")
