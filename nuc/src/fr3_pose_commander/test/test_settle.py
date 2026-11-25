#!/usr/bin/env python3
"""Tests the convergence wait (settle_verdict, is_closing), leg_outcome and load_record.

    python3 src/fr3_pose_commander/test/test_settle.py

Needs rospy importable (the node imports it). The Cartesian PD controller has no integral
term, so under a steady force the arm settles at e = F/K short of its setpoint (5-27 mm
measured on the rig); 'stopped' is a normal verdict, and the wait must not chase the error
to zero.
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
verdict, closing = mod.settle_verdict, mod.is_closing

TOL, PROGRESS, PATIENCE = 0.002, 0.0005, 0.7
FAILED = []


def check(label, condition, detail=""):
    print("%-4s %s%s" % ("ok" if condition else "FAIL", label,
                         "  (%s)" % detail if detail else ""))
    if not condition:
        FAILED.append(label)


# -- the minimum hold is honoured before any verdict --------------------------
# Otherwise a ramp ending near the goal returns before the arm can be pulled off it.
check("no verdict before the minimum hold, even sitting on the goal",
      verdict(0.0001, 9.9, False, TOL, PATIENCE) is None)

# -- arrival ------------------------------------------------------------------
check("inside the tolerance is 'arrived'",
      verdict(0.0015, 0.0, True, TOL, PATIENCE) == "arrived")
check("exactly at the tolerance is 'arrived'",
      verdict(TOL, 0.0, True, TOL, PATIENCE) == "arrived")
check("just outside it is not",
      verdict(TOL * 1.01, 0.0, True, TOL, PATIENCE) is None)

# -- the normal outcome on this rig: a flat residual, PD balanced ------------
check("a flat 21.5 mm residual is reported as 'stopped', not waited on",
      verdict(0.0215, 1.0, True, TOL, PATIENCE) == "stopped")
check("but not before patience runs out",
      verdict(0.0215, 0.5, True, TOL, PATIENCE) is None)

# -- arrival beats stalling ---------------------------------------------------
# An arm that stops dead ON the goal is arrived, not stalled.
check("stopping dead inside the tolerance still reads as 'arrived'",
      verdict(0.0005, 5.0, True, TOL, PATIENCE) == "arrived")

# -- progress ----------------------------------------------------------------
check("the first sample always counts as closing",
      closing(0.050, None, PROGRESS))
check("a real improvement counts",
      closing(0.030, 0.040, PROGRESS))
check("noise below the progress floor does not",
      not closing(0.0399, 0.040, PROGRESS),
      "0.1 mm of jitter must not reset the patience clock")
check("exactly the progress floor does not count",
      not closing(0.040 - PROGRESS, 0.040, PROGRESS))
check("getting worse does not count",
      not closing(0.045, 0.040, PROGRESS))

# -- the two together must terminate ------------------------------------------
# Noise-only drift forever: patience must still expire, or a leg never ends.
best, stalled_for, ended = 0.0215, 0.0, False
for _ in range(200):
    error = 0.0215 - 0.0001          # jitter, below the progress floor
    if closing(error, best, PROGRESS):
        best, stalled_for = error, 0.0
    else:
        stalled_for += 0.05
    if verdict(error, stalled_for, True, TOL, PATIENCE):
        ended = True
        break
check("jitter that never clears the progress floor still terminates", ended,
      "otherwise a leg waits out the full timeout every time")

# -- orientation: a roll leg never changes position, so position alone is not arrival --
ANG = 0.0087
check("a leg at position but still turning is NOT arrived",
      verdict(0.0001, 0.0, True, TOL, PATIENCE, 0.03, ANG) is None)
check("position and orientation both inside tolerance is arrived",
      verdict(0.0001, 0.0, True, TOL, PATIENCE, 0.005, ANG) == "arrived")
check("still turning but stalled past patience is 'stopped', not a hang",
      verdict(0.0001, PATIENCE + 0.1, True, TOL, PATIENCE, 0.03, ANG) == "stopped")
check("old callers (no angle) behave as before",
      verdict(0.0001, 0.0, True, TOL, PATIENCE) == "arrived")


# -- leg_outcome: a leg that did not happen is a failure ------------------------------
lo = mod.leg_outcome
check("a reflex fails the leg, naming the mode", lo(4, 0.0754, False, 0.010) is not None
      and "Reflex" in lo(4, 0.0754, False, 0.010))
check("the 2026-09-30 retreat -- 75.4 mm out under position control -- fails",
      lo(2, 0.0754, False, 0.010) is not None)
check("the move home that never started -- 382 mm out -- fails", lo(2, 0.382, False, 0.010) is not None)
check("a position leg that arrived to 1 mm passes", lo(2, 0.001, False, 0.010) is None)
check("an impedance leg that stalled 17.6 mm short -- the worst logged -- passes",
      lo(2, 0.0176, False, 0.030) is None)
check("a guarded leg stopped short by CONTACT is exempt from the distance",
      lo(2, 0.0183 + 0.1, True, 0.010) is None)
check("... but not from the mode", lo(4, 0.0, True, 0.010) is not None)
check("unknown mode (older franka_msgs) is not a failure by itself", lo(None, 0.001, False, 0.010) is None)


# -- load_record: the standing load for the weight check --------------------------------
steady = [[0.3, -0.2, -2.10 + 0.05 * ((i % 3) - 1), 0.0, 0.1, 0.0] for i in range(50)]
rec = mod.load_record(steady, [[0.4, 0.0, 0.41]] * 50)
check("load_record: the mean force of a steady window", abs(rec["force"][2] + 2.10) < 1e-3 and rec["n"] == 50, rec)
check("... its spread is small and the tool did not move", rec["force_span"] < 0.1 and rec["motion_mm"] == 0.0, rec)
moving = mod.load_record(steady, [[0.4, 0.0, 0.41 + 0.0001 * i] for i in range(50)])
check("... a tool still moving shows up as motion (4.9 mm here)", abs(moving["motion_mm"] - 4.9) < 0.01, moving)

print()
if FAILED:
    print("%d check(s) FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("all checks passed")
