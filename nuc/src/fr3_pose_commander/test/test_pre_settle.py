#!/usr/bin/env python3
"""Tests the pre-guard settle decision (wrench_settled).

    python3 src/fr3_pose_commander/test/test_pre_settle.py

Needs rospy importable (the node imports it); no robot, master or controller. Before taking
its baseline the guarded leg holds until the wrench stops moving: a still arm at the noise
floor is settled whatever its standing bias; a decaying release of held force, or one late
step in a quiet window, is not (the span is a range, not a standard deviation).
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
settled = mod.wrench_settled

F_TOL, T_TOL = 0.2, 0.15          # config/pose_commander.yaml guard/pre_settle_*
N = 9                             # ~0.3 s of franka_states at ~30 Hz
FAILED = []


def check(label, condition, detail=""):
    print("%-4s %s%s" % ("ok" if condition else "FAIL", label,
                         "  (%s)" % detail if detail else ""))
    if not condition:
        FAILED.append(label)


def w(fx=0.0, fy=0.0, fz=0.0, tx=0.0, ty=0.0, tz=0.0):
    return [fx, fy, fz, tx, ty, tz]


rng = np.random.default_rng(0)
# Static noise floor measured on the rig (idle_wrench.py): sd 0.016 N / 0.012 N.m.
STANDING = np.array(w(fx=2.1, fy=-0.4, fz=2.8, tx=0.6, ty=-0.9, tz=0.1))   # |F| ~3.5 N, |T| ~1.1 N.m


def quiet(n=N, sd_f=0.016, sd_t=0.012):
    noise = np.hstack([rng.normal(0, sd_f, (n, 3)), rng.normal(0, sd_t, (n, 3))])
    return list(STANDING + noise)


ok, f, t = settled(quiet(), F_TOL, T_TOL)
check("still arm at the static noise floor is settled", ok, "span %.3f N / %.3f N.m" % (f, t))

ok, f, t = settled([list(STANDING)] * N, F_TOL, T_TOL)
check("a large STANDING bias with no change is settled", ok and f < 1e-9)

# ~7.6 N held along -z (arm short of its setpoint) released and decaying over a few tenths of a second.
tau = 0.15
release = [list(STANDING + np.array(w(fz=-7.6 * np.exp(-k / 30.0 / tau)))) for k in range(N)]
ok, f, t = settled(release, F_TOL, T_TOL)
check("a decaying release of held force is NOT settled", not ok, "span %.2f N" % f)

# The small tail of a decay is still moving more than the tolerance.
tail = [list(STANDING + np.array(w(fz=-1.0 * np.exp(-k / 30.0 / tau)))) for k in range(N)]
ok, f, t = settled(tail, F_TOL, T_TOL)
check("the tail of a 1 N release is NOT settled", not ok, "span %.2f N" % f)

step = quiet()
step[-1] = list(np.asarray(step[-1]) + np.array(w(fx=0.5)))
ok, f, t = settled(step, F_TOL, T_TOL)
check("one late 0.5 N step in a quiet window is NOT settled", not ok, "span %.2f N" % f)

twist = quiet()
twist[-2] = list(np.asarray(twist[-2]) + np.array(w(ty=0.4)))
ok, f, t = settled(twist, F_TOL, T_TOL)
check("a torque-only step is NOT settled (either channel blocks)", not ok, "span %.2f N.m" % t)

ok, f, t = settled([], F_TOL, T_TOL)
check("no readings is NOT settled", not ok and f is None)
ok, f, t = settled([list(STANDING)], F_TOL, T_TOL)
check("one reading is NOT settled (nothing to compare)", not ok and f is None)

if FAILED:
    print("\n%d FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("\nall pre-settle checks passed")
