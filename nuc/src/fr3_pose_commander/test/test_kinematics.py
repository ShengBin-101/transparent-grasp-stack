#!/usr/bin/env python3
"""Tests kinematics.py: FR3 forward kinematics, Jacobian and conditioning. No ROS, no robot.

    python3 src/fr3_pose_commander/test/test_kinematics.py

The key check differences each Jacobian column against the forward kinematics: a joint axis
taken at the wrong point of its modified-DH step shifts every column by one joint while the
forward kinematics stays correct, so matching O_T_EE alone does not prove the Jacobian.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from fr3_pose_commander import kinematics as k  # noqa: E402

FAILED = []


def check(label, condition, detail=""):
    print("%-4s %s%s" % ("ok" if condition else "FAIL", label,
                         "  (%s)" % detail if detail else ""))
    if not condition:
        FAILED.append(label)


# franka's "ready" posture, and one measured on the rig with its reported EE position.
HOME_Q = [0.0, -0.785398163, 0.0, -2.356194490, 0.0, 1.570796327, 0.785398163]
MEASURED_Q = [0.031, -0.154, -0.005, -2.02, -0.01, 1.812, 0.816]
MEASURED_EE = (0.501, 0.0112, 0.4105)

# -- forward kinematics, against the robot's own report -----------------------

got = k.ee_position(MEASURED_Q)
check("ee_position matches the robot's O_T_EE to 1mm",
      max(abs(a - b) for a, b in zip(got, MEASURED_EE)) < 1e-3,
      "%.4f mm worst" % (1e3 * max(abs(a - b) for a, b in zip(got, MEASURED_EE))))

# -- the two reference postures: pinned sigma_min, within 15% of each other ---

home_sigma, home_dir = k.smallest_singular_value(k.jacobian(HOME_Q))
stall_sigma, stall_dir = k.smallest_singular_value(k.jacobian(MEASURED_Q))
check("sigma_min at the home pose is 0.2211",
      abs(home_sigma - 0.221060) < 1e-5, "%.6f" % home_sigma)
check("sigma_min at the 2026-09-16 17:10 stall is 0.1937",
      abs(stall_sigma - 0.193711) < 1e-5, "%.6f" % stall_sigma)
check("the stall posture is NOT meaningfully worse than home",
      home_sigma / stall_sigma < 1.2, "%.2fx" % (home_sigma / stall_sigma))

# -- every column's translation rows against finite differences of the FK -----
# The rotation rows are base-frame axes; they are checked as unit axes below.

for q in (HOME_Q, MEASURED_Q, [0.4044, 0.47, 0.0, -1.23, 0.0, 1.47, 0.7854]):
    J = k.jacobian(q)
    h = 1e-6
    worst = 0.0
    for j in range(7):
        qp, qm = list(q), list(q)
        qp[j] += h
        qm[j] -= h
        pp, pm = k.ee_position(qp), k.ee_position(qm)
        for i in range(3):
            worst = max(worst, abs(J[i][j] - (pp[i] - pm[i]) / (2 * h)))
    check("the Jacobian's translation rows match finite differences at q2=%+.2f" % q[1],
          worst < 1e-6, "%.2e" % worst)

# Joint 7 turns about the tool axis through the EE origin; a non-zero entry means shifted columns.
J = k.jacobian(HOME_Q)
check("column 7 has no translation (joint 7 turns about the tool axis)",
      max(abs(J[i][6]) for i in range(3)) < 1e-12,
      "%.2e" % max(abs(J[i][6]) for i in range(3)))

# The rotation rows are unit axes in the base frame.
for j in range(7):
    axis = [J[3 + i][j] for i in range(3)]
    if abs(math.sqrt(sum(v * v for v in axis)) - 1.0) > 1e-9:
        check("column %d's rotation row is a unit axis" % (j + 1), False)
        break
else:
    check("every rotation row is a unit axis", True)

# -- reporting ---------------------------------------------------------------

check("report() names a direction",
      "weakest in" in (k.report(HOME_Q) or ""), k.report(HOME_Q))
check("report() returns None on junk rather than raising",
      k.report([1, 2, 3]) is None and k.report(None) is None
      and k.report([float("nan")] * 7) is None)
check("describe_direction picks the dominant component",
      k.describe_direction((0, 0, 0, 0, 1, 0)) == "pitch about y"
      and k.describe_direction((0, 0.9, 0, 0, 0, 0)) == "translation along base y")

# The singular values are the same whichever way numpy is asked.
s = np.linalg.svd(np.asarray(k.jacobian(MEASURED_Q)), compute_uv=False)
check("smallest_singular_value agrees with a plain SVD",
      abs(min(s) - stall_sigma) < 1e-12)

print()
if FAILED:
    print("%d check(s) FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("all checks passed")
