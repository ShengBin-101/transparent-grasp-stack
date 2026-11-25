#!/usr/bin/env python3
"""Tests that the O_T_EE layout the mock writes matches the one pose_math reads.

    python3 src/fr3_pose_commander/test/test_mock_states.py

Needs `rospy` and `geometry_msgs` importable; no master, robot or camera. O_T_EE is 16
numbers in column-major order; written row-major it reads back as a transposed rotation,
which is still a valid rotation, so only a round trip catches it.
"""
import importlib.util
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "src"))

from fr3_pose_commander.pose_math import (matrix_from_o_t_ee,  # noqa: E402
                                          quaternion_from_matrix)

spec = importlib.util.spec_from_file_location(
    "mock_franka_states", os.path.join(PKG, "scripts", "mock_franka_states.py"))
mock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mock)

from geometry_msgs.msg import TransformStamped  # noqa: E402

FAILED = []


def check(label, condition, detail=""):
    print("%-4s %s%s" % ("ok" if condition else "FAIL", label,
                         "  (%s)" % detail if detail else ""))
    if not condition:
        FAILED.append(label)


def transform(xyz, quat):
    t = TransformStamped()
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
    (t.transform.rotation.x, t.transform.rotation.y,
     t.transform.rotation.z, t.transform.rotation.w) = quat
    return t


# A non-symmetric rotation, so a transpose shows (identity or a 180-degree turn would not).
XYZ = (0.451, -0.122, 0.377)
QUAT = (0.2331, 0.5602, -0.1976, 0.7669)
norm = math.sqrt(sum(q * q for q in QUAT))
QUAT = tuple(q / norm for q in QUAT)

m = mock.matrix_from_transform(transform(XYZ, QUAT))

check("matrix_from_transform gives a proper rotation",
      abs(np.linalg.det(m[:3, :3]) - 1.0) < 1e-9,
      "det = %.12f" % np.linalg.det(m[:3, :3]))
check("its quaternion round-trips",
      np.allclose(np.abs(quaternion_from_matrix(m)), np.abs(QUAT), atol=1e-9),
      str(quaternion_from_matrix(m)))

# Flatten the way the mock publishes, read back the way pose_commander does.
o_t_ee = list(m.flatten(order="F"))
back = matrix_from_o_t_ee(o_t_ee)

check("column-major round trip reproduces the matrix",
      np.allclose(back, m, atol=1e-12),
      "worst element %.3e" % np.abs(back - m).max())
check("the translation survives it",
      np.allclose(back[:3, 3], XYZ, atol=1e-12), str(back[:3, 3]))

# And the negative: row-major would be silently wrong, not loudly wrong.
wrong = matrix_from_o_t_ee(list(m.flatten()))
check("row-major would NOT raise (which is why this test exists)",
      wrong.shape == (4, 4))
check("row-major gives a rotation that still looks valid",
      abs(np.linalg.det(wrong[:3, :3]) - 1.0) < 1e-9,
      "det = %.12f -- plausible, and wrong" % np.linalg.det(wrong[:3, :3]))
check("row-major really is the transpose, i.e. a different pose",
      not np.allclose(wrong, m, atol=1e-6)
      and np.allclose(wrong[:3, :3], m[:3, :3].T, atol=1e-12))

# O_T_EE must be exactly 16 numbers or pose_math refuses it.
check("the mock emits exactly 16 numbers", len(o_t_ee) == 16, str(len(o_t_ee)))

print()
if FAILED:
    print("%d check(s) FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("all checks passed")
