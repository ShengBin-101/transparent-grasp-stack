"""Pose helpers for the commander: O_T_EE parsing, quaternions and slerp, all (x, y, z, w) as geometry_msgs.

franka_states' O_T_EE is 16 numbers in column-major order; read row-major it is a plausible transposed
rotation and the arm goes somewhere else, so matrix_from_o_t_ee is the only place that layout is known.
The impedance controller ignores msg.header.frame_id, so callers must resolve poses to the base frame first.
"""

import math

import numpy as np


def matrix_from_o_t_ee(o_t_ee):
    """4x4 homogeneous transform from the 16 numbers in franka_states.O_T_EE."""
    if len(o_t_ee) != 16:
        raise ValueError("O_T_EE has %d entries, expected 16" % len(o_t_ee))
    # Column-major: numpy fills rows first, so build it transposed and flip.
    return np.asarray(o_t_ee, dtype=float).reshape(4, 4).T


def quaternion_from_matrix(m):
    """(x, y, z, w) from the rotation block of a 4x4 (or a 3x3).

    Shepperd's method: pick the largest of the four diagonal combinations so the
    square root is never taken of something near zero.
    """
    r = np.asarray(m, dtype=float)[:3, :3]
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return normalize_quaternion((x, y, z, w))


def normalize_quaternion(q):
    """q scaled to unit norm, as a tuple; ValueError on a zero quaternion."""
    n = math.sqrt(sum(c * c for c in q))
    if n < 1e-12:
        raise ValueError("cannot normalize a zero quaternion")
    return tuple(c / n for c in q)


def quaternion_dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def canonicalize(q, reference):
    """Flip q's sign if it is the long way round from `reference`.

    q and -q are the same orientation but interpolate in opposite directions, so without this a ramp can
    spin the wrist the long way (the controller does the same for its own target).
    """
    return tuple(-c for c in q) if quaternion_dot(q, reference) < 0.0 else tuple(q)


def slerp(a, b, t):
    """Shortest-arc interpolation from a to b at t in [0, 1]."""
    a = normalize_quaternion(a)
    b = canonicalize(normalize_quaternion(b), a)
    dot = min(1.0, max(-1.0, quaternion_dot(a, b)))
    if dot > 0.9995:  # nearly identical: lerp, since sin(theta) underflows
        return normalize_quaternion(tuple(x + t * (y - x) for x, y in zip(a, b)))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    wa = math.sin((1.0 - t) * theta) / sin_theta
    wb = math.sin(t * theta) / sin_theta
    return normalize_quaternion(tuple(wa * x + wb * y for x, y in zip(a, b)))


def angle_between(a, b):
    """Rotation angle in radians between two quaternions."""
    dot = abs(quaternion_dot(normalize_quaternion(a), normalize_quaternion(b)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))
