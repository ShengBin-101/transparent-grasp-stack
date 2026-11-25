"""FR3 forward kinematics and base-frame Jacobian, for the planner and for conditioning reports.

    sigma, direction = smallest_singular_value(jacobian(q))

The impedance torque is J^T times the pose error, so where J is near-singular the arm has little authority,
which looks like a controller set too soft. sigma_min is reported, not enforced: the only thresholds ever
used were calibrated on a wrong Jacobian, and the postures held on the rig measure 0.10-0.22 (home 0.22).
test_kinematics.py checks every Jacobian column against a finite difference of the forward kinematics.
Ported from live_grasp/src/live_grasp/geometry.py.
"""

import math

import numpy as np

# Modified (Craig) DH for the FR3: (a, d, alpha) per joint.
FR3_DH = ((0.0, 0.333, 0.0), (0.0, 0.0, -math.pi / 2),
          (0.0, 0.316, math.pi / 2), (0.0825, 0.0, math.pi / 2),
          (-0.0825, 0.384, -math.pi / 2), (0.0, 0.0, math.pi / 2),
          (0.088, 0.0, math.pi / 2))
FLANGE_Z = 0.107
# The Franka Hand's TCP offset along flange z (franka_msgs' F_T_EE).
HAND_EE_Z = 0.1034


# The hand's rotation about flange z (F_T_EE, URDF fr3_hand_joint rpy 0 0 -pi/4): -45 deg. Reading F_T_EE
# row-wise gives +45, a pose 90 deg out. It does not enter the base-frame Jacobian but sets the fingers.
HAND_EE_YAW = -math.pi / 4

# Rot_x(alpha) . Trans_x(a) for each joint: fixed, so built once.
_FIXED = []
for _a, _d, _alpha in FR3_DH:
    _ca, _sa = math.cos(_alpha), math.sin(_alpha)
    _FIXED.append(np.array(((1.0, 0.0, 0.0, _a),
                            (0.0, _ca, -_sa, 0.0),
                            (0.0, _sa, _ca, 0.0),
                            (0.0, 0.0, 0.0, 1.0))))


def _tool(offset_z, yaw):
    """4x4 tool transform: Rz(yaw) with `offset_z` along z."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(((c, -s, 0.0, 0.0), (s, c, 0.0, 0.0),
                     (0.0, 0.0, 1.0, offset_z), (0.0, 0.0, 0.0, 1.0)))


_FLANGE = _tool(FLANGE_Z, 0.0)


def forward(joints, ee_offset_z=HAND_EE_Z, ee_yaw=HAND_EE_YAW):
    """The whole chain as arrays, which every other function here reads.

    Returns (axis_origins 7x3, axes 7x3, link_frames [7 x 4x4], flange 4x4, ee 4x4), all in the base
    frame; link_frames[i] is link i+1's frame. In the modified (Craig) convention a step is
    Rot_x(alpha) . Trans_x(a) . Rot_z(theta) . Trans_z(d), so joint i's axis is taken partway through it,
    after alpha and before theta. Taking it before the whole step shifts every Jacobian column by one joint
    while forward kinematics still matches O_T_EE, which is how that bug once went unnoticed.
    """
    t = np.eye(4)
    origins = np.empty((7, 3))
    axes = np.empty((7, 3))
    frames = []
    for i in range(7):
        t = t.dot(_FIXED[i])
        origins[i] = t[:3, 3]
        axes[i] = t[:3, 2]
        ct, st = math.cos(joints[i]), math.sin(joints[i])
        t = t.dot(np.array(((ct, -st, 0.0, 0.0),
                            (st, ct, 0.0, 0.0),
                            (0.0, 0.0, 1.0, FR3_DH[i][1]),
                            (0.0, 0.0, 0.0, 1.0))))
        frames.append(t)
    flange = t.dot(_FLANGE)
    ee = flange.dot(_tool(ee_offset_z, ee_yaw))
    return origins, axes, frames, flange, ee


def ee_position(joints, ee_offset_z=HAND_EE_Z):
    """The controlled EE origin in the base frame."""
    return tuple(forward(joints, ee_offset_z)[4][:3, 3])


def ee_matrix(joints, ee_offset_z=HAND_EE_Z, ee_yaw=HAND_EE_YAW):
    """The controlled EE pose as a 4x4: what franka_states reports as O_T_EE.

    Driving the tool to this pose does not reproduce the posture: with seven joints for six DOF that is
    the controller's choice, which planning.py models.
    """
    return forward(joints, ee_offset_z, ee_yaw)[4]


def jacobian_array(joints, ee_offset_z=HAND_EE_Z):
    """The 6x7 base-frame geometric Jacobian at the controlled EE, rows (v, omega).

    The same quantity the controller uses (getZeroJacobian(kEndEffector)), so its conditioning describes
    the torques the controller will actually produce.
    """
    origins, axes, _, _, ee = forward(joints, ee_offset_z)
    j = np.empty((6, 7))
    j[:3] = np.cross(axes, ee[:3, 3] - origins).T
    j[3:] = axes.T
    return j


def jacobian(joints, ee_offset_z=HAND_EE_Z):
    """`jacobian_array` as nested lists, for the callers written against it."""
    return jacobian_array(joints, ee_offset_z).tolist()


def link_points(joints, ee_to_camera=None):
    """Named base-frame points on the arm that must stay clear of the table.

    The origins of links 4-7, the flange, the TCP, and the camera when `ee_to_camera` ((x, y, z, ...) in
    the EE frame, m) is given. Points, not volumes: the clearance margin covers the links' radius.
    """
    _, _, frames, flange, ee = forward(joints)
    points = {"link%d" % (i + 1): frames[i][:3, 3] for i in (3, 4, 5, 6)}
    points["flange"] = flange[:3, 3]
    points["tcp"] = ee[:3, 3]
    if ee_to_camera is not None:
        points["camera"] = ee[:3, :3].dot(np.asarray(ee_to_camera[:3], float)) + ee[:3, 3]
    return points


def smallest_singular_value(rows):
    """(sigma_min, the task-space direction it belongs to).

    The direction is the left singular vector: the direction in task space the
    arm is least able to push in, which is the useful half of the number.
    """
    j = np.asarray(rows, dtype=float)
    u, s, _ = np.linalg.svd(j, full_matrices=False)
    k = int(np.argmin(s))
    return float(s[k]), tuple(float(v) for v in u[:, k])


def describe_direction(direction):
    """The dominant component of a task-space direction, in words."""
    names = ("x", "y", "z", "roll about x", "pitch about y", "yaw about z")
    big = max(range(6), key=lambda i: abs(direction[i]))
    kind = "translation along base " if big < 3 else ""
    return "%s%s" % (kind, names[big])


def report(joints, ee_offset_z=HAND_EE_Z):
    """One short phrase for a log line, or None if `joints` is unusable."""
    try:
        q = [float(v) for v in joints]
        if len(q) != 7 or not all(math.isfinite(v) for v in q):
            return None
        sigma, direction = smallest_singular_value(jacobian(q, ee_offset_z))
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    return "sigma_min %.3f, weakest in %s" % (sigma, describe_direction(direction))
