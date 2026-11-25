"""Pose maths for a camera mounted on the gripper (pure Python, no ROS).

A pose is ((x, y, z), (qx, qy, qz, qw)). The pick is resolved once, from the tool pose
at capture time:
    T_base_goal = T_base_EE(capture) . T_EE_camera . T_camera_grasp
"""
import math

IDENTITY = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


class GeometryError(Exception):
    pass


def matrix_from_quat(q):
    x, y, z, w = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def normalize(q):
    n = math.sqrt(sum(c * c for c in q))
    if n < 1e-9:
        raise GeometryError(f"quaternion {q} has zero length")
    return tuple(c / n for c in q)


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def rotate(q, v):
    r = matrix_from_quat(q)
    return tuple(sum(r[i][j] * v[j] for j in range(3)) for i in range(3))


def compose(first, second):
    """T_first . T_second."""
    t1, q1 = first
    t2, q2 = second
    r = rotate(q1, t2)
    return tuple(t1[i] + r[i] for i in range(3)), normalize(quat_mul(q1, q2))


def inverse(pose):
    """T^-1 for a pose with a unit quaternion."""
    t, (x, y, z, w) = pose
    inv = (-x, -y, -z, w)
    r = matrix_from_quat(inv)
    return tuple(-sum(r[i][j] * t[j] for j in range(3)) for i in range(3)), inv


def seven(value, what):
    """[x, y, z, qx, qy, qz, qw] from a config list -> pose."""
    try:
        nums = [float(v) for v in value]
    except (TypeError, ValueError):
        raise GeometryError(f"{what}: expected 7 numbers, got {value!r}")
    if len(nums) != 7 or not all(math.isfinite(n) for n in nums):
        raise GeometryError(f"{what}: expected 7 finite numbers [x,y,z,qx,qy,qz,qw]")
    return tuple(nums[:3]), normalize(tuple(nums[3:]))


def quat_angle(a, b):
    """Smallest rotation angle between two quaternions (rad)."""
    return 2.0 * math.acos(max(-1.0, min(1.0, abs(sum(x * y for x, y in zip(a, b))))))


def displacement(a, b):
    return math.sqrt(sum((a[0][i] - b[0][i]) ** 2 for i in range(3)))


def camera_pose_to_base(ee_pose, ee_to_camera, camera_pose, ee_to_grasp=IDENTITY):
    """Base-frame EE goal that puts the grasp frame on a camera-frame target."""
    return compose(compose(compose(ee_pose, ee_to_camera), camera_pose), inverse(ee_to_grasp))


def along_approach(pose, distance):
    """Move a pose along its own +z (approach). Negative backs off."""
    t, q = pose
    axis = rotate(q, (0.0, 0.0, 1.0))
    return tuple(t[i] + axis[i] * distance for i in range(3)), q


def jaw_flip(pose):
    """Same grasp with the fingers swapped: 180 deg about its own z."""
    t, q = pose
    return t, normalize(quat_mul(q, (0.0, 0.0, 1.0, 0.0)))


def cheaper_jaw(pose, reference_q):
    """Whichever finger order turns the wrist less from reference_q."""
    flipped = jaw_flip(pose)
    return pose if quat_angle(pose[1], reference_q) <= quat_angle(flipped[1], reference_q) else flipped


def stationary(a, b, position_tol=0.005, angle_tol=0.02):
    """True if two poses are within position_tol (m) and angle_tol (rad) of each other."""
    return displacement(a, b) <= position_tol and quat_angle(a[1], b[1]) <= angle_tol
