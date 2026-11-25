"""Five named capture viewpoints over the workspace, each checked by the planner before anything moves.

    plan = plan_viewpoint(n, q_now, pose_now, limits, cfg, ee_to_camera)

The camera is the same height above the table and aims at the same workspace centre from each:
1 above (straight down; also home, as home_pose.yaml holds its joints), 2 near (base side), 3 far,
4 left (+y), 5 right (-y). Side cameras sit height * tan(tilt) off centre. Defaults are measured with the
planner (joint 3 held); past 20 deg the far view nears the edge of reach (~0.73 m) and the near view joint 7:

    centre x  tilt  worst joint margin, rad (above, near, far, left, right)
     0.55      30   far out of reach; near 0.10
     0.50      20   0.96  0.30  0.41  0.82  0.84   <- default
     0.50      15   0.96  0.45  0.61  0.90  0.90
     0.45      15   0.78  0.34  0.65  0.90  0.84

No ROS imports; tools/goto_viewpoint.py is the ROS driver.
"""

import math

import numpy as np

from fr3_pose_commander import planning
from fr3_pose_commander.pose_math import angle_between, quaternion_from_matrix

NAMES = {1: "above", 2: "near", 3: "far", 4: "left", 5: "right"}
# Horizontal direction of the camera from the centre, per viewpoint.
_OFFSET = {1: (0.0, 0.0), 2: (-1.0, 0.0), 3: (1.0, 0.0), 4: (0.0, 1.0), 5: (0.0, -1.0)}

DEFAULT_CENTRE = (0.50, 0.0)
DEFAULT_HEIGHT = 0.50
DEFAULT_TILT_DEG = 20.0


def _matrix(pose7):
    """(x, y, z, qx, qy, qz, qw) -> 4x4."""
    return planning.matrix_from_pose((tuple(pose7[:3]), tuple(pose7[3:7])))


def camera_position(n, centre=DEFAULT_CENTRE, table_z=planning.TABLE_Z, height=DEFAULT_HEIGHT,
                    tilt_deg=DEFAULT_TILT_DEG):
    """Base-frame camera position for viewpoint n, and the point it looks at."""
    if n not in NAMES:
        raise ValueError("viewpoint must be 1-5, not %r" % (n,))
    target = np.array((centre[0], centre[1], table_z), float)
    d = height * math.tan(math.radians(tilt_deg))
    ox, oy = _OFFSET[n]
    return target + np.array((ox * d, oy * d, height)), target


def camera_pose(position, target, roll_rad):
    """4x4 camera optical frame at `position`, z looking at `target`, rolled."""
    z = target - position
    z = z / np.linalg.norm(z)
    up = np.array((1.0, 0.0, 0.0)) if abs(z[0]) < 0.9 else np.array((0.0, 1.0, 0.0))
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c, s = math.cos(roll_rad), math.sin(roll_rad)
    m = np.eye(4)
    m[:3, :3] = np.stack([x, y, z], 1).dot(np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))))
    m[:3, 3] = position
    return m


def split_move(start, goal, max_leg_deg=45.0, max_leg_m=0.20):
    """planning.split_move, with the legs named as viewpoint legs."""
    return planning.split_move(start, goal, max_leg_deg, max_leg_m, label="viewpoint leg")


def _end_and_margin(r, limits):
    """(predicted end joints, worst joint margin over every prediction) of a passing chain."""
    ends = [np.array(q) for q in r["q_end"].values()]
    margin = min(float(np.minimum(q - limits.lower, limits.upper - q).min()) for q in ends)
    return ends[-1], margin


def plan_viewpoint(n, q_now, pose_now, limits, cfg, ee_to_camera, centre=DEFAULT_CENTRE,
                   table_z=planning.TABLE_Z, height=DEFAULT_HEIGHT, tilt_deg=DEFAULT_TILT_DEG,
                   max_leg_deg=45.0, max_leg_m=0.20, roll_step_deg=15.0, margin_floor=0.30,
                   model="elbow_locked", q_ns=None):
    """The best feasible way to viewpoint n from (q_now, pose_now), or the reason there is none.

    Every roll_step_deg of camera roll is tried: the move is split into legs (max_leg_deg, max_leg_m) and
    walked by planning.check_legs under `model` ("elbow_locked" for fr3_position_control, "nullspace" with
    q_ns for impedance). Among rolls leaving margin_floor at every joint the least wrist turn wins, since
    maximising margin alone spun the camera needlessly; if none does, the most margin.
    Returns a dict: ok, name, number, camera_position, target, distance_m, roll_deg, turn_deg,
    legs [(name, pose)], q_end, joint_margin, sigma_min, reasons.
    """
    cam, target = camera_position(n, centre, table_z, height, tilt_deg)
    ee_cam = _matrix(ee_to_camera)
    cam_ee = np.linalg.inv(ee_cam)
    feasible, first_reason = [], None
    for roll in np.arange(0.0, 360.0, roll_step_deg):
        ee_goal = camera_pose(cam, target, math.radians(roll)).dot(cam_ee)
        goal = (tuple(ee_goal[:3, 3]), quaternion_from_matrix(ee_goal))
        legs = split_move(pose_now, goal, max_leg_deg, max_leg_m)
        r = planning.check_legs(q_now, pose_now, legs, model, limits, cfg, q_ns)
        if not r["ok"]:
            first_reason = first_reason or r["reasons"]
            continue
        q, margin = _end_and_margin(r, limits)
        feasible.append((margin, angle_between(pose_now[1], goal[1]), roll, legs, q))
    best = None
    if feasible:
        good = [f for f in feasible if f[0] >= margin_floor]
        margin, turn, roll, legs, q = (min(good, key=lambda f: f[1]) if good
                                       else max(feasible, key=lambda f: f[0]))
        sigma = float(np.linalg.svd(planning.kinematics.jacobian_array(q), compute_uv=False)[-1])
        best = {"ok": True, "roll_deg": float(roll), "legs": legs, "q_end": q.tolist(),
                "joint_margin": margin, "sigma_min": sigma, "turn_deg": math.degrees(turn),
                "reasons": []}
    out = best or {"ok": False, "reasons": first_reason or ["no roll reachable"]}
    out.update(name=NAMES[n], number=n, camera_position=cam.tolist(), target=target.tolist(),
               distance_m=float(np.linalg.norm(target - cam)))
    return out


def home_pose(home_joints):
    """Base-frame tool pose for home_pose.yaml's joints: where go_home ends."""
    m = planning.kinematics.ee_matrix(home_joints)
    return (tuple(m[:3, 3]), quaternion_from_matrix(m))


def plan_home(q_now, pose_now, home, limits, cfg, model="elbow_locked", q_ns=None):
    """The move to `home`, cut exactly as the commander's go_home cuts it
    (planning.home_legs), checked under the running controller's model.
    Same dict shape as plan_viewpoint, so goto_viewpoint prints both alike."""
    legs = planning.home_legs(pose_now, home, cfg)
    r = planning.check_legs(q_now, pose_now, legs, model, limits, cfg, q_ns)
    out = {"ok": r["ok"], "reasons": r["reasons"], "legs": legs, "roll_deg": None,
           "turn_deg": math.degrees(angle_between(pose_now[1], home[1]))}
    if r["ok"]:
        q, margin = _end_and_margin(r, limits)
        out.update(q_end=q.tolist(), joint_margin=margin,
                   sigma_min=float(np.linalg.svd(planning.kinematics.jacobian_array(q),
                                                 compute_uv=False)[-1]))
    return out
