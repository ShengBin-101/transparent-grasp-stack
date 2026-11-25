"""Joint-space feasibility of a pick on the FR3: IK along every leg, checked sample by sample.

    limits = load_limits(".../franka_description/robots/fr3/joint_limits.yaml")
    result = plan_legs(q_now, pose_now, candidates, model_for_topic(topic),
                       limits, PlanConfig(), q_ns=home_joints, home_pose=home)

The grasp detectors and the commander's tool checks know nothing of the joints, so each leg is walked along
the path both controllers take (straight line + slerp, every 1 cm / 2 deg), IK seeded from the previous
sample, and checked for joint limits, posture flips, joint speed and table/scene/finger collisions.
Feasible candidates are ranked; the Orin's candidates and scene are parsed here too.
No ROS imports (test_planning.py).
"""

import math
import time

import numpy as np

from fr3_pose_commander import kinematics
from fr3_pose_commander.scene import Scene, arm_spheres, finger_spheres
from fr3_pose_commander.pose_math import angle_between, normalize_quaternion, slerp

MODELS = ("nullspace", "elbow_locked")
# m, the table surface in fr3_link0, measured by guarded contact on the rig. The fallback wherever
# config/pose_commander.yaml's planner/table_z is not given; re-measure (and set both) if the table moves.
TABLE_Z = -0.041
_ELBOW = 2          # joint 3, the one libfranka's elbow command fixes


class Limits(object):
    """Joint position and velocity limits, arrays of 7."""

    def __init__(self, lower, upper, velocity):
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        self.velocity = np.asarray(velocity, float)


def load_limits(path):
    """Limits from franka_description's joint_limits.yaml, the file the robot description is built from."""
    import yaml
    with open(path) as handle:
        data = yaml.safe_load(handle)
    rows = [data["joint%d" % (i + 1)]["limit"] for i in range(7)]
    return Limits([r["lower"] for r in rows], [r["upper"] for r in rows],
                  [r["velocity"] for r in rows])


def model_for_topic(command_topic):
    """The redundancy model for the controller a commander publishes to.

    "elbow_locked" (fr3_position_control, .../target_pose): the controller holds joint 3 (hold_elbow), so
    IK drops its Jacobian column. "nullspace" (Cartesian impedance): its weak nullspace pull (stiffness 2)
    leaves the posture somewhere between no pull and fully pulled, so both ends are predicted and both
    must pass.
    """
    return "elbow_locked" if command_topic.rstrip("/").endswith("target_pose") else "nullspace"


class PlanConfig(object):
    """Every threshold the planner applies, defaulting to this rig's values.

    Keyword overrides must name an existing setting. sigma_min is recorded, not enforced (min_sigma 0): no
    failure has been traced to conditioning (see kinematics.py).
    """

    def __init__(self, **overrides):
        self.sample_step_m = 0.01
        self.sample_step_rad = math.radians(2.0)
        self.converge_m = 1e-3
        self.converge_rad = math.radians(0.5)
        self.joint_margin = 0.05            # rad inside each limit
        self.max_step_dq = 0.1              # rad per sample: more is a flip
        self.elbow_straight_margin = 0.5    # rad below joint 4's upper limit counts as the edge of reach,
        self.reach_sigma = 0.08             # as does sigma_min below this
        self.table_z = TABLE_Z              # m, measured by guarded contact on the rig
        # m above the table for clear_points; not the TCP, which the guarded grasp takes below the surface.
        self.clearance = 0.03
        self.clear_points = ("link4", "link5", "link6", "link7", "flange", "camera")
        self.ee_to_camera = None            # (x, y, z, ...) in the EE frame
        self.linear_speed = 0.012           # m/s, the commander's ramp
        self.angular_speed = 0.40           # rad/s
        self.max_speed_frac = 0.5           # of each joint's velocity limit
        self.min_sigma = 0.0                # 0 = record only
        self.rotation_max = None            # rad per leg, the commander's limits/rotation_max; None = no limit
        self.scene_margin = 0.01            # m of surface clearance from scene voxels
        # No margin from the target's own voxels: the bound already overstates every gap by up to ~22 mm,
        # and the guarded grasp stops on contact.
        self.target_margin = 0.0
        # "plane": moving parts checked against z = table_z, table voxels dropped from the scene
        # (table_plane_check, objects_only). "voxels": the table stays in the voxel scene.
        self.table_mode = "plane"
        self.table_margin = 0.005           # m, on a bound already <= 14 mm low
        self.table_band = 0.02              # m above the table; voxel centres below are table
        # Checked against nothing: link1's capsule reaches the base plate and would hit the table beside it.
        self.static_parts = ("link1",)
        # The commander's limits/z_min and limits/radius_max, prescreened before any path is walked.
        # None = not checked.
        self.z_min = None
        self.radius_max = None
        # "utility": highest confidence x comfort (comfort()); "confidence": the first feasible.
        self.rank = "utility"
        self.util_joint_ok = 0.30           # rad of joint margin that counts as comfortable
        self.util_clear_ok = 0.03           # m of table / scene clearance
        self.util_sigma_ok = 0.10           # sigma_min
        # The gripper's own volume at the grasp in the TCP frame (y = jaws, z = approach), removed from
        # the scene for the legs that end at or leave from the grasp (near_target).
        self.target_half_x = 0.06           # across the jaws
        self.target_jaw_pad = 0.03          # beyond width/2, along the jaws
        self.target_back = 0.08             # behind the TCP, toward the palm
        self.target_front = 0.03            # past the TCP
        self.default_width = 0.08           # when a candidate carries none
        # Fingers, open finger_opening apart (the sequencer's open_width), checked on labelled scenes
        # only; unlabelled scenes rely on SBG's gripper check.
        self.finger_check = True
        self.finger_opening = 0.078
        self.ns_gain = 1.0                  # nullspace pull per iteration; full step, as the projector is exact
        self.damping = 0.01                 # damped least squares
        self.max_iter = 60
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise TypeError("PlanConfig has no setting %r" % key)
            setattr(self, key, value)


# -- geometry ------------------------------------------------------------------

def matrix_from_pose(pose):
    """((x, y, z), (qx, qy, qz, qw)) -> 4x4."""
    (px, py, pz), q = pose
    x, y, z, w = normalize_quaternion(q)
    m = np.eye(4)
    m[:3, :3] = ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                 (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                 (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))
    m[:3, 3] = (px, py, pz)
    return m


def pose_error(current, target):
    """6-vector (translation, rotation vector) taking `current` to `target`,
    both 4x4, in the base frame -- the same frame as the Jacobian's rows."""
    e = np.empty(6)
    e[:3] = target[:3, 3] - current[:3, 3]
    r = target[:3, :3].dot(current[:3, :3].T)
    vee = np.array((r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]))
    cos = max(-1.0, min(1.0, (np.trace(r) - 1.0) / 2.0))
    angle = math.acos(cos)
    if angle < 1e-9:
        e[3:] = 0.5 * vee
    elif math.pi - angle < 1e-6:
        # At half a turn vee vanishes, so the axis comes from the diagonal.
        axis = np.sqrt(np.maximum((np.diag(r) + 1.0) / 2.0, 0.0))
        e[3:] = angle * axis / max(np.linalg.norm(axis), 1e-12)
    else:
        e[3:] = angle / (2.0 * math.sin(angle)) * vee
    return e


def _jacobian(origins, axes, ee):
    """6x7 base-frame geometric Jacobian at `ee` from kinematics.forward()'s axis origins and axes."""
    j = np.empty((6, 7))
    j[:3] = np.cross(axes, ee[:3, 3] - origins).T
    j[3:] = axes.T
    return j


def ik_dls(target, seed, model, q_ns=None, cfg=None):
    """Joint angles reaching `target` (4x4) from `seed`: (q, position_error_m, rotation_error_rad, iterations).

    Damped least squares, so a near-singular sample degrades gracefully and then fails the convergence
    check. "elbow_locked" never moves joint 3. "nullspace" with `q_ns` keeps pulling toward q_ns through the
    task's null space until the pull stops moving anything: the converged end of the impedance controller's
    nullspace term. The pull uses the exact projector, since the damped one leaks into the task and the two
    then trade forever; damping stays on the task step.
    """
    cfg = cfg or PlanConfig()
    if model not in MODELS:
        raise ValueError("unknown redundancy model %r" % (model,))
    q = np.array(seed, dtype=float)
    lam2 = cfg.damping ** 2
    ep = er = float("inf")
    for iteration in range(1, cfg.max_iter + 1):
        origins, axes, _, _, ee = kinematics.forward(q)
        e = pose_error(ee, target)
        ep, er = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
        j = _jacobian(origins, axes, ee)
        if model == "elbow_locked":
            jr = np.delete(j, _ELBOW, axis=1)
            dq = np.insert(jr.T.dot(np.linalg.solve(jr.dot(jr.T) + lam2 * np.eye(6), e)),
                           _ELBOW, 0.0)
            pull = 0.0
        else:
            jp = j.T.dot(np.linalg.inv(j.dot(j.T) + lam2 * np.eye(6)))
            dq = jp.dot(e)
            pull = 0.0
            if q_ns is not None:
                # J^T (J J^T)^-1 J by a 6x6 solve: pinv(J).J for a full-rank J, far cheaper than an SVD.
                try:
                    null = np.eye(7) - j.T.dot(np.linalg.solve(j.dot(j.T), j))
                except np.linalg.LinAlgError:
                    null = np.eye(7) - np.linalg.pinv(j).dot(j)
                ns = null.dot(cfg.ns_gain * (np.asarray(q_ns) - q))
                dq = dq + ns
                pull = float(np.max(np.abs(ns)))
        if ep < 1e-5 and er < 1e-4 and pull < 1e-5:
            return q, ep, er, iteration
        biggest = float(np.max(np.abs(dq)))
        if biggest > 0.2:
            dq *= 0.2 / biggest
        q = q + dq
    origins, axes, _, _, ee = kinematics.forward(q)
    e = pose_error(ee, target)
    return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:])), cfg.max_iter


def sample_leg(start_pose, goal_pose, cfg=None):
    """(targets [4x4, ...], dt_s): the path the commander's ramp takes, straight line plus shortest-arc slerp.

    Sampled every sample_step_m or sample_step_rad, whichever gives more samples; the start is excluded,
    the goal included. dt_s is the ramp's time per sample, from the same duration rule _ramp uses.
    """
    cfg = cfg or PlanConfig()
    p0, q0 = np.asarray(start_pose[0], float), start_pose[1]
    p1, q1 = np.asarray(goal_pose[0], float), goal_pose[1]
    dist = float(np.linalg.norm(p1 - p0))
    turn = angle_between(q0, q1)
    n = max(1, int(math.ceil(dist / cfg.sample_step_m)),
            int(math.ceil(turn / cfg.sample_step_rad)))
    duration = max(dist / cfg.linear_speed, turn / cfg.angular_speed, 1e-9)
    targets = []
    for i in range(1, n + 1):
        t = float(i) / n
        targets.append(matrix_from_pose((p0 + t * (p1 - p0), slerp(q0, q1, t))))
    return targets, duration / n


# -- one leg --------------------------------------------------------------------

def check_leg(q_start, start_pose, goal_pose, model, limits, cfg=None, q_ns=None,
              scene=None):
    """Walk one leg sample by sample, stopping at the first failure: {"ok", "reasons", "q_end", "stats"}.

    The wrist-turn cap comes first: it needs no IK and the commander would refuse the leg anyway. With
    `q_ns` the start posture is first settled toward q_ns at the fixed start pose, as a standing impedance
    arm does, so that relaxation is not read as a flip. Each sample then checks IK convergence, the joint
    step (a flip, or the edge of reach near full extension), joint limits, joint speed (elbow_locked only,
    where it is a firmware reflex; impedance just lags), arm points and spheres above the table, the scene
    (plus the fingers on a labelled scene) and the target's voxels. `q_end` is the prediction at the goal,
    or at the failing sample.
    """
    cfg = cfg or PlanConfig()
    if cfg.rotation_max is not None:
        turn = angle_between(start_pose[1], goal_pose[1])
        if turn > cfg.rotation_max:
            return {"ok": False,
                    "reasons": ["the leg turns the wrist %.0f deg, past the %.0f deg "
                                "per-command limit (limits/rotation_max)"
                                % (math.degrees(turn), math.degrees(cfg.rotation_max))],
                    "q_end": np.array(q_start, dtype=float),
                    "stats": {"samples": 0, "iterations": 0, "turn_deg": math.degrees(turn),
                              "min_joint_margin": None, "tightest_joint": None,
                              "min_clearance": None, "closest_point": None,
                              "min_sigma": None, "max_speed_frac": 0.0,
                              "max_step_dq": 0.0, "min_scene_clearance": None,
                              "scene_part": None, "min_table_clearance": None,
                              "table_part": None, "min_target_clearance": None,
                              "target_part": None}}
    targets, dt = sample_leg(start_pose, goal_pose, cfg)
    q = np.array(q_start, dtype=float)
    if q_ns is not None:
        settled, ep, er, _ = ik_dls(matrix_from_pose(start_pose), q, model, q_ns, cfg)
        if ep <= cfg.converge_m and er <= cfg.converge_rad:
            q = settled
    reasons = []
    stats = {"samples": len(targets), "iterations": 0,
             "min_joint_margin": float("inf"), "tightest_joint": None,
             "min_clearance": float("inf"), "closest_point": None,
             "min_sigma": float("inf"), "max_speed_frac": 0.0, "max_step_dq": 0.0,
             "min_scene_clearance": float("inf"), "scene_part": None,
             "min_table_clearance": float("inf"), "table_part": None,
             "min_target_clearance": float("inf"), "target_part": None}
    for i, target in enumerate(targets):
        where = "%d%% of the leg" % round(100.0 * (i + 1) / len(targets))
        q_new, ep, er, iterations = ik_dls(target, q, model, q_ns, cfg)
        stats["iterations"] += iterations
        if ep > cfg.converge_m or er > cfg.converge_rad:
            reasons.append("no joint solution at %s (%.1f mm, %.2f deg short)"
                           % (where, ep * 1e3, math.degrees(er)))
            break
        dq = q_new - q
        step = float(np.max(np.abs(dq)))
        stats["max_step_dq"] = max(stats["max_step_dq"], step)
        if step > cfg.max_step_dq:
            joint = int(np.argmax(np.abs(dq))) + 1
            # Near full extension joints swing fast for small tool motion: name that the edge of reach,
            # not a flip. Only the reason differs.
            sigma = float(np.linalg.svd(kinematics.jacobian_array(q_new),
                                        compute_uv=False)[-1])
            if (q_new[3] > limits.upper[3] - cfg.elbow_straight_margin
                    or sigma < cfg.reach_sigma):
                reasons.append("the target is at the edge of reach at %s: the arm is "
                               "nearly stretched out (joint 4 at %.2f rad, straight at "
                               "%.2f; sigma_min %.3f), so joint %d swings %.2f rad in "
                               "one step"
                               % (where, q_new[3], limits.upper[3], sigma, joint, step))
            else:
                reasons.append("posture flip at %s: joint %d moves %.2f rad in one step"
                               % (where, joint, step))
            break

        room = np.minimum(q_new - limits.lower, limits.upper - q_new)
        worst = int(np.argmin(room))
        if room[worst] < stats["min_joint_margin"]:
            stats["min_joint_margin"], stats["tightest_joint"] = float(room[worst]), worst + 1
        if room[worst] < cfg.joint_margin:
            reasons.append("joint %d reaches %.3f rad at %s, %s its limit [%.3f, %.3f]"
                           % (worst + 1, q_new[worst], where,
                              "past" if room[worst] < 0 else "%.1f deg inside"
                              % math.degrees(room[worst]),
                              limits.lower[worst], limits.upper[worst]))
            break

        speed = float(np.max(np.abs(dq) / dt / limits.velocity))
        stats["max_speed_frac"] = max(stats["max_speed_frac"], speed)
        if model == "elbow_locked" and speed > cfg.max_speed_frac:
            reasons.append("joint speed reaches %.0f%% of its limit at %s"
                           % (100 * speed, where))
            break

        points = kinematics.link_points(q_new, cfg.ee_to_camera)
        for name in cfg.clear_points:
            if name not in points:
                continue
            height = float(points[name][2] - cfg.table_z)
            if height < stats["min_clearance"]:
                stats["min_clearance"], stats["closest_point"] = height, name
        if stats["min_clearance"] < cfg.clearance:
            reasons.append("%s comes within %.0f mm of the table at %s"
                           % (stats["closest_point"], stats["min_clearance"] * 1e3, where))
            break

        spheres = None
        if cfg.table_mode == "plane":
            spheres = arm_spheres(q_new, cfg.ee_to_camera)
            low, part = table_plane_check(spheres, cfg)
            if low < stats["min_table_clearance"]:
                stats["min_table_clearance"], stats["table_part"] = low, part
            if low < cfg.table_margin:
                reasons.append(("%s goes %.0f mm into the table at %s" if low < 0 else
                                "%s comes within %.0f mm of the table (plane) at %s")
                               % (part, abs(low) * 1e3, where))
                break

        target = getattr(scene, "target", None) if scene is not None else None
        if scene is not None and (len(scene) or (target is not None and len(target))):
            if spheres is None:
                spheres = arm_spheres(q_new, cfg.ee_to_camera)
            moving = [sp for sp in spheres if sp[0] not in cfg.static_parts]
            if len(scene):
                obstacles_see = moving
                if cfg.finger_check and getattr(scene, "box_target_only", False):
                    obstacles_see = moving + finger_spheres(q_new, cfg.finger_opening)
                gap, part = scene.clearance(obstacles_see)
                if gap < stats["min_scene_clearance"]:
                    stats["min_scene_clearance"], stats["scene_part"] = gap, part
                if gap < cfg.scene_margin:
                    reasons.append(("%s intersects the scene (%.0f mm) at %s" if gap < 0 else
                                    "%s comes within %.0f mm of the scene at %s")
                                   % (part, abs(gap) * 1e3, where))
                    break
            if target is not None and len(target):
                gap, part = target.clearance(moving)
                if gap < stats["min_target_clearance"]:
                    stats["min_target_clearance"], stats["target_part"] = gap, part
                if gap < cfg.target_margin:
                    reasons.append(("%s intersects the target object (%.0f mm) at %s" if gap < 0
                                    else "%s comes within %.0f mm of the target object at %s")
                                   % (part, abs(gap) * 1e3, where))
                    break

        j = kinematics.jacobian_array(q_new)
        if model == "elbow_locked":
            j = np.delete(j, _ELBOW, axis=1)
        sigma = float(np.linalg.svd(j, compute_uv=False)[-1])
        stats["min_sigma"] = min(stats["min_sigma"], sigma)
        if cfg.min_sigma > 0 and sigma < cfg.min_sigma:
            reasons.append("sigma_min %.3f at %s (limit %.3f)" % (sigma, where, cfg.min_sigma))
            break
        q = q_new
    else:
        return {"ok": True, "reasons": [], "q_end": q, "stats": stats}
    return {"ok": False, "reasons": reasons, "q_end": q_new, "stats": stats}


# -- a whole pick -----------------------------------------------------------------

def _predictions(model, q_ns):
    """[(label, pull target or None)]: one for elbow_locked; no pull and fully pulled for nullspace."""
    if model == "elbow_locked":
        return [("elbow_locked", None)]
    if q_ns is None:
        return [("no_pull", None)]
    return [("no_pull", None), ("pulled", np.asarray(q_ns, float))]


# Legs cut_over_cap never cuts: the guarded grasp is a contact-seeking descent, not one to run in pieces.
NO_SPLIT = ("grasp",)


def cut_over_cap(start_pose, legs, cfg):
    """`legs` with every leg turning the wrist more than cfg.rotation_max cut into pieces of at most
    leg_split(cfg), named "<leg> i/n", exactly as the commander cuts the move it runs.

    The cap exists because one large reorientation aborted under impedance; it is not a joint limit, so a
    cut move is limited only by joint 7's range, which check_leg checks."""
    if not cfg.rotation_max:
        return list(legs)
    out, prev = [], start_pose
    for name, goal in legs:
        if name not in NO_SPLIT and angle_between(prev[1], goal[1]) > cfg.rotation_max:
            deg, m = leg_split(cfg)
            out.extend(split_move(prev, goal, deg, m, label=name))
        else:
            out.append((name, goal))
        prev = goal
    return out


def check_legs(q_start, start_pose, legs, model, limits, cfg=None, q_ns=None,
               deadline=None, scene=None, scene_near_target=None):
    """Chain legs [(name, goal_pose), ...] from (q_start, start_pose), each from the previous one's
    predicted end, cut by cut_over_cap. Every prediction must pass; ok is None if `deadline` passes.

    "grasp" and "retreat", the legs that touch the target, use `scene_near_target` (the gripper's own
    volume removed) when given; every other leg uses `scene`."""
    cfg = cfg or PlanConfig()
    legs = cut_over_cap(start_pose, legs, cfg)
    out = {"ok": True, "failed_leg": None, "prediction": None, "reasons": [],
           "legs": [], "q_end": {}}
    for label, pull in _predictions(model, q_ns):
        q, pose = np.array(q_start, float), start_pose
        for index, (name, goal) in enumerate(legs):
            if deadline is not None and time.time() > deadline:
                out.update(ok=None, failed_leg=name, prediction=label,
                           reasons=["planning budget spent before this leg"])
                return out
            leg_scene = (scene_near_target if name in ("grasp", "retreat")
                         and scene_near_target is not None else scene)
            result = check_leg(q, pose, goal, model, limits, cfg, pull, leg_scene)
            stats = dict(result["stats"], leg=name, prediction=label)
            out["legs"].append(_jsonable(stats))
            if not result["ok"]:
                out.update(ok=False, failed_leg=name, prediction=label,
                           reasons=result["reasons"])
                out["q_end"][label] = result["q_end"].tolist()
                return out
            q, pose = result["q_end"], goal
        out["q_end"][label] = q.tolist()
    return out


def scene_labels(scene):
    """How the scene tells the grasped object apart, or None.

    "instances": per candidate, the keep-out box and zero margin apply to that object only and the fingers
    are checked. "flags": one target for every candidate, and the box removes whatever is inside it."""
    if scene is None:
        return None
    if getattr(scene, "instance", None) is not None:
        return "instances"
    return "flags" if getattr(scene, "target", None) is not None else None


def candidate_scene(scene, candidate):
    """The scene one candidate is checked against: on a labelled scene its "instance" object is the target
    (Scene.for_instance; no instance = no target, so nothing gives way), otherwise the scene as sent."""
    if scene is None or getattr(scene, "instance", None) is None:
        return scene
    return scene.for_instance(candidate.get("instance"))


def near_target(scene, grasp_pose, width, cfg):
    """`scene` minus the gripper's own volume at `grasp_pose` (PlanConfig's target_* box), from both the
    obstacles and the target. A scene that is all target still gets the box, or every grasp would
    intersect its own object."""
    target = getattr(scene, "target", None) if scene is not None else None
    if scene is None or not (len(scene) or (target is not None and len(target))):
        return scene
    w = cfg.default_width if width is None else float(width)
    return scene.without_box(matrix_from_pose(grasp_pose),
                             (cfg.target_half_x, w / 2.0 + cfg.target_jaw_pad),
                             (-cfg.target_back, cfg.target_front))


def split_move(start, goal, max_leg_deg=45.0, max_leg_m=0.20, label="leg"):
    """Straight line + slerp from `start` to `goal`, cut into equal legs of at most max_leg_deg / max_leg_m.

    The commander ramps each leg along exactly this path, so a move cut here and checked with check_legs
    is the move that runs. Legs are named "<label> i/n", or just `label` when one is enough.
    """
    p0, q0 = np.asarray(start[0], float), start[1]
    p1, q1 = np.asarray(goal[0], float), goal[1]
    n = max(1, int(math.ceil(math.degrees(angle_between(q0, q1)) / max_leg_deg)),
            int(math.ceil(np.linalg.norm(p1 - p0) / max_leg_m)))
    return [(label if n == 1 else "%s %d/%d" % (label, i + 1, n),
             (tuple(p0 + (i + 1.0) / n * (p1 - p0)), slerp(q0, q1, (i + 1.0) / n)))
            for i in range(n)]


def leg_split(cfg):
    """(max_leg_deg, max_leg_m) for the moves the pipeline cuts itself (go_home, viewpoints).

    45 deg, or 5 deg under the commander's rotation_max if tighter, so no leg hits the wrist-turn limit."""
    deg = 45.0
    if cfg is not None and cfg.rotation_max:
        deg = min(deg, math.degrees(cfg.rotation_max) - 5.0)
    return max(deg, 5.0), 0.20


def home_legs(start, home, cfg=None):
    """The legs go_home runs from `start`: split_move with leg_split(cfg).

    Shared by the commander's go_home and plan_legs's return-home check, so the return the planner
    approves is the one that runs.
    """
    deg, m = leg_split(cfg)
    return split_move(start, home, deg, m, label="home")


def prescreen(legs, cfg):
    """The reason the commander would refuse one of `legs` outright (limits/z_min, limits/radius_max), or None.

    Exact and free, so a candidate that can never run does not spend the planning budget."""
    for name, goal in legs:
        x, y, z = goal[0]
        if cfg.z_min is not None and z < cfg.z_min:
            return ("%s: TCP at z = %.0f mm is below the commander's floor (%.0f mm)"
                    % (name, z * 1e3, cfg.z_min * 1e3))
        if cfg.radius_max is not None and math.hypot(x, y) > cfg.radius_max:
            return ("%s: %.0f mm from the base, past the commander's reach (%.0f mm)"
                    % (name, math.hypot(x, y) * 1e3, cfg.radius_max * 1e3))
    return None


def comfort(result, cfg):
    """(comfort in (0, 1], its terms) of a feasible check_legs result.

    The product of three terms over every leg and prediction, each capped at 1: worst joint margin /
    util_joint_ok, worst table or scene clearance / util_clear_ok, worst sigma_min / util_sigma_ok. So a
    grasp feasible by a millimetre does not beat a comfortable one, and nothing beyond comfortable counts."""
    joint = clear = sigma = float("inf")
    for st in result.get("legs") or []:
        if st.get("min_joint_margin") is not None:
            joint = min(joint, st["min_joint_margin"])
        for key in ("min_table_clearance", "min_scene_clearance"):
            if st.get(key) is not None:
                clear = min(clear, st[key])
        if st.get("min_sigma") is not None:
            sigma = min(sigma, st["min_sigma"])

    def s(value, ok):
        if not math.isfinite(value):
            return 1.0
        return max(1e-3, min(1.0, value / ok))
    terms = {"joint": s(joint, cfg.util_joint_ok), "clearance": s(clear, cfg.util_clear_ok),
             "sigma": s(sigma, cfg.util_sigma_ok)}
    return terms["joint"] * terms["clearance"] * terms["sigma"], terms


def plan_legs(q_now, pose_now, candidates, model, limits, cfg=None, q_ns=None,
              home_pose=None, budget_s=5.0, top_k=64, scene=None):
    """The best candidate whose every leg the arm can follow, as a JSON-ready dict.

    `candidates`: [{"id", "confidence", "legs": [(name, goal_pose), ...], "width_m", "instance"}]. With
    `home_pose` the return home is checked too, cut as go_home cuts it. Candidates go in descending
    confidence, at most top_k, within budget_s: prescreen (free), then check_legs. rank "confidence" takes
    the first feasible; "utility" the highest confidence x comfort(). Comfort is at most 1, so a candidate
    whose confidence is below the best utility cannot win and is not walked, which keeps ranking in budget.
    Returns the chosen id (or None), by_confidence (confidence order's choice), one entry per candidate
    with its reasons (and utility when feasible), scene counts and timing.
    """
    cfg = cfg or PlanConfig()
    scene = objects_only(scene, cfg)
    t0 = time.time()
    deadline = t0 + budget_s
    ordered = sorted(candidates, key=lambda c: -float(c.get("confidence", 0.0)))
    evaluated, best, by_confidence = [], None, None
    for c in ordered[:top_k]:
        conf = float(c.get("confidence", 0.0))
        base = {"id": c["id"], "confidence": c.get("confidence")}
        if cfg.rank == "confidence" and by_confidence is not None:
            break
        if best is not None and conf <= best[0]:
            evaluated.append(dict(base, ok=None, reasons=[
                "not evaluated: its confidence %.3f cannot beat the chosen utility %.3f"
                % (conf, best[0])]))
            continue
        if time.time() > deadline:
            evaluated.append(dict(base, ok=None, reasons=["not evaluated: budget spent"]))
            continue
        legs = list(c["legs"])
        if home_pose is not None and legs:
            legs.extend(home_legs(legs[-1][1], home_pose, cfg))
        refused = prescreen(c["legs"], cfg)
        if refused:
            evaluated.append(dict(base, ok=False, failed_leg=refused.split(":")[0],
                                  prediction=None, reasons=[refused], legs=[]))
            continue
        grasp = next((goal for name, goal in c["legs"] if name == "grasp"), None)
        cand_scene = candidate_scene(scene, c)
        near = near_target(cand_scene, grasp, c.get("width_m"), cfg) if grasp is not None else None
        r = check_legs(q_now, pose_now, legs, model, limits, cfg, q_ns, deadline,
                       cand_scene, near)
        r.update(base)
        if r["ok"]:
            k, terms = comfort(r, cfg)
            r["utility"], r["comfort"] = round(conf * k, 4), {n: round(v, 3) for n, v in terms.items()}
            if by_confidence is None:
                by_confidence = c["id"]
            if cfg.rank == "confidence":
                best = (r["utility"], c["id"])
            elif best is None or r["utility"] > best[0]:
                best = (r["utility"], c["id"])
        evaluated.append(r)
    return {"chosen": None if best is None else best[1], "rank": cfg.rank,
            "by_confidence": by_confidence, "model": model, "candidates": evaluated,
            "not_considered": max(0, len(ordered) - top_k),
            "scene_voxels": 0 if scene is None else len(scene),
            "target_voxels": 0 if getattr(scene, "target", None) is None else len(scene.target),
            "scene_labels": scene_labels(scene),
            "table_seen": getattr(scene, "table_seen", None),
            "elapsed_s": round(time.time() - t0, 3)}


def _jsonable(stats):
    """`stats` with non-finite floats as None, so it serialises to JSON."""
    out = {}
    for key, value in stats.items():
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        out[key] = value
    return out


# -- the wire format ----------------------------------------------------------------

def parse_candidates(text, base_frame):
    """/live_pipeline/pick_candidates (a JSON String) -> (plan_id, candidates).

    The Orin's sequencer sends every grasp, in both finger orders, as base-frame legs in one message:

        {"plan_id": "...", "capture_stamp": 1758..., "frame_id": "fr3_link0",
         "candidates": [{"id": "3:flipped", "confidence": 0.81, "width_m": ..., "instance": ...,
                         "legs": [{"name": "standoff", "position": [x, y, z],
                                   "orientation_xyzw": [x, y, z, w]}, ...]}]}

    Raises ValueError naming what is wrong, so plan_pick reports it rather than evaluating a misread list.
    Legs must be in `base_frame`: a camera-frame pose would pass every check while naming somewhere else.
    """
    import json
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise ValueError("pick_candidates is not JSON: %s" % exc)
    if not isinstance(payload, dict):
        raise ValueError("pick_candidates is not a JSON object")
    frame = payload.get("frame_id")
    if frame != base_frame:
        raise ValueError("pick_candidates are in %r, not %r" % (frame, base_frame))
    plan_id = payload.get("plan_id")
    if not plan_id:
        raise ValueError("pick_candidates carries no plan_id")
    out = []
    for index, c in enumerate(payload.get("candidates") or []):
        try:
            legs = []
            for leg in c["legs"]:
                position = tuple(float(v) for v in leg["position"])
                quaternion = normalize_quaternion(tuple(float(v)
                                                        for v in leg["orientation_xyzw"]))
                if len(position) != 3 or not all(math.isfinite(v) for v in position):
                    raise ValueError("position %r" % (leg["position"],))
                legs.append((str(leg["name"]), (position, quaternion)))
            if not legs:
                raise ValueError("no legs")
            width = c.get("width_m")
            inst = c.get("instance")
            out.append({"id": str(c["id"]), "confidence": float(c.get("confidence", 0.0)),
                        "legs": legs,
                        "width_m": None if width is None else float(width),
                        "instance": None if inst is None else int(inst)})
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("candidate %d is malformed: %s" % (index, exc))
    if not out:
        raise ValueError("pick_candidates holds no candidates")
    return str(plan_id), out


# Which check refused, in words a person scanning a pick log can act on. First match wins, so order matters.
CATEGORIES = (
    ("wrist turn", ("per-command limit",)),
    ("edge of reach", ("edge of reach",)),
    ("joint limit", ("its limit",)),
    ("posture flip", ("posture flip",)),
    ("no IK solution", ("no joint solution",)),
    ("joint speed", ("joint speed",)),
    ("arm vs table", ("of the table", "into the table")),
    ("arm vs target", ("the target object",)),
    ("arm vs scene", ("of the scene", "intersects the scene")),
    ("conditioning", ("sigma_min",)),
    ("commander limit", ("the commander's floor", "the commander's reach")),
    ("not evaluated", ("budget", "cannot beat")),
)


def fit_table(points, cfg, window=0.03, max_tilt_deg=3.0, max_offset=0.025):
    """(a, b, c) of the table z = a x + b y + c fitted to the voxel centres near table_z, or None.

    Depth and hand-eye error put the table in the scene off table_z and tilted, differently per viewpoint
    (measured -5 to +11 mm, 0.9-1.7 deg), so it is fitted per scene: least squares over voxels within
    `window` of table_z, refitted twice on the half nearest the plane, accepted only within max_tilt_deg
    and max_offset.
    """
    pts = np.asarray(points, float)
    sel = pts[np.abs(pts[:, 2] - cfg.table_z) < window]
    if len(sel) < 30:
        return None
    for _ in range(3):
        A = np.c_[sel[:, 0], sel[:, 1], np.ones(len(sel))]
        coef = np.linalg.lstsq(A, sel[:, 2], rcond=None)[0]
        resid = np.abs(A.dot(coef) - sel[:, 2])
        sel = sel[resid <= np.percentile(resid, 50)] if len(sel) > 60 else sel
    tilt = math.degrees(math.atan(math.hypot(coef[0], coef[1])))
    offset = float(coef[2] + coef[0] * pts[:, 0].mean() + coef[1] * pts[:, 1].mean() - cfg.table_z)
    if tilt > max_tilt_deg or abs(offset) > max_offset:
        return None
    return float(coef[0]), float(coef[1]), float(coef[2])


def objects_only(scene, cfg):
    """`scene` without its table layer, when cfg.table_mode is "plane".

    Drops voxel centres below the fitted table (fit_table, else table_z) plus table_band: the table and the
    lowest band of each object, which the plane check stands in for. The result carries table_seen =
    {"offset_mm", "tilt_deg"} or None. None, empty and already table_removed scenes pass through.
    """
    target = getattr(scene, "target", None) if scene is not None else None
    n_target = 0 if target is None else len(target)
    if scene is None or not (len(scene) + n_target) or cfg.table_mode != "plane" \
            or getattr(scene, "table_removed", False):
        return scene
    labels = getattr(scene, "instance", None)
    pts = scene.points if not n_target else np.r_[scene.points, target.points]
    plane = fit_table(pts, cfg)
    if plane is not None:
        a, b, c = plane
        surface = a * pts[:, 0] + b * pts[:, 1] + c
        seen = {"offset_mm": round((float(np.median(surface)) - cfg.table_z) * 1e3, 1),
                "tilt_deg": round(math.degrees(math.atan(math.hypot(a, b))), 2)}
    else:
        surface, seen = np.full(len(pts), cfg.table_z), None
    keep = pts[:, 2] >= surface + cfg.table_band
    if labels is not None and not n_target:
        out = Scene.labelled(pts[keep], scene.voxel_m, labels[keep])
        out.table_removed, out.table_seen = True, seen
        return out
    flags = np.r_[np.zeros(len(scene), bool), np.ones(n_target, bool)]
    out = Scene.split(pts[keep], scene.voxel_m, flags[keep])
    out.table_removed = True
    out.table_seen = seen
    return out


def table_plane_check(spheres, cfg):
    """(lowest bound on any moving part's height above table_z, its part): sphere centre minus radius,
    a true lower bound at most ~14 mm low for the mesh parts.

    A plane rather than table voxels: a voxel's half-diagonal (17 mm) and centre height made hands well
    clear of the table read as touching it, and the grasp's keep-out box removed the table beside the
    grasp, where the hand could go into it unseen.
    """
    worst, part = float("inf"), None
    for name, centres, radius in spheres:
        if name in cfg.static_parts:
            continue
        low = float(centres[:, 2].min()) - radius - cfg.table_z
        if low < worst:
            worst, part = low, name
    return worst, part


def category(reason):
    """The CATEGORIES name for one planner reason, or "other"."""
    text = str(reason or "")
    for name, keys in CATEGORIES:
        if any(k in text for k in keys):
            return name
    return "other"


def summarise(candidates):
    """({category: count}, "N of M feasible; 9 wrist turn, 4 arm vs scene, ...")."""
    counts = {}
    ok = skipped = 0
    for c in candidates:
        if c.get("ok"):
            ok += 1
            continue
        reason = str((c.get("reasons") or [""])[0])
        if "cannot beat" in reason:          # not walked: could not have won
            skipped += 1
            continue
        name = category(reason)
        counts[name] = counts.get(name, 0) + 1
    parts = ["%d %s" % (n, k) for k, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    text = "%d of %d feasible%s" % (ok, len(candidates) - skipped,
                                    "; " + ", ".join(parts) if parts else "")
    if skipped:
        text += " (%d more not walked: could not beat the choice)" % skipped
    if skipped:
        counts["not needed"] = skipped
    return counts, text


def compact(result):
    """plan_legs' result trimmed for a Trigger reply: every candidate's verdict
    and reasons, the chosen one's predicted end posture, no per-sample stats."""
    out = {k: result.get(k) for k in ("chosen", "rank", "by_confidence", "model", "not_considered",
                                      "elapsed_s", "scene_voxels", "target_voxels", "table_seen")}
    out["candidates"] = []
    for c in result["candidates"]:
        entry = {k: c.get(k) for k in ("id", "confidence", "ok", "failed_leg",
                                        "prediction", "reasons", "utility", "comfort")}
        if not c.get("ok"):
            entry["category"] = category((c.get("reasons") or [""])[0])
        if c.get("id") == result["chosen"]:
            entry["q_end"] = c.get("q_end")
        out["candidates"].append(entry)
    # One line the sequencer shows instead of the whole reply.
    out["summary"], out["summary_text"] = summarise(result["candidates"])
    return out


def parse_obstacles(text, base_frame):
    """The optional "obstacles" field of pick_candidates -> a Scene, or None.

        "obstacles": {"frame_id": "fr3_link0", "voxel_m": 0.02, "points": [[x, y, z], ...],
                      "instance": [label, ...]  or  "target": [flag, ...]}

    "instance" (a SAM 3 object label per point, 0 = none) gives a labelled scene and wins over "target"
    (true on the picked object's voxels, checked with target_margin). None when no obstacles are sent, so
    the scene check is skipped rather than passed. Raises ValueError on a field present but wrong: a scene
    in the camera frame would pass every check while describing somewhere else.
    """
    import json
    payload = json.loads(text)
    ob = payload.get("obstacles") if isinstance(payload, dict) else None
    if ob is None:
        return None
    if ob.get("frame_id") != base_frame:
        raise ValueError("obstacles are in %r, not %r" % (ob.get("frame_id"), base_frame))
    points = np.asarray(ob.get("points") or [], float)
    if points.size and (points.ndim != 2 or points.shape[1] != 3
                        or not np.all(np.isfinite(points))):
        raise ValueError("obstacles.points must be finite [x, y, z] triples")
    voxel = float(ob.get("voxel_m", 0.0))
    if not voxel > 0:
        raise ValueError("obstacles.voxel_m must be positive")
    labels = ob.get("instance")
    if labels is not None:
        if len(labels) != len(points.reshape(-1, 3)):
            raise ValueError("obstacles.instance has %d labels for %d points" % (len(labels), len(points)))
        try:
            labels = [int(v) for v in labels]
        except (TypeError, ValueError):
            raise ValueError("obstacles.instance must be integers")
        return Scene.labelled(points.reshape(-1, 3), voxel, labels)
    flags = ob.get("target")
    if flags is None:
        return Scene(points.reshape(-1, 3), voxel)
    if len(flags) != len(points.reshape(-1, 3)):
        raise ValueError("obstacles.target has %d flags for %d points" % (len(flags), len(points)))
    return Scene.split(points.reshape(-1, 3), voxel, [bool(f) for f in flags])
