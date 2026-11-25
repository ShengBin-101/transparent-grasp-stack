#!/usr/bin/env python3
"""ROS 1 motion commander for the FR3: one Cartesian move per service call, not a pick.

The ROS 2 side sends poses over ros1_bridge; this node resolves each into the base frame (the
controller ignores frame_id), checks it against workspace limits and the reachability planner,
and ramps the controller's setpoint to it at a bounded speed, since a single jump becomes a
large impedance force. The staged pick (standoff, grasp, retreat) is three calls made by the
ROS 2 sequencer. The guarded variant stops on contact, because depth completed on transparent
objects is off by centimetres. go_home moves only the tool with the controller running;
run_home.sh is the joint-space homing that sets the posture.

The trials' two views are services too: go_v1 is go_home, and go_v<name> for each ~views entry
(v2, the side view: run_viewpoint.sh 4 --tilt 45 --height 0.35) plans the view as
tools/goto_viewpoint.py does (viewpoints.plan_viewpoint with the same settings, so the same camera
roll and legs) and runs it as go_home runs: every leg checked, the whole chain through the planner
from the measured joints before anything moves, a ramp per leg, the gripper untouched. As with
go_home, the call is the go-ahead; a view the planner cannot reach is refused before the arm moves.

    /live_pipeline/grasp_pose_best      (PoseStamped)   in   the pose to go to
    /live_pipeline/grasp_width_best     (Float64)       in   gripper opening, m
    /live_pipeline/pick_candidates      (String)        in   JSON candidates and scene for plan_pick
    /live_pipeline/plan_pick            (Trigger)       srv  choose a candidate, every leg checked
    /live_pipeline/move_to_pose         (Trigger)       srv  ramp there, apply the width
    /live_pipeline/guarded_move_to_pose (Trigger)       srv  the same, but stop on contact
    /live_pipeline/release              (Trigger)       srv  open the fingers
    /live_pipeline/go_home              (Trigger)       srv  back to home (home_pose.yaml), in legs
    /live_pipeline/go_v1                (Trigger)       srv  the trials' top-down view: go_home itself
    /live_pipeline/go_v2                (Trigger)       srv  the trials' side view (~views/v2), in legs
    /live_pipeline/measure_load         (Trigger)       srv  the standing external force, JSON
    /live_pipeline/contact_pose         (PoseStamped)   out  where it touched, if it did (latched)
    /live_pipeline/contact_wrench       (WrenchStamped) out  what it felt, minus the pre-move baseline
    /live_pipeline/execution_status     (String)        out  what happened
    /franka_state_controller/franka_states              in   measured pose, wrench, joints, mode
    ~command_topic                      (PoseStamped)   out  the controller's setpoint
"""

import json
import os
import threading
import time

import actionlib
import numpy as np
import rospy
import tf2_ros
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger, TriggerResponse

from fr3_pose_commander import kinematics
from fr3_pose_commander import planning
from fr3_pose_commander import viewpoints
from fr3_pose_commander.leg_log import LegLog
from fr3_pose_commander.leg_log import record as leg_record
from fr3_pose_commander.leg_log import snapshot as leg_snapshot
from fr3_pose_commander.pose_math import (
    angle_between,
    canonicalize,
    matrix_from_o_t_ee,
    normalize_quaternion,
    quaternion_from_matrix,
    slerp,
)


def is_closing(error, best, progress):
    """Is the error still improving on its best by more than `progress`?

    `progress` keeps sensor noise on a still arm from looking like endless improvement.
    """
    return best is None or error < best - progress


def wrench_delta(wrench, baseline):
    """Magnitudes of the wrench's change from `baseline`: (force_N, torque_Nm), or (None, None).

    Contact is judged on the change, not the absolute reading, because the estimate carries a
    standing offset (3.5 N measured with the arm still and nothing touched; README 6.6).
    """
    if wrench is None or baseline is None:
        return None, None
    d = np.asarray(wrench, dtype=float) - np.asarray(baseline, dtype=float)
    return float(np.linalg.norm(d[:3])), float(np.linalg.norm(d[3:]))


def rolling_reference(history, now, lag, window, min_samples, fallback):
    """Reference wrench for the guard: the mean over [now - lag - window, now - lag].

    `history` is [(t, wrench), ...] oldest first. Returns `fallback` (the pre-move baseline) when
    that span holds fewer than `min_samples` readings, as at the start of every leg. Following the
    estimate's slow free-space drift (~0.1 N/s on the rig) avoids the false contacts a fixed
    baseline gives, and the lag keeps a fast contact step from pulling its own reference with it.
    """
    if not history:
        return fallback
    hi, lo = now - lag, now - lag - window
    picked = [w for t, w in history if lo <= t <= hi]
    if len(picked) < min_samples:
        return fallback
    return np.mean(np.asarray(picked, dtype=float), axis=0)


def contact_detected(wrench, baseline, force_threshold, torque_threshold):
    """Contact test on the wrench change: (hit, force_N, torque_Nm).

    Either channel fires: force for a head-on push, torque for an off-centre touch such as a
    fingertip catching a rim.
    """
    force, torque = wrench_delta(wrench, baseline)
    if force is None:
        return False, None, None
    hit = force >= force_threshold or torque >= torque_threshold
    return hit, force, torque


def count_over(over, hit, sample, last_sample):
    """Advance the debounce count once per NEW franka_states sample: returns (over, sample).

    The 100 Hz ramp loop sees each ~30 Hz sample about three times; counting loop ticks would let
    one spiky sample pass the debounce on its own.
    """
    if sample == last_sample:
        return over, last_sample
    return (over + 1 if hit else 0), sample


def wrench_settled(samples, force_tol, torque_tol):
    """Has the wrench stopped changing over `samples`? (settled, force_span, torque_span).

    The span is the largest distance of any sample from the window mean, per channel: a range
    rather than a standard deviation, so one late step fails it. The guard waits for this before
    taking its baseline, because a previous leg that stalled short leaves the controller pushing,
    and releasing that push at the start of the guarded ramp would otherwise trip the guard.
    """
    if not samples or len(samples) < 2:
        return False, None, None
    a = np.asarray(samples, dtype=float)
    d = a - a.mean(axis=0)
    force_span = float(np.max(np.linalg.norm(d[:, :3], axis=1)))
    torque_span = float(np.max(np.linalg.norm(d[:, 3:], axis=1)))
    return (force_span <= force_tol and torque_span <= torque_tol,
            force_span, torque_span)


def contact_hold(measured, start, goal, backoff):
    """Setpoint to hold when the guard fires: the MEASURED position, backed off along start -> goal.

    The ramp's setpoint is already past the arm at contact, so holding it would keep pushing into
    the object. `backoff` (m) leaves the fingers clear of what they touched; with no backoff or no
    approach direction the measured position is returned unchanged.
    """
    measured = np.asarray(measured, dtype=float)
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    span = goal - start
    n = float(np.linalg.norm(span))
    if n < 1e-9 or backoff <= 0.0:
        return measured.copy()
    return measured - backoff * (span / n)


# franka_msgs/FrankaState robot_mode values (libfranka RobotMode).
ROBOT_MODES = {0: "Other", 1: "Idle", 2: "Move", 3: "Guiding", 4: "Reflex",
               5: "UserStopped", 6: "AutomaticErrorRecovery"}
MODE_MOVE = 2


def load_record(wrenches, positions):
    """Standing load from O_F_ext_hat_K samples (base frame) and tool positions.

    Returns {"force", "torque", "force_span", "motion_mm", "n"}: the mean force and torque, the
    force samples' largest spread from that mean, and the tool's motion over the window. The
    caller uses the last two to reject a reading taken while the arm was moving or being pushed.
    """
    w = np.asarray(wrenches, dtype=float).reshape(-1, 6)
    mean = w.mean(axis=0)
    span = float(np.max(np.linalg.norm(w[:, :3] - mean[:3], axis=1))) if len(w) else float("inf")
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    motion = float(np.max(np.linalg.norm(pos - pos[0], axis=1))) if len(pos) else 0.0
    return {"force": [round(float(v), 4) for v in mean[:3]],
            "torque": [round(float(v), 4) for v in mean[3:]],
            "force_span": round(span, 4), "motion_mm": round(motion * 1e3, 2), "n": int(len(w))}


def leg_outcome(mode, error, contact, fail_distance):
    """Why a leg did NOT happen, or None if it did.

    After a reflex the setpoint still ramps and the settle still reports, so the leg is judged on
    the robot itself: a robot_mode other than Move means no controller is commanding the arm, and
    an `error` (measured distance to the goal after the settle, m) over `fail_distance` means it
    never got there. A guarded leg that stopped on contact is exempt from the distance check.
    """
    if mode is not None and mode != MODE_MOVE:
        return ("the robot is not being commanded (robot_mode %s) -- a reflex, a user "
                "stop or a lost controller; nothing ramped since then happened. Recover "
                "the error and restart the controller" % ROBOT_MODES.get(mode, mode))
    if not contact and error is not None and error > fail_distance:
        return ("the arm ended %.0f mm from the goal (limit %.0f mm for this controller): "
                "the leg did not happen" % (error * 1e3, fail_distance * 1e3))
    return None


def settle_verdict(error, stalled_for, min_hold_done, tolerance, patience,
                   turn=0.0, angle_tolerance=float("inf")):
    """Settle decision: 'arrived', 'stopped', or None to keep waiting.

    'arrived' needs both position (`error`, m) and orientation (`turn`, rad) inside tolerance,
    because a pure roll never changes position. 'stopped' (no progress for `patience` s) is not a
    failure: the Cartesian PD controller has no integral term and rests where K*e balances the
    opposing force, which on the grasp leg (aimed inside the object) is the wanted outcome.
    """
    if not min_hold_done:
        return None
    if error <= tolerance and turn <= angle_tolerance:
        return "arrived"
    if stalled_for > patience:
        return "stopped"
    return None


def choose_gripper_action(commanded, current, tolerance):
    """"move", "grasp", "hold" or "leave": what the fingers should be told to do.

    franka_gripper's grasp closes and searches for an object; move drives to a width. So a width
    wider than the current opening is a move (open to), a narrower one a grasp (close onto), and
    one within `tolerance` a hold (a grasp would re-squeeze, a move would let go). A strictly
    negative width means leave the fingers alone, as on the retreat; 0.0 is a real "close fully".
    With no `current` reading it falls back to grasp.
    """
    if commanded is not None and commanded < 0.0:
        return "leave"
    if current is None:
        return "grasp"
    if abs(commanded - current) <= tolerance:
        return "hold"
    return "move" if commanded > current else "grasp"


def grasp_goal_params(mode, predicted, under_width, under_eps_outer, eps_inner, eps_outer):
    """(width, epsilon_inner, epsilon_outer) for franka_gripper's grasp action.

    Mode "under" (default) commands `under_width`, narrower than any object, with an outer window
    reaching the full opening, so the fingers stop on the object before the commanded width. On
    the rig (gripper_test.py) the hand let go whenever the fingers closed well past the commanded
    width, which the detector's predicted width (wider than a compressed handle) caused every
    time. Mode "predicted" sends the detector's width and configured window, kept for comparison.
    """
    if mode == "under":
        return float(under_width), float(under_width), float(under_eps_outer)
    return float(predicted), float(eps_inner), float(eps_outer)


def grasp_verdict(ok, closed, miss_width):
    """Plain-language grasp outcome: HELD, MISSED, FAILED or no result.

    MISSED comes from the final finger width (below `miss_width`), not the action's success flag:
    a close on nothing can fall inside the "under" window, and the flag has said True for fingers
    that then opened.
    """
    at = "unknown" if closed is None else "%.0f mm" % (closed * 1e3)
    if closed is not None and closed < miss_width:
        return "MISSED -- fingers closed to %s, nothing between them" % at
    if ok:
        return "HELD at %s" % at
    if ok is False:
        return "FAILED -- fingers at %s, NOT held" % at
    return "no result (fingers at %s)" % at


def approved_leg(position, approved_positions, expected, tolerance):
    """Index of the plan_pick leg that `position` is, or None.

    Among legs within `tolerance`, the first at or after `expected` (the leg after the last one
    run) wins, else the first match. Order matters because the standoff and the retreat end at
    the same point, and the retreat must be checked as the retreat (target excluded), or it is
    refused for touching the object it has just grasped.
    """
    matches = [i for i, a in enumerate(approved_positions)
               if float(np.linalg.norm(np.asarray(position) - a)) <= tolerance]
    if not matches:
        return None
    later = [i for i in matches if i >= expected]
    return (later or matches)[0]


class Refused(Exception):
    """A check said no. The message is what the caller is told."""


class PoseCommander(object):
    """The commander node: reads ~ parameters, caches robot and gripper state from callbacks,
    and serves one motion request at a time (`_busy`), each written to the leg log."""

    def __init__(self):
        p = rospy.get_param

        self.base_frame = p("~base_frame", "fr3_link0")
        self.dry_run = bool(p("~dry_run", False))
        self.enforce_checks = bool(p("~enforce_checks", True))

        # Workspace limits (m, rad, s); see config/pose_commander.yaml. Every default here equals the
        # yaml's value, so a node started without the file behaves as the launch file's does.
        self.z_min = float(p("~limits/z_min", -0.036))
        self.radius_max = float(p("~limits/radius_max", 0.85))
        self.step_max = float(p("~limits/step_max", 0.60))
        self.rotation_max = float(p("~limits/rotation_max", 1.571))  # 90 deg
        self.state_max_age = float(p("~limits/state_max_age", 0.5))

        # Ramp shape.
        self.rate_hz = float(p("~ramp/rate_hz", 100.0))
        self.linear_speed = float(p("~ramp/linear_speed", 0.012))    # m/s
        self.angular_speed = float(p("~ramp/angular_speed", 0.40))   # rad/s
        self.settle_time = float(p("~ramp/settle_time", 0.5))
        self.settle_tolerance = float(p("~ramp/settle_tolerance", 0.002))
        self.settle_progress = float(p("~ramp/settle_progress", 0.0005))
        self.settle_patience = float(p("~ramp/settle_patience", 0.7))
        self.settle_angle_tolerance = float(p("~ramp/settle_angle_tolerance", 0.0087))
        self.settle_angle_progress = float(p("~ramp/settle_angle_progress", 0.0017))
        self.settle_timeout = float(p("~ramp/settle_timeout", 6.0))
        # A leg ending further than this from its goal failed (leg_outcome); per controller,
        # since impedance legs stall up to ~18 mm short and position legs arrive to ~1 mm.
        _position = str(p("~command_topic", "")).rstrip("/").endswith("target_pose")
        self.fail_distance = float(p("~ramp/fail_distance", 0.010 if _position else 0.030))

        # Guarded approach: stop the ramp on contact (thresholds over the reference wrench).
        self.guard_force = float(p("~guard/force_threshold", 2.0))       # N
        self.guard_torque = float(p("~guard/torque_threshold", 1.5))     # N.m
        self.guard_warmup = float(p("~guard/warmup", 1.2))               # s
        self.guard_bias_samples = int(p("~guard/bias_samples", 20))
        self.guard_bias_timeout = float(p("~guard/bias_timeout", 2.0))   # s
        self.guard_backoff = float(p("~guard/backoff", 0.003))           # m
        self.guard_mode = str(p("~guard/baseline_mode", "rolling")).strip().lower()
        self.guard_window = float(p("~guard/rolling_window", 0.5))       # s
        self.guard_lag = float(p("~guard/rolling_lag", 0.2))             # s
        self.guard_min_samples = int(p("~guard/rolling_min_samples", 5))
        self.guard_debounce = int(p("~guard/debounce", 2))
        # Pre-guard settle: hold at the measured pose until the wrench is steady (wrench_settled).
        self.pre_settle = bool(p("~guard/pre_settle", True))
        self.pre_settle_window = float(p("~guard/pre_settle_window", 0.3))       # s
        self.pre_settle_force_tol = float(p("~guard/pre_settle_force_tol", 0.2))  # N
        self.pre_settle_torque_tol = float(p("~guard/pre_settle_torque_tol", 0.15))  # N.m
        self.pre_settle_motion_tol = float(p("~guard/pre_settle_motion_tol", 0.001))  # m
        self.pre_settle_timeout = float(p("~guard/pre_settle_timeout", 3.0))     # s

        self.gripper_force = float(p("~gripper/force", 20.0))
        self.gripper_speed = float(p("~gripper/speed", 0.05))
        self.gripper_epsilon = float(p("~gripper/epsilon", 0.02))
        # Split grasp window; see config/pose_commander.yaml gripper/epsilon_inner.
        self.gripper_eps_inner = float(p("~gripper/epsilon_inner", self.gripper_epsilon))
        self.gripper_eps_outer = float(p("~gripper/epsilon_outer", self.gripper_epsilon))
        # How the grasp is commanded; see grasp_goal_params().
        self.grasp_mode = str(p("~gripper/grasp_mode", "under")).strip().lower()
        self.grasp_under_width = float(p("~gripper/under_width", 0.005))
        self.grasp_under_eps_outer = float(p("~gripper/under_epsilon_outer", 0.075))
        self.grasp_miss_width = float(p("~gripper/miss_width", 0.003))
        self.gripper_open_width = float(p("~gripper/open_width", 0.08))
        self.default_width = float(p("~gripper/default_width", 0.04))

        command_topic = p("~command_topic",
                          "/cartesian_impedance_example_controller/equilibrium_pose")
        state_topic = p("~state_topic", "/franka_state_controller/franka_states")
        self.command_topic = command_topic

        # One JSON line per leg (leg_log.py); a failed write is warned about, never fails the leg.
        # An empty directory means the workspace's own logs/.
        import rospkg
        leg_dir = p("~leg_log/directory", "") or os.path.normpath(os.path.join(
            rospkg.RosPack().get_path("fr3_pose_commander"), "..", "..", "logs"))
        self.leg_log = LegLog(leg_dir, enabled=bool(p("~leg_log/enabled", True)))

        self._init_planner(p, command_topic)

        self._lock = threading.Lock()
        self._target = None        # PoseStamped, as received
        self._width = None         # float, metres
        self._state = None         # (position, quaternion, stamp)
        self._wrench = None        # O_F_ext_hat_K: external wrench, BASE frame
        self._wrench_n = 0         # franka_states samples received (the guard's debounce)
        self._q = None             # the joint angles behind that state
        self._elbow = None         # FrankaState elbow / elbow_d / elbow_c
        self._mode = None          # FrankaState robot_mode; MODE_MOVE while commanded
        self._last_goal = None     # the base-frame goal of the leg in progress
        self._q0 = None            # the joints at the first state this node saw
        self._candidates = None    # (JSON text, arrival time) from the sequencer
        self._approved = None      # the legs plan_pick last approved
        self._plan_check = None    # the planner's verdict on the leg in progress
        self._home = None          # the tool pose go_home drives to; see _init_home
        self._home_source = None   # where it came from: "home_pose.yaml" | "startup"
        self._fingers = None       # total opening, metres, from the gripper
        self._busy = False
        self._init_home(p)
        self._init_views(p)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.goal_pub = rospy.Publisher(command_topic, PoseStamped, queue_size=1)
        self.status_pub = rospy.Publisher("/live_pipeline/execution_status",
                                          String, queue_size=10)

        rospy.Subscriber(state_topic, FrankaState, self.on_state, queue_size=1)
        rospy.Subscriber("/live_pipeline/grasp_pose_best", PoseStamped,
                         self.on_pose, queue_size=1)
        rospy.Subscriber("/live_pipeline/grasp_width_best", Float64,
                         self.on_width, queue_size=1)

        ns = p("~gripper_ns", "/franka_gripper").rstrip("/")
        ns_fingers = ns + "/joint_states"
        # How close counts as already there, as a total opening of both fingers (m).
        self.gripper_hold_tol = float(p("~gripper/hold_tolerance", 0.004))
        self.grasp_client = actionlib.SimpleActionClient(ns + "/grasp", GraspAction)
        self.move_client = actionlib.SimpleActionClient(ns + "/move", MoveAction)
        # Finger width: needed to tell "open to" from "close onto" (choose_gripper_action).
        rospy.Subscriber(ns_fingers, JointState, self.on_fingers, queue_size=1)

        # Latched so a late subscriber still gets the last contact (the ROS 2 side archives it).
        self.contact_pub = rospy.Publisher("/live_pipeline/contact_pose",
                                           PoseStamped, queue_size=1, latch=True)
        self.contact_wrench_pub = rospy.Publisher("/live_pipeline/contact_wrench",
                                                  WrenchStamped, queue_size=1, latch=True)

        rospy.Service("/live_pipeline/move_to_pose", Trigger, self.on_execute)
        rospy.Service("/live_pipeline/guarded_move_to_pose", Trigger,
                      self.on_guarded_execute)
        rospy.Service("/live_pipeline/release", Trigger, self.on_release)
        rospy.Service("/live_pipeline/go_home", Trigger, self.on_home)
        rospy.Service("/live_pipeline/go_v1", Trigger, self.on_home)
        for name in self.views:
            rospy.Service("/live_pipeline/go_%s" % name, Trigger,
                          lambda _request, name=name: self.on_view(name))
        rospy.Service("/live_pipeline/measure_load", Trigger, self.on_measure_load)
        rospy.Subscriber("/live_pipeline/pick_candidates", String,
                         self.on_candidates, queue_size=1)
        rospy.Service("/live_pipeline/plan_pick", Trigger, self.on_plan_pick)

        rospy.loginfo("pose_commander: base=%s command=%s", self.base_frame, command_topic)
        rospy.loginfo("pose_commander: limits z >= %.0fmm, radius <= %.0fmm, "
                      "step <= %.0fmm, reorientation <= %.0f deg per ramp (longer unguarded turns "
                      "are cut into pieces), state age <= %.1fs",
                      self.z_min * 1e3, self.radius_max * 1e3, self.step_max * 1e3,
                      np.degrees(self.rotation_max), self.state_max_age)
        rospy.loginfo("pose_commander: ramp %.0f mm/s, %.0f deg/s at %.0f Hz",
                      self.linear_speed * 1e3, np.degrees(self.angular_speed), self.rate_hz)
        rospy.loginfo("pose_commander: guarded_move_to_pose stops at %.1f N or "
                      "%.2f N.m over the standing wrench (warm-up %.2fs, "
                      "baseline %d samples)", self.guard_force, self.guard_torque,
                      self.guard_warmup, self.guard_bias_samples)
        rospy.loginfo("pose_commander: pre-guard settle %s (%.1fs window, "
                      "%.2f N / %.2f N.m / %.1f mm, timeout %.1fs)",
                      "ON" if self.pre_settle else "OFF", self.pre_settle_window,
                      self.pre_settle_force_tol, self.pre_settle_torque_tol,
                      self.pre_settle_motion_tol * 1e3, self.pre_settle_timeout)
        if self.dry_run:
            rospy.logwarn("pose_commander: DRY RUN -- poses are resolved and "
                          "checked, nothing is published to the controller")
        if not self.enforce_checks:
            rospy.logwarn("pose_commander: CHECKS DISABLED -- every refusal "
                          "becomes a warning and the arm goes anyway")
        rospy.loginfo("pose_commander: waiting for a pose on "
                      "/live_pipeline/grasp_pose_best and a trigger on "
                      "/live_pipeline/move_to_pose")

    # -- inputs --------------------------------------------------------------

    def on_state(self, msg):
        """Cache the measured tool pose, external wrench, joints, elbow and robot_mode.

        The first state records the startup joints and, when home did not come from
        home_pose.yaml, becomes home; otherwise it reports once how far the arm stands from home.
        """
        try:
            m = matrix_from_o_t_ee(msg.O_T_EE)
            position = m[:3, 3].copy()
            quaternion = quaternion_from_matrix(m)
        except (ValueError, ZeroDivisionError) as exc:
            rospy.logwarn_throttle(5.0, "pose_commander: unusable robot state: %s", exc)
            return
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        with self._lock:
            self._state = (position, quaternion, stamp)
            # Estimated external wrench, base frame, kept raw for the guard.
            try:
                self._wrench = [float(v) for v in msg.O_F_ext_hat_K]
            except (TypeError, ValueError):
                self._wrench = None
            self._wrench_n += 1
            # Set before the home latch, so an exception there cannot lose the joints.
            self._q = list(msg.q)
            self._mode = int(getattr(msg, "robot_mode", MODE_MOVE))
            # Measured, desired and commanded elbow: how the redundancy was resolved.
            self._elbow = {"elbow": list(msg.elbow), "elbow_d": list(msg.elbow_d),
                           "elbow_c": list(msg.elbow_c)}
            if self._q0 is None and len(self._q) == 7:
                self._q0 = list(self._q)
                self._check_startup_posture()
            if self._home is None:
                # `quaternion` is a tuple, which has no .copy().
                self._home = (position.copy(), tuple(quaternion))
                self._home_source = "startup"
                rospy.loginfo("pose_commander: home latched at [%.3f %.3f %.3f] "
                              "(where this node started)",
                              position[0], position[1], position[2])
            elif self._home_source == "home_pose.yaml" and not self._home_reported:
                self._home_reported = True
                off = float(np.linalg.norm(position - self._home[0]))
                turn = angle_between(quaternion, self._home[1])
                (rospy.loginfo if off < 0.01 and turn < 0.02 else rospy.logwarn)(
                    "pose_commander: the arm stands %.0f mm / %.0f deg from home "
                    "(home_pose.yaml)%s", off * 1e3, np.degrees(turn),
                    "" if off < 0.01 and turn < 0.02 else
                    " -- run_home.sh before the controller pins the posture there; "
                    "go_home moves only the tool")

    def on_pose(self, msg):
        with self._lock:
            self._target = msg
        frame = msg.header.frame_id or "(empty)"
        rospy.loginfo("pose_commander: target in %s at [%.3f %.3f %.3f]", frame,
                      msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def on_fingers(self, msg):
        """Total opening: franka_gripper reports the two finger joints apart."""
        try:
            self._fingers = float(msg.position[0]) + float(msg.position[1])
        except (IndexError, TypeError, ValueError):
            pass

    def on_width(self, msg):
        with self._lock:
            self._width = float(msg.data)

    # -- checks --------------------------------------------------------------

    def _current(self):
        """Measured (position, quaternion); refuses with no state, or a stale one (overrulable)."""
        with self._lock:
            state = self._state
        if state is None:
            raise Refused("no robot state on franka_states -- is the controller running?")
        position, quaternion, stamp = state
        age = (rospy.Time.now() - stamp).to_sec()
        if age > self.state_max_age:
            self._maybe_refuse("robot state is %.2fs old (limit %.2fs)"
                               % (age, self.state_max_age))
        return position, quaternion

    def _posture(self):
        """How much authority the arm has where it stands (kinematics.report), as a phrase, or None.

        Reported, never enforced: no threshold has been validated yet (see kinematics.py), so the
        number is printed only to make a stall attributable.
        """
        with self._lock:
            q = self._q
        return kinematics.report(q) if q else None

    def _require_listener(self):
        """Refuse unless a controller is subscribed to the command topic; skipped in dry run.

        A commander/controller mismatch otherwise publishes to nobody and looks like a stalled
        leg. Not overrulable by --no-checks: with no subscriber nothing would move.
        """
        if self.dry_run:
            return
        if self.goal_pub.get_num_connections() == 0:
            raise Refused(
                "no controller is subscribed to %s, so nothing would move. The "
                "controller and the commander do not match: run_controller.sh pairs "
                "with run_commander.sh, run_position_controller.sh with "
                "run_commander.sh --position." % self.goal_pub.resolved_name)

    def _require_commanding(self):
        """Refuse unless robot_mode is Move (a controller commands the arm); skipped in dry run.

        Not overrulable, like _require_listener: after a reflex nothing would move.
        """
        if self.dry_run:
            return
        with self._lock:
            mode = self._mode
        if mode is not None and mode != MODE_MOVE:
            raise Refused(leg_outcome(mode, None, True, float("inf")))

    def _outcome(self, goal_position, contact):
        """leg_outcome() on the robot as it is now; None in dry run."""
        if self.dry_run:
            return None
        with self._lock:
            mode, state = self._mode, self._state
        error = None if state is None else float(np.linalg.norm(state[0] - goal_position))
        return leg_outcome(mode, error, contact, self.fail_distance)

    def _maybe_refuse(self, reason):
        """Refuse, or warn and continue when checks are off."""
        if self.enforce_checks:
            raise Refused(reason)
        rospy.logwarn("pose_commander: OVERRULED: %s", reason)

    def _resolve(self, target):
        """Return (position, quaternion) in the base frame, or refuse.

        The controller ignores frame_id, so the frame is resolved here via tf, and a frame that
        cannot be resolved is refused rather than assumed.
        """
        frame = target.header.frame_id
        if not frame:
            self._maybe_refuse("pose has an empty frame_id; refusing to assume "
                               "it is already in %s" % self.base_frame)
            frame = self.base_frame

        if frame != self.base_frame:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame, frame, rospy.Time(0), rospy.Duration(1.0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as exc:
                raise Refused("cannot transform %s -> %s: %s"
                              % (frame, self.base_frame, exc))
            import tf2_geometry_msgs  # noqa: F401  (registers PoseStamped)
            target = tf2_geometry_msgs.do_transform_pose(target, transform)

        position = np.array([target.pose.position.x,
                             target.pose.position.y,
                             target.pose.position.z])
        q = (target.pose.orientation.x, target.pose.orientation.y,
             target.pose.orientation.z, target.pose.orientation.w)
        try:
            quaternion = normalize_quaternion(q)
        except ValueError:
            raise Refused("pose carries a zero quaternion")
        return position, quaternion

    def _check(self, position, quaternion, current_position, current_quaternion):
        """Floor, reach, step and reorientation limits for one move; returns (step_m, turn_rad)."""
        if position[2] < self.z_min:
            self._maybe_refuse("z = %.0f mm is below the floor limit of %.0f mm"
                               % (position[2] * 1e3, self.z_min * 1e3))
        radius = float(np.linalg.norm(position[:2]))
        if radius > self.radius_max:
            self._maybe_refuse("radius %.0f mm exceeds the reach limit of %.0f mm"
                               % (radius * 1e3, self.radius_max * 1e3))
        step = float(np.linalg.norm(position - current_position))
        if step > self.step_max:
            self._maybe_refuse("step of %.0f mm exceeds the limit of %.0f mm"
                               % (step * 1e3, self.step_max * 1e3))
        turn = angle_between(current_quaternion, quaternion)
        if turn > self.rotation_max:
            self._maybe_refuse("reorientation of %.0f deg exceeds the limit of %.0f deg"
                               % (np.degrees(turn), np.degrees(self.rotation_max)))
        return step, turn

    # -- motion --------------------------------------------------------------

    def _publish(self, position, quaternion):
        """Publish one base-frame setpoint and remember its position (for _relax_setpoint)."""
        self._last_setpoint = np.asarray(position, dtype=float).copy()
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.base_frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = position
        (msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w) = quaternion
        self.goal_pub.publish(msg)

    def _wrench_baseline(self):
        """Mean external wrench with the arm still, taken fresh before each guarded ramp.

        Never reused between legs, because the standing bias (the estimator's offset, and the
        centre-of-mass error of README 6.6) changes with posture; that drift is also why the guard is used on the short grasp leg only.
        Refuses when too few samples arrive rather than running unguarded.
        """
        deadline = rospy.Time.now() + rospy.Duration(self.guard_bias_timeout)
        rate = rospy.Rate(50.0)
        samples = []
        while not rospy.is_shutdown() and len(samples) < self.guard_bias_samples:
            with self._lock:
                wrench = self._wrench
            if wrench is not None:
                samples.append(list(wrench))
            if rospy.Time.now() >= deadline:
                break
            rate.sleep()
        if len(samples) < max(3, self.guard_bias_samples // 4):
            raise Refused(
                "no external wrench on franka_states (%d samples in %.1fs), so a "
                "guarded move cannot tell contact from free space. Refusing rather "
                "than running unguarded." % (len(samples), self.guard_bias_timeout))
        baseline = np.mean(np.asarray(samples, dtype=float), axis=0)
        rospy.loginfo("pose_commander: guard baseline over %d samples: "
                      "|F| %.2f N, |T| %.2f N.m (this is the standing bias, not "
                      "a contact)", len(samples),
                      float(np.linalg.norm(baseline[:3])),
                      float(np.linalg.norm(baseline[3:])))
        return baseline

    def _hold_until_settled(self):
        """Pin the setpoint to the measured pose and wait until wrench and position are steady.

        Returns (position, quaternion, report). The guarded ramp starts from this HELD pose, not a
        fresh measurement, so its first step is small and does not re-open the gap just absorbed.
        An unsettled wrench does not refuse: on timeout the leg goes ahead and the report says
        "NOT settled", since a guard that trips early only stops the arm.
        """
        position, quaternion = self._current()
        position = np.asarray(position, dtype=float).copy()
        self._publish(position, quaternion)

        rate = rospy.Rate(50.0)
        start = rospy.Time.now()
        deadline = start + rospy.Duration(self.pre_settle_timeout)
        history = []   # (t, wrench, measured position)
        spans = (None, None, None)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            with self._lock:
                wrench = self._wrench
                state = self._state
            if wrench is not None and state is not None:
                history.append((now.to_sec(), list(wrench), np.asarray(state[0], dtype=float)))
            lo = now.to_sec() - self.pre_settle_window
            recent = [h for h in history if h[0] >= lo]
            window_full = (now - start).to_sec() >= self.pre_settle_window and len(recent) >= 3
            if window_full:
                ok, f_span, t_span = wrench_settled([h[1] for h in recent],
                                                    self.pre_settle_force_tol,
                                                    self.pre_settle_torque_tol)
                pos = np.asarray([h[2] for h in recent])
                motion = float(np.max(np.linalg.norm(pos - pos.mean(axis=0), axis=1)))
                spans = (f_span, t_span, motion)
                if ok and motion <= self.pre_settle_motion_tol:
                    return position, quaternion, (
                        "settled in %.1fs (|dF| %.2f N, |dT| %.2f N.m, %.1f mm)"
                        % ((now - start).to_sec(), f_span, t_span, motion * 1e3))
            if now >= deadline:
                break
            rate.sleep()
        f_span, t_span, motion = spans
        if f_span is None:
            return position, quaternion, ("NOT settled: no wrench/state in %.1fs"
                                          % self.pre_settle_timeout)
        rospy.logwarn("pose_commander: wrench not settled after %.1fs "
                      "(|dF| %.2f N, |dT| %.2f N.m, %.1f mm); guarding anyway",
                      self.pre_settle_timeout, f_span, t_span, motion * 1e3)
        return position, quaternion, (
            "NOT settled after %.1fs (|dF| %.2f N, |dT| %.2f N.m, %.1f mm)"
            % (self.pre_settle_timeout, f_span, t_span, motion * 1e3))

    def _report_contact(self, position, quaternion, wrench, baseline,
                        force, torque, goal_position):
        """Publish the contact pose and wrench minus `baseline`; return the distance short of goal (m).

        The distance is reported raw: it is the depth estimate's error only when the goal was meant
        to lie on the surface, and the grasp leg deliberately aims inside the object.
        """
        stamp = rospy.Time.now()
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = position
        (msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w) = quaternion
        self.contact_pub.publish(msg)

        wmsg = WrenchStamped()
        wmsg.header.stamp = stamp
        wmsg.header.frame_id = self.base_frame
        d = np.asarray(wrench, dtype=float) - np.asarray(baseline, dtype=float)
        (wmsg.wrench.force.x, wmsg.wrench.force.y, wmsg.wrench.force.z) = d[:3]
        (wmsg.wrench.torque.x, wmsg.wrench.torque.y, wmsg.wrench.torque.z) = d[3:]
        self.contact_wrench_pub.publish(wmsg)

        short_by = float(np.linalg.norm(goal_position - position))
        return short_by

    def _ramp(self, start_position, start_quaternion, goal_position, goal_quaternion,
              guard=False):
        """Walk the setpoint from start to goal at a bounded speed; return a status string.

        A single jump would become a large impedance force, so the pose is interpolated (lerp and
        slerp) over max(distance / linear_speed, turn / angular_speed). With `guard`: hold until
        the wrench settles (pre_settle), take a baseline, then after a warm-up compare the wrench
        with the rolling reference; `debounce` consecutive franka_states samples over threshold
        stop the ramp and hold at the MEASURED pose (contact_hold). No contact is a normal outcome
        too, since a top-down grasp's fingers straddle the object; the peak is reported either way.
        """
        distance = float(np.linalg.norm(goal_position - start_position))
        turn = angle_between(start_quaternion, goal_quaternion)
        duration = max(distance / max(self.linear_speed, 1e-6),
                       turn / max(self.angular_speed, 1e-6))
        steps = max(1, int(round(duration * self.rate_hz)))

        # Goal quaternion on the start's sign, so the interpolation takes the short way round.
        goal_quaternion = canonicalize(goal_quaternion, start_quaternion)

        rospy.loginfo("pose_commander: ramping %.0f mm / %.0f deg over %.1fs (%d steps)",
                      distance * 1e3, np.degrees(turn), duration, steps)
        if self.dry_run:
            rospy.loginfo("pose_commander: DRY RUN -- would end at "
                          "[%.3f %.3f %.3f] q=[%.3f %.3f %.3f %.3f]",
                          goal_position[0], goal_position[1], goal_position[2],
                          *goal_quaternion)
            if guard:
                rospy.loginfo("pose_commander: DRY RUN -- the guard would be "
                              "ACTIVE at %.1f N / %.2f N.m over baseline",
                              self.guard_force, self.guard_torque)
            return "dry run"

        # Start from the held pose, so neither baseline nor warm-up sees a previous push released.
        pre = None
        if guard and self.pre_settle:
            start_position, start_quaternion, pre = self._hold_until_settled()
            rospy.loginfo("pose_commander: pre-guard hold: %s", pre)
            distance = float(np.linalg.norm(goal_position - start_position))
            turn = angle_between(start_quaternion, goal_quaternion)
            duration = max(distance / max(self.linear_speed, 1e-6),
                           turn / max(self.angular_speed, 1e-6))
            steps = max(1, int(round(duration * self.rate_hz)))
            goal_quaternion = canonicalize(goal_quaternion, start_quaternion)
        pre_note = "; pre-guard hold %s" % pre if pre else ""

        # Also the rolling reference's fallback until it has enough history.
        baseline = self._wrench_baseline() if guard else None
        history = []
        seen_f = seen_t = 0.0
        over, last_sample = 0, None

        rate = rospy.Rate(self.rate_hz)
        started = rospy.Time.now()
        for i in range(1, steps + 1):
            if rospy.is_shutdown():
                rospy.logwarn("pose_commander: shutdown mid-ramp; "
                              "the equilibrium stays where it last was")
                return "shutdown mid-ramp"
            t = float(i) / steps
            setpoint = start_position + t * (goal_position - start_position)
            self._publish(setpoint, slerp(start_quaternion, goal_quaternion, t))

            if guard:
                now_t = rospy.Time.now()
                with self._lock:
                    wrench = self._wrench
                    state = self._state
                    sample = self._wrench_n
                if wrench is not None:
                    history.append((now_t.to_sec(), list(wrench)))
                    # Keep only what the reference can use.
                    cutoff = now_t.to_sec() - (self.guard_lag + self.guard_window) * 2
                    while history and history[0][0] < cutoff:
                        history.pop(0)

            # Warm-up: the estimator reads the ramp's initial acceleration as an external force.
            if guard and (rospy.Time.now() - started).to_sec() >= self.guard_warmup:
                reference = baseline
                if self.guard_mode == "rolling":
                    reference = rolling_reference(
                        history, rospy.Time.now().to_sec(), self.guard_lag,
                        self.guard_window, self.guard_min_samples, baseline)
                hit, force, torque = contact_detected(
                    wrench, reference, self.guard_force, self.guard_torque)
                if force is not None:
                    seen_f = max(seen_f, force)
                    seen_t = max(seen_t, torque)
                # Debounce per franka_states sample: free-space torque spikes last one sample.
                over, last_sample = count_over(over, hit, sample, last_sample)
                hit = over >= max(1, self.guard_debounce)
                if hit and state is not None:
                    # Hold where the arm IS; the ramp's setpoint would keep pushing.
                    hold = contact_hold(state[0], start_position, goal_position,
                                        self.guard_backoff)
                    self._publish(hold, state[1])
                    short_by = self._report_contact(
                        state[0], state[1], wrench, baseline, force, torque,
                        goal_position)
                    rospy.sleep(self.settle_time)
                    return ("CONTACT at %.0f%% of the way in: %.1f N / %.2f N.m "
                            "over baseline, stopped %.1f mm short of the "
                            "commanded pose%s" % (100.0 * t, force, torque,
                                                  short_by * 1e3, pre_note))
            rate.sleep()

        if guard:
            arrival = self._settle(goal_position, goal_quaternion)
            # The peak shows how much headroom the thresholds have on this rig.
            return ("no contact, reached the commanded pose (%s) [peak %.2f N / "
                    "%.2f N.m against %.1f / %.1f]%s"
                    % (arrival, seen_f, seen_t, self.guard_force, self.guard_torque,
                       pre_note))

        return self._settle(goal_position, goal_quaternion)

    def _settle(self, goal_position, goal_quaternion):
        """Hold the setpoint until the arm arrives or stops closing; report, never refuse.

        Waits at least settle_time, then returns on settle_verdict ('arrived' within position and
        angle tolerance, 'stopped' after settle_patience s without progress) or after
        settle_timeout. The error is not chased to zero: the PD controller has no integral term, so
        the arm rests at e = F/K, and a flat residual is a force balance, not a fault.
        """
        rate = rospy.Rate(20.0)
        start = rospy.Time.now()
        floor = start + rospy.Duration(self.settle_time)
        deadline = start + rospy.Duration(self.settle_timeout)
        best = best_turn = None
        best_at = start
        error = None

        while not rospy.is_shutdown():
            with self._lock:
                state = self._state
            now = rospy.Time.now()
            if state is not None:
                error = float(np.linalg.norm(state[0] - goal_position))
                turn = angle_between(state[1], goal_quaternion)
                verdict = settle_verdict(error, (now - best_at).to_sec(),
                                         now >= floor, self.settle_tolerance,
                                         self.settle_patience, turn,
                                         self.settle_angle_tolerance)
                if verdict == "arrived":
                    return "arrived, %.1f mm, %.2f deg out" % (error * 1e3,
                                                                np.degrees(turn))
                if verdict == "stopped":
                    return ("stopped closing at %.1f mm, %.1f deg out "
                            "after %.1fs" % (error * 1e3, np.degrees(turn),
                                             (now - start).to_sec()))
                # Still closing if EITHER is: a roll improves only the angle.
                closing_p = is_closing(error, best, self.settle_progress)
                closing_r = is_closing(turn, best_turn, self.settle_angle_progress)
                if closing_p or closing_r:
                    best_at = now
                if closing_p:
                    best = error
                if closing_r:
                    best_turn = turn
            if now >= deadline:
                break
            rate.sleep()

        if error is None:
            return "no robot state, so where it got to is unknown"
        return "still %.1f mm out after %.1fs" % (error * 1e3,
                                                  self.settle_timeout)

    def _gripper(self, width):
        """Open to, close onto, hold or leave the fingers (choose_gripper_action); return a status.

        The action's result is reported plainly but not acted on: a grasp that finds nothing is
        information for the caller, not a reason to abandon the sequence here.
        """
        current = self._fingers
        action = choose_gripper_action(width, current, self.gripper_hold_tol)
        at = "unknown" if current is None else "%.0f mm" % (current * 1e3)

        if current is None:
            rospy.logwarn("pose_commander: no /franka_gripper/joint_states, so "
                          "'open to' cannot be told from 'close onto'; falling "
                          "back to a grasp at %.0f mm", width * 1e3)

        if action == "leave":
            rospy.loginfo("pose_commander: this leg does not command the "
                          "fingers; leaving them at %s", at)
            return "left alone at %s" % at

        if action == "hold":
            rospy.loginfo("pose_commander: fingers already at %s, leaving them "
                          "(commanded %.0f mm)", at, width * 1e3)
            return "held at %s" % at

        if self.dry_run:
            rospy.loginfo("pose_commander: DRY RUN -- would %s to %.0f mm "
                          "(from %s)", action, width * 1e3, at)
            return "%s to %.0f mm (dry run)" % (action, width * 1e3)

        if action == "move":
            if not self.move_client.wait_for_server(rospy.Duration(2.0)):
                rospy.logwarn("pose_commander: no franka_gripper move action; "
                              "leaving the fingers alone")
                return "no move action"
            goal = MoveGoal()
            goal.width = width
            goal.speed = self.gripper_speed
            client = self.move_client
        else:
            if not self.grasp_client.wait_for_server(rospy.Duration(2.0)):
                rospy.logwarn("pose_commander: no franka_gripper grasp action; "
                              "leaving the fingers alone")
                return "no grasp action"
            goal = GraspGoal()
            goal.width, goal.epsilon.inner, goal.epsilon.outer = grasp_goal_params(
                self.grasp_mode, width, self.grasp_under_width,
                self.grasp_under_eps_outer, self.gripper_eps_inner, self.gripper_eps_outer)
            goal.speed = self.gripper_speed
            goal.force = self.gripper_force
            client = self.grasp_client

        rospy.loginfo("pose_commander: %s fingers to %.0f mm (from %s)",
                      action, width * 1e3, at)
        client.send_goal(goal)
        finished = client.wait_for_result(rospy.Duration(10.0))
        result = client.get_result()
        ok = getattr(result, "success", None) if finished else None
        rospy.sleep(0.2)                       # let joint_states catch up
        closed = self._fingers
        if action == "move":
            outcome = ("" if ok else " (FAILED)" if ok is False else " (no result)")
            return "%s to %.0f mm%s" % (action, width * 1e3, outcome)
        verdict = grasp_verdict(ok, closed, self.grasp_miss_width)
        if not verdict.startswith("HELD"):
            rospy.logwarn("pose_commander: grasp %s", verdict)
        return "grasp (%s mode, predicted %.0f mm): %s" % (self.grasp_mode, width * 1e3, verdict)

    # -- services ------------------------------------------------------------

    def on_execute(self, _request):
        """move_to_pose: one unguarded leg to the latest pose, then the commanded width."""
        with self._lock:
            if self._busy:
                return TriggerResponse(success=False, message="already moving")
            self._busy = True
        try:
            return self._logged("move", self._execute)
        finally:
            with self._lock:
                self._busy = False

    def on_guarded_execute(self, _request):
        """guarded_move_to_pose: like move_to_pose, but stop the ramp if something is touched.

        A separate service rather than a flag, since Trigger carries no payload and a mode would
        persist between legs.
        """
        with self._lock:
            if self._busy:
                return TriggerResponse(success=False, message="already moving")
            self._busy = True
        try:
            return self._logged("guarded", lambda: self._execute(guarded=True))
        finally:
            with self._lock:
                self._busy = False

    def _execute(self, guarded=False):
        """Run one leg to the latest target pose, then apply the width; returns a TriggerResponse.

        Every check (listener, robot_mode, limits, planner re-check) runs before anything moves.
        An unguarded turn over rotation_max is ramped in the pieces planning.cut_over_cap gives
        (the cut plan_pick checks), each limit-checked; a guarded descent is never cut. The fingers
        are not commanded after a leg that did not happen (leg_outcome), and the robot is checked
        again after them, since closing on an object can raise a reflex too.
        """
        with self._lock:
            target = self._target
            width = self._width
        if target is None:
            return self._done(False, "no pose received on /live_pipeline/grasp_pose_best")

        try:
            self._require_listener()
            self._require_commanding()
            current_position, current_quaternion = self._current()
            position, quaternion = self._resolve(target)
            self._last_goal = (position, quaternion)
            pieces = [("leg", (position, quaternion))]
            turn_total = angle_between(current_quaternion, quaternion)
            if (not guarded and self.rotation_max
                    and turn_total > self.rotation_max):
                pieces = planning.cut_over_cap((current_position, current_quaternion),
                                               [("leg", (position, quaternion))],
                                               self.planner_cfg)
            start = (current_position, current_quaternion)
            for _, (p_goal, q_goal) in pieces:
                self._check(np.asarray(p_goal), q_goal, start[0], start[1])
                start = (np.asarray(p_goal), q_goal)
            step = float(np.linalg.norm(position - current_position))
            turn = turn_total
            self._check_leg((position, quaternion),
                            (current_position, current_quaternion))
        except Refused as exc:
            return self._done(False, str(exc))

        rospy.loginfo("pose_commander: going to [%.3f %.3f %.3f] "
                      "(%.0f mm away, %.0f deg turn%s)%s [%s]",
                      position[0], position[1], position[2],
                      step * 1e3, np.degrees(turn),
                      "" if len(pieces) == 1 else ", in %d pieces under the %.0f deg cap"
                      % (len(pieces), np.degrees(self.rotation_max)),
                      ", GUARDED at %.1f N / %.2f N.m" % (self.guard_force,
                                                          self.guard_torque)
                      if guarded else "",
                      self._posture() or "posture unknown")
        try:
            for index, (_, (p_goal, q_goal)) in enumerate(pieces):
                if index:
                    current_position, current_quaternion = self._current()
                arrival = self._ramp(current_position, current_quaternion,
                                     np.asarray(p_goal), q_goal, guard=guarded)
                if index < len(pieces) - 1:
                    failed = self._outcome(np.asarray(p_goal), False)
                    if failed:
                        return self._done(False, "stopped at piece %d/%d: %s (%s); fingers "
                                          "left alone" % (index + 1, len(pieces), failed, arrival))
        except Refused as exc:
            # E.g. no wrench for the guard's baseline, or a stale state between pieces.
            return self._done(False, str(exc))
        rospy.loginfo("pose_commander: %s", arrival)
        contact = str(arrival).startswith("CONTACT")
        failed = self._outcome(position, contact)
        if failed:
            return self._done(False, "%s (%s); fingers left alone" % (failed, arrival))

        if width is None:
            width = self.default_width
            rospy.loginfo("pose_commander: no width published; closing to the "
                          "default %.0f mm", width * 1e3)
        did = self._gripper(width)
        # Closing on an object can raise a reflex too.
        failed = self._outcome(position, True)
        if failed:
            return self._done(False, "%s -- after the fingers moved (%s)" % (failed, did))

        return self._done(True, "at [%.3f %.3f %.3f] (%s), fingers %s"
                          % (position[0], position[1], position[2],
                             arrival, did))

    def on_measure_load(self, _request):
        """measure_load: the standing external force on the arm, averaged over load/window s.

        The sequencer reads it at the standoff and again after the retreat (the same pose), so the
        posture-dependent bias cancels and the difference is the weight carried; this catches an
        object held that the finger width calls a miss, e.g. a mug caged through its handle. A
        carried load raises the base-frame z force (measured on the rig). The setpoint is first
        relaxed onto the arm (_relax_setpoint). Reply JSON: load_record() plus position and
        relaxed_mm; success=false while moving, with no wrench samples, or when the reading is not
        steady (force spread over 0.5 N or tool motion over 1 mm).
        """
        with self._lock:
            if self._busy:
                return TriggerResponse(success=False, message="already moving")
            self._busy = True
        try:
            relaxed_mm = self._relax_setpoint()
            wrenches, positions = [], []
            rate = rospy.Rate(100.0)
            end = time.time() + float(rospy.get_param("~load/window", 0.5))
            while time.time() < end and not rospy.is_shutdown():
                with self._lock:
                    wrench, state = self._wrench, self._state
                if wrench is not None and state is not None:
                    wrenches.append(list(wrench))
                    positions.append(list(state[0]))
                rate.sleep()
            if not wrenches:
                return TriggerResponse(success=False, message="no external wrench on franka_states")
            rec = load_record(wrenches, positions)
            rec["position"] = [round(float(v), 4) for v in positions[-1]]
            rec["relaxed_mm"] = relaxed_mm
            steady = rec["force_span"] <= 0.5 and rec["motion_mm"] <= 1.0
            rospy.loginfo("pose_commander: load at [%.3f %.3f %.3f]: F = [%.2f %.2f %.2f] N "
                          "(spread %.2f N, moved %.1f mm)%s", *(rec["position"] + rec["force"] + [
                              rec["force_span"], rec["motion_mm"], "" if steady else " -- NOT steady"]))
            return TriggerResponse(success=steady, message=json.dumps(rec))
        finally:
            with self._lock:
                self._busy = False

    def _relax_setpoint(self):
        """Put the setpoint on the MEASURED pose and wait; return the gap it closed (mm), or None.

        Under impedance control an arm stalled short of its setpoint is pulled by the spring K*e
        (~1 N per mm), which the wrench estimate includes and a load reading would mistake for
        weight. Skipped under position control, which tracks to ~1 mm, and in dry run.
        """
        if self.dry_run or self.command_topic.rstrip("/").endswith("target_pose"):
            return None
        with self._lock:
            state = self._state
        if state is None:
            return None
        goal = self._last_setpoint if getattr(self, "_last_setpoint", None) is not None else None
        gap = None if goal is None else round(float(np.linalg.norm(np.asarray(goal) - state[0])) * 1e3, 1)
        self._publish(state[0], state[1])
        rospy.sleep(float(rospy.get_param("~load/relax_time", 0.5)))
        return gap

    def _init_home(self, p):
        """Set home to the forward kinematics of home_pose.yaml's joints (run_home.sh's home).

        With home/source `startup`, or no readable home_joints, home stays None and on_state
        latches the first measured pose instead.
        """
        self._home_reported = False
        source = str(p("~home/source", "home_pose")).strip().lower()
        if source not in ("home_pose", "startup"):
            rospy.logwarn("pose_commander: home/source %r is not home_pose|startup; "
                          "using home_pose", source)
            source = "home_pose"
        if source == "startup":
            rospy.loginfo("pose_commander: home/source startup -- home is wherever "
                          "this node first sees the arm")
            return
        if not self.home_joints or len(self.home_joints) != 7:
            rospy.logwarn("pose_commander: no home_joints (%s), so home falls back to "
                          "where this node first sees the arm",
                          self.home_error or "home_pose.yaml not read")
            return
        m = kinematics.ee_matrix(self.home_joints)
        self._home = (m[:3, 3].copy(), tuple(quaternion_from_matrix(m)))
        self._home_source = "home_pose.yaml"
        rospy.loginfo("pose_commander: home = home_pose.yaml's joints, tool at "
                      "[%.3f %.3f %.3f]", *self._home[0])

    def on_home(self, _request):
        """go_home: ramp the tool back to home in planned legs; the gripper is untouched.

        It moves the tool, not the joints, so it does not set the posture the way run_home.sh does.
        """
        with self._lock:
            if self._busy:
                return TriggerResponse(success=False, message="already moving")
            self._busy = True
            home = self._home
        try:
            return self._logged("home", lambda: self._home_leg(home))
        finally:
            with self._lock:
                self._busy = False

    def _home_leg(self, home):
        """Drive to `home` in the legs planning.home_legs cuts (the same cut plan_pick checks).

        Each leg gets the floor, reach, step and turn limits, and the planner checks the whole
        chain from the measured joints before anything moves; each leg's outcome is checked after.
        """
        if home is None:
            return self._done(False, "no home pose yet -- no robot state has "
                                     "arrived since this node started")
        try:
            self._require_listener()
            self._require_commanding()
            current_position, current_quaternion = self._current()
            self._last_goal = home
            legs = planning.home_legs((current_position, current_quaternion), home,
                                      self.planner_cfg)
            start = (current_position, current_quaternion)
            for _, goal in legs:
                self._check(np.asarray(goal[0]), goal[1], start[0], start[1])
                start = (np.asarray(goal[0]), goal[1])
            self._check_leg(legs, (current_position, current_quaternion), approved=False)
        except Refused as exc:
            return self._done(False, str(exc))

        rospy.loginfo("pose_commander: going home to [%.3f %.3f %.3f] from %s "
                      "(%.0f mm away, %.0f deg turn, %d leg%s) [%s]",
                      home[0][0], home[0][1], home[0][2], self._home_source,
                      np.linalg.norm(np.asarray(home[0]) - current_position) * 1e3,
                      np.degrees(angle_between(current_quaternion, home[1])),
                      len(legs), "" if len(legs) == 1 else "s",
                      self._posture() or "posture unknown")
        arrival = None
        for name, goal in legs:
            if name != legs[0][0]:
                current_position, current_quaternion = self._current()
            arrival = self._ramp(current_position, current_quaternion,
                                 np.asarray(goal[0]), goal[1])
            failed = self._outcome(np.asarray(goal[0]), False)
            if failed:
                return self._done(False, "home stopped at %s: %s (%s)" % (name, failed, arrival))
        return self._done(True, "home: at [%.3f %.3f %.3f] in %d leg%s (%s)"
                          % (home[0][0], home[0][1], home[0][2], len(legs),
                             "" if len(legs) == 1 else "s", arrival))

    # -- the trials' views: go_v1 (= go_home), go_v2 -----------------------------------

    def _init_views(self, p):
        """~views/<name>: {viewpoint, tilt_deg, height[, centre]}, served as /live_pipeline/go_<name>.
        Without the parameter: v2, the trials' side view (run_viewpoint.sh 4 --tilt 45 --height
        0.35). v1 is not configurable: it is home."""
        raw = p("~views", None)
        if raw is None:
            raw = {"v2": {"viewpoint": 4, "tilt_deg": 45.0, "height": 0.35}}
        self.views = {}
        for name, spec in sorted(raw.items()):
            name = str(name).strip().lower()
            if name in ("v1", "home"):
                rospy.logwarn("pose_commander: views/%s ignored: v1 is home (go_v1 = go_home)", name)
                continue
            try:
                view = {"viewpoint": int(spec["viewpoint"]),
                        "tilt_deg": float(spec.get("tilt_deg", viewpoints.DEFAULT_TILT_DEG)),
                        "height": float(spec.get("height", viewpoints.DEFAULT_HEIGHT)),
                        "centre": tuple(float(v) for v in spec.get("centre", viewpoints.DEFAULT_CENTRE))}
                if view["viewpoint"] not in viewpoints.NAMES or len(view["centre"]) != 2:
                    raise ValueError("viewpoint 1-5 and a 2-value centre")
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                rospy.logerr("pose_commander: views/%s ignored (%s): %r", name, exc, spec)
                continue
            self.views[name] = view
            rospy.loginfo("pose_commander: go_%s = viewpoint %d (%s), tilt %.0f deg, camera %.2f m "
                          "above the table, centre (%.2f, %.2f)", name, view["viewpoint"],
                          viewpoints.NAMES[view["viewpoint"]], view["tilt_deg"], view["height"],
                          view["centre"][0], view["centre"][1])

    def on_view(self, name):
        """Drive to view `name` in planned legs. The gripper is untouched."""
        with self._lock:
            if self._busy:
                return TriggerResponse(success=False, message="already moving")
            self._busy = True
        try:
            return self._logged("view", lambda: self._view_legs(name))
        finally:
            with self._lock:
                self._busy = False

    def _plan_view(self, view, q, pose_now):
        """viewpoints.plan_viewpoint with tools/goto_viewpoint.py's settings, so the roll and the legs
        are the ones run_viewpoint.sh would choose from the same joints."""
        cfg = planning.PlanConfig(
            ee_to_camera=tuple(self.ee_to_camera[:3]), rotation_max=self.rotation_max,
            joint_margin=self.planner_cfg.joint_margin, table_z=self.planner_cfg.table_z,
            linear_speed=self.linear_speed, angular_speed=self.angular_speed)
        q_ns = self.home_joints if self.planner_model == "nullspace" else None
        return viewpoints.plan_viewpoint(
            view["viewpoint"], q, pose_now, self.planner_limits, cfg, self.ee_to_camera,
            view["centre"], cfg.table_z, view["height"], view["tilt_deg"],
            max_leg_deg=planning.leg_split(cfg)[0], model=self.planner_model, q_ns=q_ns)

    def _view_legs(self, name):
        view = self.views[name]
        try:
            # Never driven to unplanned: without the planner there is no roll, and no legs.
            if not self._planner_ready():
                raise Refused("the planner is %s (%s), and a view is driven to only when it has "
                              "planned the whole way" % (self.planner_mode, self.planner_error or "off"))
            if self.ee_to_camera is None:
                raise Refused("no ~ee_to_camera: the view aims the camera, so it needs the camera's pose")
            self._require_listener()
            self._require_commanding()
            with self._lock:
                q = None if self._q is None else [float(v) for v in self._q]
            if q is None:
                raise Refused("no joint angles to plan from")
            current_position, current_quaternion = self._current()
            plan = self._plan_view(view, q, (current_position, current_quaternion))
            if not plan["ok"]:
                raise Refused("out of reach from here: %s" % plan["reasons"][0])
            legs = plan["legs"]
            self._last_goal = legs[-1][1]
            start = (current_position, current_quaternion)
            for _, goal in legs:
                self._check(np.asarray(goal[0]), goal[1], start[0], start[1])
                start = (np.asarray(goal[0]), goal[1])
            self._check_leg(legs, (current_position, current_quaternion), approved=False)
        except Refused as exc:
            return self._done(False, "%s: %s" % (name, exc))

        rospy.loginfo("pose_commander: going to %s (viewpoint %d, tilt %.0f deg): camera roll %.0f "
                      "deg, wrist turn %.0f deg in %d leg%s, worst joint margin %.2f rad [%s]",
                      name, view["viewpoint"], view["tilt_deg"], plan["roll_deg"], plan["turn_deg"],
                      len(legs), "" if len(legs) == 1 else "s", plan["joint_margin"],
                      self._posture() or "posture unknown")
        arrival = None
        for index, (leg_name, goal) in enumerate(legs):
            if index:
                current_position, current_quaternion = self._current()
            arrival = self._ramp(current_position, current_quaternion,
                                 np.asarray(goal[0]), goal[1])
            failed = self._outcome(np.asarray(goal[0]), False)
            if failed:
                return self._done(False, "%s stopped at %s: %s (%s)" % (name, leg_name, failed, arrival))
        end = legs[-1][1][0]
        return self._done(True, "%s: at [%.3f %.3f %.3f] in %d leg%s, camera roll %.0f deg (%s)"
                          % (name, end[0], end[1], end[2], len(legs), "" if len(legs) == 1 else "s",
                             plan["roll_deg"], arrival))

    # -- the reachability planner ------------------------------------------------

    def _init_planner(self, p, command_topic):
        """Read the planner settings (planning.PlanConfig), FR3 joint limits and home_pose.yaml.

        A planner whose joint limits cannot load is reported once and stands down: the commander's
        own checks still run, and its verdicts say "unavailable" rather than passing.
        """
        mode = str(p("~planner/mode", "enforce")).strip().lower()
        if mode not in ("off", "report", "enforce"):
            rospy.logwarn("pose_commander: planner/mode %r is not off|report|enforce; "
                          "using report", mode)
            mode = "report"
        self.planner_mode = mode
        self.planner_model = planning.model_for_topic(command_topic)
        self.planner_budget = float(p("~planner/budget_s", 5.0))
        self.planner_top_k = int(p("~planner/top_k", 64))
        self.candidates_max_age = float(p("~planner/candidates_max_age", 5.0))
        self.approved_tolerance = float(p("~planner/approved_tolerance", 0.002))
        self.nullspace_reference = str(p("~planner/nullspace_reference", "startup")).strip().lower()
        if self.nullspace_reference not in ("startup", "home"):
            rospy.logwarn("pose_commander: planner/nullspace_reference %r is not "
                          "startup|home; using startup", self.nullspace_reference)
            self.nullspace_reference = "startup"
        camera = p("~ee_to_camera", None)
        # The whole pose, not just the planner's offset: the views aim the camera (go_v2).
        self.ee_to_camera = [float(v) for v in camera] if camera and len(camera) == 7 else None
        self.planner_cfg = planning.PlanConfig(
            joint_margin=float(p("~planner/joint_margin", 0.05)),
            max_step_dq=float(p("~planner/max_step_dq", 0.1)),
            table_z=float(p("~planner/table_z", planning.TABLE_Z)),
            clearance=float(p("~planner/clearance", 0.03)),
            max_speed_frac=float(p("~planner/max_speed_frac", 0.5)),
            min_sigma=float(p("~planner/min_sigma", 0.0)),
            scene_margin=float(p("~planner/scene_margin", 0.01)),
            target_margin=float(p("~planner/target_margin", 0.0)),
            finger_check=bool(p("~planner/finger_check", True)),
            finger_opening=float(p("~planner/finger_opening", 0.078)),
            table_mode=str(p("~planner/table_mode", "plane")),
            table_margin=float(p("~planner/table_margin", 0.005)),
            table_band=float(p("~planner/table_band", 0.02)),
            linear_speed=self.linear_speed, angular_speed=self.angular_speed,
            # The commander's own limits, so plan_pick never approves a leg _check refuses.
            rotation_max=getattr(self, "rotation_max", None),
            z_min=getattr(self, "z_min", None),
            radius_max=getattr(self, "radius_max", None),
            rank=str(p("~planner/rank", "utility")).strip().lower(),
            ee_to_camera=tuple(camera[:3]) if camera else None)
        self.planner_limits = self.home_joints = None
        self.planner_error = None
        try:
            import rospkg
            pack = rospkg.RosPack()
            limits = p("~planner/joint_limits", "") or os.path.join(
                pack.get_path("franka_description"), "robots", "fr3", "joint_limits.yaml")
            self.planner_limits = planning.load_limits(limits)
        except Exception as exc:  # noqa: BLE001 -- reported, and the planner stands down
            self.planner_error = "%s: %s" % (type(exc).__name__, exc)
            rospy.logerr("pose_commander: planner unavailable (%s); legs are checked "
                         "by the tool-pose limits only", self.planner_error)
        # Read separately: go_home needs it even when the planner is down.
        self.home_error = None
        try:
            import rospkg
            import yaml
            home_file = p("~planner/home_pose_file", "") or os.path.join(
                rospkg.RosPack().get_path("fr3_pose_commander"), "config", "home_pose.yaml")
            with open(home_file) as handle:
                self.home_joints = [float(v) for v in yaml.safe_load(handle)["home_joints"]]
        except Exception as exc:  # noqa: BLE001 -- reported; home falls back to startup
            self.home_error = "%s: %s" % (type(exc).__name__, exc)
            rospy.logwarn("pose_commander: home_pose.yaml unreadable (%s)", self.home_error)
        rospy.loginfo("pose_commander: planner %s, model %s%s", self.planner_mode,
                      self.planner_model,
                      "" if self.planner_cfg.ee_to_camera else
                      " (no ~ee_to_camera: the camera's clearance is not checked)")

    def _q_ns(self):
        """The nullspace posture the impedance controller holds, or None for other planner models.

        The controller latches q at its own start and publishes it nowhere, so a proxy is used
        (planner/nullspace_reference): `startup` (default) is the joints this node first saw, right
        when the controller started first with the arm untouched; `home` is home_pose.yaml, right
        only if the arm was homed before the controller started.
        """
        if self.planner_model != "nullspace":
            return None
        if self.nullspace_reference == "startup" and getattr(self, "_q0", None):
            return self._q0
        return self.home_joints

    def _check_startup_posture(self):
        """Log how far the startup joints are from home_pose.yaml; warn if the planner assumes home.

        With nullspace_reference `home` the planner predicts a pull toward home_joints, which is
        wrong if the controller started elsewhere. Called from on_state with the lock held, so it
        does nothing beyond one log line.
        """
        if self.planner_model != "nullspace" or not self.home_joints:
            return
        off = max(abs(a - b) for a, b in zip(self._q0, self.home_joints))
        if self.nullspace_reference == "startup":
            rospy.loginfo("pose_commander: nullspace reference = the startup posture "
                          "(%.3f rad from home_pose.yaml at most)", off)
            return
        if off > 0.05:
            rospy.logwarn("pose_commander: the arm started %.2f rad from home_pose.yaml. "
                          "The planner predicts a nullspace pull toward home_joints; "
                          "re-home (run_home.sh) before the controller if that is not "
                          "where it will pull.", off)

    def _planner_ready(self):
        return self.planner_mode != "off" and self.planner_limits is not None

    def on_candidates(self, msg):
        with self._lock:
            self._candidates = (msg.data, time.time())

    def on_plan_pick(self, _request):
        """plan_pick: choose the most confident candidate whose every leg the arm can follow.

        Uses the latest pick_candidates, which must have arrived within candidates_max_age by THIS
        machine's clock (list and trigger cross the bridge separately, so a leftover list is the
        risk; Orin stamps are not compared). Replies JSON echoing plan_id, and stores the chosen
        legs with the candidate's own scene for the per-leg re-check (_check_leg). success=false
        when the list is missing, stale or malformed, there is no robot state, the arm is moving,
        or in enforce mode nothing passes; in report mode "chosen": null still succeeds and the
        sequencer falls back to the detector's best grasp.
        """
        with self._lock:
            if self._busy:
                return TriggerResponse(success=False, message="already moving")
            self._busy = True
            candidates, state, q = self._candidates, self._state, self._q
            home = self._home
            # Cleared first, so a refusal cannot leave the previous pick's legs approved.
            self._approved = None
        try:
            reply = {"mode": self.planner_mode, "model": self.planner_model}
            if self.planner_mode == "off":
                reply.update(chosen=None, note="planner is off")
                return TriggerResponse(success=True, message=json.dumps(reply))
            if self.planner_limits is None:
                reply.update(chosen=None, error="planner unavailable: %s" % self.planner_error)
                return TriggerResponse(success=self.planner_mode != "enforce",
                                       message=json.dumps(reply))
            if candidates is None:
                return TriggerResponse(success=False,
                                       message="no /live_pipeline/pick_candidates received")
            age = time.time() - candidates[1]
            if age > self.candidates_max_age:
                return TriggerResponse(success=False, message=(
                    "the last pick_candidates arrived %.1fs ago (limit %.1fs); it "
                    "describes an earlier pick" % (age, self.candidates_max_age)))
            if state is None or q is None or len(q) != 7:
                return TriggerResponse(success=False,
                                       message="no robot state yet -- is the controller up?")
            try:
                plan_id, parsed = planning.parse_candidates(candidates[0], self.base_frame)
                # None when no scene was sent: the scene check is skipped (scene_voxels 0).
                scene = planning.parse_obstacles(candidates[0], self.base_frame)
                # The table is checked as a plane; only objects stay as voxels.
                scene = planning.objects_only(scene, self.planner_cfg)
            except ValueError as exc:
                return TriggerResponse(success=False, message=str(exc))
            reply["plan_id"] = plan_id
            try:
                result = planning.plan_legs(
                    q, (state[0], state[1]), parsed, self.planner_model,
                    self.planner_limits, self.planner_cfg, q_ns=self._q_ns(),
                    home_pose=home, budget_s=self.planner_budget, top_k=self.planner_top_k,
                    scene=scene)
            except Exception as exc:  # noqa: BLE001 -- unknown is reported, not crashed on
                rospy.logwarn("pose_commander: plan_pick failed: %s", exc)
                reply.update(chosen=None, error="planner raised: %s" % exc)
                return TriggerResponse(success=self.planner_mode != "enforce",
                                       message=json.dumps(reply))
            reply.update(planning.compact(result))
            chosen = result["chosen"]
            if chosen is not None:
                cand = next(c for c in parsed if c["id"] == chosen)
                legs = cand["legs"]
                grasp = next((goal for name, goal in legs if name == "grasp"), None)
                # The candidate's own scene and target exclusion, for each leg's re-check.
                cand_scene = planning.candidate_scene(scene, cand)
                self._approved = {
                    "plan_id": plan_id, "id": chosen,
                    "positions": [np.asarray(goal[0]) for _, goal in legs],
                    "names": [name for name, _ in legs],
                    "next": 0,                 # the leg expected next; see approved_leg
                    "scene": cand_scene,
                    "near": (planning.near_target(cand_scene, grasp, cand.get("width_m"),
                                                  self.planner_cfg)
                             if grasp is not None else None)}
            rospy.loginfo("pose_commander: plan_pick %s: %s of %d evaluated in %.2fs%s",
                          plan_id, chosen or "NOTHING feasible",
                          len(result["candidates"]), result["elapsed_s"],
                          "" if chosen else " -- " + "; ".join(
                              "%s: %s" % (c["id"], (c["reasons"] or ["?"])[0])
                              for c in result["candidates"]))
            ok = chosen is not None or self.planner_mode != "enforce"
            return TriggerResponse(success=ok, message=json.dumps(reply))
        finally:
            with self._lock:
                self._busy = False

    def _check_leg(self, goal, start, approved=True):
        """Planner re-check of the next leg from the MEASURED joints; the verdict goes to the log.

        plan_pick predicted each leg from the previous one's predicted end, but a guarded leg can
        stop early and an impedance arm settles short, so each leg is re-checked from where the arm
        is, against the matched approved leg's candidate scene (target excluded on grasp and
        retreat). `goal` is one pose, or a list of (name, pose) legs checked as a chain; go_home
        passes approved=False, since it is no plan_pick leg and carries the object. Enforce mode
        refuses via _maybe_refuse (so --no-checks overrules it); report mode only logs.
        """
        if not self._planner_ready():
            if self.planner_mode != "off":
                self._plan_check = {"ok": None, "error": "planner unavailable"}
            return
        with self._lock:
            q = self._q
            approved = self._approved if approved else None
        if q is None or len(q) != 7:
            self._plan_check = {"ok": None, "error": "no joint angles"}
            return
        leg_name, leg_scene = "leg", None
        chain = goal if isinstance(goal, list) else None
        matched, next_before = None, None
        if chain is None and approved is not None:
            matched = approved_leg(goal[0], approved["positions"],
                                   approved.get("next", 0), self.approved_tolerance)
            if matched is None:
                gap = min(float(np.linalg.norm(np.asarray(goal[0]) - a))
                          for a in approved["positions"])
                rospy.logwarn("pose_commander: this leg is %.0f mm from every leg "
                              "plan_pick approved (%s)", gap * 1e3, approved["id"])
            else:
                # Advance `next` now; it is restored below if this leg is refused.
                leg_name = approved["names"][matched]
                leg_scene = (approved["near"] if leg_name in ("grasp", "retreat")
                             else approved["scene"])
                with self._lock:
                    next_before = approved.get("next", 0)
                    approved["next"] = matched + 1
        try:
            r = planning.check_legs(q, start, chain or [(leg_name, goal)], self.planner_model,
                                    self.planner_limits, self.planner_cfg,
                                    q_ns=self._q_ns(), scene=leg_scene)
        except Exception as exc:  # noqa: BLE001 -- unknown is reported, not crashed on
            self._plan_check = {"ok": None, "error": "planner raised: %s" % exc}
            rospy.logwarn("pose_commander: leg check failed: %s", exc)
            return
        self._plan_check = {"ok": r["ok"], "reasons": r["reasons"],
                            "prediction": r["prediction"], "q_end": r["q_end"],
                            "model": self.planner_model}
        if r["ok"]:
            return
        why = "planner (%s%s): %s" % (self.planner_model,
                                      "" if r["prediction"] in (None, self.planner_model)
                                      else ", " + r["prediction"], r["reasons"][0])
        if self.planner_mode == "enforce":
            try:
                self._maybe_refuse(why)
            except Refused:
                if matched is not None:        # refused: it did not run
                    with self._lock:
                        approved["next"] = next_before
                raise
        else:
            rospy.logwarn("pose_commander: report only, would refuse: %s", why)

    # -- the leg log -----------------------------------------------------------

    def _robot_snapshot(self):
        with self._lock:
            state, q, elbow = self._state, self._q, self._elbow
        # Wall-clock, not ROS time: the log is read beside other machines' logs.
        return leg_snapshot(state, q, elbow, time.time())

    def _logged(self, kind, leg):
        """Run one leg and append its line to the leg log, whatever happens.

        The start snapshot is taken before the leg's checks, so refused legs are logged too
        (`target` null if none was resolved). The write is in a finally and never raises; the
        leg's own exception still propagates.
        """
        self._last_goal = None
        self._plan_check = None
        start = self._robot_snapshot()
        response = None
        try:
            response = leg()
            return response
        finally:
            try:
                entry = leg_record(kind, self.command_topic, self.dry_run,
                                   self._last_goal, start, self._robot_snapshot(),
                                   response is not None and response.success,
                                   response.message if response is not None
                                   else "raised", planner=self._plan_check)
                if self.leg_log.write(entry) is None and self.leg_log.enabled:
                    # Not logwarn_throttle: it reads ROS time, which can raise here.
                    rospy.logwarn("pose_commander: leg log not written: %s",
                                  self.leg_log.last_error)
            except Exception as exc:  # noqa: BLE001 -- the log must never cost a leg
                try:
                    rospy.logwarn("pose_commander: leg log failed: %s", exc)
                except Exception:  # noqa: BLE001 -- not even by failing to say so
                    pass

    def on_release(self, _request):
        """release: open the fingers to gripper/open_width with a move action."""
        if self.dry_run:
            return self._done(True, "DRY RUN -- would open the fingers")
        if not self.move_client.wait_for_server(rospy.Duration(2.0)):
            return self._done(False, "no franka_gripper move action")
        goal = MoveGoal()
        goal.width = self.gripper_open_width
        goal.speed = self.gripper_speed
        self.move_client.send_goal(goal)
        self.move_client.wait_for_result(rospy.Duration(10.0))
        return self._done(True, "fingers opened to %.0f mm" % (self.gripper_open_width * 1e3))

    def _done(self, success, message):
        """Add the posture report, log, publish on execution_status; return the TriggerResponse."""
        posture = self._posture()
        if posture:
            message = "%s [%s]" % (message, posture)
        if success:
            rospy.loginfo("pose_commander: %s", message)
        else:
            rospy.logwarn("pose_commander: refused: %s", message)
        self.status_pub.publish(String(data=("ok: " if success else "refused: ") + message))
        return TriggerResponse(success=success, message=message)


def main():
    rospy.init_node("pose_commander")
    PoseCommander()
    rospy.spin()


if __name__ == "__main__":
    main()
