"""Run a pick on the FR3 through the ROS 1 bridge. MOVES THE ARM.

    ros2 service call /live_pipeline/execute_pick std_srvs/srv/Trigger

The camera is on the gripper, so the grasp is resolved into the base frame once, from the tool pose at capture:
T_base_goal = T_base_EE(capture) . T_EE_camera . T_camera_grasp. The NUC planner (plan_pick) chooses a candidate the
arm can follow; then standoff (open) -> grasp (guarded, closes to the width) -> retreat, a load check, optionally
home, and a pick record in the run's archive.

In  (ROS 2)  /live_pipeline/grasp_pose_camera, grasp_width_camera, grasp_candidates
In  (NUC)    /live_pipeline/tool_pose, execution_status, contact_pose
Out (NUC)    /live_pipeline/pick_candidates, grasp_pose_best + grasp_width_best (one leg at a time)
Calls (NUC)  plan_pick, move_to_pose, guarded_move_to_pose, measure_load, go_home
This stops the sequence on a problem; it is not an emergency stop.
"""
import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger

from . import eye_in_hand as eh
from .run_export import spawn as spawn_export
from .common import CAMERA_FRAME, LIVE_DIR, RUNS_DIR, declare, latched_qos, run_node

# Refusal kinds as the NUC's planning.CATEGORIES names them, for an older NUC reply without its own summary.
_KINDS = (("wrist turn", ("per-command limit",)), ("edge of reach", ("edge of reach",)),
          ("joint limit", ("its limit",)), ("posture flip", ("posture flip",)),
          ("no IK solution", ("no joint solution",)), ("joint speed", ("joint speed",)),
          ("arm vs table", ("of the table", "into the table")),
          ("arm vs scene", ("of the scene", "intersects the scene")),
          ("not evaluated", ("budget",)))


def plan_summary(reply):
    """One line for a person from a plan_pick reply: "3 of 16 feasible; 9 wrist turn, ..."."""
    if reply.get("summary_text"):
        return reply["summary_text"]
    cands = reply.get("candidates") or []
    counts = {}
    for c in cands:
        if c.get("ok"):
            continue
        text = str((c.get("reasons") or [""])[0])
        kind = c.get("category") or next((k for k, keys in _KINDS if any(x in text for x in keys)), "other")
        counts[kind] = counts.get(kind, 0) + 1
    ok = sum(1 for c in cands if c.get("ok"))
    parts = ", ".join(f"{n} {k}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    return f"{ok} of {len(cands)} feasible" + (f"; {parts}" if parts else "") + (
        f" ({reply['error']})" if reply.get("error") else "")


def funnel_line(funnel, reply, n_candidates):
    """The whole pruning funnel in one line: "SBG 22 raw -> 13 collision-free -> 8 accepted
    -> 6 on target -> 12 candidates -> 3 feasible -> chose 5:direct (utility 0.061)"."""
    f = funnel or {}
    parts = []
    if f.get("raw") is not None:
        parts.append(f"SBG {f['raw']} raw")
    for key, label in (("after_nms", "after NMS"), ("collision_free", "collision-free"),
                       ("accepted", "accepted")):
        if f.get(key) is not None:
            parts.append(f"{f[key]} {label}")
    if f.get("executable") is not None:
        extra = [f"{f[k]} {n}" for k, n in (("not_on_target", "off target"), ("unseen_side", "from the unseen side"))
                 if f.get(k)]
        parts.append(f"{f['executable']} executable" + (f" ({', '.join(extra)} dropped)" if extra else ""))
    parts.append(f"{n_candidates} candidates")
    if reply is not None:
        cands = reply.get("candidates") or []
        parts.append(f"{sum(1 for c in cands if c.get('ok'))} feasible")
        chosen = reply.get("chosen")
        if chosen:
            u = next((c.get("utility") for c in cands if c.get("id") == chosen), None)
            by_c = reply.get("by_confidence")
            parts.append(f"chose {chosen}" + (f" (utility {u:.3f})" if u is not None else "")
                         + (f", by confidence {by_c}" if by_c and by_c != chosen else ""))
        else:
            parts.append("chose nothing")
    return " -> ".join(parts)


G = 9.81


def load_verdict(before, after, gripper_text="", threshold_n=1.0):
    """(verdict, one line) from two measure_load replies at the same pose: the standoff before the grasp and the
    retreat after it.

    `lifted` is the rise in the external force's z (base frame); the sign was measured on the rig (a 0.23 kg mug
    read +2.29 N). At or over `threshold_n` (~100 g; the idle reading spreads 0.1-0.2 N) something is carried. It is
    compared with the gripper's width verdict because a cage (a finger through a handle) closes the fingers fully
    and reads MISSED while it carries the object."""
    if not before or not after:
        return "unknown", "load not measured"
    lifted = float(after["force"][2]) - float(before["force"][2])
    kg = lifted / G
    missed = "MISSED" in (gripper_text or "")
    if lifted >= threshold_n:
        if missed:
            return "caged", (f"CARRYING {kg:.2f} kg ({lifted:.1f} N) although the fingers closed fully -- "
                             f"caged (e.g. through a handle), not pinched")
        return "held", f"holding {kg:.2f} kg ({lifted:.1f} N)"
    if "HELD" in (gripper_text or ""):
        return "not lifted", (f"the gripper says HELD but nothing was lifted ({kg:.2f} kg) -- pinched "
                              f"something fixed, or it slipped")
    return "nothing", f"nothing lifted ({kg:.2f} kg, {lifted:.1f} N)"


def run_dir_for(runs_dir, capture_stamp, slack=10.0):
    """The archived run this pick's grasps came from, or None: the newest run directory written no earlier than the
    grasp's stamp (less `slack` s), as the orchestrator archives right after the chain that published it."""
    runs_dir = Path(runs_dir)
    if capture_stamp is None or not runs_dir.is_dir():
        return None
    dirs = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "grasp_poses.json").exists()]
    dirs = [d for d in dirs if d.stat().st_mtime >= capture_stamp - slack]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def pose_of(msg):
    """PoseStamped -> ((x, y, z), unit quaternion (x, y, z, w))."""
    p, o = msg.pose.position, msg.pose.orientation
    return (p.x, p.y, p.z), eh.normalize((o.x, o.y, o.z, o.w))


def build_pick_candidates(payload, grasp_stamp, capture_tool, tool, ee_to_camera, ee_to_grasp,
                          standoff, retreat, do_retreat, max_width, camera_frame):
    """grasp_candidates -> (legs for pick_candidates, {id: (goal, width)}), most confident first.

    Each candidate with a usable width is resolved into the base frame from the capture pose and offered in both
    jaw orders ("direct", "flipped"), the one closer to the current tool orientation first. The payload's stamp
    must match the held grasp's, so the candidates come from the same grasp_poses call."""
    if payload.get("frame_id") != camera_frame:
        raise ValueError(f"grasp_candidates are in {payload.get('frame_id')!r}, not {camera_frame!r}")
    stamp = payload.get("stamp")
    if stamp is None or grasp_stamp is None or abs(stamp - grasp_stamp) > 1e-6:
        raise ValueError("grasp_candidates are not from the same call as the held grasp")
    wire, lookup = [], {}
    for c in sorted(payload.get("candidates") or [], key=lambda c: -float(c.get("confidence", 0.0))):
        width = c.get("width_m")
        if width is None or not 0.0 < width <= max_width:
            continue
        pose = c["grasp_pose"]
        goal = eh.camera_pose_to_base(capture_tool, ee_to_camera,
                                      (tuple(pose["position"]), eh.normalize(tuple(pose["orientation_xyzw"]))),
                                      ee_to_grasp)
        orders = sorted([("direct", goal), ("flipped", eh.jaw_flip(goal))],
                        key=lambda o: eh.quat_angle(o[1][1], tool[1]))
        for jaw, g in orders:
            legs = [("standoff", eh.along_approach(g, -standoff)), ("grasp", g)]
            if do_retreat:
                legs.append(("retreat", eh.along_approach(g, -retreat)))
            cid = f"{c['index']}:{jaw}"
            wire.append({"id": cid, "confidence": float(c.get("confidence", 0.0)), "width_m": float(width),
                         "instance": c.get("instance"), "transparent": c.get("transparent"),
                         "legs": [{"name": n, "position": list(q[0]), "orientation_xyzw": list(q[1])}
                                  for n, q in legs]})
            lookup[cid] = (g, float(width))
    return wire, lookup


def obstacles_to_base(obstacles, capture_tool, ee_to_camera, camera_frame, base_frame):
    """Scene voxels (camera frame) -> base frame, from the same capture pose as the grasps."""
    if not obstacles:
        return None
    if obstacles.get("frame_id") != camera_frame:
        raise ValueError(f"obstacles are in {obstacles.get('frame_id')!r}")
    origin, q = eh.compose(capture_tool, ee_to_camera)
    pts = np.asarray(obstacles.get("points") or [], float).reshape(-1, 3)
    return {"frame_id": base_frame, "voxel_m": float(obstacles["voxel_m"]),
            "depth_source": obstacles.get("depth_source"),
            # the per-voxel target marks are unchanged; only the points change frame
            **({} if obstacles.get("target") is None else {"target": list(obstacles["target"])}),
            **({} if obstacles.get("instance") is None else {"instance": list(obstacles["instance"])}),
            "points": np.round(pts @ np.asarray(eh.matrix_from_quat(q)).T + np.asarray(origin), 4).tolist()}


class SequenceError(Exception):
    pass


class Sequencer(Node):
    """The /live_pipeline/execute_pick service, holding the latest grasp, candidates and tool pose stream."""

    def __init__(self):
        super().__init__("grasp_sequencer")
        default_cfg = str(Path(get_package_share_directory("remake_grasp")) / "config" / "eye_in_hand.yaml")
        p = declare(self, config=default_cfg, base_frame="fr3_link0", camera_frame=CAMERA_FRAME, dry_run=False,
                    runs_dir=str(RUNS_DIR), live_dir=str(LIVE_DIR))
        self.base_frame, self.camera_frame, self.dry_run = p["base_frame"], p["camera_frame"], p["dry_run"]
        self.runs_dir = Path(p["runs_dir"])
        self.live_dir = Path(p["live_dir"])
        self.record = None            # this pick's record, written into its run's archive
        cfg = yaml.safe_load(Path(p["config"]).read_text()) or {}
        self.ee_to_camera = eh.seven(cfg["ee_to_camera"], "ee_to_camera")
        self.ee_to_grasp = eh.seven(cfg.get("ee_to_grasp", [0, 0, 0, 0, 0, 0, 1]), "ee_to_grasp")
        s = cfg.get("sequence", {})
        self.standoff = float(s.get("standoff", 0.10))
        self.retreat = float(s.get("retreat", 0.10))
        self.do_retreat = bool(s.get("do_retreat", True))
        self.open_width = float(s.get("open_width", 0.078))
        self.max_width = float(s.get("max_width", 0.080))
        self.drift_position = float(s.get("drift_position", 0.005))
        self.drift_angle = float(s.get("drift_angle", 0.02))
        self.cheaper_jaw = bool(s.get("cheaper_jaw", True))
        self.settle = float(s.get("settle", 1.0))
        self.pose_timeout = float(s.get("pose_timeout", 120.0))
        self.tool_timeout = float(s.get("tool_timeout", 2.0))
        self.leg_timeout = float(s.get("leg_timeout", 90.0))
        self.guarded_grasp = bool(s.get("guarded_grasp", True))
        self.pick_from = str(s.get("pick_from", "capture")).lower()
        self.return_home = bool(s.get("return_home", False))
        self.plan_pick = str(s.get("plan_pick", "auto")).lower()

        self.lock = threading.Lock()
        self.grasp = self.grasp_stamp = self.capture_tool = self.width = self.candidates = None
        self.grasp_at = 0.0
        self.tool, self.tool_at, self.tool_seen = None, 0.0, 0
        self.tool_stamp = self.tool_stamp_prev = None
        self.contact, self.contact_at = None, 0.0
        self.busy = False

        g = self.group = ReentrantCallbackGroup()
        # To the NUC (through ros1_bridge, which subscribes with volatile QoS).
        self.pose_pub = self.create_publisher(PoseStamped, "/live_pipeline/grasp_pose_best", 10)
        self.width_pub = self.create_publisher(Float64, "/live_pipeline/grasp_width_best", 10)
        self.cand_pub = self.create_publisher(String, "/live_pipeline/pick_candidates", 10)
        # From perception (latched).
        self.create_subscription(PoseStamped, "/live_pipeline/grasp_pose_camera", self._on_grasp,
                                 latched_qos(), callback_group=g)
        self.create_subscription(Float64, "/live_pipeline/grasp_width_camera", self._on_width,
                                 latched_qos(), callback_group=g)
        self.create_subscription(String, "/live_pipeline/grasp_candidates", self._on_candidates,
                                 latched_qos(), callback_group=g)
        # From the NUC (bridge publishes volatile).
        self.create_subscription(PoseStamped, "/live_pipeline/tool_pose", self._on_tool, 10, callback_group=g)
        self.create_subscription(String, "/live_pipeline/execution_status",
                                 lambda m: self.get_logger().info(f"arm: {m.data}"), 10, callback_group=g)
        self.create_subscription(PoseStamped, "/live_pipeline/contact_pose", self._on_contact, 10, callback_group=g)
        self.move = self.create_client(Trigger, "/live_pipeline/move_to_pose", callback_group=g)
        self.move_guarded = self.create_client(Trigger, "/live_pipeline/guarded_move_to_pose", callback_group=g)
        self.planner = self.create_client(Trigger, "/live_pipeline/plan_pick", callback_group=g)
        self.home = self.create_client(Trigger, "/live_pipeline/go_home", callback_group=g)
        self.load = self.create_client(Trigger, "/live_pipeline/measure_load", callback_group=g)
        self.create_service(Trigger, "/live_pipeline/execute_pick", self._handle_pick, callback_group=g)
        self.get_logger().info(f"execute_pick ready (standoff {self.standoff * 1e3:.0f}mm, pick from "
                               f"{self.pick_from}{', DRY RUN' if self.dry_run else ''})")

    # ------------------------------------------------------------ inputs

    def _tool_problem(self):
        """Why the tool pose cannot be trusted, or None. The grasp is placed from the tool pose, so a dead or frozen
        stream would place it wrong; judged by arrivals and stamp advance, not by comparing clocks across machines."""
        if self.tool_seen == 0:
            return "no tool pose on /live_pipeline/tool_pose -- is the bridge up and the NUC publishing?"
        idle = time.monotonic() - self.tool_at
        if self.tool_seen < 2:
            return f"only one tool pose has ever arrived ({idle:.1f}s ago); not a live stream"
        if idle > self.tool_timeout:
            return f"the tool pose stopped arriving {idle:.1f}s ago"
        if self.tool_stamp_prev is not None and self.tool_stamp == self.tool_stamp_prev:
            return "the tool pose stamp is not advancing"
        return None

    def _on_grasp(self, msg):
        """Hold the new best grasp with the tool pose at its arrival (the capture pose), unless that is not live."""
        with self.lock:
            if msg.header.frame_id != self.camera_frame:
                self.grasp = self.capture_tool = None     # nothing executable this frame
                return
            stale = self._tool_problem()
            self.grasp, self.grasp_at = pose_of(msg), time.monotonic()
            self.grasp_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.capture_tool = None if stale else self.tool
        if stale:
            self.get_logger().warning(f"grasp received but the tool pose is not live: {stale}")

    def _on_width(self, msg):
        with self.lock:
            self.width = float(msg.data)

    def _on_candidates(self, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            payload = None
        with self.lock:
            self.candidates = payload if isinstance(payload, dict) else None

    def _on_tool(self, msg):
        with self.lock:
            self.tool, self.tool_at = pose_of(msg), time.monotonic()
            self.tool_seen += 1
            self.tool_stamp_prev = self.tool_stamp
            self.tool_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _on_contact(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        self.contact, self.contact_at = ((p.x, p.y, p.z), (o.x, o.y, o.z, o.w)), time.monotonic()
        self.get_logger().info(f"contact at [{p.x:.3f} {p.y:.3f} {p.z:.3f}]")

    # ------------------------------------------------------------ plan

    def _plan(self):
        """Check the held grasp and the arm, choose the goal, and return (legs, goal, tool) with each leg
        (name, pose, width or None, guarded). Refuses a grasp already used, one from a run that failed, a tool pose
        that is not live, a grasp older than pose_timeout, a bad width, or an arm that moved since capture (the grasp
        is placed from the capture pose)."""
        with self.lock:
            grasp, capture_tool, width, tool = self.grasp, self.capture_tool, self.width, self.tool
            stale = self._tool_problem()
            stamp = self.grasp_stamp
            self.grasp = None            # one grasp, one pick: the next pick needs a new run
        if grasp is None:
            raise SequenceError("no executable grasp on /live_pipeline/grasp_pose_camera -- run the chain "
                                "(each grasp is used by one pick)")
        if not (self.live_dir / "grasp_poses.json").exists():
            # every run clears the live folder first, so a run that failed leaves none
            raise SequenceError("the last run did not finish (no grasp_poses.json in the live folder), so this "
                                "grasp is from an earlier run -- run the chain again")
        if stale:
            raise SequenceError(stale)
        if capture_tool is None:
            raise SequenceError("the tool pose was not live when the grasp arrived; re-run the chain")
        age = self.get_clock().now().nanoseconds * 1e-9 - stamp
        if age > self.pose_timeout:
            raise SequenceError(f"the grasp was captured {age:.0f}s ago (limit {self.pose_timeout:.0f}s)")
        if width is None or not 0.0 < width <= self.max_width:
            raise SequenceError(f"width {width} is outside (0, {self.max_width}] m")
        if not eh.stationary(capture_tool, tool, self.drift_position, self.drift_angle):
            raise SequenceError(f"the arm moved {eh.displacement(capture_tool, tool) * 1e3:.0f} mm since "
                                f"capture; re-run the chain from here")

        goal = eh.camera_pose_to_base(capture_tool, self.ee_to_camera, grasp, self.ee_to_grasp)
        if self.pick_from == "home":
            tool = self._go_home("before the pick")
        chosen = self._planned(capture_tool, tool, stamp)
        if chosen is not None:
            goal, width = chosen
        elif self.cheaper_jaw:
            goal = eh.cheaper_jaw(goal, tool[1])
        legs = [("standoff", eh.along_approach(goal, -self.standoff), self.open_width, False),
                ("grasp", goal, width, self.guarded_grasp)]
        if self.do_retreat:
            legs.append(("retreat", eh.along_approach(goal, -self.retreat), None, False))  # fingers untouched
        return legs, goal, tool

    def _planned(self, capture_tool, tool, grasp_stamp):
        """Ask the NUC planner which candidate the arm can follow: (goal, width), or None to use the detector's best.

        Sends every candidate with the scene voxels in the base frame and reads the reply for this plan_id. With
        plan_pick=required a missing planner, a missing scene or no choice refuses the pick, so only a grasp the
        planner checked against the scene moves the arm."""
        if self.plan_pick == "off":
            return None
        if not self.planner.service_is_ready():
            if self.plan_pick == "required":
                raise SequenceError("plan_pick is required but /live_pipeline/plan_pick is not available")
            return None
        with self.lock:
            payload = self.candidates
        if payload is None:
            raise SequenceError("no /live_pipeline/grasp_candidates received")
        try:
            wire, lookup = build_pick_candidates(payload, grasp_stamp, capture_tool, tool, self.ee_to_camera,
                                                 self.ee_to_grasp, self.standoff, self.retreat,
                                                 self.do_retreat, self.max_width, self.camera_frame)
        except (ValueError, KeyError, TypeError, eh.GeometryError) as exc:
            raise SequenceError(f"cannot build the candidate list: {exc}")
        if not wire:
            raise SequenceError("no candidate has a usable width")
        try:
            obstacles = obstacles_to_base(payload.get("obstacles"), capture_tool, self.ee_to_camera,
                                          self.camera_frame, self.base_frame)
        except (ValueError, KeyError, TypeError) as exc:
            self.get_logger().warning(f"the scene could not be moved into {self.base_frame}: {exc}")
            obstacles = None
        if obstacles is None and self.plan_pick == "required":
            raise SequenceError("no scene for the planner's collision check (the candidates carried no obstacles, "
                                "or they could not be converted) -- run the chain again")
        plan_id = uuid.uuid4().hex[:12]
        self.cand_pub.publish(String(data=json.dumps({
            "plan_id": plan_id, "frame_id": self.base_frame, "capture_stamp": grasp_stamp, "candidates": wire,
            **({} if obstacles is None else {"obstacles": obstacles})})))
        if self.record is not None:
            self.record.update(plan_id=plan_id, candidates_sent=wire,
                               scene_voxels_sent=0 if obstacles is None else len(obstacles["points"]),
                               start_tool=[list(tool[0]), list(tool[1])])
        for attempt in range(3):
            time.sleep(self.settle if attempt == 0 else 0.3)
            # A refusal is an answer: in enforce mode plan_pick returns success=false with the verdict as JSON.
            ok, text = self._call_raw(self.planner, "plan_pick")
            try:
                reply = json.loads(text)
            except ValueError:
                if not ok:
                    raise SequenceError(f"plan_pick refused: {text}")
                raise SequenceError("plan_pick replied with something that is not JSON")
            if reply.get("plan_id") in (plan_id, None):
                break
        else:
            raise SequenceError("plan_pick kept evaluating an older candidate list")
        summary = plan_summary(reply)
        funnel = funnel_line(payload.get("funnel"), reply, len(wire))
        self.get_logger().info(f"funnel: {funnel}")
        if self.record is not None:
            self.record.update(plan_reply=reply, plan_summary=summary, funnel=funnel)
        seen = reply.get("table_seen")
        if seen:
            self.get_logger().info(f"planner: the table in this scene reads {seen['offset_mm']:+.0f} mm "
                                   f"from the NUC's table_z, tilted {seen['tilt_deg']:.1f} deg")
        chosen = reply.get("chosen")
        if chosen is None:
            if not ok:
                raise SequenceError(f"no grasp the arm can do: {summary} [{funnel}]")
            if self.plan_pick == "required":
                # a planner in report mode answers without choosing; required means only its choice moves the arm
                raise SequenceError(f"plan_pick is required but the planner chose nothing: {summary} [{funnel}]")
            self.get_logger().warning(f"planner chose nothing ({summary}); running the detector's best grasp")
            return None
        if chosen not in lookup:
            raise SequenceError(f"plan_pick chose {chosen!r}, which was not in the list")
        self.get_logger().info(f"planner chose {chosen} of {len(wire)}: {summary}")
        return lookup[chosen]

    # ------------------------------------------------------------ act

    def _call_raw(self, client, label, timeout=None):
        """(success, message) of a Trigger call; raises only on no reply."""
        if not client.service_is_ready():
            raise SequenceError(f"{label} is not available -- is the bridge up and the NUC commander running?")
        future = client.call_async(Trigger.Request())
        end = time.time() + (timeout or self.leg_timeout)
        while rclpy.ok() and not future.done() and time.time() < end:
            time.sleep(0.02)
        if not future.done():
            raise SequenceError(f"{label} gave no reply in time")
        reply = future.result()
        return reply.success, reply.message

    def _call(self, client, label, timeout=None):
        """The message of a successful Trigger call; raises SequenceError on a refusal or no reply."""
        ok, message = self._call_raw(client, label, timeout)
        if not ok:
            raise SequenceError(f"{label} refused: {message}")
        return message

    def _leg(self, name, pose, width, guarded):
        """Publish one waypoint (base frame) and finger width, then trigger the (guarded) move. MOVES THE ARM
        unless dry_run."""
        self.get_logger().info(f"leg {name}: [{pose[0][0]:.3f} {pose[0][1]:.3f} {pose[0][2]:.3f}] in "
                               f"{self.base_frame}, fingers {'untouched' if width is None else f'{width * 1e3:.0f} mm'}"
                               f"{', guarded' if guarded else ''}")
        msg = PoseStamped()
        msg.header.stamp, msg.header.frame_id = self.get_clock().now().to_msg(), self.base_frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[0]
        o = msg.pose.orientation
        o.x, o.y, o.z, o.w = pose[1]
        self.pose_pub.publish(msg)
        self.width_pub.publish(Float64(data=-1.0 if width is None else float(width)))   # -1 = leave fingers
        time.sleep(self.settle)            # let the waypoint cross the bridge before triggering
        if self.dry_run:
            return f"{name}: dry run"
        if guarded:
            return f"{name}: {self._call(self.move_guarded, 'guarded_move_to_pose')}"
        return f"{name}: {self._call(self.move, 'move_to_pose')}"

    def _measure_load(self):
        """measure_load's reply (dict), or None if the NUC has no such service or the
        reading is not steady twice running. Never stops the pick."""
        if self.dry_run or not self.load.service_is_ready():
            return None
        for _ in range(2):
            try:
                ok, text = self._call_raw(self.load, "measure_load", timeout=5.0)
                reading = json.loads(text) if text.startswith("{") else None
            except (SequenceError, ValueError):
                return None
            if ok and reading:
                return reading
            time.sleep(1.0)                        # not steady yet: an impedance arm still settling
        return None

    def _go_home(self, why):
        """Send the arm home (MOVES THE ARM) and return the first tool pose that arrives after it."""
        if self.dry_run:
            with self.lock:
                return self.tool
        self.get_logger().info(f"going home {why}")
        self._call(self.home, "go_home")
        after = time.monotonic()
        while time.monotonic() < after + max(2.0, 2 * self.tool_timeout):
            with self.lock:
                if self.tool is not None and self.tool_at > after:
                    return self.tool
            time.sleep(0.05)
        raise SequenceError("no tool pose arrived after go_home")

    def _handle_pick(self, request, response):
        """Service: plan, run the legs, judge the load, optionally go home, and record the pick. One at a time."""
        with self.lock:
            if self.busy:
                response.success, response.message = False, "a pick is already running"
                return response
            self.busy = True
        self._start_record()
        try:
            legs, goal, _ = self._plan()
            done, before, after, gripper = [], None, None, ""
            for leg in legs:
                done.append(self._leg(*leg))
                if leg[0] == "standoff":
                    before = self._measure_load()          # fingers open, nothing held yet
                elif leg[0] == "grasp":
                    gripper = done[-1]
                elif leg[0] == "retreat":
                    after = self._measure_load()           # the same pose, carrying or not
            verdict, line = load_verdict(before, after, gripper)
            if self.record is not None:
                self.record["load"] = {"verdict": verdict, "line": line, "before": before, "after": after}
            self.get_logger().info(f"load: {line}")
            done.append(f"load: {line}")
            if self.return_home:
                self._go_home("after the pick")
                done.append("home")
            response.success, response.message = True, "; ".join(done)
        except (SequenceError, eh.GeometryError) as exc:
            response.success, response.message = False, str(exc)
            self.get_logger().warning(f"pick stopped: {exc}")
        finally:
            with self.lock:
                self.busy = False
            self._finish_record(response)
        return response

    # ------------------------------------------------------------ record

    def _start_record(self):
        """Begin this pick's record: what was captured, from where."""
        with self.lock:
            stamp, capture_tool = self.grasp_stamp, self.capture_tool
        self.record = {"schema": 1, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "capture_stamp": stamp, "pick_from": self.pick_from,
                       "capture_tool": None if capture_tool is None else
                       [list(capture_tool[0]), list(capture_tool[1])]}

    def _finish_record(self, response):
        """Write pick_<time>.json into the run the grasps came from, plus capture_tool.json so
        nuc/tools/replay_plan.py can replay the run from its capture pose. Never fails the pick."""
        rec, self.record = self.record, None
        if rec is None:
            return
        try:
            rec.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"), success=bool(response.success),
                       message=response.message)
            run = run_dir_for(self.runs_dir, rec.get("capture_stamp"))
            if run is None:
                self.get_logger().warning("pick record not written: no archived run matches this grasp")
                return
            (run / f"pick_{time.strftime('%H%M%S')}.json").write_text(json.dumps(rec, indent=1))
            ct = run / "capture_tool.json"
            stale = False
            if ct.exists():
                try:
                    stale = not json.loads(ct.read_text()).get("live", True)
                except ValueError:
                    stale = True
            # the orchestrator writes it at capture; replace it only if it says the pose was not live
            if rec.get("capture_tool") and (not ct.exists() or stale):
                pos, quat = rec["capture_tool"]
                ct.write_text(json.dumps({"live": True, "frame_id": self.base_frame, "position": pos,
                                          "orientation_xyzw": quat,
                                          "source": "sequencer: tool pose when the grasp arrived"}))
            self.get_logger().info(f"pick recorded in {run.name}")
            spawn_export(run, final=True)          # grasp_final.pcd, in the background
        except Exception as exc:  # noqa: BLE001 -- a record must never cost a pick
            self.get_logger().warning(f"pick record not written: {exc}")


def main():
    run_node(Sequencer, threads=4)


if __name__ == "__main__":
    main()
