#!/usr/bin/env python3
"""The trials' views as services: go_v1 is go_home, go_v2 drives to ~views/v2 in the legs
run_viewpoint.sh would plan from the same joints, every leg checked, the gripper untouched. Needs rospy
and the workspace's messages importable (run it on the NUC, devel sourced); no robot, master or
controller -- the ramp is a fake that moves a simulated arm.

    python3 src/fr3_pose_commander/test/test_view_services.py
"""
import importlib.util
import math
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
WS_SRC = os.path.dirname(PKG)
sys.path.insert(0, os.path.join(PKG, "src"))
spec = importlib.util.spec_from_file_location("pose_commander_node", os.path.join(PKG, "scripts", "pose_commander_node.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from fr3_pose_commander import kinematics as k  # noqa: E402
from fr3_pose_commander import planning as P  # noqa: E402
from fr3_pose_commander import viewpoints as V  # noqa: E402
from fr3_pose_commander.pose_math import quaternion_from_matrix  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

FAILED = []


def check(label, condition, detail=""):
    print("%-4s %s%s" % ("ok" if condition else "FAIL", label, "  (%s)" % (detail,) if detail else ""))
    if not condition:
        FAILED.append(label)


LIMITS = os.path.join(WS_SRC, "franka_ros", "franka_description", "robots", "fr3", "joint_limits.yaml")
HOME_FILE = os.path.join(PKG, "config", "home_pose.yaml")
HOME = [float(v) for v in yaml.safe_load(open(HOME_FILE))["home_joints"]]
CONFIG = yaml.safe_load(open(os.path.join(PKG, "config", "pose_commander.yaml")))
EE_CAM = yaml.safe_load(open(os.path.join(PKG, "config", "camera_extrinsics.yaml")))["ee_to_camera"]
IMPEDANCE = "/cartesian_impedance_example_controller/equilibrium_pose"
POSITION = "/cartesian_pose_tracking_controller/target_pose"
TABLE_Z = float(CONFIG["planner"]["table_z"])


class _Pub(object):
    def publish(self, *_):
        pass


def commander(topic=IMPEDANCE, views="config", planner_mode="enforce"):
    """A PoseCommander without ROS, at home, with the config file's settings and a fake ramp."""
    c = mod.PoseCommander.__new__(mod.PoseCommander)
    c._lock, c._busy = threading.Lock(), False
    c.base_frame, c.enforce_checks, c.command_topic = "fr3_link0", True, topic
    c.linear_speed = float(CONFIG["ramp"]["linear_speed"])
    c.angular_speed = float(CONFIG["ramp"]["angular_speed"])
    c.rotation_max = float(CONFIG["limits"]["rotation_max"])
    params = {"~planner/mode": planner_mode, "~planner/joint_limits": LIMITS, "~planner/home_pose_file": HOME_FILE,
              "~planner/table_z": TABLE_Z, "~planner/joint_margin": float(CONFIG["planner"]["joint_margin"]),
              "~ee_to_camera": EE_CAM}
    c._init_planner(lambda name, default=None: params.get(name, default), topic)
    c._home = c._home_source = None
    c._init_home(lambda name, default=None: {"~home/source": "home_pose"}.get(name, default))
    view_params = {"~views": CONFIG["views"]} if views == "config" else ({"~views": views} if views is not None else {})
    c._init_views(lambda name, default=None: view_params.get(name, default))
    c.dry_run, c.status_pub = False, _Pub()
    c.z_min = float(CONFIG["limits"]["z_min"])
    c.radius_max, c.step_max = float(CONFIG["limits"]["radius_max"]), float(CONFIG["limits"]["step_max"])
    c.fail_distance, c._mode, c.default_width = 0.030, 2, 0.078
    c._posture = lambda: ""
    m = k.ee_matrix(HOME)
    c._q, c._q0, c._state = list(HOME), list(HOME), (m[:3, 3].copy(), quaternion_from_matrix(m), None)
    c._current = lambda: (c._state[0], c._state[1])
    c._require_listener = c._require_commanding = lambda: None
    c._candidates = c._approved = c._plan_check = None
    c._elbow, c.leg_log = None, mod.LegLog("/nonexistent", enabled=False)   # on_view/on_home log their leg
    c.ramps, c.widths = [], []
    c._gripper = lambda width: (c.widths.append(width), "moved")[1]

    def ramp(p0, q0, p1, q1, guard=False):
        c.ramps.append((np.asarray(p1), tuple(q1)))
        with c._lock:
            c._state = (np.asarray(p1), tuple(q1), None)
        return "arrived"
    c._ramp = ramp
    return c


def goto_viewpoint_plan(n, tilt, height, q, pose_now, model):
    """What tools/goto_viewpoint.py plans for `run_viewpoint.sh n --tilt tilt --height height`."""
    cfg = P.PlanConfig(ee_to_camera=tuple(EE_CAM[:3]), rotation_max=float(CONFIG["limits"]["rotation_max"]),
                       joint_margin=float(CONFIG["planner"]["joint_margin"]), table_z=TABLE_Z,
                       linear_speed=float(CONFIG["ramp"]["linear_speed"]),
                       angular_speed=float(CONFIG["ramp"]["angular_speed"]))
    q_ns = HOME if model == "nullspace" else None
    return V.plan_viewpoint(n, q, pose_now, P.load_limits(LIMITS), cfg, EE_CAM, tuple(V.DEFAULT_CENTRE),
                            TABLE_Z, height, tilt, max_leg_deg=P.leg_split(cfg)[0], model=model, q_ns=q_ns)


def camera_at(pose):
    m = P.matrix_from_pose((tuple(pose[0]), tuple(pose[1]))).dot(
        P.matrix_from_pose((tuple(EE_CAM[:3]), tuple(EE_CAM[3:]))))
    return m[:3, 3], m[:3, 2]


# -- the views offered -------------------------------------------------------------------
c = commander()
check("config: go_v2 = viewpoint 4 (left), tilt 45, camera 0.35 m up, centre (0.50, 0.00) -- "
      "run_viewpoint.sh 4 --tilt 45 --height 0.35",
      c.views == {"v2": {"viewpoint": 4, "tilt_deg": 45.0, "height": 0.35, "centre": (0.5, 0.0)}}, c.views)
check("no ~views parameter: the same v2", commander(views=None).views == c.views)
bad = commander(views={"v1": {"viewpoint": 1}, "v9": {"tilt_deg": 10}, "v8": {"viewpoint": 7}, "v3": {"viewpoint": 4}})
check("v1 (it is home), a view with no viewpoint and one out of 1-5 are ignored; the rest kept",
      sorted(bad.views) == ["v3"], sorted(bad.views))

# -- go_v2 from home, impedance ----------------------------------------------------------------
for topic, model in ((IMPEDANCE, "nullspace"), (POSITION, "elbow_locked")):
    c = commander(topic=topic)
    start = (c._state[0].copy(), c._state[1])
    want = goto_viewpoint_plan(4, 45.0, 0.35, list(HOME), start, model)
    r = c.on_view("v2")
    check("%s: go_v2 from home succeeds" % model, r.success, r.message[:140])
    check("... in exactly the legs run_viewpoint.sh 4 --tilt 45 --height 0.35 plans from the same joints",
          want["ok"] and len(c.ramps) == len(want["legs"]) and all(
              np.allclose(p, w[1][0], atol=1e-9) and np.allclose(q, w[1][1], atol=1e-9)
              for (p, q), w in zip(c.ramps, want["legs"])), (len(c.ramps), len(want.get("legs", []))))
    cam, axis = camera_at(c.ramps[-1])
    tilt = math.degrees(math.acos(-axis[2]))
    hit = cam + axis * (TABLE_Z - cam[2]) / axis[2]
    check("... the camera ends 0.35 m above the table, tilted 45 deg, looking at the centre",
          abs(cam[2] - TABLE_Z - 0.35) < 1e-6 and abs(tilt - 45.0) < 0.01 and np.allclose(hit[:2], (0.50, 0.0), atol=1e-6),
          "height %.3f, tilt %.2f, hits %s" % (cam[2] - TABLE_Z, tilt, np.round(hit[:2], 4).tolist()))
    check("... on the +y side (viewpoint 4, left)", cam[1] > 0.3, np.round(cam, 3).tolist())
    check("... the gripper untouched", c.widths == [])
    check("... the planner's verdict on the whole chain is recorded, and it passed",
          c._plan_check is not None and c._plan_check.get("ok") is True, c._plan_check)
    n = len(c.ramps)
    c._q = list(want["q_end"])          # the fake ramp moves the tool only; the joints the planner predicted
    r = c.on_home(None)
    check("... and go_home (= go_v1) brings it back to home_pose.yaml's pose", r.success
          and np.allclose(c._state[0], c._home[0], atol=1e-9) and len(c.ramps) > n, r.message[:120])

# -- refusals: nothing moves -------------------------------------------------------------------
c = commander()
c._busy = True
r = c.on_view("v2")
check("while another leg runs: refused, nothing moved", not r.success and r.message == "already moving" and not c.ramps)

c = commander(planner_mode="off")
r = c.on_view("v2")
check("planner off: refused (a view is never driven to unplanned), nothing moved",
      not r.success and "planner" in r.message and not c.ramps, r.message[:120])

c = commander()
c.ee_to_camera = None
r = c.on_view("v2")
check("no ee_to_camera: refused, nothing moved", not r.success and "ee_to_camera" in r.message and not c.ramps, r.message[:120])

c = commander(views={"far30": {"viewpoint": 3, "tilt_deg": 30.0, "height": 0.50, "centre": [0.55, 0.0]}})
r = c.on_view("far30")
check("a view out of reach (far side, 30 deg, centre x 0.55, as measured): refused before moving",
      not r.success and "out of reach" in r.message and not c.ramps, r.message[:140])

c = commander()
c.z_min = 10.0                      # every leg below the floor: _check refuses the chain
r = c.on_view("v2")
check("a leg failing the commander's own limits: the chain is refused before the first leg",
      not r.success and "floor" in r.message and not c.ramps, r.message[:120])

c = commander()
real = c._ramp


def stall(p0, q0, p1, q1, guard=False):
    real(p0, q0, p1, q1, guard)
    with c._lock:
        c._state = (np.asarray(p1) + (0.0, 0.0, 0.05), tuple(q1), None)   # ends 50 mm off its goal
    return "stopped closing"


c._ramp = stall
r = c.on_view("v2")
check("a leg that ends 50 mm off its goal fails the call, and no later leg runs",
      not r.success and "stopped at" in r.message and len(c.ramps) == 1, r.message[:140])

if FAILED:
    print("\n%d FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("\nall view service checks passed")
