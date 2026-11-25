#!/usr/bin/env python3
"""Move the arm to capture viewpoint 1-5 through the commander.   MOVES THE ARM

    ./run_viewpoint.sh 3            plan, show it, ask, then move
    ./run_viewpoint.sh 3 --dry-run  plan and show only
    ./run_viewpoint.sh 4 --tilt 15 --height 0.55 --centre 0.50 0.05

    1 above   2 near (base side)   3 far   4 left (+y)   5 right (-y)

Viewpoint 1 with the default --centre/--height is home: it calls /live_pipeline/go_home
(home_pose.yaml's joints). Given either knob, it is planned like the others and is not home.
The camera ends --height above the table (default 0.50 m) with its optical axis on the
workspace centre (default x 0.50, y 0.00 in fr3_link0); side views tilt --tilt deg (default 20).

Needs a controller and its commander (run_position_controller.sh + run_commander.sh --position,
or run_controller.sh + run_commander.sh). Each leg goes through /live_pipeline/move_to_pose, so
the commander's limits, re-check and leg log apply; this refuses before moving if the planner
says the viewpoint is unreachable. The planner model follows the controller: position control
holds joint 3; impedance pulls toward the posture it started in (taken as home_pose.yaml's,
so start run_controller.sh at home) and both its predictions must pass. Hand on the stop.
"""
import argparse
import os
import sys
import time

import numpy as np
import rospy
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from std_srvs.srv import Trigger

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "fr3_pose_commander", "src"))
from fr3_pose_commander import planning, viewpoints  # noqa: E402
from fr3_pose_commander.pose_math import matrix_from_o_t_ee, quaternion_from_matrix  # noqa: E402

COMMANDER = "/pose_commander"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("viewpoint", type=int, choices=range(1, 6), metavar="N",
                    help="1 above, 2 near, 3 far, 4 left, 5 right")
    ap.add_argument("--centre", type=float, nargs=2, default=viewpoints.DEFAULT_CENTRE,
                    metavar=("X", "Y"), help="workspace centre on the table, fr3_link0")
    ap.add_argument("--height", type=float, default=viewpoints.DEFAULT_HEIGHT,
                    help="camera height above the table, m")
    ap.add_argument("--tilt", type=float, default=viewpoints.DEFAULT_TILT_DEG,
                    help="side-view tilt toward the centre, deg")
    ap.add_argument("--table-z", type=float, default=None,
                    help="table surface z in fr3_link0 (default: the commander's planner/table_z)")
    ap.add_argument("--dry-run", action="store_true", help="plan and print; move nothing")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args(rospy.myargv()[1:])

    rospy.init_node("goto_viewpoint", anonymous=True)
    p = lambda name, default=None: rospy.get_param(COMMANDER + "/" + name, default)

    topic = p("command_topic", "")
    if not topic:
        sys.exit("ERROR: no %s/command_topic parameter -- is the commander running?" % COMMANDER)
    model = planning.model_for_topic(topic)
    ee_to_camera = p("ee_to_camera")
    if not ee_to_camera or len(ee_to_camera) != 7:
        sys.exit("ERROR: no %s/ee_to_camera parameter -- is the commander running?" % COMMANDER)
    table_z = args.table_z if args.table_z is not None else float(p("planner/table_z", planning.TABLE_Z))
    is_home = (args.viewpoint == 1 and tuple(args.centre) == tuple(viewpoints.DEFAULT_CENTRE)
               and args.height == viewpoints.DEFAULT_HEIGHT)

    limits = planning.load_limits(p("planner/joint_limits", "") or os.path.join(
        HERE, "..", "src", "franka_ros", "franka_description", "robots", "fr3",
        "joint_limits.yaml"))
    cfg = planning.PlanConfig(
        ee_to_camera=tuple(ee_to_camera[:3]),
        rotation_max=float(p("limits/rotation_max", 1.571)),
        joint_margin=float(p("planner/joint_margin", 0.05)),
        table_z=table_z, linear_speed=float(p("ramp/linear_speed", 0.012)),
        angular_speed=float(p("ramp/angular_speed", 0.40)))

    state = rospy.wait_for_message("/franka_state_controller/franka_states", FrankaState, 5.0)
    if getattr(state, "robot_mode", 2) != 2:
        sys.exit("ERROR: the robot is not being commanded (robot_mode %d): recover the error and "
                 "restart the controller" % state.robot_mode)
    m = matrix_from_o_t_ee(state.O_T_EE)
    pose_now = (tuple(m[:3, 3]), quaternion_from_matrix(m))
    q_ns = _home_joints(p) if model == "nullspace" else None
    print("controller model: %s%s" % (model, " (posture pulled toward home_pose.yaml)" if q_ns else ""))
    if is_home:
        home_joints = _home_joints(p)
        home = viewpoints.home_pose(home_joints)
        plan = viewpoints.plan_home(list(state.q), pose_now, home, limits, cfg, model, q_ns)
        cam_home = planning.matrix_from_pose(home).dot(planning.matrix_from_pose(
            (tuple(ee_to_camera[:3]), tuple(ee_to_camera[3:]))))
        target = cam_home[:3, 3] + cam_home[:3, 2] * (cam_home[2, 3] - table_z) / -cam_home[2, 2]
        plan.update(number=1, name="above = home", camera_position=cam_home[:3, 3].tolist(),
                    target=target.tolist(),
                    distance_m=float(np.linalg.norm(target - cam_home[:3, 3])))
    else:
        plan = viewpoints.plan_viewpoint(args.viewpoint, list(state.q), pose_now, limits, cfg,
                                         ee_to_camera, tuple(args.centre), table_z, args.height,
                                         args.tilt, max_leg_deg=planning.leg_split(cfg)[0],
                                         model=model, q_ns=q_ns)
        if args.viewpoint == 1:
            print("NOTE: viewpoint 1 with a custom --centre/--height is not home")

    cam = np.round(plan["camera_position"], 3)
    print("viewpoint %d (%s): camera at %s, looking at %s, %.2f m away"
          % (plan["number"], plan["name"], cam.tolist(), np.round(plan["target"], 3).tolist(),
             plan["distance_m"]))
    if not plan["ok"]:
        sys.exit("REFUSED before moving: %s" % plan["reasons"][0])
    print("  %s; wrist turn %.0f deg in %d leg(s); predicted joints %s"
          % ("home_pose.yaml's pose (go_home's legs)" if is_home
             else "camera roll %.0f deg" % plan["roll_deg"], plan["turn_deg"], len(plan["legs"]),
             np.round(plan["q_end"], 3).tolist()))
    if plan["turn_deg"] > 120:
        print("  NOTE: a long wrist spin. From home, the near view's only roll that keeps "
              "0.30 rad at every joint needs ~180 deg (measured 2026-09-26).")
    print("  worst joint margin %.2f rad, sigma_min %.3f" % (plan["joint_margin"], plan["sigma_min"]))
    if args.dry_run:
        print("DRY RUN: nothing moved")
        return

    if not args.yes:
        reply = input("THIS WILL MOVE THE ARM. Hand on the stop. Type 'yes' to go: ")
        if reply.strip() != "yes":
            sys.exit("aborted")

    if is_home:
        rospy.wait_for_service("/live_pipeline/go_home", 5.0)
        t = time.time()
        r = rospy.ServiceProxy("/live_pipeline/go_home", Trigger)()
        print("  go_home: %s in %.1fs -- %s" % ("ok" if r.success else "REFUSED",
                                               time.time() - t, r.message[:160]))
        if not r.success:
            sys.exit("stopped; the arm is where go_home left it")
        _report(plan, ee_to_camera)
        return

    pose_pub = rospy.Publisher("/live_pipeline/grasp_pose_best", PoseStamped, queue_size=1, latch=True)
    width_pub = rospy.Publisher("/live_pipeline/grasp_width_best", Float64, queue_size=1, latch=True)
    rospy.wait_for_service("/live_pipeline/move_to_pose", 5.0)
    move = rospy.ServiceProxy("/live_pipeline/move_to_pose", Trigger)
    width_pub.publish(Float64(data=-1.0))          # fingers left alone
    for name, (pos, quat) in plan["legs"]:
        msg = PoseStamped()
        msg.header.frame_id, msg.header.stamp = "fr3_link0", rospy.Time.now()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos
        (msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w) = quat
        pose_pub.publish(msg)
        rospy.sleep(1.0)                           # let the latched target land first
        t = time.time()
        r = move()
        print("  %s: %s in %.1fs -- %s" % (name, "ok" if r.success else "REFUSED",
                                           time.time() - t, r.message[:120]))
        if not r.success:
            sys.exit("stopped at %s; the arm is where that leg left it" % name)
    _report(plan, ee_to_camera)


def _home_joints(p):
    """home_pose.yaml's joints, from the file the commander was pointed at."""
    import yaml
    path = p("planner/home_pose_file", "") or os.path.join(
        HERE, "..", "src", "fr3_pose_commander", "config", "home_pose.yaml")
    return [float(v) for v in yaml.safe_load(open(path))["home_joints"]]


def _report(plan, ee_to_camera):
    """Print the reached camera's distance from the plan (mm) and its axis error to the target (deg)."""
    state = rospy.wait_for_message("/franka_state_controller/franka_states", FrankaState, 5.0)
    m = matrix_from_o_t_ee(state.O_T_EE)
    ee_cam = planning.matrix_from_pose((tuple(ee_to_camera[:3]), tuple(ee_to_camera[3:])))
    c = m.dot(ee_cam)
    look = c[:3, 2]
    want = np.asarray(plan["target"]) - c[:3, 3]
    err = np.degrees(np.arccos(np.clip(look.dot(want) / np.linalg.norm(want), -1, 1)))
    print("at viewpoint %d: camera %s (%.0f mm from the plan), optical axis %.1f deg off the centre"
          % (plan["number"], np.round(c[:3, 3], 3).tolist(),
             1000 * np.linalg.norm(c[:3, 3] - np.asarray(plan["camera_position"])), err))


if __name__ == "__main__":
    main()
