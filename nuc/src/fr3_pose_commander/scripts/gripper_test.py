#!/usr/bin/env python3
"""Gripper-only test: does a successful grasp stay closed? Moves the fingers, never the arm.

    roslaunch franka_gripper franka_gripper.launch robot_ip:=172.16.0.2    # if not already up
    rosrun fr3_pose_commander gripper_test.py                     # in the air
    rosrun fr3_pose_commander gripper_test.py --width 0.057       # an object 57 mm wide

Opens to --open, sends one grasp with pose_commander's parameters (width, speed, force,
epsilon), logs the finger width for --hold seconds and prints the timeline. Fingers that stay
at the object's width mean the gripper holds; fingers that reopen mean the hand itself released.
Asks before every finger move. Put an object between the fingers (skin clear) for the real
test; nothing between them tests the in-air case.
"""
import argparse
import sys
import time

import actionlib
import rospy
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal
from sensor_msgs.msg import JointState

LOG = []   # (t, width_m)


def on_joints(msg):
    """Log (time, gripper width in m); the width is the sum of both finger positions."""
    try:
        LOG.append((time.time(), float(msg.position[0]) + float(msg.position[1])))
    except (IndexError, TypeError, ValueError):
        pass


def width_now():
    return LOG[-1][1] if LOG else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=float, default=0.04, help="grasp width (m)")
    ap.add_argument("--force", type=float, default=None, help="N; default pose_commander's")
    ap.add_argument("--speed", type=float, default=None, help="m/s; default pose_commander's")
    ap.add_argument("--eps-inner", type=float, default=None)
    ap.add_argument("--eps-outer", type=float, default=None)
    ap.add_argument("--open", type=float, default=0.08, help="open width before/after (m)")
    ap.add_argument("--hold", type=float, default=6.0, help="s to keep logging after the result")
    ap.add_argument("--ns", default="/franka_gripper")
    a = ap.parse_args(rospy.myargv()[1:])
    rospy.init_node("gripper_test")
    p = lambda k, d: rospy.get_param("/pose_commander/gripper/" + k, d)   # noqa: E731
    force = a.force if a.force is not None else float(p("force", 20.0))
    speed = a.speed if a.speed is not None else float(p("speed", 0.05))
    # Fallbacks match config/pose_commander.yaml, for when its parameters are not on the server.
    eps_in = a.eps_inner if a.eps_inner is not None else float(p("epsilon_inner", 0.055))
    eps_out = a.eps_outer if a.eps_outer is not None else float(p("epsilon_outer", 0.02))

    rospy.Subscriber(a.ns + "/joint_states", JointState, on_joints, queue_size=50)
    grasp = actionlib.SimpleActionClient(a.ns + "/grasp", GraspAction)
    move = actionlib.SimpleActionClient(a.ns + "/move", MoveAction)
    if not (grasp.wait_for_server(rospy.Duration(5)) and move.wait_for_server(rospy.Duration(5))):
        sys.exit("no %s grasp/move actions -- start franka_gripper (see the usage line)" % a.ns)
    t_end = time.time() + 3.0
    while not LOG and time.time() < t_end:
        time.sleep(0.05)
    if not LOG:
        sys.exit("no %s/joint_states" % a.ns)

    print("gripper at %.1f mm; grasp %.1f mm, window %.1f-%.1f mm, force %.0f N, speed %.0f mm/s"
          % (width_now() * 1e3, a.width * 1e3, (a.width - eps_in) * 1e3, (a.width + eps_out) * 1e3,
             force, speed * 1e3))
    if width_now() < a.open - 0.004:
        input(">>> open the fingers to %.0f mm first? Enter to go, Ctrl-C to abort: " % (a.open * 1e3))
        move.send_goal(MoveGoal(width=a.open, speed=speed))
        move.wait_for_result(rospy.Duration(10))
    input(">>> GRASP now (object between the fingers, hands clear). Enter to go, Ctrl-C to abort: ")

    t0 = time.time()
    goal = GraspGoal(width=a.width, speed=speed, force=force)
    goal.epsilon.inner, goal.epsilon.outer = eps_in, eps_out
    grasp.send_goal(goal)
    grasp.wait_for_result(rospy.Duration(15))
    t_res = time.time()
    res = grasp.get_result()
    ok = getattr(res, "success", None)
    err = getattr(res, "error", "")
    time.sleep(a.hold)

    # Timeline: every change of more than 0.5 mm, relative to the grasp command.
    print("\n  t (s)   width (mm)")
    last = None
    stop_w = None
    for t, w in LOG:
        if t < t0 - 0.5:
            continue
        if last is None or abs(w - last) > 0.0005:
            mark = ""
            if last is not None and t_res - 0.05 <= t <= t_res + 0.5:
                mark = "   <- around the result"
            print("  %+6.2f  %6.1f%s" % (t - t0, w * 1e3, mark))
            last = w
        if t <= t_res:
            stop_w = w
    final = width_now()
    # Judge by the narrowest width, not the width at the result: the hand can reopen inside the
    # grasp action and still return success.
    after = [(t, w) for t, w in LOG if t >= t0]
    t_min, w_min = min(after, key=lambda x: x[1]) if after else (None, None)
    print("\nresult at %+.2f s: success=%s%s" % (t_res - t0, ok, (" error=%r" % err) if err else ""))
    print("narrowest: %.1f mm at %+.2f s; at the result: %s mm; %.1f s later: %.1f mm"
          % (w_min * 1e3, t_min - t0, "%.1f" % (stop_w * 1e3) if stop_w else "?", a.hold, final * 1e3))
    if w_min is not None and final - w_min > 0.005:
        when = "BEFORE the result (inside the grasp action)" if stop_w - w_min > 0.005 else "after the result"
        print("=> the HAND RELEASED: closed to %.1f mm, then reopened to %.1f mm %s"
              % (w_min * 1e3, final * 1e3, when))
    else:
        print("=> the gripper HOLDS at %.1f mm" % (final * 1e3))
    try:
        input("\n>>> open the fingers to %.0f mm? Enter to open, Ctrl-C to leave them: " % (a.open * 1e3))
        move.send_goal(MoveGoal(width=a.open, speed=speed))
        move.wait_for_result(rospy.Duration(10))
    except KeyboardInterrupt:
        print("\nleft as they are")


if __name__ == "__main__":
    main()
