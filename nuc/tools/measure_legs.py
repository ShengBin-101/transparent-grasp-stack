#!/usr/bin/env python3
"""Small measurement legs from the current pose, each followed by go_home.

    tools/measure_legs.py [path to fr3_pose_commander/src]    MOVES THE ARM

Needs a controller and a live commander (run_controller.sh or run_position_controller.sh, then
run_commander.sh [--position]). Ten legs, +/-3 cm along x, y, z and +/-15, +/-30 deg of tool
roll: each target goes on /live_pipeline/grasp_pose_best (base frame, width -1 leaves the
fingers alone) and runs with move_to_pose, then go_home. Stops at the first refusal. Checks the
planner's posture models: read the leg log with tools/compare_legs.py. Hand on the stop.
"""
import math
import os
import sys
import time

import numpy as np
import rospy
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from std_srvs.srv import Trigger

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "fr3_pose_commander", "src"))
from fr3_pose_commander.pose_math import matrix_from_o_t_ee, quaternion_from_matrix  # noqa: E402

LEGS = [("x+3cm", (0.03, 0, 0), 0), ("x-3cm", (-0.03, 0, 0), 0),
        ("y+3cm", (0, 0.03, 0), 0), ("y-3cm", (0, -0.03, 0), 0),
        ("z+3cm", (0, 0, 0.03), 0), ("z-3cm", (0, 0, -0.03), 0),
        ("roll+15", (0, 0, 0), 15), ("roll-15", (0, 0, 0), -15),
        ("roll+30", (0, 0, 0), 30), ("roll-30", (0, 0, 0), -30)]

rospy.init_node("measure_legs")
start = matrix_from_o_t_ee(rospy.wait_for_message(
    "/franka_state_controller/franka_states", FrankaState, 10).O_T_EE)
pose_pub = rospy.Publisher("/live_pipeline/grasp_pose_best", PoseStamped, queue_size=1, latch=True)
width_pub = rospy.Publisher("/live_pipeline/grasp_width_best", Float64, queue_size=1, latch=True)
rospy.wait_for_service("/live_pipeline/move_to_pose", 10)
move = rospy.ServiceProxy("/live_pipeline/move_to_pose", Trigger)
home = rospy.ServiceProxy("/live_pipeline/go_home", Trigger)
width_pub.publish(Float64(data=-1.0))


def target(offset, roll_deg):
    """The start pose shifted by offset (m, base frame) and rolled roll_deg about tool z."""
    m = start.copy()
    c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    m[:3, :3] = m[:3, :3].dot(np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))))  # about tool z
    m[:3, 3] += offset
    msg = PoseStamped()
    msg.header.frame_id, msg.header.stamp = "fr3_link0", rospy.Time.now()
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = m[:3, 3]
    (msg.pose.orientation.x, msg.pose.orientation.y,
     msg.pose.orientation.z, msg.pose.orientation.w) = quaternion_from_matrix(m)
    return msg


print("start tcp [%.3f %.3f %.3f]" % tuple(start[:3, 3]), flush=True)
for name, offset, roll in LEGS:
    pose_pub.publish(target(offset, roll))
    rospy.sleep(1.0)
    t = time.time()
    r = move()
    print("%-8s move  %s %.1fs  %s" % (name, "ok " if r.success else "REFUSED", time.time() - t,
                                       r.message[:110]), flush=True)
    if not r.success:
        break
    t = time.time()
    r = home()
    print("%-8s home  %s %.1fs  %s" % (name, "ok " if r.success else "REFUSED", time.time() - t,
                                       r.message[:110]), flush=True)
    if not r.success:
        break
