#!/usr/bin/env python3
"""Watch the arm track its equilibrium, in a terminal.

    rosrun fr3_pose_commander watch_tracking.py

One line per sample: the gains, the error from the last commanded setpoint, the force that
error asks for (error x gain), and the posture's conditioning. With
`run_stiffness.sh rotational <n>` in a second shell it is the stall diagnostic without a GUI:
an error that shrinks as the gain rises was gain or friction; one that does not move means no
authority in that direction (re-home). A steady error alone is not a fault.

Start this before the move: pose_commander publishes equilibrium_pose only while ramping, unlatched.
"""

import math
import sys

import numpy as np
import rospy
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import PoseStamped

from fr3_pose_commander import kinematics
from fr3_pose_commander.pose_math import (matrix_from_o_t_ee, normalize_quaternion,
                                          quaternion_from_matrix)

COMPLIANCE = ("/cartesian_impedance_example_controller"
              "/dynamic_reconfigure_compliance_param_node")


class Watcher(object):
    """Pairs the last equilibrium_pose with the measured O_T_EE and formats a status line."""

    def __init__(self):
        self.target = None      # (position, quaternion) last commanded
        self.state = None       # (position, quaternion, q) measured
        self.stamp = None
        rospy.Subscriber(rospy.get_param("~command_topic",
                         "/cartesian_impedance_example_controller/equilibrium_pose"),
                         PoseStamped, self.on_target, queue_size=1)
        rospy.Subscriber(rospy.get_param("~state_topic",
                         "/franka_state_controller/franka_states"),
                         FrankaState, self.on_state, queue_size=1)

    def on_target(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        try:
            self.target = (np.array([p.x, p.y, p.z]),
                           np.array(normalize_quaternion((o.x, o.y, o.z, o.w))))
        except ValueError:
            self.target = None

    def on_state(self, msg):
        try:
            m = matrix_from_o_t_ee(msg.O_T_EE)
        except (ValueError, ZeroDivisionError):
            return
        self.state = (m[:3, 3].copy(), np.array(quaternion_from_matrix(m)),
                      list(msg.q))
        self.stamp = msg.header.stamp

    def gains(self):
        """Translational, rotational and nullspace stiffness as strings ("--" when unset)."""
        names = ("translational_stiffness", "rotational_stiffness",
                 "nullspace_stiffness")
        out = []
        for n in names:
            v = rospy.get_param("%s/%s" % (COMPLIANCE, n), None)
            out.append("--" if v is None else ("%g" % v))
        return out

    def line(self):
        if self.state is None:
            return "waiting for franka_states"
        position, quaternion, q = self.state
        age = (rospy.Time.now() - self.stamp).to_sec() if self.stamp else float("nan")
        trans, rot, null = self.gains()
        posture = kinematics.report(q) or "posture unknown"

        if self.target is None:
            # Not an error: nothing has commanded a pose since this started.
            return ("K %s/%s null %s | no setpoint seen yet (start this before "
                    "the move) | %s | state %.2fs old" % (trans, rot, null, posture, age))

        dp = float(np.linalg.norm(self.target[0] - position))
        dot = abs(float(np.dot(self.target[1], quaternion)))
        dq = 2.0 * math.acos(max(-1.0, min(1.0, dot)))
        # Commanded force = error x gain; with no controller loaded there is none ("--").
        if trans == "--" or rot == "--":
            asks = "asks    --      --   "
        else:
            asks = ("asks %5.1f N %5.2f N.m"
                    % (dp * float(trans), math.sin(dq / 2.0) * float(rot)))
        return ("K %s/%s null %s | err %6.1f mm %5.1f deg | %s | %s | state %.2fs old"
                % (trans, rot, null, dp * 1e3, math.degrees(dq), asks, posture, age))


def main():
    rospy.init_node("watch_tracking")
    rate_hz = float(rospy.get_param("~rate", 2.0))
    watcher = Watcher()
    rospy.loginfo("watch_tracking: gains | tracking error | commanded force "
                  "| conditioning.  Ctrl-C to stop.")
    rate = rospy.Rate(rate_hz)
    while not rospy.is_shutdown():
        sys.stdout.write(watcher.line() + "\n")
        sys.stdout.flush()
        rate.sleep()


if __name__ == "__main__":
    main()
