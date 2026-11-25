#!/usr/bin/env python3
"""Publish the measured tool pose so the ROS 2 side can resolve camera-frame poses itself.

    /franka_state_controller/franka_states  ->  /live_pipeline/tool_pose

The camera is on the gripper, so a camera-frame grasp only means something with the tool pose
it was seen from. The ROS 2 side resolves a whole trajectory once, from the tool pose at
capture, and sends base-frame waypoints; this supplies that pose over ros1_bridge. It reads
O_T_EE rather than tf because pose_commander's step checks measure from franka_states too.
The stamp is the robot's, so a stale pose does not look fresh (the machines keep separate
clocks). Only a PoseStamped is sent: franka_msgs does not exist on the ROS 2 side.
"""

import rospy
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import PoseStamped

from fr3_pose_commander.pose_math import matrix_from_o_t_ee, quaternion_from_matrix


class ToolPosePublisher(object):
    """Republishes O_T_EE as a base-frame PoseStamped, throttled to ~rate Hz."""

    def __init__(self):
        p = rospy.get_param
        arm_id = p("~arm_id", "fr3")
        self.base_frame = p("~base_frame", "%s_link0" % arm_id)
        self.topic = p("~topic", "/live_pipeline/tool_pose")
        state_topic = p("~state_topic", "/franka_state_controller/franka_states")
        # franka_states arrives at ~30 Hz (franka_control's publish_rate); the consumer samples once per capture over a websocket.
        self.min_period = 1.0 / max(float(p("~rate", 30.0)), 1e-6)

        self.publisher = rospy.Publisher(self.topic, PoseStamped, queue_size=1)
        self.last = rospy.Time(0)
        self.warned = False
        self.count = 0
        rospy.Subscriber(state_topic, FrankaState, self.on_state, queue_size=1)

        rospy.loginfo("tool_pose: %s -> %s in %s, at most %.0f Hz",
                      state_topic, self.topic, self.base_frame,
                      1.0 / self.min_period)

    def on_state(self, msg):
        now = rospy.Time.now()
        if (now - self.last).to_sec() < self.min_period:
            return
        self.last = now

        try:
            m = matrix_from_o_t_ee(msg.O_T_EE)
            quaternion = quaternion_from_matrix(m)
        except (ValueError, ZeroDivisionError) as exc:
            rospy.logwarn_throttle(5.0, "tool_pose: unusable robot state: %s", exc)
            return

        out = PoseStamped()
        # The robot's stamp, not now(), so the consumer can tell a live pose from a stale one.
        out.header.stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else now
        out.header.frame_id = self.base_frame
        out.pose.position.x, out.pose.position.y, out.pose.position.z = m[:3, 3]
        (out.pose.orientation.x, out.pose.orientation.y,
         out.pose.orientation.z, out.pose.orientation.w) = quaternion
        self.publisher.publish(out)

        self.count += 1
        if not self.warned:
            self.warned = True
            rospy.loginfo("tool_pose: first pose published: [%.3f %.3f %.3f]",
                          m[0, 3], m[1, 3], m[2, 3])


def main():
    rospy.init_node("tool_pose_publisher")
    ToolPosePublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
