#!/usr/bin/env python3
"""Publish franka_states for an arm that is not there, so pose_commander runs with no robot.

A kinematic echo, not a simulator: it looks up <arm_id>_link0 -> <arm_id>_EE on tf (the arm
drawn by mock_joint_states.py) and reports it as O_T_EE, with q from /joint_states. Nothing
responds to commands. O_T_EE is written column-major, as pose_math reads it; row-major would
give a plausible but transposed rotation (test_mock_states.py pins the round trip). Only
header, O_T_EE, q and elbow are filled; every other field is zero and means nothing.

    <node name="mock_franka_states" pkg="fr3_pose_commander"
          type="mock_franka_states.py" output="screen">
      <param name="arm_id" value="fr3" />
    </node>
"""

import numpy as np
import rospy
import tf2_ros
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import JointState


def matrix_from_transform(transform):
    """4x4 homogeneous transform from a geometry_msgs/TransformStamped."""
    t = transform.transform.translation
    q = transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm < 1e-9:
        raise ValueError("zero quaternion on tf")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    m = np.eye(4)
    m[0, 0] = 1.0 - 2.0 * (y * y + z * z)
    m[0, 1] = 2.0 * (x * y - z * w)
    m[0, 2] = 2.0 * (x * z + y * w)
    m[1, 0] = 2.0 * (x * y + z * w)
    m[1, 1] = 1.0 - 2.0 * (x * x + z * z)
    m[1, 2] = 2.0 * (y * z - x * w)
    m[2, 0] = 2.0 * (x * z - y * w)
    m[2, 1] = 2.0 * (y * z + x * w)
    m[2, 2] = 1.0 - 2.0 * (x * x + y * y)
    m[0, 3], m[1, 3], m[2, 3] = t.x, t.y, t.z
    return m


def main():
    rospy.init_node("mock_franka_states")

    arm_id = rospy.get_param("~arm_id", "fr3")
    base_frame = rospy.get_param("~base_frame", "%s_link0" % arm_id)
    ee_frame = rospy.get_param("~ee_frame", "%s_EE" % arm_id)
    rate_hz = float(rospy.get_param("~rate", 30.0))
    topic = rospy.get_param("~topic", "/franka_state_controller/franka_states")

    # q comes from the same /joint_states the drawn arm is built from.
    joints = {"q": None}

    def on_joints(msg):
        want = ["%s_joint%d" % (arm_id, i + 1) for i in range(7)]
        try:
            joints["q"] = [msg.position[msg.name.index(n)] for n in want]
        except (ValueError, IndexError):
            joints["q"] = None

    rospy.Subscriber("/joint_states", JointState, on_joints, queue_size=1)

    buffer_ = tf2_ros.Buffer()
    tf2_ros.TransformListener(buffer_)
    publisher = rospy.Publisher(topic, FrankaState, queue_size=1)

    rospy.logwarn("mock_franka_states: THIS IS NOT THE ROBOT. Reporting "
                  "%s -> %s from tf as O_T_EE on %s so the commander can run "
                  "a whole command with no arm present.",
                  base_frame, ee_frame, topic)

    rate = rospy.Rate(rate_hz)
    warned = False
    while not rospy.is_shutdown():
        try:
            transform = buffer_.lookup_transform(base_frame, ee_frame,
                                                 rospy.Time(0),
                                                 rospy.Duration(0.1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            # Normal while the tf tree is being built; warn once.
            if not warned:
                warned = True
                rospy.logwarn("mock_franka_states: waiting for %s -> %s: %s",
                              base_frame, ee_frame, exc)
            rate.sleep()
            continue

        if warned:
            warned = False
            rospy.loginfo("mock_franka_states: %s -> %s resolved; publishing",
                          base_frame, ee_frame)

        try:
            m = matrix_from_transform(transform)
        except ValueError as exc:
            rospy.logwarn_throttle(5.0, "mock_franka_states: %s", exc)
            rate.sleep()
            continue

        message = FrankaState()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = base_frame
        # Column-major: pose_math.matrix_from_o_t_ee reads it back with .reshape(4, 4).T.
        message.O_T_EE = list(m.flatten(order="F"))
        if joints["q"] is not None:
            message.q = list(joints["q"])
            # libfranka's elbow: joint 3's angle and the joint-4 flip sign. The sign (-1) is
            # assumed, not measured; the leg log records the real robot's value.
            elbow = [float(joints["q"][2]), -1.0]
            message.elbow = list(elbow)
            message.elbow_d = list(elbow)
            message.elbow_c = list(elbow)
        publisher.publish(message)
        rate.sleep()


if __name__ == "__main__":
    main()
