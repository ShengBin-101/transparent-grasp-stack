#!/usr/bin/env python3
"""Publish joint states for an arm that is not there, to bring the visualization up without it.

robot_state_publisher turns these into the /tf tree Foxglove draws. Not a simulator: the arm
holds ~posture (default franka's "ready" pose) or, with ~animate, sweeps gently around it so a
frozen panel can be told from a still arm. The fingers are published too, because
robot_state_publisher leaves the hand at the origin without them.
"""

import math

import rospy
from sensor_msgs.msg import JointState

ARM_JOINTS = 7

# franka's "ready" posture (move_to_start's target): clear of joint limits and the table.
READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]


def main():
    rospy.init_node("mock_joint_states")

    arm_id = rospy.get_param("~arm_id", "fr3")
    rate_hz = float(rospy.get_param("~rate", 30.0))
    animate = bool(rospy.get_param("~animate", True))
    amplitude = float(rospy.get_param("~amplitude", 0.25))   # rad
    period = float(rospy.get_param("~period", 12.0))         # s
    finger = float(rospy.get_param("~finger_width", 0.04)) / 2.0

    posture = rospy.get_param("~posture", None)
    if posture is None:
        posture = list(READY)
    elif len(posture) != ARM_JOINTS:
        rospy.logfatal("mock_joint_states: ~posture has %d values, expected %d",
                       len(posture), ARM_JOINTS)
        return
    posture = [float(v) for v in posture]

    names = ["%s_joint%d" % (arm_id, i + 1) for i in range(ARM_JOINTS)]
    names += ["%s_finger_joint1" % arm_id, "%s_finger_joint2" % arm_id]

    publisher = rospy.Publisher("/joint_states", JointState, queue_size=1)

    rospy.logwarn("mock_joint_states: THIS IS NOT THE ROBOT. Publishing a "
                  "%s posture on /joint_states so the arm can be drawn "
                  "without hardware.", "sweeping" if animate else "fixed")
    rospy.loginfo("mock_joint_states: %d arm joints + 2 fingers at %.0f Hz",
                  ARM_JOINTS, rate_hz)

    start = rospy.Time.now()
    rate = rospy.Rate(rate_hz)
    while not rospy.is_shutdown():
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = names

        if animate:
            phase = 2.0 * math.pi * (rospy.Time.now() - start).to_sec() / period
            # Only the joints that swing the arm visibly.
            offsets = [amplitude * math.sin(phase),
                       0.5 * amplitude * math.sin(phase),
                       0.0,
                       0.5 * amplitude * math.sin(phase + math.pi / 2),
                       0.0,
                       0.5 * amplitude * math.sin(phase),
                       amplitude * math.sin(phase)]
            positions = [p + o for p, o in zip(posture, offsets)]
        else:
            positions = list(posture)

        message.position = positions + [finger, finger]
        message.velocity = []
        message.effort = []
        publisher.publish(message)
        rate.sleep()


if __name__ == "__main__":
    main()
