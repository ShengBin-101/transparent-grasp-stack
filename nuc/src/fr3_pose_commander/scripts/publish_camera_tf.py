#!/usr/bin/env python3
"""Publish the wrist camera's mounting as a static transform, ~ee_frame -> ~camera_frame.

The camera rides on the gripper, so this fixed offset on top of the arm's EE frame is what
resolves camera-frame poses into the base frame (and puts the camera on Foxglove's 3D panel).
~ee_to_camera ([x, y, z, qx, qy, qz, qw]) comes only from config/camera_extrinsics.yaml, so
there is one copy of the calibration.
"""

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped


def main():
    rospy.init_node("camera_tf_publisher")

    ee_frame = rospy.get_param("~ee_frame", "fr3_EE")
    camera_frame = rospy.get_param("~camera_frame", "camera_color_optical_frame")
    values = rospy.get_param("~ee_to_camera", None)

    if values is None:
        rospy.logfatal("camera_tf: no ~ee_to_camera parameter. Load "
                       "config/camera_extrinsics.yaml into this node.")
        return
    if len(values) != 7:
        rospy.logfatal("camera_tf: ee_to_camera has %d numbers, expected 7 "
                       "([x, y, z, qx, qy, qz, qw])", len(values))
        return

    x, y, z, qx, qy, qz, qw = (float(v) for v in values)
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if abs(norm - 1.0) > 1e-3:
        # A non-unit quaternion is a typo; tf would publish it and distort every pose.
        rospy.logfatal("camera_tf: ee_to_camera quaternion has norm %.6f, not "
                       "1.0 -- refusing to publish a bad calibration", norm)
        return

    transform = TransformStamped()
    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = ee_frame
    transform.child_frame_id = camera_frame
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.translation.z = z
    transform.transform.rotation.x = qx
    transform.transform.rotation.y = qy
    transform.transform.rotation.z = qz
    transform.transform.rotation.w = qw

    broadcaster = tf2_ros.StaticTransformBroadcaster()
    broadcaster.sendTransform(transform)

    rospy.loginfo("camera_tf: %s -> %s at [%.3f %.3f %.3f] q=[%.4f %.4f %.4f %.4f]",
                  ee_frame, camera_frame, x, y, z, qx, qy, qz, qw)
    rospy.spin()


if __name__ == "__main__":
    main()
