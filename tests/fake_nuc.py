#!/usr/bin/env python3
"""A stand-in for the NUC's ROS 1 side, for testing the bridge without the robot.

Publishes /live_pipeline/tool_pose at 30 Hz, serves the Trigger services the sequencer
calls, and logs every waypoint it receives. Run it next to a roscore:
    roscore &  python3 tests/fake_nuc.py
"""
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger, TriggerResponse


def main():
    rospy.init_node("fake_nuc")
    tool = rospy.Publisher("/live_pipeline/tool_pose", PoseStamped, queue_size=1)
    status = rospy.Publisher("/live_pipeline/execution_status", String, queue_size=10)
    last = {"pose": None, "width": None}
    rospy.Subscriber("/live_pipeline/grasp_pose_best", PoseStamped,
                     lambda m: last.update(pose=m) or rospy.loginfo(
                         "waypoint [%.3f %.3f %.3f] in %s", m.pose.position.x, m.pose.position.y,
                         m.pose.position.z, m.header.frame_id))
    rospy.Subscriber("/live_pipeline/grasp_width_best", Float64, lambda m: last.update(width=m.data))

    def service(name):
        def handle(_req):
            p = last["pose"]
            msg = f"{name}: ok" + ("" if p is None else f" to [{p.pose.position.x:.3f} "
                                                        f"{p.pose.position.y:.3f} {p.pose.position.z:.3f}]")
            rospy.loginfo(msg)
            status.publish(String(data=msg))
            return TriggerResponse(success=True, message=msg)
        rospy.Service(f"/live_pipeline/{name}", Trigger, handle)

    for name in ("move_to_pose", "guarded_move_to_pose", "release", "go_home"):
        service(name)
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        m = PoseStamped()
        m.header.stamp, m.header.frame_id = rospy.Time.now(), "fr3_link0"
        m.pose.position.x, m.pose.position.z, m.pose.orientation.x = 0.45, 0.40, 1.0
        tool.publish(m)
        rate.sleep()


if __name__ == "__main__":
    main()
