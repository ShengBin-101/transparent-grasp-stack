"""ROS 1 bridge to the NUC plus the pick sequencer.

    ros2 launch remake_grasp bridge.launch.py
    ros2 launch remake_grasp bridge.launch.py dry_run:=true      # resolve legs, never move
Needs ROS_MASTER_URI (default http://172.16.0.1:11311) and ROS_IP (this machine) set.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("dry_run", default_value="false"),
        Node(package="remake_grasp", executable="ros1_bridge_node", output="screen",
             emulate_tty=True, respawn=True, respawn_delay=2.0),
        Node(package="remake_grasp", executable="sequencer", output="screen", emulate_tty=True,
             parameters=[{"dry_run": LaunchConfiguration("dry_run")}]),
    ])
