"""Foxglove bridge (connect Foxglove to ws://<this machine>:8765) and pins_node, which draws
world-fixed poses on /live_pipeline/pins (config/pins.yaml, plus each capture's grasps in base).

    ros2 launch remake_grasp foxglove.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    xml = os.path.join(get_package_share_directory("foxglove_bridge"), "launch", "foxglove_bridge_launch.xml")
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="8765"),
        IncludeLaunchDescription(AnyLaunchDescriptionSource(xml),
                                 launch_arguments={"port": LaunchConfiguration("port")}.items()),
        Node(package="remake_grasp", executable="pins_node", output="screen", emulate_tty=True),
    ])
