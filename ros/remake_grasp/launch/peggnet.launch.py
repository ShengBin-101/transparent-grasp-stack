"""PEGG-Net grasp detector (one top-down grasp per SAM 3 object mask).

    ros2 launch remake_grasp peggnet.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arg = LaunchConfiguration
    return LaunchDescription([
        DeclareLaunchArgument("depth_name", default_value="remake_depth_mm.npy"),
        # Accepted so tmuxp can pass profile:= to either detector; only SBG uses it.
        DeclareLaunchArgument("profile", default_value="picking", description="unused by PEGG-Net"),
        DeclareLaunchArgument("pipeline", default_value="adapted", description="unused by PEGG-Net"),
        DeclareLaunchArgument("quality_gate", default_value="0.5"),
        DeclareLaunchArgument("aperture_source", default_value="feature"),
        Node(package="remake_grasp", executable="peggnet_node", output="screen", emulate_tty=True,
             parameters=[{"depth_name": arg("depth_name"), "mask_name": "sam3_masks_all.npy",
                          "selection": "per_object", "quality_gate": arg("quality_gate"),
                          "aperture_offset_mm": -16.0, "max_aperture_mm": 70.0,
                          "aperture_source": arg("aperture_source")}]),
    ])
