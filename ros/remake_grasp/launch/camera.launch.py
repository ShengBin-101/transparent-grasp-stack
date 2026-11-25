"""Camera driver: the RealSense L515, publishing /camera/color/image_raw, /camera/depth/image_raw
and their camera_info.

    ros2 launch remake_grasp camera.launch.py
    ros2 launch remake_grasp camera.launch.py initial_reset:=true   # L515 stuck at 0 Hz depth
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _camera(context):
    """L515: realsense2_camera with config/l515.yaml, depth remapped to depth/image_raw."""
    camera = LaunchConfiguration("camera").perform(context).lower()
    if camera == "l515":
        params = os.path.join(get_package_share_directory("remake_grasp"), "config", "l515.yaml")
        reset = LaunchConfiguration("initial_reset").perform(context).lower() == "true"
        return [Node(package="realsense2_camera", executable="realsense2_camera_node",
                     name="camera", namespace="camera", output="screen", emulate_tty=True,
                     parameters=[params, {"initial_reset": reset}],
                     remappings=[("depth/image_rect_raw", "depth/image_raw")],
                     additional_env={"LRS_LOG_LEVEL": "error"}, cwd="/tmp")]  # SDK drops empty .log files in cwd
    raise RuntimeError(f"camera:={camera!r}: only the L515 (l515) is supported")


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("camera", default_value=os.environ.get("CAMERA", "l515")),
        DeclareLaunchArgument("initial_reset", default_value="false"),
        OpaqueFunction(function=_camera),
    ])
