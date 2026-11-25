"""Scale-Balanced Grasp detector: 6-DoF grasps from the completed point cloud.

    ros2 launch remake_grasp sbg.launch.py                      # profile:=picking (default)
    ros2 launch remake_grasp sbg.launch.py profile:=perception

profile:= (default $PROFILE, else picking):
  picking     30 deg approach cone about "down" (from the arm's tool pose) plus the empty-grasp
              check; on the rig 30 deg gave the highest pickable share and feasible rim-wall grasps.
              The trials' setting (blocks 1-3): keep it the default
  picking20   the same with a 20 deg cone, and picking15 with 15 deg (offline replay of 31 frames):
              steeper grasps the planner accepts more often; feasible objects at the side view (V2)
              1 -> 3 of 6, every table-clearing frame with a feasible grasp, top-down unchanged.
              PROFILE=picking20 rg-adapted
  perception  SBG as upstream (no cone, no empty check), for comparing the detector across viewpoints
remake:= (default $REMAKE, else the pipeline:= preset's) picks the depth when depth_name:= is empty:
none -> depth_aligned_mm.npy (raw sensor depth), else remake_depth_mm.npy
(ReMake's output). pipeline:= (default $PIPELINE, else adapted) matters only through that.
grasps_sbg.json records both choices (approach_cone, depth_source).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# pipeline:= presets whose remake is none (remake.launch.py's PIPELINES; tests/test_remake_variants.py checks)
RAW_PIPELINES = ("raw",)
PROFILES = {
    "picking": {"approach_cone_deg": 30.0, "empty_check": True},       # the trials
    "picking20": {"approach_cone_deg": 20.0, "empty_check": True},
    "picking15": {"approach_cone_deg": 15.0, "empty_check": True},
    "perception": {"approach_cone_deg": 0.0, "empty_check": False},
}


def _nodes(context):
    """sbg_node with the profile's cone / empty-check settings and the chosen depth file."""
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    profile = arg("profile")
    if profile not in PROFILES:
        raise RuntimeError(f"profile:={profile!r}; use {' or '.join(PROFILES)}")
    remake = arg("remake") or ("none" if arg("pipeline") in RAW_PIPELINES else "")
    depth_name = arg("depth_name") or ("depth_aligned_mm.npy" if remake == "none" else "remake_depth_mm.npy")
    return [Node(package="remake_grasp", executable="sbg_node", output="screen", emulate_tty=True,
                 parameters=[{"depth_name": depth_name, "top_k": int(arg("top_k")),
                              "num_points": 20000, "max_aperture_mm": 70.0, **PROFILES[profile]}])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("depth_name", default_value="",
                              description="the depth file SBG reads (empty: from remake:=)"),
        DeclareLaunchArgument("remake", default_value=os.environ.get("REMAKE", ""),
                              description="none = SBG on the raw sensor depth (match remake.launch.py; empty: the preset's)"),
        DeclareLaunchArgument("pipeline", default_value=os.environ.get("PIPELINE", "adapted"),
                              description="the preset (match remake.launch.py): raw reads the raw depth"),
        DeclareLaunchArgument("top_k", default_value="50"),
        DeclareLaunchArgument("profile", default_value=os.environ.get("PROFILE", "picking"),
                              description=" | ".join(PROFILES)),
        OpaqueFunction(function=_nodes),
    ])
