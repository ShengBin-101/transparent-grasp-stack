"""Capture + chain runner (orchestrator), grasp-pose node and point clouds.

    ros2 launch remake_grasp pipeline.launch.py detector:=sbg   # or peggnet
    ros2 service call /live_pipeline/run_chain std_srvs/srv/Trigger

detector:= starts no detector: it picks the detector service the chain calls and the output the
grasp-pose node reads (start the detector with sbg.launch.py or peggnet.launch.py).
pipeline:= (default $PIPELINE, else adapted) sets mask:=, remake:= and targets:= from PIPELINES, one per
trial condition (adapted, baseline, raw);
each one given (or $MASK / $REMAKE / $TARGETS) overrides its preset value. mask:= and targets:=
must match remake.launch.py's (same defaults); a mismatch fails loudly (ReMake refuses a missing mask).
  mask:=     sam3: segment_sam3? -> sam3_masks.npy;  seg: segment_trans (seg_node) -> seg_masks.npy;
             none: no mask stage (only with remake:=none and targets:=depth or none)
  remake:=   only recorded here (run_config.json), except none: the raw-depth floor, with no
             depth_remake stage and the mask stage only if the targets need it
  targets:=  depth        the segmenter's transparent objects + opaque blobs in the raw sensor depth; the same for
                          every mask / remake condition, since neither reads ReMake's output (the trials' rule)
             prompt       SAM 3's grasp prompt (data/live/sam3_grasp_prompt.txt) + the transparency
                          mask: PEGG-Net's input, and only with detector:=peggnet
             transparent  the transparency mask's objects only (the rule before the trials)
             none         no target filter: every grasp counts and the planner gets no object labels,
                          as in a plain RGB-D pipeline (pipeline:=raw)
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Detector stages; they follow the mask stage, the targets' stage and depth_remake.
CHAINS = {
    "peggnet": "grasps_peggnet,grasp_poses,pointcloud?",
    "sbg": "grasps_sbg,grasp_poses,pointcloud?",
}
REMAKE_NAMES = ("official", "L_combo_both", "G_maskdrop", "A_ft", "none")   # remake.launch.py's REMAKE keys
# targets:= -> the stage that finds them, skipped when the mask stage is the same one.
# segment_sam3_all (~2-4 s) finds the grasp prompt's objects; segment_trans (~0.3 s) the transparent ones.
TARGET_STAGES = {"depth": "segment_trans", "prompt": "segment_sam3_all?", "transparent": None, "none": None}
# mask:= -> (chain stage, the file it writes). segment_sam3 is optional ("?") because SAM 3 reports
# an empty scene as a failure; the segmenter returns an empty mask, so its failures are real.
MASK_STAGES = {"sam3": ("segment_sam3?", "sam3_masks.npy"), "seg": ("segment_trans", "seg_masks.npy"),
               "none": (None, "")}

# pipeline:= presets, one per trial condition; remake.launch.py holds the same table
# (tests/test_remake_variants.py checks).
PIPELINES = {
    # the segmenter's mask, the fine-tuned weights, prompt-free targets (transparent + raw-depth objects)
    "adapted": {"mask": "seg", "remake": "L_combo_both", "targets": "depth"},
    # ReMake as released: SAM 3's transparency mask and the authors' weights, with the same targets as ADAPTED,
    # so only the depth differs between the two (recorded as "baseline (targets=depth)" in the trials)
    "baseline": {"mask": "sam3", "remake": "official", "targets": "depth"},
    # a plain RGB-D pipeline: the raw sensor depth into SBG, no mask, no target filter
    "raw": {"mask": "none", "remake": "none", "targets": "none"},
}


def _settings(arg):
    """(pipeline, {mask, remake, targets}): the preset, with any mask:= / remake:= / targets:= given."""
    pipeline = arg("pipeline")
    if pipeline not in PIPELINES:
        raise RuntimeError(f"pipeline:={pipeline!r}; use {' or '.join(PIPELINES)}")
    return pipeline, {k: arg(k) or v for k, v in PIPELINES[pipeline].items()}


def _nodes(context):
    """Validate the arguments, build the chain (mask, targets, depth_remake, detector, grasp poses,
    point cloud) and start the orchestrator, grasp-pose node and point-cloud node."""
    detector = LaunchConfiguration("detector").perform(context)
    if detector not in CHAINS:
        raise RuntimeError(f"detector:={detector!r}; use peggnet or sbg")
    pipeline, chosen = _settings(lambda name: LaunchConfiguration(name).perform(context))
    mask, targets = chosen["mask"], chosen["targets"]
    if mask not in MASK_STAGES:
        raise RuntimeError(f"mask:={mask!r}; use {' or '.join(MASK_STAGES)}")
    stage, mask_file = MASK_STAGES[mask]
    if targets not in TARGET_STAGES:
        raise RuntimeError(f"targets:={targets!r}; use {' or '.join(TARGET_STAGES)}")
    if detector == "peggnet" and targets != "prompt":
        raise RuntimeError("detector:=peggnet takes SAM 3's grasp-prompt masks as its input: use targets:=prompt")
    if targets == "prompt" and detector != "peggnet":
        raise RuntimeError("targets:=prompt is PEGG-Net's input; SBG uses depth, transparent or none")
    remake = chosen["remake"]
    if remake not in REMAKE_NAMES:
        raise RuntimeError(f"remake:={remake!r}; use {' or '.join(REMAKE_NAMES)}")
    if detector == "peggnet" and remake == "none":
        raise RuntimeError("remake:=none (raw depth) is wired for SBG only")
    floor = remake == "none"
    # the transparency mask: ReMake's input, or (prompt / transparent) part of the targets
    needs_mask = not floor or targets in ("prompt", "transparent")
    if mask == "none" and needs_mask:
        raise RuntimeError(f"mask:=none, but remake:={remake} / targets:={targets} need a transparency mask")
    head = [stage] if needs_mask else []
    stages = head + [t for t in (TARGET_STAGES[targets],) if t and t not in head] \
        + ([] if floor else ["depth_remake"]) + [CHAINS[detector]]
    changed = [f"{k}={v}" for k, v in chosen.items() if v != PIPELINES[pipeline][k]]
    label = pipeline + (f" ({', '.join(changed)})" if changed else "")
    archive = LaunchConfiguration("archive").perform(context)
    return [
        Node(package="remake_grasp", executable="orchestrator", output="screen", emulate_tty=True,
             parameters=[{"chain": ",".join(stages), "archive": archive, "pipeline": label}]),
        Node(package="remake_grasp", executable="grasp_pose_node", output="screen", emulate_tty=True,
             parameters=[{"source": detector, "base_frame": "camera_color_optical_frame",
                          "targets": targets, "target_mask": MASK_STAGES["seg"][1],
                          "transparent_mask": mask_file, "depth_completed": not floor}]),
        # cloud_completed colours the mask ReMake completed magenta, so it must read ReMake's mask file
        Node(package="remake_grasp", executable="pointcloud_node", output="screen", emulate_tty=True,
             parameters=[{"depth_name": "depth_aligned_mm.npy" if floor else "remake_depth_mm.npy",
                          "colour": "source", "mask_name": "" if floor else mask_file}]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("detector", default_value=os.environ.get("DETECTOR", "sbg")),
        DeclareLaunchArgument("pipeline", default_value=os.environ.get("PIPELINE", "adapted"),
                              description="adapted | baseline | raw: sets mask and targets (match remake.launch.py)"),
        DeclareLaunchArgument("mask", default_value=os.environ.get("MASK", ""),
                              description="sam3 | seg | none (empty: the preset's): the transparency mask (match remake.launch.py)"),
        DeclareLaunchArgument("remake", default_value=os.environ.get("REMAKE", ""),
                              description="(empty: the preset's) none = the raw-depth floor; else only recorded"),
        DeclareLaunchArgument("targets", default_value=os.environ.get("TARGETS", ""),
                              description="depth | prompt | transparent | none (empty: the preset's): where the grasp targets come from"),
        DeclareLaunchArgument("archive", default_value="always", description="off | success | always"),
        OpaqueFunction(function=_nodes),
    ])
