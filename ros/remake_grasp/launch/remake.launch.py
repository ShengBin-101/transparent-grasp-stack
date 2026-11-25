"""ReMake depth completion, plus the masks it needs.

    ros2 launch remake_grasp remake.launch.py pipeline:=adapted      # segmenter mask, fine-tuned weights
    ros2 launch remake_grasp remake.launch.py pipeline:=baseline     # SAM 3 mask, the authors' weights
    ros2 launch remake_grasp remake.launch.py pipeline:=raw          # no mask, no ReMake: starts nothing
    ros2 launch remake_grasp remake.launch.py pipeline:=adapted remake:=official   # a preset, one setting changed
    ros2 launch remake_grasp remake.launch.py sam3:=false            # SAM 3 running elsewhere

pipeline:= (default $PIPELINE, else adapted) sets mask:=, remake:= and targets:= from PIPELINES, one
preset per trial condition (ADAPTED, BASELINE, RAW);
each one given (or $MASK / $REMAKE / $TARGETS) overrides its preset value. pipeline.launch.py
takes the same arguments, and they must agree (same defaults).
  mask:=     the transparency mask ReMake completes inside: sam3 (SAM 3 "transparent object"
             prompt -> sam3_masks.npy), seg (the prompt-free segmenter, seg_node -> seg_masks.npy)
             or none (no mask at all: only with remake:=none and targets:=depth or none)
  remake:=   official      the authors' release (TransCG, as the paper trains; the release does not say)
             L_combo_both  fine-tuned on TransCG with in-mask blanking, mask dropout and
                           rel_rgb_fix; the only arm that passed every pre-registered TransCG row
             G_maskdrop    fine-tuned with mask dropout only (benchmark rows)
             A_ft          the control fine-tune: same schedule, nothing changed (paired replay)
             none          no ReMake: SBG plans on the raw sensor depth; the mask
                           node then starts only if the targets need it
  targets:=  depth (the trials' rule: the segmenter's transparent objects plus the objects standing in
             the raw depth; seg_node starts whatever mask:= is), prompt (SAM 3's grasp prompt, PEGG-Net's
             input; SAM 3 starts whatever mask:= is), transparent (the transparency mask's objects only,
             the rule before the trials) or none (no target filter: every grasp counts, as in a plain
             RGB-D pipeline; pipeline:=raw)
SAM 3 starts only when used (mask:=sam3 or targets:=prompt); without it, ~5.7 GB less memory.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CKPT_DIR = os.path.join(os.environ.get("RG_ROOT", "/opt/remake_grasp"), "checkpoints")
MASKS = {"sam3": "sam3_masks.npy", "seg": "seg_masks.npy", "none": None}
# preset -> (checkpoint file in checkpoints/, rel_rgb_fix: fed DepthAnything RGB in training)
REMAKE = {"official": ("remake_official.tar", False),
          "L_combo_both": ("L_combo_both_checkpoint43.tar", True),
          "G_maskdrop": ("G_maskdrop_checkpoint45.tar", False),
          # the control fine-tune: the released checkpoint, six epochs on TransCG (39 -> 45) with nothing changed;
          # L_combo_both's same-data comparison, used offline (trials/paired_replay.py *-AFT), not in the trials
          "A_ft": ("A_ft_checkpoint45.tar", False),
          "none": (None, False)}                 # the floor: no completion

# pipeline:= presets, one per trial condition; pipeline.launch.py holds the same table
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
    """Validate the arguments; start remake_node (unless remake:=none) plus seg_node and/or
    sam3_node, whichever the mask and the targets need."""
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    _, chosen = _settings(arg)
    mask, remake, targets = chosen["mask"], chosen["remake"], chosen["targets"]
    if mask not in MASKS:
        raise RuntimeError(f"mask:={mask!r}; use {' or '.join(MASKS)}")
    if targets not in ("depth", "prompt", "transparent", "none"):
        raise RuntimeError(f"targets:={targets!r}; use depth, prompt, transparent or none")
    if remake not in REMAKE:
        raise RuntimeError(f"remake:={remake!r}; use {' or '.join(REMAKE)}")
    # the transparency mask feeds ReMake and, with targets:=transparent / prompt, the targets
    needs_mask = remake != "none" or targets in ("prompt", "transparent")
    if mask == "none" and needs_mask:
        raise RuntimeError(f"mask:=none, but remake:={remake} / targets:={targets} need a transparency mask")
    ckpt, rel_rgb_fix = REMAKE[remake]
    nodes = []
    if ckpt is not None:
        ckpt = os.path.join(CKPT_DIR, ckpt)
        if not os.path.exists(ckpt):
            raise RuntimeError(f"remake:={remake} needs {ckpt} (see checkpoints/README.md)")
        nodes.append(Node(package="remake_grasp", executable="remake_node", output="screen", emulate_tty=True,
                          parameters=[{"mask_name": MASKS[mask], "mask_dilate": int(arg("mask_dilate")),
                                       "checkpoint": ckpt, "rel_rgb_fix": rel_rgb_fix,
                                       "fill_holes": arg("fill_holes").lower() == "true"}]))
    if (needs_mask and mask == "seg") or targets == "depth":
        nodes.append(Node(package="remake_grasp", executable="seg_node", output="screen", emulate_tty=True,
                          parameters=[{"mask_name": MASKS["seg"]}]))
    if arg("sam3").lower() == "true" and ((needs_mask and mask == "sam3") or targets == "prompt"):
        nodes.append(Node(package="remake_grasp", executable="sam3_node", output="screen", emulate_tty=True,
                          # float(): sam3_node declares threshold a double and dies on a string
                          parameters=[{"prompt": arg("prompt"), "threshold": float(arg("threshold"))}]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("pipeline", default_value=os.environ.get("PIPELINE", "adapted"),
                              description="adapted | baseline | raw: sets mask, remake and targets together"),
        DeclareLaunchArgument("mask", default_value=os.environ.get("MASK", ""),
                              description="sam3 | seg | none (empty: the preset's): which transparency mask ReMake completes inside"),
        DeclareLaunchArgument("remake", default_value=os.environ.get("REMAKE", ""),
                              description="official | L_combo_both | G_maskdrop | A_ft | none (empty: the preset's): ReMake weights, or none = raw depth"),
        DeclareLaunchArgument("targets", default_value=os.environ.get("TARGETS", ""),
                              description="depth | prompt | transparent | none (empty: the preset's): which of seg_node / SAM 3 they need"),
        DeclareLaunchArgument("sam3", default_value="true"),
        DeclareLaunchArgument("prompt", default_value="transparent object",
                              description="SAM 3 prompt for the transparent-object mask (mask:=sam3)"),
        DeclareLaunchArgument("threshold", default_value="0.4"),
        DeclareLaunchArgument("mask_dilate", default_value="5"),
        # false: ReMake's depth inside the mask only; the sensor's holes elsewhere stay empty, as in RAW. At the
        # side view (V2) the L515 leaves 8 % of the frame unread past the table's edge, and filling it made a false
        # surface up to about 215 mm over the table. true = every hole filled, as in all the top-down (V1) attempts,
        # where it changes about 0.2 % of the pixels per frame; replayed offline as trials/paired_replay.py *-FULL-FILL.
        DeclareLaunchArgument("fill_holes", default_value="false"),
        OpaqueFunction(function=_nodes),
    ])
