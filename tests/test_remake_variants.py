"""Tests the pipelines (PIPELINE=adapted|baseline|raw, the trials' three conditions) and the settings they preset:
MASK=sam3|seg|none x REMAKE=official|L_combo_both|G_maskdrop|A_ft|none x TARGETS=depth|prompt|transparent|none,
rel_rgb_fix, and the segmenter service. Loads the segmenter and ReMake once each (~2 GB GPU).

    docker exec remake_grasp python3 /opt/remake_grasp/tests/test_remake_variants.py
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("RG_RUNS", ROOT / "runs"))
sys.path.insert(0, str(ROOT / "ros/remake_grasp"))
fail = []


def ck(name, ok, detail=""):
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  -- %s" % (detail,)))
    if not ok:
        fail.append(name)


def load_launch(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "ros/remake_grasp/launch" / f"{name}.launch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def params(node, context):
    """The parameters launch_ros would hand the node, as plain {name: value}."""
    from launch_ros.utilities import evaluate_parameters
    out = {}
    for d in evaluate_parameters(context, getattr(node, "_Node__parameters", [])):
        if isinstance(d, dict):
            out.update(d)
    return out


# --- the launch presets ------------------------------------------------------------------------
from launch import LaunchContext  # noqa: E402

remake_l, pipe_l = load_launch("remake"), load_launch("pipeline")
LAUNCH_ARGS = dict(mask_dilate="5", sam3="true", prompt="transparent object", threshold="0.4", fill_holes="false")
TAIL = ["depth_remake", "grasps_sbg", "grasp_poses", "pointcloud?"]


def launch_remake(**cfg):
    ctx = LaunchContext()
    ctx.launch_configurations.update({**LAUNCH_ARGS, "mask": "", "remake": "", "targets": "", **cfg})
    return remake_l._nodes(ctx), ctx


def launch_pipe(**cfg):
    ctx = LaunchContext()
    ctx.launch_configurations.update({"detector": "sbg", "archive": "always",
                                      "mask": "", "remake": "", "targets": "", **cfg})
    return {n.node_executable: params(n, ctx) for n in pipe_l._nodes(ctx)}


# --- the two pipelines -----------------------------------------------------------------------------
ck("PIPELINES: the same presets in remake.launch.py and pipeline.launch.py", remake_l.PIPELINES == pipe_l.PIPELINES,
   (remake_l.PIPELINES, pipe_l.PIPELINES))
PRESET_WANT = {   # pipeline -> (checkpoint, rel_rgb_fix, ReMake's mask, seg_node, SAM 3, chain head, targets)
    "adapted": ("L_combo_both_checkpoint43.tar", True, "seg_masks.npy", True, False, ["segment_trans"], "depth"),
    "baseline": ("remake_official.tar", False, "sam3_masks.npy", True, True, ["segment_sam3?", "segment_trans"],
                 "depth"),
}
for name, (want_ck, want_fix, want_mask, want_seg, want_sam3, head, want_targets) in PRESET_WANT.items():
    nodes, ctx = launch_remake(pipeline=name)
    execs = [n.node_executable for n in nodes]
    rp = params(nodes[0], ctx)
    ck(f"pipeline={name}: ReMake {want_ck} (rel_rgb_fix {want_fix}) on {want_mask}; seg_node "
       f"{'on' if want_seg else 'off'}, SAM 3 {'on' if want_sam3 else 'off'}",
       Path(rp["checkpoint"]).name == want_ck and rp["rel_rgb_fix"] is want_fix and rp["mask_name"] == want_mask
       and ("seg_node" in execs) == want_seg and ("sam3_node" in execs) == want_sam3, (execs, rp))
    ck(f"pipeline={name}: ReMake fills the object only (fill_holes false)", rp["fill_holes"] is False, rp)
    pn = launch_pipe(pipeline=name)
    chain = pn["orchestrator"]["chain"].split(",")
    ck(f"pipeline={name}: chain {','.join(head + TAIL)}, targets={want_targets}, runs labelled {name!r}",
       chain == head + TAIL and pn["grasp_pose_node"]["targets"] == want_targets
       and pn["grasp_pose_node"]["transparent_mask"] == want_mask and pn["orchestrator"]["pipeline"] == name,
       (chain, pn["grasp_pose_node"], pn["orchestrator"]))
    ck(f"pipeline={name}: cloud_completed's magenta uses ReMake's mask ({want_mask})",
       pn["pointcloud_node"].get("mask_name") == want_mask, pn["pointcloud_node"])
# raw: a plain RGB-D pipeline -- nothing in remake.launch, SBG alone on the raw depth, no target filter
nodes, ctx = launch_remake(pipeline="raw")
pn = launch_pipe(pipeline="raw")
gp = pn["grasp_pose_node"]
ck("pipeline=raw: remake.launch starts nothing (no ReMake, no segmenter, no SAM 3)", nodes == [],
   [n.node_executable for n in nodes])
ck("pipeline=raw: chain grasps_sbg,grasp_poses,pointcloud? only; targets=none, nothing completed, labelled 'raw'",
   pn["orchestrator"]["chain"].split(",") == TAIL[1:] and gp["targets"] == "none" and gp["depth_completed"] is False
   and gp["transparent_mask"] == "" and pn["orchestrator"]["pipeline"] == "raw"
   and pn["pointcloud_node"]["depth_name"] == "depth_aligned_mm.npy" and pn["pointcloud_node"]["mask_name"] == "",
   (pn["orchestrator"], gp, pn["pointcloud_node"]))
for mask in ("sam3", "seg"):
    pc = launch_pipe(pipeline="adapted", mask=mask)["pointcloud_node"]
    ck(f"pipeline.launch mask={mask}: the point-cloud node colours by {pipe_l.MASK_STAGES[mask][1]}",
       pc.get("mask_name") == pipe_l.MASK_STAGES[mask][1], pc)
# one setting overridden: the baseline's models with the transparency mask's objects as the only targets
# (the target rule before the trials)
nodes, ctx = launch_remake(pipeline="baseline", targets="transparent")
execs = [n.node_executable for n in nodes]
pn = launch_pipe(pipeline="baseline", targets="transparent")
ck("pipeline=baseline targets=transparent: official ReMake on SAM 3's mask, no segmenter, "
   "labelled 'baseline (targets=transparent)'",
   Path(params(nodes[0], ctx)["checkpoint"]).name == "remake_official.tar" and "seg_node" not in execs
   and "sam3_node" in execs and pn["orchestrator"]["chain"].split(",") == ["segment_sam3?"] + TAIL
   and pn["grasp_pose_node"]["targets"] == "transparent"
   and pn["orchestrator"]["pipeline"] == "baseline (targets=transparent)", (execs, pn["orchestrator"]))
pn = launch_pipe(pipeline="adapted", remake="official")
ck("pipeline=adapted remake=official: labelled 'adapted (remake=official)'",
   pn["orchestrator"]["pipeline"] == "adapted (remake=official)", pn["orchestrator"])

# --- parameter types ---------------------------------------------------------------------------------
# rclpy refuses an override whose type is not the declared default's (common.declare) and the node
# dies at start-up, so every parameter a launch file passes must match its declared type.
import ast  # noqa: E402


def declared(module):
    """{name: default} of every literal default the node module passes to declare(...)."""
    out = {}
    for n in ast.walk(ast.parse((ROOT / f"ros/remake_grasp/remake_grasp/{module}.py").read_text())):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "declare":
            for kw in n.keywords:
                try:
                    out[kw.arg] = ast.literal_eval(kw.value)
                except ValueError:
                    pass
    return out


sbg_l = load_launch("sbg")


def launch_sbg(**cfg):
    ctx = LaunchContext()
    ctx.launch_configurations.update({"profile": "picking", "top_k": "50", "depth_name": "", "remake": "",
                                      "pipeline": "adapted", **cfg})
    return sbg_l._nodes(ctx), ctx


mismatch, seen = [], set()
for pipeline in ("adapted", "baseline", "raw"):
    for targets in ("", "depth", "prompt", "transparent", "none"):
        for remake in ("", "none"):
            if pipeline == "raw" and targets in ("prompt", "transparent"):
                continue                                  # refused: raw has no transparency mask
            if targets == "prompt" and remake == "none":
                continue                                  # refused: prompt is PEGG-Net's, which needs ReMake
            nodes, ctx = launch_remake(pipeline=pipeline, targets=targets, remake=remake)
            launched = [(n.node_executable, params(n, ctx)) for n in nodes]
            launched += list(launch_pipe(pipeline=pipeline, targets=targets, remake=remake,
                                         detector="peggnet" if targets == "prompt" else "sbg").items())
            snodes, sctx = launch_sbg(remake=remake, pipeline=pipeline)
            launched += [(n.node_executable, params(n, sctx)) for n in snodes]
            for exe, prm in launched:
                seen.add(exe)
                want = declared(exe)
                mismatch += [(pipeline, targets, remake, exe, k, repr(v), type(want[k]).__name__)
                             for k, v in prm.items() if k in want and type(v) is not type(want[k])]
ck(f"every parameter the launch files pass has its node's declared type ({', '.join(sorted(seen))})",
   not mismatch and {"sam3_node", "seg_node", "remake_node", "grasp_pose_node", "orchestrator", "sbg_node",
                     "pointcloud_node"} <= seen, mismatch[:6] or seen)

# --- every combination of the three settings ---------------------------------------------------------
ck("pipeline.launch knows exactly remake.launch's REMAKE names", pipe_l.REMAKE_NAMES == tuple(remake_l.REMAKE),
   (pipe_l.REMAKE_NAMES, tuple(remake_l.REMAKE)))
for remake in tuple(remake_l.REMAKE):
    floor = remake == "none"
    bad = []
    for targets in ("depth", "prompt", "transparent", "none"):
        for mask in ("sam3", "seg", "none"):
            needs_mask = not floor or targets in ("prompt", "transparent")
            if mask == "none" and needs_mask:
                for launcher in (launch_remake, launch_pipe):
                    try:
                        launcher(pipeline="adapted", mask=mask, remake=remake, targets=targets)
                        bad.append((launcher.__name__, "accepted mask=none", targets))
                    except RuntimeError:
                        pass
                continue
            nodes, ctx = launch_remake(pipeline="adapted", mask=mask, remake=remake, targets=targets)
            execs = [n.node_executable for n in nodes]
            want_seg = (needs_mask and mask == "seg") or targets == "depth"
            want_sam3 = (needs_mask and mask == "sam3") or targets == "prompt"
            if floor:
                ok = "remake_node" not in execs
            else:
                rp = params(nodes[0], ctx)
                want_ck, want_fix = remake_l.REMAKE[remake]
                ok = (execs[0] == "remake_node" and Path(rp["checkpoint"]).name == want_ck
                      and rp["rel_rgb_fix"] is want_fix and rp["mask_name"] == remake_l.MASKS[mask]
                      and Path(rp["checkpoint"]).exists())
            if not (ok and ("seg_node" in execs) == want_seg and ("sam3_node" in execs) == want_sam3):
                bad.append(("remake.launch", mask, targets, execs))
            if targets == "prompt":                       # PEGG-Net's input: SBG refuses it
                try:
                    launch_pipe(pipeline="adapted", mask=mask, targets=targets, remake=remake)
                    bad.append(("pipeline.launch", "accepted targets=prompt with SBG", mask))
                except RuntimeError:
                    pass
                continue
            pn = launch_pipe(pipeline="adapted", mask=mask, targets=targets, remake=remake)
            chain = pn["orchestrator"]["chain"].split(",")
            first = (["segment_sam3?" if mask == "sam3" else "segment_trans"]) if needs_mask else []
            tstage = pipe_l.TARGET_STAGES[targets]
            want = first + ([] if tstage is None or tstage in first else [tstage]) \
                + ([] if floor else ["depth_remake"]) + ["grasps_sbg", "grasp_poses", "pointcloud?"]
            gp, pc = pn["grasp_pose_node"], pn["pointcloud_node"]
            if not (chain == want and gp["targets"] == targets and gp["target_mask"] == "seg_masks.npy"
                    and gp["transparent_mask"] == pipe_l.MASK_STAGES[mask][1] == (remake_l.MASKS[mask] or "")
                    and gp["depth_completed"] is (not floor)
                    and pc["depth_name"] == ("depth_aligned_mm.npy" if floor else "remake_depth_mm.npy")
                    and pc["mask_name"] == ("" if floor else pipe_l.MASK_STAGES[mask][1])):
                bad.append(("pipeline.launch", mask, targets, chain, gp.get("depth_completed"), pc))
    what = ("no ReMake node, no depth_remake stage, the mask stage only when the targets need it, the pose node "
            "told nothing was completed, the point cloud on the raw depth; mask=none only where nothing needs one") \
        if floor else (f"ReMake with {remake_l.REMAKE[remake][0]}, chain mask -> targets -> depth_remake -> SBG, cloud on "
                       f"ReMake's depth; mask=none refused")
    ck(f"remake={remake}, every mask x targets: {what}", not bad, bad[:3])
for pipeline, remake, depth_name, want in (("adapted", "", "", "remake_depth_mm.npy"),
                                           ("adapted", "official", "", "remake_depth_mm.npy"),
                                           ("adapted", "none", "", "depth_aligned_mm.npy"),
                                           ("adapted", "none", "x.npy", "x.npy"),
                                           ("raw", "", "", "depth_aligned_mm.npy"),
                                           ("baseline", "", "", "remake_depth_mm.npy")):
    snodes, sctx = launch_sbg(pipeline=pipeline, remake=remake, depth_name=depth_name)
    got = params(snodes[0], sctx)["depth_name"]
    ck(f"sbg.launch pipeline={pipeline} remake={remake or '(unset)'} depth_name={depth_name or '(unset)'}: SBG reads {want}",
       got == want, got)
ck("sbg.launch: RAW_PIPELINES are exactly the presets whose remake is none",
   set(sbg_l.RAW_PIPELINES) == {k for k, v in remake_l.PIPELINES.items() if v["remake"] == "none"}, sbg_l.RAW_PIPELINES)
ctx = LaunchContext()
decl = {a.name: a for a in sbg_l.generate_launch_description().entities if hasattr(a, "default_value")}
d = "".join(sub.perform(ctx) for sub in decl["remake"].default_value)
ck("sbg.launch: remake defaults to $REMAKE (unset here: '')", d == os.environ.get("REMAKE", ""), d)
decl = {a.name: a for a in remake_l.generate_launch_description().entities if hasattr(a, "default_value")}
d = "".join(sub.perform(LaunchContext()) for sub in decl["fill_holes"].default_value)
ck("remake.launch: fill_holes defaults to false", d == "false", d)
ck("remake_node and align.composite_depth: fill_holes defaults to False too",
   declared("remake_node").get("fill_holes") is False
   and "fill_holes=False" in (ROOT / "ros/remake_grasp/remake_grasp/align.py").read_text())
ck("remake.launch: no keep-nearer compositing (removed)", "keep_nearer_mm" not in decl, sorted(decl))
try:
    launch_pipe(detector="peggnet", pipeline="baseline", targets="prompt", remake="none")
    ck("pipeline.launch refuses detector=peggnet with remake=none", False)
except RuntimeError:
    ck("pipeline.launch refuses detector=peggnet with remake=none", True)
lab = launch_pipe(pipeline="adapted", remake="none")["orchestrator"]["pipeline"]
ck("pipeline=adapted remake=none: runs labelled 'adapted (remake=none)'", lab == "adapted (remake=none)", lab)
nodes, ctx = launch_remake(pipeline="adapted", sam3="false", mask="sam3")
ck("remake.launch sam3:=false starts no SAM 3", "sam3_node" not in [n.node_executable for n in nodes])
for targets, ok in (("depth", False), ("transparent", False), ("prompt", True)):
    what = "allowed" if ok else "refused (PEGG-Net needs the grasp prompt)"
    try:
        chain = launch_pipe(detector="peggnet", pipeline="baseline", targets=targets)["orchestrator"]["chain"]
        ck(f"pipeline.launch detector=peggnet targets={targets}: {what}",
           ok and chain.startswith("segment_sam3?,segment_sam3_all?,depth_remake,grasps_peggnet"), chain)
    except RuntimeError as e:
        ck(f"pipeline.launch detector=peggnet targets={targets}: {what}", not ok, e)
try:
    launch_pipe(pipeline="adapted", targets="prompt")
    ck("pipeline.launch refuses targets=prompt with detector=sbg (PEGG-Net's input only)", False)
except RuntimeError:
    ck("pipeline.launch refuses targets=prompt with detector=sbg (PEGG-Net's input only)", True)
for name, mod, env, want in (("pipeline", pipe_l, "PIPELINE", "adapted"), ("remake", remake_l, "PIPELINE", "adapted"),
                             ("pipeline", pipe_l, "MASK", ""),
                             ("pipeline", pipe_l, "TARGETS", ""), ("remake", remake_l, "MASK", ""),
                             ("remake", remake_l, "REMAKE", ""), ("remake", remake_l, "TARGETS", "")):
    ctx = LaunchContext()
    decl = {a.name: a for a in mod.generate_launch_description().entities if hasattr(a, "default_value")}
    default = "".join(sub.perform(ctx) for sub in decl[env.lower()].default_value)
    ck(f"{name}.launch: {env.lower()} defaults to {want!r} (with ${env} unset{'' if want else ': the preset decides'})",
       default == os.environ.get(env, want), default)
for bad in ({"pipeline": "fancy"}, {"pipeline": "adapted", "targets": "named"}):
    try:
        launch_pipe(**bad)
        ck(f"pipeline.launch refuses {bad}", False)
    except RuntimeError:
        ck(f"pipeline.launch refuses {bad}", True)

for bad in ({"pipeline": "fancy"}, {"pipeline": "adapted", "mask": "sam4"}, {"pipeline": "adapted", "remake": "L"},
            {"pipeline": "adapted", "targets": "named"}, {"pipeline": "raw", "remake": "official"},
            {"pipeline": "raw", "targets": "transparent"}):
    try:
        launch_remake(**bad)
        ck(f"remake.launch refuses {bad}", False)
    except RuntimeError as e:
        ck(f"remake.launch refuses {bad}", True, e)

# --- rel_rgb_fix in remake_node -----------------------------------------------------------------
import cv2  # noqa: E402
import torch  # noqa: E402
from remake_grasp import remake_node as RN  # noqa: E402
from remake_grasp.common import CKPT_DIR  # noqa: E402

run = next((r for r in sorted(RUNS.glob("20260930_*_auto")) if (r / "remake_depth_net_mm.npy").exists()
            and (r / "sam3_masks.npy").exists() and (np.load(r / "sam3_masks.npy") > 0).any()), None)
if run is None:
    print(f"  skip  rel_rgb_fix against an archived network output: no 20260930_*_auto run under {RUNS} "
          "(the rig's runs of 30 Sep 2026; set RG_RUNS)")
else:
    n = RN.RemakeNode.__new__(RN.RemakeNode)
    n.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n.model, n.reldepth, n.process_data, n.epoch = RN.load_remake(
        CKPT_DIR / "remake_official.tar", CKPT_DIR / "depth_anything_v2_vits.pth", n.device)
    n.mask_dilate = 5
    bgr = cv2.imread(str(run / "rgb.png"))
    depth = np.nan_to_num(np.load(run / "depth_aligned_mm.npy").astype(np.float32))
    K = np.load(run / "color_intrinsics.npy").astype(np.float32)
    n.live_dir, n.mask_name = run, "sam3_masks.npy"
    mask, tight, _ = n._mask(bgr.shape[:2])
    n.rel_rgb_fix = False
    off = n.complete(bgr, depth, K, mask)
    archived = np.load(run / "remake_depth_net_mm.npy")
    d = np.abs(off - archived)
    ck(f"rel_rgb_fix off reproduces the archived network output of {run.name} (median {np.median(d):.3f} mm, "
       f"max {d.max():.2f} mm)", np.median(d) < 0.05 and d.max() < 2.0, (np.median(d), d.max()))
    n.rel_rgb_fix = True
    on = n.complete(bgr, depth, K, mask)
    dd = np.abs(on - off)[tight]
    ck(f"rel_rgb_fix on changes only DepthAnything's input: in-mask change median {np.median(dd):.3f} mm, "
       f"p95 {np.percentile(dd, 95):.2f} mm (rgbfix_test: < 0.1 mm typical)",
       0 < np.median(dd) < 0.5 and np.percentile(dd, 95) < 2.0, (np.median(dd), np.percentile(dd, 95)))
    del n
    torch.cuda.empty_cache()

# --- the segmenter service --------------------------------------------------------------------
from remake_grasp import seg_node as SN  # noqa: E402


class _Log:
    def info(self, *a, **k): pass
    error = warning = info


s = SN.SegNode.__new__(SN.SegNode)
if str(SN.ROOT) not in sys.path:
    sys.path.insert(0, str(SN.ROOT))
from segmenter.live_segment import segment  # noqa: E402
from segmenter.model import load_checkpoint  # noqa: E402

s.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
s.model, ckp = load_checkpoint(CKPT_DIR / "seg_vits_rgb_best.pt", s.device, weights=CKPT_DIR / "depth_anything_v2_vits.pth")
s.segment, s.meta = segment, {"epoch": ckp.get("epoch"), "val_iou": ckp.get("val_iou")}
s.checkpoint, s.mask_name, s.thresh, s.dilate, s.fp16 = CKPT_DIR / "seg_vits_rgb_best.pt", "seg_masks.npy", 0.5, 0, True
s.get_logger = lambda: _Log()
# the overlay topic: a publisher that drops it, and a clock for its stamp
from builtin_interfaces.msg import Time as _Time  # noqa: E402
s.pub_overlay = type("_Pub", (), {"publish": lambda self, msg: None})()
s.get_clock = lambda: type("_Clock", (), {"now": lambda self: type("_Now", (), {"to_msg": lambda self: _Time()})()})()
s.frame_id = "camera_color_optical_frame"
# the archived run above, or on a fresh clone (no runs/) the three_transparent sample: SAM 3 found all three objects there
if run is not None:
    src_rgb, src_name, sam = run / "rgb.png", run.name, np.load(run / "sam3_masks.npy") > 0
else:
    smp = ROOT / "samples" / "three_transparent"
    src_rgb, src_name, sam = smp / "rgb.png", "samples/three_transparent", np.load(smp / "arrays.npz")["sam3_masks"] > 0
bgr = cv2.imread(str(src_rgb))
live = Path(tempfile.mkdtemp())
shutil.copy2(src_rgb, live / "rgb.png")
s.live_dir = live


class _Resp:
    success, message = None, ""


r = s._handle(None, _Resp())
m = np.load(live / "seg_masks.npy") if (live / "seg_masks.npy").exists() else None
rc = json.loads((live / "run_config.json").read_text()) if (live / "run_config.json").exists() else {}
ck(f"segment_trans on {src_name}: success, seg_masks.npy int32 at the colour size, overlay and run record "
   f"({r.message})", r.success and m is not None and m.dtype == np.int32 and m.shape == bgr.shape[:2]
   and (live / "seg_masks.png").exists() and "segment_trans" in json.dumps(rc), (r.message, None if m is None else m.dtype))
ck("segment_trans covers SAM 3's transparent object (>= 60% of its pixels)",
   m is not None and (m[sam] > 0).mean() >= 0.6, None if m is None else float((m[sam] > 0).mean()))

print("\nFAILED: " + ", ".join(fail) if fail else "\nall passed")
sys.exit(1 if fail else 0)
