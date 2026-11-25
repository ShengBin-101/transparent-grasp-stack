#!/usr/bin/env python3
"""Run ReMake over the frozen 1,177-frame TransCG test subset under one mask condition, saving the
predictions that eval/score.py and eval/measure_concavity.py read.

    python3 eval/predict_transcg.py --remake L_combo_both --condition blanked
    python3 eval/predict_transcg.py --remake official --condition dir --mask-dir <pngs> --name official_seg_d5
    python3 eval/predict_transcg.py --checkpoint checkpoints/other.tar [--rel-rgb-fix] --condition dataset --name other

The network gets what the training code fed it for the ladder numbers: upstream ReMake's TransCG reader
(third_party/remake, unchanged) at the release test settings (640x480, 0.3-1.5 m) and the frozen
DepthAnything-V2 prior. The model loads through remake_node.load_remake, as in the live pipeline, and
--remake reads remake.launch.py's (file, rel_rgb_fix) table, so the benchmark and the robot agree.

A condition changes only the network's input (the mask, and for blanked the depth inside it); scoring always
uses the dataset's mask.
  dataset   the benchmark protocol: the dataset's transparent mask
  blanked   the dataset mask, with the sensor's depth inside it removed  (the training code's --zero-in-mask)
  nomask    an empty mask                                               (its --no-mask)
  dir       <key>.png masks from --mask-dir, e.g. the segmenter's       (its --mask-dir)
"""
import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
sys.path.insert(0, str(EVAL))
import common as C  # noqa: E402

CONDITIONS = ("dataset", "blanked", "nomask", "dir")


def remake_table():
    """REMAKE = {name: (file, rel_rgb_fix)} from launch/remake.launch.py, without importing launch."""
    tree = ast.parse((ROOT / "ros/remake_grasp/launch/remake.launch.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "REMAKE" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("REMAKE table not found in remake.launch.py")


def main():
    """Load the model and upstream's reader, predict each subset frame with the condition's mask and
    depth, and save the PNGs plus run.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--remake", help="a name from remake.launch.py's REMAKE table: official, L_combo_both, ...")
    src.add_argument("--checkpoint", help="any ReMake checkpoint file (then say --rel-rgb-fix if it was trained so)")
    ap.add_argument("--rel-rgb-fix", action="store_true")
    ap.add_argument("--condition", choices=CONDITIONS, default="dataset")
    ap.add_argument("--mask-dir", default=None, help="for --condition dir: <key>.png masks, any size, nonzero = mask")
    ap.add_argument("--name", default=None, help="predictions/<name>/; default <remake>_<condition>")
    ap.add_argument("--limit", type=int, default=0, help="first N subset frames only (a quick check)")
    args = ap.parse_args()

    if args.remake:
        table = remake_table()
        if args.remake not in table:
            raise SystemExit(f"--remake {args.remake}: not in remake.launch.py's REMAKE ({', '.join(table)})")
        ckpt_file, rel_rgb_fix = table[args.remake]
        checkpoint = ROOT / "checkpoints" / ckpt_file
    else:
        checkpoint, rel_rgb_fix = Path(args.checkpoint), args.rel_rgb_fix
    if args.condition == "dir" and not args.mask_dir:
        raise SystemExit("--condition dir needs --mask-dir")
    name = args.name or f"{args.remake or checkpoint.stem}_{args.condition}"

    os.environ.setdefault("RG_ROOT", str(ROOT))
    sys.path.insert(0, str(ROOT / "ros/remake_grasp"))
    from remake_grasp.remake_node import load_remake            # the live pipeline's loader (installs the stubs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, reldepth, _, epoch = load_remake(checkpoint, ROOT / "checkpoints/depth_anything_v2_vits.pth", device)

    import utils.data_preparation as DP                          # third_party/remake, now on sys.path
    if rel_rgb_fix:
        # The reader's image is already RGB: reverse it so img_rel_preprocess's BGR->RGB swap cancels
        # (the training code's input_is_rgb=True), without editing upstream's file.
        upstream = DP.img_rel_preprocess
        DP.img_rel_preprocess = lambda img, input_size=518: upstream(np.ascontiguousarray(img[..., ::-1]), input_size)
    from datasets.transcg import TransCG                       # upstream's reader, unchanged

    dataset = TransCG(data_dir=str(C.DATA_DIR), split="test", image_size=(640, 480), use_augmentation=False,
                      depth_min=0.3, depth_max=1.5, depth_norm=1.0, use_depth_augmentation=False,
                      no_mask_depth=False, reldepth_model="depthanything")
    index = {}
    for i, info in enumerate(dataset.sample_info):
        parts = Path(info[0]).parts
        index[f"s{int(parts[-2].replace('scene', '')):03d}_p{int(parts[-1]):03d}_c{info[1]}"] = i
    samples = C.load_subset()["samples"][: args.limit or None]
    missing = [s["key"] for s in samples if s["key"] not in index]
    if missing:
        raise SystemExit(f"{len(missing)} subset frames are not in {C.DATA_DIR}, e.g. {missing[:3]}")

    print(f"{name}: {checkpoint.name} (epoch {epoch}), rel_rgb_fix {rel_rgb_fix}, condition {args.condition}, "
          f"{len(samples)} frames from {C.DATA_DIR}, on {device}", flush=True)
    t0 = time.time()
    for n, s in enumerate(samples, 1):
        d = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in dataset[index[s["key"]]].items()}
        mask = d["depth_gt_mask"]
        if args.condition == "nomask":
            mask = torch.zeros_like(mask)
        elif args.condition == "dir":
            lab = cv2.imread(str(Path(args.mask_dir) / f"{s['key']}.png"), cv2.IMREAD_UNCHANGED)
            if lab is None:
                raise SystemExit(f"{args.mask_dir}/{s['key']}.png is missing")
            m = cv2.resize((lab > 0).astype(np.uint8), (mask.shape[-1], mask.shape[-2]), interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy(m.astype(bool)).to(device).reshape(mask.shape).to(mask.dtype)
        depth = d["depth"]
        if args.condition == "blanked":
            depth = depth * (1 - mask.to(depth.dtype).reshape(depth.shape))
        with torch.no_grad():
            rel = reldepth.forward(d["rgb_relat"]).unsqueeze(1)
            rel = F.interpolate(rel, size=(480, 640), mode="bilinear", align_corners=False)
            res = model(d["rgb"], rel, depth, mask)
        C.save_pred(name, s["key"], res[0, 0].float().cpu().numpy())
        if n % 100 == 0:
            print(f"  {n}/{len(samples)}  {time.time() - t0:.0f}s", flush=True)

    meta = {"checkpoint": checkpoint.name, "epoch": epoch, "rel_rgb_fix": rel_rgb_fix, "condition": args.condition,
            "mask_dir": args.mask_dir, "frames": len(samples), "data_dir": str(C.DATA_DIR),
            "seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = C.pred_path(name, "x").parent
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(samples)} predictions to {out} in {meta['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
