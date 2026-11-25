#!/usr/bin/env python3
"""Run the transparency segmenter and write masks in the formats the rest of
the repo consumes, plus mask quality against ground truth.

    # the frozen 1,177-frame TransCG test subset -> eval/predict_transcg.py's dir condition
    python3 segmenter/predict.py --checkpoint checkpoints/seg_vits_rgb_best.pt --out datasets/seg_masks_vits_rgb
    # cross-domain gate
    python3 segmenter/predict.py --checkpoint ... --dataset cleargrasp --cg-split real-test --out /tmp/cg_masks
    # one live-pipeline frame, in the sam3_masks.npy contract
    python3 segmenter/predict.py --checkpoint ... --live-dir data/live --live-name seg_masks.npy

PNG masks are 0/255 at the ground truth's native size, the format
eval/common.py::load_external_mask reads. `--dilate k` applies the same square
(2k+1) dilation remake_node uses (mask_dilate 5 there). The live-dir output is
int32 connected-component labels with 0 = background, which remake_node reads
as `m > 0`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import common as C  # noqa: E402
from segmenter import mask_quality as MQ  # noqa: E402
from segmenter.data import ClearGraspReal, TransCGSeg  # noqa: E402
from segmenter.live_segment import grow, predict_mask  # noqa: E402
from segmenter.model import load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="transcg", choices=["transcg", "cleargrasp"])
    ap.add_argument("--sample-list", default=None, help="transcg: eval/common-style list; default the frozen subset")
    ap.add_argument("--cg-split", default="real-test", choices=["real-test", "real-val"])
    ap.add_argument("--out", default=None, help="directory for <key>.png masks")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--dilate", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", default=None, help="per-frame IoU/recall/precision JSONL")
    ap.add_argument("--live-dir", default=None, help="write one live-pipeline frame's mask instead")
    ap.add_argument("--live-rgb", default="rgb.png")
    ap.add_argument("--live-name", default="seg_masks.npy")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(args.checkpoint, device)
    print(f"{args.checkpoint}: encoder {ck['encoder']}, use_valid {ck['use_valid']}, epoch {ck.get('epoch')}, "
          f"val IoU {ck.get('val_iou', float('nan')):.4f}, device {device}")

    if args.live_dir:
        live = Path(args.live_dir)
        bgr = cv2.imread(str(live / args.live_rgb))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        valid = np.ones(rgb.shape[:2], bool)
        dm = live / "depth_aligned_mm.npy"
        if model.use_valid and dm.exists():
            valid = np.load(dm) > 0
        m = grow(predict_mask(model, rgb, device, rgb.shape[:2], args.thresh, valid=valid), args.dilate)
        n, labels = cv2.connectedComponents(m.astype(np.uint8))
        np.save(live / args.live_name, labels.astype(np.int32))
        print(f"wrote {live / args.live_name}: {n - 1} component(s), {m.mean() * 100:.1f}% of frame")
        return

    if args.dataset == "transcg":
        if args.sample_list:
            with open(args.sample_list) as f:
                samples = json.load(f)["samples"]
        else:
            samples = C.load_subset()["samples"]
        ds = TransCGSeg("samples", samples=samples, aug=False, limit=args.limit)
    else:
        ds = ClearGraspReal(args.cg_split, need_valid=model.use_valid, limit=args.limit)
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i in range(len(ds)):
        rgb, gt, valid, key = ds.load(i)
        m = grow(predict_mask(model, rgb, device, gt.shape, args.thresh, valid=valid), args.dilate)
        if out:
            cv2.imwrite(str(out / f"{key}.png"), (m * 255).astype(np.uint8))
        rows.append(MQ.frame_row(m, gt, key))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(ds)}  {time.time() - t0:.0f}s", flush=True)
    label = f"{args.dataset}{'/' + args.cg_split if args.dataset == 'cleargrasp' else ''} thresh {args.thresh} dilate {args.dilate}"
    text, summary = MQ.summarise(rows, label)
    print("\n" + text)
    print(f"  {len(rows)} frames in {time.time() - t0:.0f}s" + (f", masks -> {out}" if out else ""))
    if args.report:
        MQ.write_report(rows, args.report)
        print(f"  per-frame report -> {args.report}")


if __name__ == "__main__":
    main()
