#!/usr/bin/env python3
"""The prompt-free segmenter's masks for the frozen TransCG subset, for predict_transcg.py --mask-dir.

    python3 eval/make_seg_masks.py --dilate 5      # -> data/eval/masks/seg_vits_rgb_d5/<key>.png

Makes the live seg_node's call (segmenter/live_segment.segment: threshold 0.5, fp16, square
(2k+1) dilation) on each frame's rgb{camera}.png, as the training code did for its masks. Writes 0/255
PNGs at native 1280x720 and prints the IoU against the dataset's mask (on the test subset 0.887 as is,
0.850 grown 5 px).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
sys.path.insert(0, str(EVAL))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/seg_vits_rgb_best.pt"))
    ap.add_argument("--dilate", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="default data/eval/masks/seg_vits_rgb_d<dilate>")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from segmenter.live_segment import segment
    from segmenter.model import load_checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(args.checkpoint, device, weights=str(ROOT / "checkpoints/depth_anything_v2_vits.pth"))
    out = Path(args.out or C.ANALYSIS / "masks" / f"seg_vits_rgb_d{args.dilate}")
    out.mkdir(parents=True, exist_ok=True)
    samples = C.load_subset()["samples"][: args.limit or None]
    print(f"{Path(args.checkpoint).name} (epoch {ck.get('epoch')}), dilate {args.dilate} -> {out}", flush=True)

    inter = union = 0
    ious = []
    t0 = time.time()
    for n, s in enumerate(samples, 1):
        d, c = C.sample_dir(s), s["camera"]
        bgr = cv2.imread(str(d / f"rgb{c}.png"))
        labels, _, _ = segment(model, bgr, device, args.thresh, args.dilate, True)
        m = labels > 0
        cv2.imwrite(str(out / f"{s['key']}.png"), (m * 255).astype(np.uint8))
        gt = cv2.imread(str(d / f"depth{c}-gt-mask.png"), cv2.IMREAD_UNCHANGED) > 0
        i, u = int((m & gt).sum()), int((m | gt).sum())
        inter, union = inter + i, union + u
        if u:
            ious.append(i / u)
        if n % 200 == 0:
            print(f"  {n}/{len(samples)}  {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {len(samples)} masks in {time.time() - t0:.0f} s; IoU against the dataset mask: mean per frame "
          f"{np.mean(ious):.3f}, pooled {inter / max(union, 1):.3f}", flush=True)


if __name__ == "__main__":
    main()
