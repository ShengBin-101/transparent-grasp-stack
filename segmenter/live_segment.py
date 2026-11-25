#!/usr/bin/env python3
"""segment(), which seg_node calls, and a standalone runner for one live frame
-> seg_masks.npy without ROS. Imports only torch / cv2 / numpy, segmenter/model.py
and the Depth Anything V2 code in the ReMake submodule.

    python3 segmenter/live_segment.py --checkpoint checkpoints/seg_vits_rgb_best.pt \
        --backbone-weights checkpoints/depth_anything_v2_vits.pth --live-dir data/live [--watch]

Reads <live-dir>/rgb.png (BGR on disk, as the capture stage writes it) and
writes <live-dir>/seg_masks.npy: int32 connected-component labels, 0 =
background, the sam3_masks.npy contract remake_node reads (`m > 0`). remake_node
grows the mask by its own mask_dilate (5 px); --dilate here is for a consumer
that does not.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from segmenter.model import load_checkpoint, preprocess  # noqa: E402


def predict_mask(model, rgb, device, hw=None, thresh=0.5, fp16=True, valid=None):
    """uint8 RGB -> bool transparency mask at hw (default: the input's size).

    valid is the sensor-validity map, needed only by a head trained with it (use_valid).
    """
    x = preprocess(rgb).unsqueeze(0).to(device)
    v = None
    if model.use_valid:
        if valid is None:
            raise ValueError("this head was trained with the sensor-validity map: pass valid=")
        v = torch.from_numpy(cv2.resize(valid.astype(np.uint8), (x.shape[-1], x.shape[-2]),
                                        interpolation=cv2.INTER_NEAREST).astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast("cuda", enabled=fp16 and device.type == "cuda"):
        y = model(x, v)
    prob = torch.sigmoid(y.float())[0]
    prob = torch.nn.functional.interpolate(prob[None, None], size=hw or rgb.shape[:2], mode="bilinear",
                                           align_corners=False)[0, 0]
    return (prob > thresh).cpu().numpy()


def grow(mask, k):
    """Square (2k+1) dilation, as remake_node's mask_dilate; k <= 0 returns the mask unchanged."""
    if k <= 0:
        return mask
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool)


def segment(model, bgr, device, thresh=0.5, dilate=0, fp16=True):
    """BGR frame -> (int32 component labels, 0 = background; number of components; mask share of the frame)."""
    m = grow(predict_mask(model, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), device, None, thresh, fp16), dilate)
    m = m.astype(np.uint8)
    n, labels = cv2.connectedComponents(m)
    return labels.astype(np.int32), n - 1, float(m.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--live-dir", required=True)
    ap.add_argument("--rgb-name", default="rgb.png")
    ap.add_argument("--mask-name", default="seg_masks.npy")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--dilate", type=int, default=0, help="the depth service already dilates; leave 0 unless it does not")
    ap.add_argument("--watch", action="store_true", help="loop: re-segment whenever rgb.png's mtime changes")
    ap.add_argument("--backbone-weights", default="default",
                    help="path to depth_anything_v2_vits.pth if not at checkpoints/")
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(args.checkpoint, device, weights=args.backbone_weights)
    live = Path(args.live_dir)
    src, dst = live / args.rgb_name, live / args.mask_name
    print(f"segmenter {args.checkpoint} (epoch {ck.get('epoch')}, val IoU {ck.get('val_iou', float('nan')):.3f}) "
          f"on {device}; {src} -> {dst}", flush=True)
    seen = None
    while True:
        if not src.exists():
            if not args.watch:
                raise SystemExit(f"{src} not found")
            time.sleep(0.2); continue
        mt = src.stat().st_mtime
        if mt != seen:
            t0 = time.time()
            bgr = cv2.imread(str(src))
            labels, n, frac = segment(model, bgr, device, args.thresh, args.dilate, not args.no_fp16)
            tmp = dst.with_suffix(".tmp.npy")
            np.save(tmp, labels); tmp.replace(dst)        # atomic for the reader
            print(f"[{time.strftime('%H:%M:%S')}] {dst.name}: {n} component(s), {frac * 100:.1f}% of frame, "
                  f"{(time.time() - t0) * 1000:.0f} ms", flush=True)
            seen = mt
        if not args.watch:
            break
        time.sleep(0.1)


if __name__ == "__main__":
    main()
