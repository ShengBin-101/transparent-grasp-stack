#!/usr/bin/env python3
"""Mask quality against the TransCG ground-truth transparent mask: per-frame
IoU / recall / precision, the numbers every mask in this project is described by.

    python3 segmenter/mask_quality.py --mask-dir datasets/sam_masks_box [--report out.jsonl]
    python3 segmenter/mask_quality.py --mask-dir datasets/seg_masks_vits_rgb

SAM masks, degraded masks and the segmenter are all scored by this one
implementation. A missing mask file counts as an EMPTY mask (IoU 0), not a
skipped frame: a segmenter that finds nothing on a frame has failed that frame.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def frame_row(pred, gt, key, **extra):
    """One frame's row. pred, gt: bool HxW at the same size."""
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    inter = float((pred & gt).sum())
    row = {"key": key,
           "iou": inter / max(float((pred | gt).sum()), 1.0),
           "recall": inter / max(float(gt.sum()), 1.0),
           "precision": inter / max(float(pred.sum()), 1.0),
           "pred_px": int(pred.sum()), "gt_px": int(gt.sum())}
    row.update(extra)
    return row


def summarise(rows, label="mask quality against the ground-truth mask"):
    iou = np.array([r["iou"] for r in rows])
    rec = np.array([r["recall"] for r in rows])
    pre = np.array([r["precision"] for r in rows])
    empty = sum(1 for r in rows if r["pred_px"] == 0)
    lines = [f"{label}, {len(rows)} frames",
             f"  IoU        mean {iou.mean():.3f}  median {np.median(iou):.3f}",
             f"  recall     mean {rec.mean():.3f}  (of GT transparent pixels found)",
             f"  precision  mean {pre.mean():.3f}  (of predicted pixels that are transparent)",
             f"  frames with an empty prediction: {empty}"]
    return "\n".join(lines), {"n": len(rows), "iou": float(iou.mean()), "iou_median": float(np.median(iou)),
                              "recall": float(rec.mean()), "precision": float(pre.mean()), "empty": empty}


def write_report(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    import sys
    sys.path.insert(0, str(ROOT / "eval"))
    import common as C
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--sample-list", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    if args.sample_list:
        with open(args.sample_list) as f:
            samples = json.load(f)["samples"]
    else:
        samples = C.load_subset()["samples"]
    if args.limit:
        samples = samples[:args.limit]
    rows, missing = [], 0
    for s in samples:
        _, _, _, gt = C.load_sample(s)
        m = C.load_external_mask(args.mask_dir, s["key"], gt.shape)
        if m is None:
            missing += 1
            m = np.zeros(gt.shape, bool)
        rows.append(frame_row(m, gt, s["key"]))
    text, _ = summarise(rows, f"{args.mask_dir}")
    print(text)
    if missing:
        print(f"  missing mask files (scored as empty): {missing}")
    if args.report:
        write_report(rows, args.report)
        print(f"  per-frame report -> {args.report}")


if __name__ == "__main__":
    main()
