#!/usr/bin/env python3
"""How much of an object's concavity does a prediction reproduce? (From the project's training code;
the import path changed, and unreadable frames are skipped and counted.)

    python3 eval/measure_concavity.py --model L_combo_both_dataset
    python3 eval/measure_concavity.py --model official_dataset --limit 200      # 200 random frames
    python3 eval/measure_concavity.py --model official_dataset --model-b L_combo_both_dataset

Masked RMSE cannot see this: a model that renders an open bowl as a smooth dome of the right mean
depth scores well while losing the shape a gripper needs (released ReMake flattens slopes by 7-10%).

Per transparent instance, concavity = median(depth[r > 0.60]) - median(depth[r < 0.25]), where r is
the mask's distance transform over its maximum (0 at the rim, 1 innermost); positive means the
interior is farther than the rim, as in an open vessel. The score is pred / GT concavity, a ratio
because GT concavity spans 10 mm to over 100 mm. Instances under --min-area px are skipped (too few
rim pixels for a stable median), and those under --min-concavity mm are left out of the overall %.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

BANDS = [(10, 25), (25, 40), (40, 60), (60, 90), (90, 400)]


def concavity_mm(depth_m, blob, dist, r_in=0.60, r_out=0.25, min_px=40):
    """Interior depth minus rim depth, in mm; None if the blob is too thin (radius < 12 px) or
    either ring has fewer than min_px valid pixels."""
    radius = dist.max()
    if radius < 12:
        return None
    r = dist / radius
    inner = depth_m[blob & (r > r_in)]
    outer = depth_m[blob & (r < r_out)]
    inner = inner[inner > C.GT_VALID_MIN]
    outer = outer[outer > C.GT_VALID_MIN]
    if inner.size < min_px or outer.size < min_px:
        return None
    return float((np.median(inner) - np.median(outer)) * 1000.0)


def instances(gt, mask, min_area):
    """Yield (blob, dist) per transparent-object instance in the frame."""
    m = (mask & (gt > C.GT_VALID_MIN)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    for i in range(1, n):
        if st[i, 4] < min_area:
            continue
        blob = lab == i
        yield blob, cv2.distanceTransform(blob.astype(np.uint8), cv2.DIST_L2, 5)


def score(model, samples, min_area):
    """(gt_mm, pred_mm, camera, key) per measurable instance, over the samples with a saved
    prediction; both concavities use the GT instance's rings."""
    rows, unreadable = [], 0
    for s in samples:
        try:
            _, raw, gt, mask = C.load_sample(s)
        except Exception:
            unreadable += 1
            continue
        pred = C.load_pred(model, s["key"])
        if pred is None:
            continue
        pred = C.resize_to(pred, gt.shape)
        for blob, dist in instances(gt, mask, min_area):
            valid = blob & (gt > C.GT_VALID_MIN)
            g = concavity_mm(gt, valid, dist)
            p = concavity_mm(pred, valid, dist)
            if g is None or p is None:
                continue
            rows.append((g, p, s["camera"], s["key"]))
    if unreadable:
        print(f"  {unreadable} sample(s) could not be read and were skipped", file=sys.stderr)
    return rows


def report(name, rows, min_conc):
    """Print reproduced % per GT band, overall (GT > min_conc) and per camera; returns the overall %
    (None when no instance is deep enough)."""
    a = np.array([[r[0], r[1]] for r in rows])
    cam = np.array([r[2] for r in rows])
    print(f"\n=== {name} ===")
    print(f"  instances scored: {len(rows)}  (D435 {(cam==1).sum()}, L515 {(cam==2).sum()})")
    if not len(a):
        return None
    print(f"  all instances, GT concavity median {np.median(a[:,0]):+.1f} mm, "
          f"prediction {np.median(a[:,1]):+.1f} mm")
    print(f"\n  {'GT band':>12}  {'n':>4}  {'GT med':>8}  {'pred med':>9}  {'reproduced':>11}")
    for lo, hi in BANDS:
        sel = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if sel.sum() < 3:
            continue
        g, p = np.median(a[sel, 0]), np.median(a[sel, 1])
        print(f"  {lo:5d}-{hi:4d} mm  {sel.sum():4d}  {g:7.1f}  {p:8.1f}  {100*p/g:10.0f}%")
    deep = a[a[:, 0] > min_conc]
    if len(deep):
        overall = 100 * np.median(deep[:, 1]) / np.median(deep[:, 0])
        print(f"\n  overall (GT > {min_conc:.0f} mm, n={len(deep)}): "
              f"{overall:.0f}% of concavity reproduced")
        print(f"  instances keeping under half: {100*np.mean(deep[:,1] < 0.5*deep[:,0]):.0f}%")
        for c, lbl in ((1, "D435"), (2, "L515")):
            sel = (cam == c) & (a[:, 0] > min_conc)
            if sel.sum() > 3:
                print(f"    {lbl}: n={sel.sum():3d}  "
                      f"{100*np.median(a[sel,1])/np.median(a[sel,0]):.0f}%")
        return overall
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="remake",
                    help="predictions/<model>/ to score (default: remake)")
    ap.add_argument("--model-b", default=None,
                    help="a second model to score on the SAME instances, for A/B")
    ap.add_argument("--limit", type=int, default=0, help="N random subset frames (seeded by --seed)")
    ap.add_argument("--concave-only", action="store_true",
                    help="restrict to frames whose model_list has a bowl/cup/glass/"
                         "fruitdish. Faster, but the full subset is the fair test: a "
                         "change must not flatten the objects that are already right")
    ap.add_argument("--min-area", type=int, default=3000)
    ap.add_argument("--min-concavity", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = C.load_subset()["samples"]
    if args.concave_only:
        names = C.object_names()
        concave = {k for k, v in names.items()
                   if v.startswith(("bowl", "cup", "glass", "fruitdish"))}
        samples = [s for s in samples if set(s["model_list"]) & concave]
        print(f"frames containing a bowl/cup/glass/fruitdish: {len(samples)}")
    if args.limit:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(samples), min(args.limit, len(samples)), replace=False)
        samples = [samples[i] for i in sorted(idx)]
    print(f"frames: {len(samples)}")

    a_overall = report(args.model, score(args.model, samples, args.min_area),
                       args.min_concavity)
    if args.model_b:
        b_overall = report(args.model_b, score(args.model_b, samples, args.min_area),
                           args.min_concavity)
        if a_overall and b_overall:
            print(f"\n  {args.model} {a_overall:.0f}%  ->  "
                  f"{args.model_b} {b_overall:.0f}%   "
                  f"({b_overall - a_overall:+.0f} points)")


if __name__ == "__main__":
    main()
