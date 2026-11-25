#!/usr/bin/env python3
"""Score saved predictions with the unified metric: prints masked RMSE and writes per-frame rows.
(From the project's training code: the import path changed and its shape metrics are left out.)

    python3 eval/score.py --model L_combo_both_dataset
    python3 eval/score.py --model official_dataset --model L_combo_both_dataset   # compare

Reads predictions/<model>/*.png and writes metrics/<model>.jsonl (under data/eval) through
common.unified_metrics, the code behind the published ladder numbers. Masked RMSE is RMSE over
(transparent mask AND GT > 1 cm) at native 1280x720. It cannot see doming, so report it next to
measure_concavity.py. The reflection / refraction / normal split (no return / wrong return / sensor
already right) shows whether a change helps where transparency actually breaks the sensor.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def score_model(model, samples, verbose=True):
    """Unified metrics for every sample with a saved prediction; rewrites metrics/<model>.jsonl and
    returns the rows."""
    out_path = C.ANALYSIS / "metrics" / f"{model}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    rows, missing, t0 = [], 0, time.time()
    for i, s in enumerate(samples):
        pred = C.load_pred(model, s["key"])
        if pred is None:
            missing += 1
            continue
        try:
            _, raw, gt, mask = C.load_sample(s)
        except Exception as e:
            missing += 1
            if verbose and missing < 3:
                print(f"  cannot load {s['key']}: {e}")
            continue
        pred = C.resize_to(pred, gt.shape)
        row = C.unified_metrics(pred, gt, raw, mask)
        row["key"] = s["key"]
        row["camera"] = s["camera"]
        C.append_jsonl(out_path, row)
        rows.append(row)
        if verbose and (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(samples)}  {time.time()-t0:.0f}s")
    if missing:
        print(f"  WARNING: {missing} frames had no prediction or no data; "
              f"scores cover {len(rows)} of {len(samples)}")
    return rows


def summarise(name, rows, fail=0.025, good=0.012):
    """Print masked RMSE (mean / median / p90, % above fail and below good, both in metres), per
    camera and per sensor region; returns the mean in metres."""
    m = np.array([r["mask_rmse"] for r in rows if r.get("mask_rmse") is not None])
    cam = np.array([r["camera"] for r in rows if r.get("mask_rmse") is not None])
    print(f"\n=== {name}  (n={len(m)}) ===")
    if not len(m):
        print("  no scored frames")
        return None
    print(f"  masked RMSE   mean {1000*m.mean():6.1f} mm   median {1000*np.median(m):6.1f} mm"
          f"   p90 {1000*np.percentile(m,90):6.1f} mm")
    print(f"  FAIL >{1000*fail:.0f}mm  {100*(m>fail).mean():5.1f}%        "
          f"GOOD <{1000*good:.0f}mm  {100*(m<good).mean():5.1f}%")
    for c, lbl in ((1, "D435"), (2, "L515")):
        sel = cam == c
        if sel.sum():
            print(f"    {lbl}: mean {1000*m[sel].mean():6.1f} mm   "
                  f"median {1000*np.median(m[sel]):6.1f}   FAIL {100*(m[sel]>fail).mean():5.1f}%  (n={sel.sum()})")
    for key, lbl in (("reflection_rmse", "reflection (no return)"),
                     ("refraction_rmse", "refraction (wrong return)"),
                     ("normal_rmse", "normal (sensor already right)")):
        v = np.array([r[key] for r in rows if r.get(key) is not None])
        if len(v):
            print(f"    {lbl:<30} {1000*v.mean():6.1f} mm  (n={len(v)})")
    return float(m.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True,
                    help="predictions/<model>/ to score; repeat to compare")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fail", type=float, default=0.025)
    ap.add_argument("--good", type=float, default=0.012)
    args = ap.parse_args()

    samples = C.load_subset()["samples"]
    if args.limit:
        samples = samples[:args.limit]
    print(f"frames: {len(samples)}")

    means = {}
    for m in args.model:
        means[m] = summarise(m, score_model(m, samples), args.fail, args.good)

    if len(means) > 1:
        print("\n=== comparison (masked RMSE mean) ===")
        base = args.model[0]
        for m in args.model:
            if means[m] is None:
                continue
            d = "" if m == base else f"   {1000*(means[m]-means[base]):+6.1f} mm vs {base}"
            print(f"  {m:<26} {1000*means[m]:6.1f} mm{d}")
        print("\n  Concavity, the shape measure: eval/measure_concavity.py.")
    print("\nwrote metrics/<model>.jsonl (one row per frame) under data/eval")


if __name__ == "__main__":
    main()
