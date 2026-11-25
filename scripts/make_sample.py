#!/usr/bin/env python3
"""Pack an archived capture into samples/<name>/ for scripts/demo_offline.sh (runs anywhere: numpy + json).

    python3 scripts/make_sample.py runs/20260928_171157_auto three_transparent --note "three transparent objects"
    python3 scripts/make_sample.py samples/<name> --unpack <dir>   # the reverse: what the demo does

A sample is the capture stage's output (rgb.png, plus arrays.npz with depth_aligned_mm, depth_coverage
and color_intrinsics), so every later stage runs for real. SAM 3 is not re-run (gated weights, ~5.7 GB),
so its archived masks travel with the sample when the run has them: sam3_masks (PIPELINE=baseline) and
sam3_masks_all (TARGETS=prompt). np.savez_compressed keeps the arrays bit-identical (~0.6 MB a capture).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ARRAYS = ("depth_aligned_mm", "depth_coverage", "color_intrinsics", "sam3_masks", "sam3_masks_all")
REQUIRED = ("depth_aligned_mm", "depth_coverage", "color_intrinsics")
ROOT = Path(__file__).resolve().parents[1]


def pack(run, name, note):
    """Copy a run's rgb.png and capture arrays into samples/<name>/, and write sample.json with the
    presets the sample supports (baseline only when it has a SAM 3 mask)."""
    run, out = Path(run), ROOT / "samples" / name
    missing = [a for a in REQUIRED if not (run / f"{a}.npy").exists()] + ([] if (run / "rgb.png").exists() else ["rgb.png"])
    if missing:
        raise SystemExit(f"{run}: missing {missing}")
    out.mkdir(parents=True, exist_ok=True)
    arrays = {a: np.load(run / f"{a}.npy") for a in ARRAYS if (run / f"{a}.npy").exists()}
    np.savez_compressed(out / "arrays.npz", **arrays)
    shutil.copy2(run / "rgb.png", out / "rgb.png")
    meta = {"source_run": run.name, "note": note, "arrays": {k: [v.dtype.str, list(v.shape)] for k, v in arrays.items()},
            "presets": ["adapted", "raw"] + (["baseline"] if "sam3_masks" in arrays else [])}
    (out / "sample.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{out}: {', '.join(arrays)}; presets {', '.join(meta['presets'])}")


def unpack(sample, live):
    """Write a sample back out as rgb.png and <array>.npy files, as the capture stage leaves them."""
    sample, live = Path(sample), Path(live)
    live.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample / "rgb.png", live / "rgb.png")
    with np.load(sample / "arrays.npz") as z:
        for k in z.files:
            np.save(live / f"{k}.npy", z[k])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="an archived run (pack) or a samples/<name> folder (with --unpack)")
    ap.add_argument("name", nargs="?", help="sample name (pack)")
    ap.add_argument("--note", default="")
    ap.add_argument("--unpack", metavar="DIR", help="write the sample's files into DIR, as the capture stage would")
    a = ap.parse_args()
    if a.unpack:
        unpack(a.src, a.unpack)
    elif a.name:
        pack(a.src, a.name, a.note)
    else:
        ap.error("give a sample name to pack, or --unpack DIR")
