#!/usr/bin/env python3
"""Copy just what the benchmark reads out of a full TransCG release: ~2.6 GB instead of 204 GB.

    python3 eval/make_transcg_subset.py /path/to/transcg/transcg     # -> datasets/transcg/transcg (not in git)
    python3 eval/make_transcg_subset.py /path/to/transcg/transcg /elsewhere/transcg/transcg
                                                    # then RG_DATASETS=/elsewhere ./start.sh --recreate

Keeps metadata.json, camera_intrinsics/, every scene's metadata.json (upstream's reader indexes the
whole test split before opening a frame), and the rgb, raw depth, GT depth and GT mask of the 1,177
frames in eval/sample_list.json. The object meshes (models/) are left out; only measure_concavity.py
--concave-only reads their names. TransCG's licence applies to the copy (graspnet.net/transcg).
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="TransCG root: the folder holding metadata.json and scene*/")
    ap.add_argument("dst", nargs="?", help="default: datasets/transcg/transcg in this repo, where the benchmark looks first")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    src, dst = Path(a.src), Path(a.dst or root / "datasets" / "transcg" / "transcg")
    if not (src / "metadata.json").exists():
        raise SystemExit(f"{src}/metadata.json not found: point at the folder holding metadata.json and scene*/")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "metadata.json", dst)
    shutil.copytree(src / "camera_intrinsics", dst / "camera_intrinsics", dirs_exist_ok=True)
    meta = json.loads((src / "metadata.json").read_text())
    for i in range(1, meta["total_scenes"] + 1):
        (dst / f"scene{i}").mkdir(exist_ok=True)
        shutil.copy2(src / f"scene{i}" / "metadata.json", dst / f"scene{i}")
    samples = json.loads((Path(__file__).resolve().parent / "sample_list.json").read_text())["samples"]
    n = 0
    for s in samples:
        rel = Path(f"scene{s['scene']}") / str(s["perspective"])
        (dst / rel).mkdir(parents=True, exist_ok=True)
        c = s["camera"]
        for f in (f"rgb{c}.png", f"depth{c}.png", f"depth{c}-gt.png", f"depth{c}-gt-mask.png"):
            shutil.copy2(src / rel / f, dst / rel / f)
            n += 1
    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    print(f"{len(samples)} frames, {n} files, {size / 1e9:.1f} GB -> {dst}")


if __name__ == "__main__":
    main()
