"""Re-segment archived frames with the SAM 3 node and compare masks (IoU) with the archive.

    source /opt/ros/humble/setup.bash
    python tests/replay_sam3.py <run_dir> [<run_dir> ...]
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import rclpy

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RG_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "ros/remake_grasp"))
from remake_grasp.sam3_node import Sam3Node  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402


def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / max((a | b).sum(), 1)


def main(runs):
    live = Path(tempfile.mkdtemp())
    rclpy.init(args=["--ros-args", "-p", f"live_dir:={live}", "-p", "threshold:=0.4"])
    node = Sam3Node()
    ok_all = True
    for run in map(Path, runs):
        shutil.copy2(run / "rgb.png", live / "rgb.png")
        for stem, meta, handler in (("sam3_masks", "sam3_meta.json", node._handle),
                                    ("sam3_masks_all", "sam3_meta_all.json", node._handle_all)):
            if not (run / f"{stem}.npy").exists():
                continue
            prompt = json.loads((run / meta).read_text())["prompt"]
            if stem == "sam3_masks_all":
                (live / "sam3_grasp_prompt.txt").write_text(prompt)
            else:
                node.set_parameters([rclpy.Parameter("prompt", value=prompt)])
            res = handler(None, Trigger.Response())
            got, ref = np.load(live / f"{stem}.npy"), np.load(run / f"{stem}.npy")
            m = iou(got, ref)
            ok = res.success and m >= 0.95 and got.max() == ref.max()
            ok_all &= ok
            print(f"{'OK ' if ok else 'BAD'} {run.name} {stem}: {got.max()} vs {ref.max()} instances, "
                  f"IoU {m:.4f}  ({prompt!r})")
    node.destroy_node()
    rclpy.shutdown()
    print("PASS" if ok_all else "FAIL")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
