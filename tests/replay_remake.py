"""Re-run ReMake on archived runs and compare with the depth they recorded.

    source /opt/ros/humble/setup.bash
    python tests/replay_remake.py <run_dir> [<run_dir> ...]
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RG_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "ros/remake_grasp"))

from remake_grasp.align import composite_depth  # noqa: E402
from remake_grasp.common import CKPT_DIR  # noqa: E402
from remake_grasp.remake_node import RemakeNode, load_remake  # noqa: E402


def main(runs):
    node = RemakeNode.__new__(RemakeNode)
    node.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {}
    node.mask_dilate = 5
    ok_all = True
    for run in map(Path, runs):
        dil = json.loads((run / "run_config.json").read_text())["stages"]["depth_remake"]["settings"]
        node.mask_dilate, node.mask_name, node.live_dir = dil["mask_dilate_px"], dil["mask_name"], run
        ck = dil.get("checkpoint", "remake_official.tar")               # the weights this run used
        if ck not in loaded:
            loaded[ck] = load_remake(CKPT_DIR / ck, CKPT_DIR / "depth_anything_v2_vits.pth", node.device)[:3]
        node.model, node.reldepth, node.process_data = loaded[ck]
        node.rel_rgb_fix = bool(dil.get("rel_rgb_fix", False))
        fill = bool(dil.get("fill_holes", True))   # a run that does not record it is from before the setting and filled every hole
        bgr = cv2.imread(str(run / "rgb.png"))
        depth = np.nan_to_num(np.load(run / "depth_aligned_mm.npy").astype(np.float32))
        K = np.load(run / "color_intrinsics.npy").astype(np.float32)
        mask, tight, _ = node._mask(depth.shape)
        net = node.complete(bgr, depth, K, mask)
        cov = np.load(run / "depth_coverage.npy").astype(bool)
        out, _ = composite_depth(net, depth, tight, valid_region=cov, fill_holes=fill)
        ref_net = np.load(run / "remake_depth_net_mm.npy")
        ref = np.load(run / "remake_depth_mm.npy")
        e_net = np.abs(net - ref_net)
        e_in = np.abs(out - ref)[tight]
        outside_same = np.array_equal(out[~tight], ref[~tight])
        ok = outside_same and np.median(e_in) < 1.0 and np.percentile(e_in, 99) < 5.0
        ok_all &= ok
        print(f"{'OK ' if ok else 'BAD'} {run.name} ({ck}, rel_rgb_fix {node.rel_rgb_fix}, fill_holes {fill}): net max {e_net.max():.3f}mm  inside mask median "
              f"{np.median(e_in):.3f} p99 {np.percentile(e_in, 99):.3f} max {e_in.max():.3f}mm  "
              f"outside mask identical: {outside_same}")
    print("PASS" if ok_all else "FAIL")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
