"""Re-run SBG on an archived run and compare with the grasps it recorded.

    python tests/replay_sbg.py <run_dir> [--compiled] [--stub-open3d]

--compiled     use the compiled pointnet2._ext instead of pointnet2_torch (reference)
--stub-open3d  force the numpy voxel_down_sample even if open3d is installed
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RG_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "ros/remake_grasp"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--compiled", action="store_true")
    ap.add_argument("--stub-open3d", action="store_true")
    a = ap.parse_args()
    run = Path(a.run_dir)

    from remake_grasp import stubs
    if a.stub_open3d:
        sys.modules["open3d"] = stubs._open3d()
    from remake_grasp.sbg_core import ScaleBalancedGrasp, SbgConfig

    rc = json.loads((run / "run_config.json").read_text())["stages"]["sbg"]["settings"]
    depth_name = rc.get("depth_name", "remake_depth_mm.npy")
    # Every recorded SbgConfig field, so profile settings added later (approach cone, empty check) replay too.
    fields = {f.name for f in dataclasses.fields(SbgConfig)}
    cfg = SbgConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in rc.items() if k in fields})
    t0 = time.time()
    eng = ScaleBalancedGrasp(config=cfg, compiled_pointnet2=a.compiled)
    print(f"load+warm-up {time.time() - t0:.1f}s  (pointnet2: {'compiled' if a.compiled else 'torch'})")

    depth = np.load(run / depth_name).astype(np.float32) / 1000.0
    K = np.load(run / "color_intrinsics.npy")
    rgb = cv2.imread(str(run / "rgb.png"))[:, :, ::-1]
    cone = json.loads((run / "grasps_sbg.json").read_text()).get("approach_cone") or {}
    axis = cone.get("axis_camera")   # the cone's "down", as the live node computed it from the arm's pose
    times = []
    for _ in range(3):
        t0 = time.time()
        grasps, info = eng.grasps(depth, K, rgb, approach_axis=axis)
        times.append(time.time() - t0)
    got = info["array"].astype(np.float32)
    ref = np.load(run / "grasps_sbg.npy")
    print(f"grasps() {np.median(times):.2f}s median  {info['timings_s']}")
    print(f"rows: got {len(got)} ref {len(ref)}")
    n = min(len(got), len(ref))
    if n:
        err = np.abs(got[:n] - ref[:n]).max()
        same_order = np.allclose(got[:10, 13:16], ref[:10, 13:16], atol=1e-4)
        print(f"max abs diff {err:.2e}; top-10 translations {'match' if same_order else 'DIFFER'}")
        ok = len(got) == len(ref) and err < 1e-4
    else:
        ok = len(got) == len(ref)
    ref_json = json.loads((run / "grasps_sbg.json").read_text())["grasps"]
    acc_ok = [g["accepted"] for g in grasps] == [g["accepted"] for g in ref_json]
    print(f"accepted flags {'match' if acc_ok else 'DIFFER'}")
    print("PASS" if ok and acc_ok else "FAIL")
    return 0 if ok and acc_ok else 1


if __name__ == "__main__":
    sys.exit(main())
