"""Re-run PEGG-Net on archived runs and compare with the grasps they recorded.

    source /opt/ros/humble/setup.bash
    python tests/replay_peggnet.py <run_dir> [<run_dir> ...]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RG_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "ros/remake_grasp"))

from remake_grasp.peggnet_core import FITTED_OFFSET_SCHEDULE, GraspConfig, PeggNet  # noqa: E402
from remake_grasp.peggnet_node import PeggnetNode  # noqa: E402

KEYS_EXACT = ("label", "recommendation", "accepted", "aperture_source", "z_centre_measured")
KEYS_CLOSE = ("quality", "angle_rad", "aperture_m", "aperture_raw_m", "z_m",
              "local_width_m", "local_feature_width_m")


def node_for(settings, engine_cache={}):
    """A PeggnetNode without ROS: just the attributes _infer needs."""
    cfg = GraspConfig(**{k: settings[k] for k in GraspConfig.__dataclass_fields__ if k in settings
                         and k != "aperture_offset_schedule"},
                      aperture_offset_schedule=FITTED_OFFSET_SCHEDULE if settings["aperture_offset_schedule"] else ())
    key = json.dumps(cfg.as_dict(), sort_keys=True)
    if key not in engine_cache:
        engine_cache[key] = PeggNet(config=cfg)
    n = PeggnetNode.__new__(PeggnetNode)
    n.cfg, n.engine = cfg, engine_cache[key]
    n.measure_depth_name = settings.get("measure_depth_name", "depth_aligned_mm.npy")
    n.max_aperture_m = settings.get("max_aperture_m", 0.07)
    n.width_tol_m = 0.010
    return n


def main(runs):
    all_ok = True
    for run in map(Path, runs):
        stages = json.loads((run / "run_config.json").read_text())["stages"]
        if "peggnet" not in stages:
            print(f"skip {run.name}: no PEGG-Net stage in this run (the trials used SBG)")
            continue
        rc = stages["peggnet"]["settings"]
        ref = json.loads((run / "grasps_peggnet.json").read_text())
        depth_name = ref.get("depth_source", "remake_depth_mm.npy")
        node = node_for(rc)
        node.live_dir = run
        depth = np.load(run / depth_name).astype(np.float32) / 1000.0
        masks = np.load(run / "sam3_masks_all.npy").astype(np.int32)
        K = np.load(run / "color_intrinsics.npy")
        got, _ = node._infer(depth, masks, K)
        ok = len(got) == len(ref["grasps"])
        worst = 0.0
        for g, r in zip(got, ref["grasps"]):
            for k in KEYS_EXACT:
                if g.get(k) != r.get(k):
                    ok = False
                    print(f"   {k}: got {g.get(k)} ref {r.get(k)}")
            for k in KEYS_CLOSE:
                a, b = g.get(k), r.get(k)
                if (a is None) != (b is None):
                    ok = False
                    print(f"   {k}: got {a} ref {b}")
                elif a is not None:
                    worst = max(worst, abs(a - b))
            worst = max(worst, max(abs(x - y) for x, y in zip(g["center_px"], r["center_px"])))
        ok &= worst < 1e-3
        all_ok &= ok
        print(f"{'OK ' if ok else 'BAD'} {run.name}: {len(got)} grasps, max abs diff {worst:.1e}, "
              f"recs {[g['recommendation'] for g in got]}")
    print("PASS" if all_ok else "FAIL")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
