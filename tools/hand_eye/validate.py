"""Judge an ee_to_camera on data it was NOT fitted to: ordinary pipeline captures.

    python3 tools/hand_eye/validate.py RESULT.json RUNS_DIR

RUNS_DIR is the pipeline's runs/ folder; the captures named below are the rig's calibration captures,
which are not part of this repository.

1. The table: the raw L515 table points of every capture with a known pose should land on
   z = table_z (the robot measured it by touching) and be level, from every viewpoint.
2. The mug: the survey saw one stationary mug from viewpoints 1, 3, 4 and 5; its
   rim circle (completed depth, top 12 mm, circle fit) should resolve to one place.
"""
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "nuc", "tools"))
import replay_plan as R  # noqa: E402

RUNS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("runs")
SURVEY = ["20260929_223844_auto", "20260929_223921_auto", "20260929_224011_auto", "20260929_224102_auto"]


def table_check(e2c, cfg):
    T_ec = R.mat((e2c[:3], e2c[3:7]))
    rows = []
    for d in sorted(RUNS.glob("2026093*_auto")) + sorted(RUNS.glob("20260929_2*_auto")):
        ct = d / "capture_tool.json"
        if not ct.exists() or not (d / "depth_aligned_mm.npy").exists():
            continue
        c = json.load(open(ct))
        T = R.mat((c["position"], c["orientation_xyzw"])).dot(T_ec)
        K = np.load(d / "color_intrinsics.npy")
        dep = np.load(d / "depth_aligned_mm.npy").astype(float)
        mask = np.zeros(dep.shape, bool)
        if (d / "sam3_masks.npy").exists():
            m = np.load(d / "sam3_masks.npy")
            mask = (m.any(0) if m.ndim == 3 else m).astype(bool)
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
        v, u = np.nonzero((dep > 0) & ~mask)
        sel = np.random.RandomState(0).choice(len(v), min(4000, len(v)), replace=False)
        v, u = v[sel], u[sel]
        z = dep[v, u] / 1000.0
        b = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1).dot(T[:3, :3].T) + T[:3, 3]
        b = b[np.abs(b[:, 2] - cfg.table_z) < 0.04]
        if len(b) < 500:
            continue
        A = np.c_[b[:, 0], b[:, 1], np.ones(len(b))]
        co = np.linalg.lstsq(A, b[:, 2], rcond=None)[0]
        rows.append((np.median(b[:, 2] - cfg.table_z) * 1e3, math.degrees(math.atan(math.hypot(co[0], co[1])))))
    h = np.array([r[0] for r in rows])
    t = np.array([r[1] for r in rows])
    return len(rows), float(np.median(np.abs(h))), float(np.abs(h).max()), float(np.median(t)), float(t.max())


def rim_check(e2c, cfg):
    cs = []
    for run in SURVEY:
        d = RUNS / run
        c = json.load(open(d / "capture_tool.json"))
        T = R.mat((c["position"], c["orientation_xyzw"])).dot(R.mat((e2c[:3], e2c[3:7])))
        K = np.load(d / "color_intrinsics.npy")
        m = np.load(d / "sam3_masks.npy")
        m = (m.any(0) if m.ndim == 3 else m).astype(bool)
        dep = np.load(d / "remake_depth_mm.npy").astype(float)
        v, u = np.nonzero(m & (dep > 0))
        z = dep[v, u] / 1000.0
        b = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1).dot(T[:3, :3].T) + T[:3, 3]
        top = np.percentile(b[:, 2], 97)
        ring = b[b[:, 2] > top - 0.012]
        A = np.c_[2 * ring[:, 0], 2 * ring[:, 1], np.ones(len(ring))]
        sol = np.linalg.lstsq(A, (ring[:, :2] ** 2).sum(1), rcond=None)[0]
        cs.append(sol[:2])
    cs = np.array(cs)
    dev = np.linalg.norm(cs - cs.mean(0), axis=1)
    return 1e3 * math.sqrt((dev ** 2).mean()), 1e3 * dev.max()


def main(result):
    cfg, ee_cam, _ = R.load_config()
    new = json.load(open(result))["ee_to_camera"]
    for label, e2c in (("current", list(ee_cam)), ("calibrated", new)):
        n, hmed, hmax, tmed, tmax = table_check(e2c, cfg)
        rrms, rmax = rim_check(e2c, cfg)
        print("%-10s table over %d captures: height |median| %.1f mm (max %.1f), tilt median %.2f deg (max %.2f)"
              " | mug rim from 4 viewpoints: spread rms %.1f mm (max %.1f)" % (label, n, hmed, hmax, tmed, tmax, rrms, rmax))


if __name__ == "__main__":
    main(sys.argv[1])
