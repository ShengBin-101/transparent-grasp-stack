"""Height-above-the-table maps for the side-view analysis. Reads files only (in the container: cv2, numpy).

    python3 fig.py views OUT.jpg RUN=TITLE ...            photo | L515 | ReMake composite, one row per run
    python3 fig.py conditions OUT.jpg PAIRED_DIR RUN=TITLE ...   RAW | BASELINE | ADAPTED | ADAPTED object-only fill
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RUNS = os.environ.get("RG_RUNS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs"))
from sensor import TABLE_Z, cam_pose  # noqa: E402

W, H = 400, 225


def height_map(depth_mm, K, Rc, pc):
    h_, w_ = depth_mm.shape
    v, u = np.mgrid[0:h_, 0:w_]
    d = depth_mm.astype(np.float32) / 1000
    P = np.stack([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d], -1) @ Rc.T + pc
    h = (P[..., 2] - TABLE_Z) * 1000
    h[d <= 0] = np.nan
    return h


def colour(h):
    """0-200 mm above the table on the turbo scale; magenta where there is no depth."""
    img = np.full(h.shape + (3,), (255, 0, 255), np.uint8)
    ok = np.isfinite(h)
    x = np.clip(np.nan_to_num(h) / 200.0, 0, 1)
    c = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[ok] = c[ok]
    return img


def tile(img, label):
    t = cv2.resize(img, (W, H))
    cv2.rectangle(t, (0, 0), (W, 22), (0, 0, 0), -1)
    cv2.putText(t, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return t


def scale_bar(width):
    bar = np.zeros((26, width, 3), np.uint8)
    bar[6:20, 40:440] = cv2.applyColorMap(np.linspace(0, 255, 400).astype(np.uint8)[None].repeat(14, 0), cv2.COLORMAP_TURBO)
    cv2.putText(bar, "0", (20, 18), 0, 0.45, (255, 255, 255), 1)
    cv2.putText(bar, "200 mm above the table   (magenta: no depth)", (450, 18), 0, 0.45, (255, 255, 255), 1)
    return bar


def views(out, specs):
    rows = []
    for run, title in specs:
        r = os.path.join(RUNS, run)
        Rc, pc = cam_pose(r)
        K = np.load(r + "/color_intrinsics.npy")
        rows.append(np.hstack([tile(cv2.imread(r + "/rgb.png"), title),
                               tile(colour(height_map(np.load(r + "/depth_aligned_mm.npy"), K, Rc, pc)), "L515"),
                               tile(colour(height_map(np.load(r + "/remake_depth_mm.npy"), K, Rc, pc)), "ReMake composite, as run")]))
    cv2.imwrite(out, np.vstack(rows + [scale_bar(rows[0].shape[1])]))


def conditions(out, paired, specs):
    rows = []
    for run, title in specs:
        tiles = []
        for c, lab in (("RAW", "RAW (L515)"), ("BASELINE", "BASELINE"), ("ADAPTED", "ADAPTED"),
                       ("ADAPTED-OBJECT-FILL", "ADAPTED, object-only fill")):
            d = f"{paired}/{run}/{c}"
            Rc, pc = cam_pose(d)
            K = np.load(d + "/color_intrinsics.npy")
            depth = np.load(d + ("/depth_aligned_mm.npy" if c == "RAW" else "/remake_depth_mm.npy"))
            tiles.append(tile(colour(height_map(depth, K, Rc, pc)), f"{title}: {lab}"))
        rows.append(np.hstack(tiles))
    cv2.imwrite(out, np.vstack(rows + [scale_bar(rows[0].shape[1])]))


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "views":
        views(out, [tuple(a.split("=", 1)) for a in sys.argv[3:]])
    else:
        conditions(out, sys.argv[3], [tuple(a.split("=", 1)) for a in sys.argv[4:]])
