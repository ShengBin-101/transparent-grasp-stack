#!/usr/bin/env python3
"""Slide figures and verified numbers for one view of the robot trials. Reads the trial records; writes only under --out.

    python3 trials/make_trial_figures.py --view V1 --out figures/trials_v1 [--clips <folder of trial videos>]

Reads (never writes): trials/trial_runs.csv, runs/trial/<run>/ (grasps_sbg.json, run_config.json, capture_tool.json,
color_intrinsics.npy, the depth SBG read, pick_*.json), ros/remake_grasp/config/eye_in_hand.yaml, and the SBG source for the overlay legend.
Filter: view == --view and status starting with "counted".

Writes, with <v> = the view in lower case:
  <v>_success.png/.pdf      held rate by condition x object kind, as percentages with their counts
  <v>_stats.md               k/n and %; outcome counts; per-object held table
  <v>_outcomes.csv           tidy counts: condition, object_kind, outcome, count
  overlay_legend.md          what grasps_sbg_overlay.png draws, quoted from the code with file:line
  bowl_cloud_<a>.png         the cloud SBG planned on, one per condition, same 3-D viewpoint, accepted grasps drawn
                             (open3d, rendered off-screen)
  bowl_cloud_3up.png         the three side by side, labelled
  bowl_bowl_outcomes.md      the bowl's outcome in every attempt of the view, with the planner's reasons for a refusal
  README.md                  every file, the numbers in one place, and checks against the numbers given for V1
The "bowl" files use --cloud-object (default T3) and its --cloud-attempt (default 1) under each condition.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

try:
    import open3d  # noqa: F401
except ImportError:
    sys.exit("open3d is needed for the clouds: pip install open3d (in a virtual environment of its own, if the "
             "system Python also runs ROS)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONDS = ["RAW", "BASELINE", "ADAPTED"]                   # as trial_runs.csv records them

COLOURS = {"RAW": "#2B79D6", "BASELINE": "#EC6A37", "ADAPTED": "#1BAF7A"}
BG, INK, INK2, GRID = "#FCFCFB", "#16202a", "#52514e", "#e4e3df"
KINDS = OrderedDict([("transparent", "T"), ("opaque", "O")])
OUTCOME_NAMES = OrderedDict([("held", "held"), ("N", "no executable grasp"), ("P", "planner refused"), ("W", "wrong depth"),
                             ("C", "collision"), ("S", "slip"), ("E", "execution fault"), ("X", "other")])
TABLE_Z = -0.041                     # m in fr3_link0, measured by guarded contact (nuc/.../pose_commander.yaml)
SPOT = (0.50, 0.00)                  # the object spot in fr3_link0
# The numbers the slides were drafted with (V1), checked against the records when --view V1.
CLAIMS_V1 = {
    "held": {("RAW", "transparent"): (1, 12), ("BASELINE", "transparent"): (8, 12), ("ADAPTED", "transparent"): (7, 12),
             ("RAW", "opaque"): (4, 6), ("BASELINE", "opaque"): (3, 6), ("ADAPTED", "opaque"): (3, 6)},
    "outcomes": {("RAW", "transparent"): {"held": 1, "N": 6, "W": 3, "P": 2},
                 ("BASELINE", "transparent"): {"held": 8, "N": 3, "C": 1},
                 ("ADAPTED", "transparent"): {"held": 7, "P": 3, "N": 1, "S": 1},
                 ("RAW", "opaque"): {"held": 4, "P": 2}, ("BASELINE", "opaque"): {"held": 3, "N": 2, "P": 1},
                 ("ADAPTED", "opaque"): {"held": 3, "P": 2, "C": 1}},
    "sbg": {"a8": (0, 0), "a22": (25, 7), "a38": (15, 9)},
}


# ---------------------------------------------------------------- records and statistics

def load(view):
    path = ROOT / "trials/trial_runs.csv"
    rows = [r for r in csv.DictReader(open(path)) if r["view"] == view and r["status"].startswith("counted")]
    if not rows:
        sys.exit(f"no counted attempts for view {view} in {path}")
    return path, rows


def dash(x):
    """A count SBG did not compute (nothing to thin or check) prints as a dash."""
    return "–" if x is None else x


def kind(r):
    return "transparent" if r["object"].startswith("T") else "opaque"


def outcome(r):
    return "held" if r["held"] == "Y" else (r["fail"] or "N")


def tallies(rows):
    held = {}
    for c in CONDS:
        for kd in list(KINDS) + ["all"]:
            sel = [r for r in rows if r["condition"] == c and (kd == "all" or kind(r) == kd)]
            held[(c, kd)] = (sum(r["held"] == "Y" for r in sel), len(sel))
    outs = {(c, kd): Counter(outcome(r) for r in rows if r["condition"] == c and kind(r) == kd) for c in CONDS for kd in KINDS}
    return held, outs


# ---------------------------------------------------------------- 1. the chart

def chart(held, view, out):
    """Held rate per condition and object kind, as percentages with their counts."""
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]})
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    w, gap = 0.25, 0.03
    for gi, kd in enumerate(KINDS):
        for ci, c in enumerate(CONDS):
            k, n = held[(c, kd)]
            p = 100 * k / n
            x = gi + (ci - 1) * (w + gap)
            ax.bar(x, p, width=w, color=COLOURS[c], label=c if gi == 0 else None, zorder=3)
            ax.text(x, p + 2, f"{k}/{n}", ha="center", va="bottom", fontsize=26, color=INK2, zorder=5)
            ax.text(x, p + 7.5, f"{p:.0f} %", ha="center", va="bottom", fontsize=32, color=INK, fontweight="bold", zorder=5)
    n_t, n_o = held[(CONDS[0], "transparent")][1], held[(CONDS[0], "opaque")][1]
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Transparent objects\n({n_t} attempts each)", f"Opaque objects\n({n_o} attempts each)"],
                       fontsize=30, color=INK)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{t} %" for t in (0, 25, 50, 75, 100)], fontsize=28, color=INK2)
    ax.set_ylabel("Lifted and held for 3 s", fontsize=30, color=INK2, labelpad=16)
    ax.yaxis.grid(True, color=GRID, linewidth=1.5, zorder=0)
    ax.set_axisbelow(True)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color("#bdbcb6")
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(axis="both", length=0, pad=10)
    ax.legend(ncol=3, frameon=False, fontsize=30, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.2,
              columnspacing=2.0, borderaxespad=0.3)
    fig.tight_layout()
    base = out / f"{view.lower()}_success"
    fig.savefig(str(base) + ".png", facecolor=BG)
    fig.savefig(str(base) + ".pdf", facecolor=BG)
    plt.close(fig)
    return [base.name + ".png", base.name + ".pdf"]


# ---------------------------------------------------------------- 2-3. stats and tidy counts

def stats_md(rows, held, outs, view, src, out):
    L = [f"# Trial statistics, view {view}", "",
         f"Source: `{Path(src).relative_to(ROOT) if Path(src).is_absolute() else src}`. Filter: `view == \"{view}\"` and `status` starting with `\"counted\"` "
         f"({len(rows)} attempts). Held = `held == \"Y\"`; any other attempt counts under its `fail` code.", "",
         "## Held, as percentages with their counts", "",
         "| condition | object kind | held k/n | % |", "|---|---|---|---|"]
    for c in CONDS:
        for kd in list(KINDS) + ["all"]:
            k, n = held[(c, kd)]
            L.append(f"| {c} | {kd} | {k}/{n} | {100 * k / n:.0f} % |")
    L += ["", "## Outcomes, recomputed from the records", "",
          "| condition | object kind | " + " | ".join(OUTCOME_NAMES) + " |", "|---|---|" + "---|" * len(OUTCOME_NAMES)]
    for c in CONDS:
        for kd in KINDS:
            L.append(f"| {c} | {kd} | " + " | ".join(str(outs[(c, kd)].get(o, 0)) for o in OUTCOME_NAMES) + " |")
    L += ["", "Codes: " + "; ".join(f"{k} {v}" for k, v in OUTCOME_NAMES.items() if k != "held") + ".", "",
          "## Per object: held of 3 (attempts in order)", "",
          "| object | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    objs = sorted({(r["object"], r["object_name"]) for r in rows}, key=lambda o: (o[0][0] != "T", o[0]))
    for o, name in objs:
        cells = []
        for c in CONDS:
            sel = sorted([r for r in rows if r["object"] == o and r["condition"] == c], key=lambda r: int(r["object_attempt"]))
            cells.append(f"{sum(r['held'] == 'Y' for r in sel)}/{len(sel)} ({' '.join('✓' if r['held'] == 'Y' else (r['fail'] or 'N') for r in sel)})")
        L.append(f"| {o} {name} | " + " | ".join(cells) + " |")
    (out / f"{view.lower()}_stats.md").write_text("\n".join(L) + "\n")
    with open(out / f"{view.lower()}_outcomes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "object_kind", "outcome", "count"])
        for c in CONDS:
            for kd in KINDS:
                for o in OUTCOME_NAMES:
                    if outs[(c, kd)].get(o, 0):
                        w.writerow([c, kd, o, outs[(c, kd)][o]])
    return [f"{view.lower()}_stats.md", f"{view.lower()}_outcomes.csv"]


# ---------------------------------------------------------------- 4. the overlay legend, from the code

def quote(path, pattern, after=0):
    lines = Path(path).read_text().splitlines()
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            rel = Path(path).relative_to(ROOT)
            return "\n".join(f"{rel}:{j + 1}: {lines[j]}" for j in range(i, min(len(lines), i + 1 + after)))
    return f"(pattern {pattern!r} not found in {path})"


def overlay_md(rows, out):
    core = ROOT / "ros/remake_grasp/remake_grasp/sbg_core.py"
    node = ROOT / "ros/remake_grasp/remake_grasp/sbg_node.py"
    s = json.load(open(ROOT / rows[0]["folder"] / "run_config.json"))["stages"]["sbg"]["settings"]
    L = ["# What `grasps_sbg_overlay.png` shows", "",
         "Each run folder keeps it as `grasps_sbg_overlay.png` (a `.jpg` only in exported copies). It is drawn by "
         "`draw_grasps()` in `sbg_core.py`, called from the SBG node right after detection, on the run's colour photo.", "",
         "```", quote(node, r"overlay = draw_grasps\(canvas, arr, K, accepted\)", 1), "```", "",
         "## Which grasps are drawn: the header \"top N (M accepted)\"", "",
         f"- **N** is the length of the list SBG returns: its raw grasps, sorted by score, thinned by non-maximum "
         f"suppression, ranked collision-free first and then by score, cut to `top_k` (= {s.get('top_k')} in these runs). "
         "So N = min(`n_after_nms`, top_k) in `grasps_sbg.json`, **not** the raw count `n_raw`.",
         "- **M** is how many of those N are accepted: no collision, not empty, within the hand's aperture and not "
         f"saturated, and at or above the score gate (`score_gate` = {s.get('score_gate')}, `max_aperture_m` = "
         f"{s.get('max_aperture_m')}, `empty_check` = {s.get('empty_check')}, approach cone {s.get('approach_cone_deg')}°).", "",
         "```", quote(core, r"top = arr\[:cfg.top_k\]"), quote(core, r"if col:", 7), quote(core, r'"accepted": not reasons'),
         quote(core, r'n_ok = n if accepted is None'), quote(core, r'cv2.putText\(img, f"Scale-Balanced-Grasp', 1), "```", "",
         "## Colours", "",
         "- **Red → orange → yellow → green: accepted grasps, coloured by SBG's score** (0 red, 0.5 yellow, 1 green), "
         "2 px lines. Red or orange is an *accepted* grasp with a low score, not a rejected one.",
         "- **Grey (128, 128, 128), 1 px: rejected grasps** (collision, empty, too wide or saturated, low score).",
         "- **Cyan text (BGR 255, 255, 0): labels on the first 10 accepted grasps in list order**, as "
         "`index:score widthmm`, at the grasp's centre. The width is the network's raw width (`width_raw_m`, which "
         "includes SBG's 1.2 clearance), not the object width `width_m`.",
         "- **White text, top left:** the header above.", "",
         "```", quote(core, r"def score_colour_bgr", 5), quote(core, r"labelled = \["), quote(core, r"colour = score_colour_bgr", 0),
         quote(core, r"colour, 2 if ok else 1"), quote(core, r'cv2.putText\(img, f"\{i\}:', 2), "```", "",
         "## The wireframe", "",
         "Two finger lines from the fingertips (the grasp's depth along the approach axis) back to 2 cm behind the "
         "grasp point, a bar joining them, and a 4 cm tail behind the bar; opened to the raw width.", "",
         "```", quote(core, r"def gripper_segments", 10), "```", ""]
    (out / "overlay_legend.md").write_text("\n".join(L) + "\n")
    return ["overlay_legend.md"]


# ---------------------------------------------------------------- 5. the cloud SBG planned on, in the robot's frame

def quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def camera_in_base(run_dir):
    cfg = yaml.safe_load(open(ROOT / "ros/remake_grasp/config/eye_in_hand.yaml"))
    e2c = cfg["ee_to_camera"]
    t = json.load(open(run_dir / "capture_tool.json"))
    Rb = quat_mat(t["orientation_xyzw"])
    return Rb @ quat_mat(e2c[3:]), Rb @ np.array(e2c[:3]) + np.array(t["position"])


def planned_cloud(run_dir):
    """The depth SBG read (its settings' depth_name), back-projected with the run's intrinsics in SBG's z range."""
    s = json.load(open(run_dir / "run_config.json"))["stages"]["sbg"]["settings"]
    need = [run_dir / s["depth_name"], run_dir / "color_intrinsics.npy", run_dir / "capture_tool.json"]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        return None, f"missing {', '.join(missing)}"
    d = np.load(need[0]).astype(np.float32) / 1000.0
    K = np.load(need[1])
    v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    ok = np.isfinite(d) & (d > s.get("z_min_m", 0.15)) & (d < s.get("z_max_m", 1.5))
    z = d[ok]
    P = np.stack([(u[ok] - K[0, 2]) * z / K[0, 0], (v[ok] - K[1, 2]) * z / K[1, 1], z], 1)
    Rc, pc = camera_in_base(run_dir)
    return {"points": P @ Rc.T + pc, "Rc": Rc, "pc": pc, "depth_name": s["depth_name"]}, None


def grasp_lines(g, Rc, pc):
    """SBG's wireframe (sbg_core.gripper_segments) for one grasp dict, in the base frame."""
    R, t = np.array(g["rotation"]), np.array(g["translation_m"])
    w2, d = g["width_raw_m"] / 2.0, g["depth_m"]
    local = np.array([[[d, -w2, 0], [-0.02, -w2, 0]], [[d, w2, 0], [-0.02, w2, 0]],
                      [[-0.02, -w2, 0], [-0.02, w2, 0]], [[-0.02, 0, 0], [-0.06, 0, 0]]])
    cam = local @ R.T + t
    return cam @ Rc.T + pc


def render_cloud(cloud, grasps, path, crop, elev=35.0, azim=-150.0, size=(1600, 1200)):
    """open3d off-screen render: points coloured by height above the table, SBG's accepted grasps as magenta lines.
    The camera sits `elev` degrees above the table, at azimuth `azim` (degrees from +x towards +y, robot frame),
    looking at the crop's centre on the table, so every render shares one viewpoint."""
    import open3d as o3d
    from open3d.visualization import rendering
    lo, hi = np.array([c[0] for c in crop]), np.array([c[1] for c in crop])
    P = cloud["points"]
    P = P[np.all((P >= lo) & (P <= hi), axis=1)]
    h = (P[:, 2] - TABLE_Z) * 1000
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    pc.colors = o3d.utility.Vector3dVector(matplotlib.cm.turbo(np.clip(h, 0, 120) / 120.0)[:, :3])
    # On the Orin (open3d 0.20, Vulkan) the off-screen renderer returns a black frame for large scenes: above ~65,000
    # points at 800x600, and already at ~34,000 points at 1600x1200. So the cloud is thinned to a 1.5 mm voxel grid
    # (still covering the surface), and the full-size image is rendered as 800x600 tiles of one virtual camera.
    pc = pc.voxel_down_sample(0.0015)
    if len(pc.points) > 60000:
        pc = pc.random_down_sample(60000 / len(pc.points))
    W, H = size
    tw, th = W // 2, H // 2
    r = rendering.OffscreenRenderer(tw, th)
    bg = [int(BG[i:i + 2], 16) / 255 for i in (1, 3, 5)] + [1.0]
    r.scene.set_background(bg)
    r.scene.view.set_post_processing(False)   # no tone mapping: the colours and background come out as given
    mp = rendering.MaterialRecord()
    mp.shader, mp.point_size = "defaultUnlit", 6.0
    r.scene.add_geometry("cloud", pc, mp)
    ml = rendering.MaterialRecord()
    ml.shader, ml.line_width = "unlitLine", 5.0     # colour per vertex: a material colour renders darker here
    for i, g in enumerate(grasps):
        segs = grasp_lines(g, cloud["Rc"], cloud["pc"])
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(segs.reshape(-1, 3).astype(np.float64)),
                                  o3d.utility.Vector2iVector(np.arange(2 * len(segs)).reshape(-1, 2)))
        ls.colors = o3d.utility.Vector3dVector(np.tile([0xE0 / 255, 0x21 / 255, 0x8A / 255], (len(segs), 1)))
        r.scene.add_geometry(f"grasp{i}", ls, ml)
    centre = np.array([SPOT[0], SPOT[1], TABLE_Z + 0.03])
    e, a = np.radians(elev), np.radians(azim)
    eye = centre + 0.50 * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    # one pinhole camera for the whole W x H image (30 deg vertical field of view), world -> camera with x right,
    # y down, z forward; each tile is that camera with its principal point shifted
    fwd = (centre - eye) / np.linalg.norm(centre - eye)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    Rwc = np.stack([right, down, fwd])
    ext = np.eye(4)
    ext[:3, :3], ext[:3, 3] = Rwc, -Rwc @ eye
    f = (H / 2) / np.tan(np.radians(30.0) / 2)
    full = np.zeros((H, W, 3), np.uint8)
    r.render_to_image()                       # warm-up: the first frames after start-up can come back black
    for ty in range(2):
        for tx in range(2):
            K = np.array([[f, 0, W / 2 - tx * tw], [0, f, H / 2 - ty * th], [0, 0, 1]], float)
            r.setup_camera(K, ext, tw, th)
            for _ in range(6):                # a solid black tile is a failed frame (the background is never black)
                tile = np.asarray(r.render_to_image())
                if tile.max() > 0:
                    break
            else:
                raise RuntimeError(f"open3d kept rendering black tiles for {path.name}")
            full[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw] = tile
    if len(np.unique(full.reshape(-1, 3)[::97], axis=0)) < 2:
        raise RuntimeError(f"open3d rendered a blank frame for {path.name}")
    o3d.io.write_image(str(path), o3d.geometry.Image(full))
    del r
    return len(pc.points)


def mask_tile(run_dir, colour, size, crop_w=900):
    """The run's photo around the object spot, with the transparency mask ReMake was given drawn over it."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    rgb = np.asarray(Image.open(run_dir / "rgb.png").convert("RGB")).astype(float)
    H0, W0 = rgb.shape[:2]
    K = np.load(run_dir / "color_intrinsics.npy")
    Rc, pc = camera_in_base(run_dir)
    q = Rc.T @ (np.array([SPOT[0], SPOT[1], TABLE_Z + 0.03]) - pc)          # the spot, in this capture's image
    u, v = K[0, 0] * q[0] / q[2] + K[0, 2], K[1, 1] * q[1] / q[2] + K[1, 2]
    crop_h = int(round(crop_w * size[1] / size[0]))
    x0 = int(np.clip(u - crop_w / 2, 0, W0 - crop_w))
    y0 = int(np.clip(v - crop_h / 2, 0, H0 - crop_h))
    stage = json.load(open(run_dir / "run_config.json"))["stages"].get("depth_remake")
    mname = stage["settings"]["mask_name"] if stage else None
    if mname and (run_dir / mname).exists():
        m = np.load(run_dir / mname)
        m = (m.max(0) if m.ndim == 3 else m) > 0
        col = np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], float)
        rgb[m] = 0.45 * rgb[m] + 0.55 * col
        out8 = np.ascontiguousarray(rgb.astype(np.uint8))
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out8, cs, -1, tuple(int(x) for x in col), 2, cv2.LINE_AA)
        label = f"{'SAM 3' if 'sam3' in mname else 'our segmenter'}: {100 * m.mean():.1f} % of the image"
    else:
        out8 = rgb.astype(np.uint8)
        label = "no mask: the raw depth goes in as it is"
    tile = Image.fromarray(out8).crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize(size, Image.LANCZOS)
    d = ImageDraw.Draw(tile)
    fr = glob.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True)
    f = ImageFont.truetype(fr[0], 54) if fr else ImageFont.load_default()
    tw, th = d.textbbox((0, 0), label, font=f)[2:]
    d.rectangle((0, size[1] - th - 28, tw + 36, size[1]), fill=(20, 26, 32))
    d.text((18, size[1] - th - 16), label, font=f, fill=(255, 255, 255))
    return tile


def mask_panel(rows3, cloud_crops, bgc, path, fpath):
    from PIL import Image, ImageDraw, ImageFont
    cw, chh = cloud_crops[0].size
    gap, head, rt, mrow = 24, 110, 76, 620
    scale = 0.8
    cw2, ch2 = int(cw * scale), int(chh * scale)
    H = head + rt + mrow + gap + rt + ch2
    img = Image.new("RGB", (3 * cw + 2 * gap, H), tuple(int(x) for x in bgc))
    d = ImageDraw.Draw(img)
    fb = ImageFont.truetype(fpath[0], 64) if fpath else ImageFont.load_default()
    fr = glob.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True)
    ft = ImageFont.truetype(fr[0], 56) if fr else ImageFont.load_default()
    d.text((8, head + 8), "The transparency mask ReMake was given", font=ft, fill=INK2)
    d.text((8, head + rt + mrow + gap + 8), "The point cloud the grasp planner received", font=ft, fill=INK2)
    for i, (r, crop) in enumerate(zip(rows3, cloud_crops)):
        c = r["condition"]
        x = i * (cw + gap)
        text = f"{c} · {r['attempt']}"
        tw = d.textbbox((0, 0), text, font=fb)[2]
        d.text((x + (cw - tw) // 2, 18), text, font=fb, fill=COLOURS[c])
        img.paste(mask_tile(ROOT / r["folder"], COLOURS[c], (cw, mrow)), (x, head + rt))
        img.paste(crop.resize((cw2, ch2), Image.LANCZOS), (x + (cw - cw2) // 2, head + rt + mrow + gap + rt))
    img.save(path)


def bowl_section(rows, view, obj, k, out):
    from PIL import Image, ImageDraw, ImageFont
    picks = {}
    for c in CONDS:
        sel = [r for r in rows if r["condition"] == c and r["object"] == obj and int(r["object_attempt"]) == k]
        if sel:
            picks[c] = sel[0]
    crop = ((SPOT[0] - 0.12, SPOT[0] + 0.12), (SPOT[1] - 0.12, SPOT[1] + 0.12), (TABLE_Z - 0.02, TABLE_Z + 0.12))
    made, notes, sbg = [], {}, {}
    for c, r in picks.items():
        run_dir = ROOT / r["folder"]
        s = json.load(open(run_dir / "grasps_sbg.json")) if (run_dir / "grasps_sbg.json").exists() else {}
        sbg[r["attempt"]] = {kk: s.get(kk) for kk in ("n_raw", "n_after_nms", "n_grasps", "n_collision_free", "n_accepted")}
        cloud, why = planned_cloud(run_dir)
        if cloud is None:
            notes[r["attempt"]] = why
            continue
        acc = [g for g in s.get("grasps", []) if g.get("accepted")]
        name = f"{obj.lower() if obj != 'T3' else 'bowl'}_cloud_{r['attempt']}.png"
        n = render_cloud(cloud, acc, out / name, crop)
        notes[r["attempt"]] = f"{cloud['depth_name']}, {n:,} points drawn (1.5 mm voxels of the crop), {len(acc)} accepted grasps drawn"
        made.append((c, r["attempt"], out / name))
    if len(made) == 3:
        ims = [Image.open(p).convert("RGB") for _, _, p in made]
        W, H = ims[0].size
        band, foot = 130, 90
        comp = Image.new("RGB", (3 * W, H + band + foot), ims[0].getpixel((4, 4)))   # the render's own background
        dr = ImageDraw.Draw(comp)
        fpath = glob.glob("/usr/share/fonts/**/DejaVuSans-Bold.ttf", recursive=True)
        fr = glob.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True)
        fb = ImageFont.truetype(fpath[0], 72) if fpath else ImageFont.load_default()
        fs = ImageFont.truetype(fr[0], 40) if fr else ImageFont.load_default()
        for i, ((c, a, _), im) in enumerate(zip(made, ims)):
            comp.paste(im, (i * W, band))
            text = f"{c} · {a}"
            tw = dr.textbbox((0, 0), text, font=fb)[2]
            dr.text((i * W + (W - tw) // 2, 28), text, font=fb, fill=COLOURS[c])
        dr.text((40, H + band + 22), "Points: height above the table, 0 mm (blue) to 120 mm (red). Magenta: SBG's accepted "
                                     "grasps. Same 3-D viewpoint, 35° above the table, robot frame.", font=fs, fill=INK2)
        name3 = f"{'bowl' if obj == 'T3' else obj.lower()}_cloud_3up.png"
        comp.save(out / name3)
        made.append(("3up", "", out / name3))
        # for a slide: every panel cropped to one shared box around what was drawn (same scale and viewpoint)
        bgc = np.array(ims[0].getpixel((4, 4)))
        boxes = []
        for im in ims:
            ys, xs = np.where(np.abs(np.asarray(im).astype(int) - bgc).sum(2) > 24)
            boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        x0, y0 = max(0, min(b[0] for b in boxes) - 30), max(0, min(b[1] for b in boxes) - 30)
        x1, y1 = min(W - 1, max(b[2] for b in boxes) + 30), min(H - 1, max(b[3] for b in boxes) + 30)
        crops = [im.crop((x0, y0, x1, y1)) for im in ims]
        cw, chh, gap, band2 = crops[0].size[0], crops[0].size[1], 24, 110
        slide = Image.new("RGB", (3 * cw + 2 * gap, chh + band2), tuple(int(x) for x in bgc))
        ds = ImageDraw.Draw(slide)
        fb2 = ImageFont.truetype(fpath[0], 64) if fpath else ImageFont.load_default()
        for i, ((c, a, _), cr) in enumerate(zip(made[:3], crops)):
            slide.paste(cr, (i * (cw + gap), band2))
            text = f"{c} · {a}"
            tw = ds.textbbox((0, 0), text, font=fb2)[2]
            ds.text((i * (cw + gap) + (cw - tw) // 2, 18), text, font=fb2, fill=COLOURS[c])
        name_s = name3.replace(".png", "_slide.png")
        slide.save(out / name_s)
        made.append(("3up-slide", "", out / name_s))
        # the masks ReMake was given (top) over the clouds the grasp planner received (bottom), one column per condition
        name_m = name3.replace("_cloud_3up.png", "_masks_clouds.png")
        mask_panel([picks[c] for c, _, _ in made[:3]], crops, bgc, out / name_m, fpath)
        made.append(("masks", "", out / name_m))
    return made, notes, sbg, picks


# ---------------------------------------------------------------- 6. the object's outcome in every attempt

def outcomes_md(rows, view, obj, out):
    sel = sorted([r for r in rows if r["object"] == obj], key=lambda r: (CONDS.index(r["condition"]), int(r["object_attempt"])))
    name = sel[0]["object_name"] if sel else obj
    L = [f"# {obj} {name}: every {view} attempt", "",
         "| attempt | condition | # | outcome | SBG raw → after NMS → accepted | planner (first pick record) |", "|---|---|---|---|---|---|"]
    detail = []
    for r in sel:
        d = ROOT / r["folder"]
        s = json.load(open(d / "grasps_sbg.json")) if (d / "grasps_sbg.json").exists() else {}
        picks = sorted(glob.glob(str(d / "pick_*.json")))
        pk = json.load(open(picks[0])) if picks else None
        summ = (pk.get("plan_summary") or (pk.get("plan_reply") or {}).get("summary_text") or "") if pk else "no pick"
        L.append(f"| {r['attempt']} | {r['condition']} | {r['object_attempt']} | {'held' if r['held'] == 'Y' else r['fail'] or 'N'} | "
                 f"{dash(s.get('n_raw'))} → {dash(s.get('n_after_nms'))} → {dash(s.get('n_accepted'))} | {summ} |")
        if pk and not pk.get("success"):
            detail += [f"## {r['attempt']} ({r['condition']}): why the planner refused", "",
                       f"Run `{r['run']}`, record `{Path(picks[0]).name}` ({len(picks)} pick record(s); the first counts).", "",
                       f"Message: `{pk.get('message')}`", "", "| candidate | category | failed leg | reason |", "|---|---|---|---|"]
            for cnd in (pk.get("plan_reply") or {}).get("candidates", []):
                detail.append(f"| {cnd.get('id')} | {cnd.get('category') or ('feasible' if cnd.get('ok') else '')} | "
                              f"{cnd.get('failed_leg') or ''} | {'; '.join(cnd.get('reasons') or [])} |")
            detail.append("")
    fname = "bowl_bowl_outcomes.md" if obj == "T3" else f"{obj.lower()}_outcomes.md"
    (out / fname).write_text("\n".join(L + [""] + detail) + "\n")
    return fname, sel


# ---------------------------------------------------------------- 7. clips

def find_clips(folder):
    """Videos under `folder` named by an attempt id, trial id or view; none without a folder."""
    hits = []
    for base in ([Path(folder)] if folder else []):
        for p in base.rglob("*"):
            if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm") and not any(
                    x in p.parts for x in ("site-packages", "node_modules", ".cache", "venv")):
                if re.search(r"(^|[^a-z0-9])a\d{1,3}([^0-9]|$)|_V[123]_|trial", p.name, re.I):
                    hits.append(p)
    return hits


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--view", default="V1")
    ap.add_argument("--out", required=True, type=lambda s: Path(os.path.expanduser(s)))
    ap.add_argument("--cloud-object", default="T3")
    ap.add_argument("--cloud-attempt", type=int, default=1)
    ap.add_argument("--clips", default=None, help="a folder of trial videos to list (nothing is searched without it)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    src, rows = load(a.view)
    held, outs = tallies(rows)
    files = chart(held, a.view, a.out)
    files += stats_md(rows, held, outs, a.view, src, a.out)
    files += overlay_md(rows, a.out)
    made, notes, sbg, picks = bowl_section(rows, a.view, a.cloud_object, a.cloud_attempt, a.out)
    files += [p.name for _, _, p in made]
    oname, osel = outcomes_md(rows, a.view, a.cloud_object, a.out)
    files.append(oname)
    clips = find_clips(a.clips)

    # checks against the numbers given for V1
    mism = []
    if a.view == "V1":
        for key, (k, n) in CLAIMS_V1["held"].items():
            if held[key] != (k, n):
                mism.append(f"held {key}: given {k}/{n}, records {held[key][0]}/{held[key][1]}")
        for key, d in CLAIMS_V1["outcomes"].items():
            rec = {o: c for o, c in outs[key].items() if c}
            if rec != d:
                mism.append(f"outcomes {key}: given {d}, records {rec}")
        for att, (n_top, n_acc) in CLAIMS_V1["sbg"].items():
            s = sbg.get(att)
            if s is not None and (s["n_grasps"], s["n_accepted"]) != (n_top, n_acc):
                mism.append(f"SBG {att}: given {n_top} with {n_acc} accepted, records top {s['n_grasps']}, {s['n_accepted']} accepted")
    v = a.view.lower()
    R = [f"# Slide export: robot trials, view {a.view}", "",
         f"Made by `python3 trials/make_trial_figures.py --view {a.view} --out {a.out}` in `{ROOT}`. "
         "Nothing in the trial records was changed.", "",
         "## Paths read", "", f"- Attempts: `{src}` (filter: view == \"{a.view}\", status starting with \"counted\"; {len(rows)} attempts).",
         f"- Run folders: `{ROOT}/runs/trial/<run>/` (the CSV's `folder` column).",
         f"- Hand-eye calibration: `{ROOT}/ros/remake_grasp/config/eye_in_hand.yaml`; table at z = {TABLE_Z} m.",
         f"- Overlay code: `{ROOT}/ros/remake_grasp/remake_grasp/sbg_core.py`, `sbg_node.py`.", "",
         "## Files", "",
         f"- `{v}_success.png` / `.pdf`: held rate by condition × object kind, as percentages with their counts (`k/n` under each %); 1920×1080, no title.",
         f"- `{v}_stats.md`: k/n and %; outcome counts; the per-object table.",
         f"- `{v}_outcomes.csv`: tidy counts (condition, object_kind, outcome, count).",
         "- `overlay_legend.md`: what each colour and label in `grasps_sbg_overlay.png` means, quoted from the code.",
         ]
    for c, att, p in made:
        R.append(f"- `{p.name}`: " + ("the three clouds side by side, labelled." if c == "3up" else
                                     "the same three cropped to one shared box, for a slide." if c == "3up-slide" else
                                     "top: each condition's photo with the transparency mask ReMake was given (SAM 3 for "
                                     "BASELINE, the segmenter for ADAPTED, none for RAW); bottom: the cloud the planner "
                                     "received." if c == "masks" else
                                     f"{c} {att}, the cloud SBG planned on ({notes.get(att)}); 1600×1200."))
    for att, why in notes.items():
        if why.startswith("missing"):
            R.append(f"- Not rendered: {att}, {why}.")
    R += [f"- `{oname}`: the {a.cloud_object} outcome in every {a.view} attempt, with the planner's reasons for each refusal.", "",
          "## The numbers", ""]
    for kd in list(KINDS) + ["all"]:
        R.append(f"- Held, {kd}: " + ", ".join(f"{c} {held[(c, kd)][0]}/{held[(c, kd)][1]} ({100 * held[(c, kd)][0] / held[(c, kd)][1]:.0f} %)"
                                                for c in CONDS) + ".")
    for c in CONDS:
        for kd in KINDS:
            R.append(f"- {c}, {kd}: " + ", ".join(f"{o} {n}" for o, n in outs[(c, kd)].most_common()) + ".")
    for att, s in sbg.items():
        R.append(f"- SBG on {att}: {dash(s['n_raw'])} raw → {dash(s['n_after_nms'])} after NMS → top {dash(s['n_grasps'])} listed, "
                 f"{dash(s['n_collision_free'])} collision-free, {dash(s['n_accepted'])} accepted.")
    R += ["", "## Checks against the numbers given", ""]
    if a.view != "V1":
        R.append("- No numbers were given for this view.")
    elif mism:
        R += [f"- MISMATCH: {m}" for m in mism]
    else:
        R.append("- Every number given for V1 matches the records: held counts, outcome counts, and the SBG counts for "
                 "a8, a22 and a38.")
    if a.view == "V1":
        R.append("- Note: the SBG counts given (0; 25 with 7 accepted; 15 with 9 accepted) are the overlay's "
                 "\"top N (M accepted)\": N is after non-maximum suppression, not SBG's raw output (" +
                 ", ".join(f"{att} {s['n_raw']} raw" for att, s in sbg.items()) + ").")
    R += ["", "## Clips", "",
          ("- No external-camera trial clips: no video named by an attempt id, trial id or view was found "
           "under --clips." if not clips else "- Clips found:\n" + "\n".join(f"  - `{p}`" for p in clips)), ""]
    R += ["## Caveats", "",
          "- Success is reported as percentages with their counts. No intervals are drawn.",
          "- The clouds are the depth SBG read, back-projected with the run's own intrinsics and placed in the robot's frame "
          "with the capture pose and the hand-eye calibration; SBG itself sampled 20,000 points from the whole frame. For "
          "BASELINE and ADAPTED this is the composite as run at V1 (ReMake also filled sensor holes outside the mask).",
          f"- The clouds are rendered off-screen with open3d {__import__('open3d').__version__}.", ""]
    (a.out / "README.md").write_text("\n".join(R) + "\n")
    print("\n".join(R))


if __name__ == "__main__":
    main()
