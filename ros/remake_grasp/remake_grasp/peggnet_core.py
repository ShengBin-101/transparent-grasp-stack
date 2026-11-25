"""PEGG-Net inference on a metric depth map (no ROS).

depth (m) -> centre square crop, resize to 480, inpaint, normalise -> network ->
quality / angle / width maps -> one grasp per object mask (argmax inside the mask).

Adaptations over the published method (all in GraspConfig):
  selection 'per_object'      one grasp per SAM 3 mask instead of one per frame
  normalisation 'meanonly'    d/mean - 1 (the paper's constants were tuned for another camera)
  aperture_source 'feature'   width of the surface under the jaws, measured on the sensor map
                              (the wall of an open cup), falling back to the mask silhouette
  aperture offset schedule    clearance removed from the predicted width depends on the width
  depth_readback_px 3         grasp z = median of a 3x3 window, not one pixel
"""
import sys
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import scipy.ndimage as ndimage
import torch

from .common import CKPT_DIR, THIRD_PARTY

DEFAULT_CKPT = CKPT_DIR / "peggnet_jacquard_d_480.pt"
MAX_WIDTH_PX = 150.0   # Cornell/Jacquard width normaliser
# (upper bound on the measured width in m, offset in m); first matching band wins.
FITTED_OFFSET_SCHEDULE = ((0.016, -0.002), (0.030, -0.012), (0.060, -0.012), (float("inf"), 0.0))


@dataclass
class GraspConfig:
    selection: str = "per_object"            # 'per_object' | 'global'
    normalisation: str = "meanonly"          # 'meanonly' | 'paper' | 'centred'
    aperture_offset_m: float = -0.016        # used when the schedule is off or nothing was measured
    aperture_offset_schedule: tuple = ()
    feature_width_tol_m: float = 0.0         # depth step that ends the surface walk; 0 = off
    min_feature_width_m: float = 0.002       # shorter walks count as unmeasurable
    feature_retry_below_m: float = 0.005     # re-walk on the scoring map below this
    min_finger_gap_m: float = 0.015          # re-walk if no room for a finger past a drop
    aperture_source: str = "predicted"       # 'predicted' | 'measured' | 'feature'
    depth_readback_px: int = 1
    quality_gate: float = 0.8
    input_size: int = 480
    min_mask_px: int = 50

    def as_dict(self):
        return asdict(self)


def _normalise(x, mode):
    """Network input normalisation: 'paper' constants, 'meanonly' d/mean - 1, or 'centred' z-score."""
    if mode == "paper":
        return (x / x.mean() - 1.3) / 2.5
    if mode == "meanonly":
        return x / x.mean() - 1.0
    if mode == "centred":
        return (x - x.mean()) / (x.std() + 1e-6)
    raise ValueError(f"unknown normalisation {mode!r}")


def median_depth(depth, r, c, window_px=1):
    """(z, n_valid, centre_measured): median of the valid readings in a window at (r, c)."""
    centre = float(depth[r, c])
    centre_ok = bool(np.isfinite(centre) and centre > 0)
    if int(window_px) <= 1:
        return (centre if centre_ok else 0.0), int(centre_ok), centre_ok
    h = int(window_px) // 2
    win = depth[max(0, r - h):r + h + 1, max(0, c - h):c + h + 1]
    vals = win[np.isfinite(win) & (win > 0)]
    if vals.size == 0:
        return 0.0, 0, False
    return float(np.median(vals)), int(vals.size), centre_ok


def _step(r, c, angle, sgn, step):
    """Pixel `step` along the jaw axis (sin a, -cos a) from (r, c)."""
    return int(round(r + sgn * np.sin(angle) * step)), int(round(c + sgn * -np.cos(angle) * step))


def local_mask_width_px(mask, r, c, angle, max_px=400):
    """Silhouette width through (r, c) along the jaw axis, in pixels."""
    total = 0.0
    for sgn in (1, -1):
        for step in range(1, max_px):
            rr, cc = _step(r, c, angle, sgn, step)
            if not (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]) or not mask[rr, cc]:
                break
            total += 1.0
    return total


def offset_for_width(schedule, width_m, default_m):
    """Aperture offset (m) of the first schedule band above width_m; default_m if no schedule or width."""
    if not schedule or width_m is None or not np.isfinite(width_m) or width_m <= 0:
        return default_m
    for hi_m, off_m in schedule:
        if width_m < hi_m:
            return off_m
    return schedule[-1][1]


def walk_surface(mask, depth_m, r, c, angle, tol_m, max_px=400, max_gap_px=2):
    """Walk the jaw axis until the mask edge or a depth step > tol_m.

    Returns None if a side runs out of valid depth (gaps up to max_gap_px are bridged),
    else (width_px, [(sgn, reason, step), ...]) with reason frame|mask|drop|rise|max.
    """
    if tol_m <= 0:
        return None
    r0, c0 = int(round(r)), int(round(c))
    if not (0 <= r0 < mask.shape[0] and 0 <= c0 < mask.shape[1]):
        return None
    z0 = float(depth_m[r0, c0])
    if not np.isfinite(z0) or z0 <= 0:
        return None
    total, stops = 0.0, []
    for sgn in (1, -1):
        gap, reason, at = 0, "max", max_px
        for step in range(1, max_px):
            rr, cc = _step(r, c, angle, sgn, step)
            if not (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]):
                reason, at = "frame", step
                break
            if not mask[rr, cc]:
                reason, at = "mask", step
                break
            z = float(depth_m[rr, cc])
            if not np.isfinite(z) or z <= 0:
                gap += 1
                if gap > max_gap_px:
                    return None
                continue
            if abs(z - z0) > tol_m:
                reason, at = ("drop" if z > z0 else "rise"), step
                break
            total += 1.0 + gap
            gap = 0
        stops.append((sgn, reason, at))
    return total, stops


def finger_gap_px(mask, depth_m, r, c, angle, z0, tol_m, sgn, start, max_px=400):
    """Room past a depth drop (pixels) before the surface resumes or the mask ends."""
    n = 0
    for step in range(start, max_px):
        rr, cc = _step(r, c, angle, sgn, step)
        if not (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]) or not mask[rr, cc]:
            break
        z = float(depth_m[rr, cc])
        if np.isfinite(z) and z > 0 and abs(z - z0) <= tol_m:
            break
        n += 1
    return n


def feature_width_with_retry(mask, meas_depth, scoring_depth, r, c, angle, cfg, z, fx):
    """(width, source): surface width on the sensor map, re-walked on the scoring map when
    it is implausibly short or a finger would not fit past its drop."""
    tol = cfg.feature_width_tol_m
    if tol <= 0:
        return None, None
    to_m = (z / fx) if fx else 1.0
    walk = walk_surface(mask, meas_depth, r, c, angle, tol)
    width = None if walk is None else walk[0] * to_m
    source = "measure"
    if fx and width is not None and scoring_depth is not None and scoring_depth is not meas_depth:
        too_short = cfg.feature_retry_below_m > 0 and width < cfg.feature_retry_below_m
        no_room = False
        if cfg.min_finger_gap_m > 0:
            z0 = float(meas_depth[int(round(r)), int(round(c))])
            for sgn, reason, at in walk[1]:
                if reason == "drop" and finger_gap_px(
                        mask, meas_depth, r, c, angle, z0, tol, sgn, at) * to_m < cfg.min_finger_gap_m:
                    no_room = True
        if too_short or no_room:
            w2 = walk_surface(mask, scoring_depth, r, c, angle, tol)
            if w2 is not None:
                width, source = w2[0] * to_m, "scoring"
    if width is None or (fx and width < cfg.min_feature_width_m):
        return None, None
    return width, source


class PeggNet:
    """PEGG-Net (depth-only, 1 channel) loaded once; checks the config combination and warms up."""

    def __init__(self, checkpoint=DEFAULT_CKPT, config=None, device=None):
        self.cfg = config or GraspConfig()
        c = self.cfg
        if c.input_size % 8:
            raise ValueError("input_size must be a multiple of 8")
        if c.aperture_source == "feature" and c.feature_width_tol_m <= 0:
            raise ValueError("aperture_source='feature' needs feature_width_tol_m > 0")
        if c.aperture_source in ("measured", "feature") and c.selection != "per_object":
            raise ValueError(f"aperture_source={c.aperture_source!r} needs selection='per_object'")
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        models_dir = str(THIRD_PARTY / "peggnet" / "models")
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
        from peggnet import PEGG_NET
        self.net = PEGG_NET(input_channels=1).to(self.device)
        self.net.load_state_dict(torch.load(str(checkpoint), map_location=self.device))
        self.net.eval()
        with torch.no_grad():
            self.net(torch.zeros(1, 1, c.input_size, c.input_size, device=self.device))

    def preprocess(self, depth_m):
        """Centre square crop -> resize -> inpaint holes -> normalise.

        Returns (network input, resized crop, crop origin (r0, c0), full-frame px per network px).
        """
        n = self.cfg.input_size
        h, w = depth_m.shape
        side = min(h, w)
        r0, c0 = (h - side) // 2, (w - side) // 2
        crop = cv2.resize(depth_m[r0:r0 + side, c0:c0 + side], (n, n), interpolation=cv2.INTER_NEAREST)
        x = crop.copy().astype(np.float32)
        x[np.isnan(x)] = 0
        if (x == 0).any():
            x = cv2.copyMakeBorder(x, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
            hole = (x == 0).astype(np.uint8)
            s = np.abs(x).max()
            x = cv2.inpaint(x / max(s, 1e-6), hole, 1, cv2.INPAINT_NS)[1:-1, 1:-1] * s
        return _normalise(x, self.cfg.normalisation).astype(np.float32), crop, (r0, c0), side / n

    def forward(self, net_input):
        """-> smoothed quality, angle (rad) and width (px) maps."""
        n = self.cfg.input_size
        with torch.no_grad():
            p, cs, sn, wd = self.net(torch.from_numpy(net_input.reshape(1, 1, n, n)).to(self.device))
        pos = ndimage.gaussian_filter(p.squeeze().cpu().numpy(), 1.0)
        ang = ndimage.gaussian_filter(np.arctan2(sn.squeeze().cpu().numpy(), cs.squeeze().cpu().numpy()) / 2.0, 2.0)
        return pos, ang, wd.squeeze().cpu().numpy() * MAX_WIDTH_PX

    def grasps(self, depth_m, masks=None, fx=None, measure_depth_m=None):
        """One grasp per mask label (or one per frame for 'global'). Returns (grasps nearest first, maps).

        Each grasp is the quality peak inside its mask, with z read back from depth_m. The aperture
        is the width head plus the offset ('predicted'), the mask silhouette ('measured'), or the
        surface walk on measure_depth_m ('feature', else the silhouette). Peaks with no depth are
        listed in maps['dropped'].
        """
        cfg = self.cfg
        net_in, crop, (r0, c0), scale = self.preprocess(depth_m)
        pos, ang, wid = self.forward(net_in)
        n = cfg.input_size
        meas = depth_m if measure_depth_m is None else measure_depth_m
        if meas.shape != depth_m.shape:
            raise ValueError("measure_depth_m must match depth_m")

        mask_crop = None
        if cfg.selection == "global":
            picks = [(None, np.unravel_index(int(np.argmax(pos)), pos.shape), None)]
        else:
            if masks is None:
                raise ValueError("selection='per_object' needs masks")
            side = min(depth_m.shape)
            mask_crop = cv2.resize(masks[r0:r0 + side, c0:c0 + side].astype(np.int32), (n, n),
                                   interpolation=cv2.INTER_NEAREST)
            picks = []
            for label in (l for l in np.unique(mask_crop) if l != 0):
                m = mask_crop == label
                if m.sum() >= cfg.min_mask_px:
                    peak = np.unravel_index(int(np.argmax(np.where(m, pos, -np.inf))), pos.shape)
                    picks.append((int(label), peak, m))

        out, dropped = [], []
        for label, (pr, pc), m in picks:
            q, a = float(pos[pr, pc]), float(ang[pr, pc])
            fr, fc = pr * scale + r0, pc * scale + c0
            rr = int(np.clip(round(fr), 0, depth_m.shape[0] - 1))
            cc = int(np.clip(round(fc), 0, depth_m.shape[1] - 1))
            z, n_valid, z_centre = median_depth(depth_m, rr, cc, cfg.depth_readback_px)
            if n_valid == 0:
                dropped.append({"label": label, "quality": q, "peak_rc": [int(pr), int(pc)],
                                "full_rc": [float(fr), float(fc)],
                                "readback_px": int(cfg.depth_readback_px), "reason": "no_depth_at_peak"})
                continue
            to_m = (scale * z / fx) if fx else 1.0
            raw = float(wid[pr, pc]) * to_m
            local = (local_mask_width_px(m, pr, pc, a) * to_m) if m is not None else None
            feature, feature_src = None, None
            if cfg.feature_width_tol_m > 0 and masks is not None and label is not None:
                feature, feature_src = feature_width_with_retry(
                    masks == label, meas, None if measure_depth_m is None else depth_m,
                    fr, fc, a, cfg, z, fx)
            band_on = feature if feature is not None else (local if local is not None else raw)
            offset = offset_for_width(cfg.aperture_offset_schedule, band_on, cfg.aperture_offset_m)
            predicted = raw + offset if raw + offset > 0 else None
            measured = local if (local is not None and local > 0) else None
            if cfg.aperture_source == "predicted":
                aperture = predicted
            elif cfg.aperture_source == "measured":
                aperture = measured
            else:
                aperture = feature if (feature is not None and feature > 0) else measured
            g = {"label": label, "quality": q, "accepted": bool(q >= cfg.quality_gate),
                 "angle_rad": a, "peak_rc": [int(pr), int(pc)], "full_rc": [float(fr), float(fc)],
                 "z_m": z, "z_readback_px": int(cfg.depth_readback_px), "z_valid_px": int(n_valid),
                 "z_centre_measured": bool(z_centre), "aperture_raw_m": raw,
                 "aperture_offset_applied_m": offset, "aperture_predicted_m": predicted,
                 "aperture_source": cfg.aperture_source, "aperture_m": aperture,
                 "aperture_undefined": aperture is None}
            if feature is not None:
                g["local_feature_width_m"] = feature
                g["feature_depth_source"] = feature_src
            if local is not None:
                g["local_width_m"] = local
            out.append(g)
        out.sort(key=lambda g: g["z_m"])
        return out, {"pos": pos, "ang": ang, "wid": wid, "dropped": dropped, "crop": crop,
                     "offset": (r0, c0), "scale": scale, "mask_crop": mask_crop}
