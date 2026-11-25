"""Depth geometry: warp depth into the colour frame, and splice model depth onto sensor depth.

No ROS in here, so it can be tested on its own.
"""
import cv2
import numpy as np


def quat_to_rot(x, y, z, w):
    """Unit quaternion -> 3x3 rotation matrix."""
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def footprint(Kc, Kd, cap=8):
    """(sy, sx): colour pixels covered by one depth pixel (ratio of focal lengths)."""
    sx = int(np.ceil(Kc[0, 0] / Kd[0, 0])) if Kd[0, 0] > 0 else 1
    sy = int(np.ceil(Kc[1, 1] / Kd[1, 1])) if Kd[1, 1] > 0 else 1
    return min(max(1, sy), cap), min(max(1, sx), cap)


def align_depth_to_color(depth_mm, Kd, Kc, R, T, shape_color, splat=True):
    """Forward-warp depth (mm, 0 = none) into the colour image, nearest surface wins.

    R, T take a point from the depth optical frame to the colour optical frame.
    Each sample is written to its whole footprint so the lower-resolution depth
    image does not leave a lattice of fake holes. Returns float32 mm.
    """
    depth_mm = np.asarray(depth_mm, dtype=np.float32)
    Hc, Wc = shape_color
    Hd, Wd = depth_mm.shape

    v, u = np.indices((Hd, Wd))
    z = depth_mm / 1000.0
    valid = z > 0
    x = (u - Kd[0, 2]) / Kd[0, 0] * z
    y = (v - Kd[1, 2]) / Kd[1, 1] * z
    pts = np.stack([x[valid], y[valid], z[valid]], axis=-1) @ np.asarray(R).T + np.asarray(T)

    zc = pts[:, 2]
    front = zc > 0
    zc = zc[front]
    uc = np.round(pts[front, 0] / zc * Kc[0, 0] + Kc[0, 2]).astype(np.int64)
    vc = np.round(pts[front, 1] / zc * Kc[1, 1] + Kc[1, 2]).astype(np.int64)

    sy, sx = footprint(Kc, Kd) if splat else (1, 1)
    du, dv = np.meshgrid(np.arange(sx) - (sx - 1) // 2, np.arange(sy) - (sy - 1) // 2)
    du, dv = du.ravel(), dv.ravel()

    # Far-to-near order: numpy keeps the last write per pixel, so the nearest wins.
    order = np.argsort(-zc)
    uc, vc, zc = uc[order], vc[order], zc[order]
    uu = (uc[:, None] + du[None, :]).ravel()
    vv = (vc[:, None] + dv[None, :]).ravel()
    zz = np.repeat(zc, du.size)
    keep = (uu >= 0) & (uu < Wc) & (vv >= 0) & (vv < Hc)

    out = np.zeros((Hc, Wc), dtype=np.float32)
    out[vv[keep], uu[keep]] = zz[keep] * 1000.0
    return out


def coverage_mask(depth_mm, Kd, Kc, R, T, shape_color, splat=True):
    """Bool mask of the colour pixels the depth camera can see at all.

    A hole inside this region is a sensor miss; outside it there is no evidence.
    """
    depth_mm = np.asarray(depth_mm, dtype=np.float32)
    valid = depth_mm > 0
    if not valid.any():
        return np.zeros(shape_color, bool)
    filled = np.where(valid, depth_mm, np.median(depth_mm[valid])).astype(np.float32)
    cov = align_depth_to_color(filled, Kd, Kc, R, T, shape_color, splat=splat) > 0

    # Close small gaps, then fill interior holes so the mask is one region.
    sy, sx = footprint(Kc, Kd)
    cov = cv2.morphologyEx(cov.astype(np.uint8), cv2.MORPH_CLOSE,
                           np.ones((2 * sy + 1, 2 * sx + 1), np.uint8))
    Hc, Wc = cov.shape
    outside = np.zeros((Hc + 2, Wc + 2), np.uint8)
    outside[1:-1, 1:-1] = cov
    cv2.floodFill(outside, np.zeros((Hc + 4, Wc + 4), np.uint8), (0, 0), 1)
    return (cov > 0) | (outside[1:-1, 1:-1] == 0)


def composite_depth(model_mm, sensor_mm, mask=None, ring_px=5, valid_region=None, fill_holes=False):
    """Keep sensor depth outside the mask and model depth inside it. Returns (out_mm, stats).

    Nothing is filled outside `valid_region` (the depth camera's field of view).

    `fill_holes` False (the default, and the trials' rule after the first side-view attempts): model
    depth inside the mask only; the sensor's holes elsewhere stay empty, as in RAW. At the side view
    (V2) the L515 leaves 8 % of the frame unread past the table's edge, and filling it with model depth
    made a false surface up to about 215 mm above the table, which the planner then avoided. True:
    model depth in every hole too, as in all the top-down (V1) attempts, where it changes about 0.2 %
    of the pixels per frame.
    """
    model_mm = np.nan_to_num(np.asarray(model_mm, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sensor_mm = np.nan_to_num(np.asarray(sensor_mm, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if model_mm.shape != sensor_mm.shape:
        raise ValueError(f"shape mismatch: model {model_mm.shape} vs sensor {sensor_mm.shape}")
    mask = np.zeros(model_mm.shape, bool) if mask is None else np.asarray(mask, bool)
    valid_region = (np.ones(model_mm.shape, bool) if valid_region is None
                    else np.asarray(valid_region, bool))
    if mask.shape != model_mm.shape or valid_region.shape != model_mm.shape:
        raise ValueError("mask / valid_region shape does not match the depth")

    sensor_ok = sensor_mm > 0
    take_sensor = sensor_ok & ~mask
    fill = valid_region if fill_holes else (valid_region & mask)
    out = np.where(take_sensor, sensor_mm, np.where(fill, model_mm, 0.0)).astype(np.float32)
    out[out <= 0] = 0.0

    delta = np.abs(out - model_mm)[take_sensor]
    stats = {
        "px_from_sensor": float(take_sensor.mean()),
        "px_from_model": float(((~take_sensor) & (out > 0)).mean()),
        "px_no_data": float((out <= 0).mean()),
        "px_out_of_field": float((~valid_region).mean()),
        "changed_frac": float((delta > 1.0).mean() * take_sensor.mean()),
        "changed_p95_mm": float(np.percentile(delta, 95)) if delta.size else 0.0,
        "seam_mm": None,
    }
    # Model-vs-sensor disagreement just outside the mask. Large = bad mask.
    if mask.any() and ring_px > 0:
        k = np.ones((2 * ring_px + 1,) * 2, np.uint8)
        ring = (cv2.dilate(mask.astype(np.uint8), k).astype(bool)
                & ~mask & sensor_ok & (model_mm > 0) & valid_region)
        if ring.any():
            stats["seam_mm"] = float(np.median(np.abs(model_mm[ring] - sensor_mm[ring])))
    return out, stats
