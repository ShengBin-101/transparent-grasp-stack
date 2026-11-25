"""Turn SBG grasps into gripper poses, keep those on a target object, and build the scene for the NUC planner.

Service /live_pipeline/grasp_poses reads grasps_sbg.json (or grasps_peggnet.json), writes grasp_poses.json and
publishes, latched: grasp_poses (every executable grasp), grasp_pose_camera, grasp_pregrasp_best, grasp_width_camera
and grasp_best (the best grasp, camera frame; empty frame, -1 or "null" = none), grasp_candidates (every executable
grasp plus scene voxels, for the NUC planner) and grasp_pose_markers.

Gripper frame: z = approach, y = jaw closing axis. Widths are object widths (no clearance). A grasp is not
executable if its width is unknown, saturated or over max_aperture, it approaches from the side the camera did not
see, or (require_target) it is not on a target. Targets (`targets`): depth (default and the trials' rule, no
prompt: the segmenter's transparent objects plus whatever stands on the table in the raw depth), prompt (SAM 3's
grasp-prompt objects, PEGG-Net's input), transparent (the transparency mask's objects only, the rule before the
trials) or none (no target filter and no object labels: every grasp counts, as in a plain RGB-D pipeline;
pipeline:=raw).
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Point, PoseArray, PoseStamped, Quaternion
from rclpy.node import Node
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .align import quat_to_rot
from .common import CAMERA_FRAME, LIVE_DIR, declare, latched_qos, record_stage, run_node, stamp


def rot_to_quat(R):
    """3x3 rotation -> (x, y, z, w), Shepperd's method."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s])


def gripper_frame(approach, jaw):
    """Rotation with z = approach and y = jaw (made orthogonal)."""
    z = np.asarray(approach, float)
    z = z / np.linalg.norm(z)
    y = np.asarray(jaw, float)
    y = y - np.dot(y, z) * z
    if np.linalg.norm(y) < 1e-9:
        y = np.cross(z, [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x / np.linalg.norm(x), y, z], axis=1)


def from_peggnet(g):
    """PEGG-Net grasp -> (centre, R, width or None): top-down along the optical axis, jaws at the in-image angle."""
    a = float(g["angle_rad"])
    ap = g.get("aperture_m")
    return (np.array(g["center_m"], float), gripper_frame([0.0, 0.0, 1.0], [-math.cos(a), math.sin(a), 0.0]),
            None if ap is None else float(ap))


def from_sbg(g):
    """SBG grasp -> (fingertip centre, R, width); SBG's x = approach, y = jaw is re-ordered to z = approach."""
    R = np.array(g["rotation"], float)
    return np.array(g["tcp_m"], float), gripper_frame(R[:, 0], R[:, 1]), float(g["width_m"])


def scene_voxels(depth_mm, K, voxel_m=0.02, min_points=3, z_min=0.15, z_max=1.5, stride=2, max_voxels=8000):
    """Occupied voxels of the detector's depth, for the NUC's arm-vs-scene check: (Nx3 centres in m, camera frame,
    voxel size in m). A voxel needs `min_points` points; the size doubles until at most `max_voxels` remain."""
    z = np.asarray(depth_mm, np.float32)[::stride, ::stride] / 1000.0
    h, w = z.shape
    vv, uu = np.mgrid[0:h, 0:w]
    uu, vv = uu * stride, vv * stride
    ok = np.isfinite(z) & (z > z_min) & (z < z_max)
    z, uu, vv = z[ok], uu[ok], vv[ok]
    pts = np.stack([(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z], axis=1)
    while True:
        idx, counts = np.unique(np.floor(pts / voxel_m).astype(np.int64), axis=0, return_counts=True)
        idx = idx[counts >= min_points]
        if len(idx) <= max_voxels:
            return (idx + 0.5) * voxel_m, voxel_m
        voxel_m *= 2.0


def mask_labels(path):
    """A saved mask (NxHxW stack or HxW labels, SAM 3's or the segmenter's) as an int label image (1..N, 0 = none),
    or None if absent or empty."""
    if not Path(path).is_file():
        return None
    m = np.load(path)
    if m.ndim == 3:
        lab = np.zeros(m.shape[1:], np.int32)
        for k in range(m.shape[0]):
            lab[(m[k] > 0) & (lab == 0)] = k + 1
    else:
        lab = m.astype(np.int32)
    return lab if lab.any() else None


def _present(labels):
    """The nonzero labels in an int label image, ascending."""
    return [int(k) for k in np.nonzero(np.bincount(labels.ravel()))[0] if k]


def merge_objects(transparent, graspable, match=0.5):
    """One object per thing the transparency mask or SAM 3's grasp prompt found: (labels, {label: transparent?}),
    or None.

    `transparent` (the mask ReMake completed) and `graspable` (sam3_masks_all.npy) are label images or None. A
    graspable mask joins the transparent mask it overlaps most if the overlap covers `match` of the smaller one, else
    it is a new opaque object. Transparent masks are never merged with each other, keep their labels, come first and
    win where masks disagree; "transparent" means the object's depth is ReMake's."""
    if graspable is None or (transparent is not None and graspable.shape != transparent.shape):
        if transparent is None:
            return None
        return transparent, {k: True for k in _present(transparent)}
    t_ids = [] if transparent is None else _present(transparent)
    width = int(graspable.max()) + 1
    t_map = np.zeros((max(t_ids) + 1) if t_ids else 1, np.int32)
    kinds = {}
    for label, t in enumerate(t_ids, start=1):
        t_map[t], kinds[label] = label, True
    if t_ids:
        both = (transparent > 0) & (graspable > 0)
        inter = np.bincount(transparent[both].astype(np.int64) * width + graspable[both],
                            minlength=len(t_map) * width).reshape(len(t_map), width)
        t_area = np.bincount(transparent.ravel(), minlength=len(t_map))
    a_area = np.bincount(graspable.ravel(), minlength=width)
    a_map = np.zeros(width, np.int32)
    for a in (k for k in range(1, width) if a_area[k]):
        t = int(np.argmax(inter[:, a])) if t_ids else 0
        if t and inter[t, a] >= match * min(t_area[t], a_area[a]):
            a_map[a] = t_map[t]
        else:
            label = len(kinds) + 1
            a_map[a], kinds[label] = label, False
    labels = a_map[graspable]
    if t_ids:
        labels = np.where(transparent > 0, t_map[transparent], labels)
    return labels.astype(np.int32), kinds


def scene_objects(live_dir, graspable=True, transparent_name="sam3_masks.npy"):
    """The run's mask objects for targets=prompt/transparent: the transparency mask and, with `graspable`, SAM 3's
    grasp-prompt masks, merged by merge_objects."""
    live_dir = Path(live_dir)
    return merge_objects(mask_labels(live_dir / transparent_name),
                         mask_labels(live_dir / "sam3_masks_all.npy") if graspable else None)


def target_masks(live_dir, dilate_px, graspable=True, objects=None):
    """The target objects as one boolean image dilated by `dilate_px`, or None when there are none.
    `objects` is the node's _scene_objects result; without it, scene_objects is loaded (~75 ms at 1280x720)."""
    objects = scene_objects(live_dir, graspable) if objects is None else objects
    if not objects:
        return None
    m = (objects[0] > 0).astype(np.uint8)
    if dilate_px > 0:
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)
    return m.astype(bool)


def target_labels(live_dir, dilate_px, graspable=True, objects=None):
    """The target objects as an int label image (1..N, 0 = none), each grown by `dilate_px` into the background
    (a pixel near two objects goes to the nearer), or None when there are none. `objects` as for target_masks.
    Per-object labels, not one union mask, so the NUC applies its keep-out box and zero target margin to the grasped
    object only, not to its neighbours (a union let a finger come down on a nearby mug handle)."""
    objects = scene_objects(live_dir, graspable) if objects is None else objects
    if not objects:
        return None
    lab = objects[0]
    if dilate_px > 0:
        from scipy.ndimage import distance_transform_edt
        dist, (iy, ix) = distance_transform_edt(lab == 0, return_indices=True)
        grown = lab[iy, ix]
        lab = np.where(dist <= dilate_px, grown, 0).astype(np.int32)
    return lab


def table_plane(depth_mm, K, stride=8, iters=200, tol_m=0.008, z_min=0.15, z_max=1.5, seed=0):
    """The table, as the depth's dominant plane in the camera frame: (n, d, inlier fraction), with n the unit normal
    toward the camera so n.X + d is a point's height above it; None with too few points. RANSAC on every
    `stride`-th pixel, then a least-squares refit on the inliers."""
    z = np.asarray(depth_mm, np.float32)[::stride, ::stride] / 1000.0
    vv, uu = np.mgrid[0:z.shape[0], 0:z.shape[1]] * stride
    ok = np.isfinite(z) & (z > z_min) & (z < z_max)
    if ok.sum() < 100:
        return None
    z, uu, vv = z[ok], uu[ok], vv[ok]
    X, Y, Z = (uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z
    pts = np.stack([X, Y, Z], axis=1).astype(float)
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    # Element-wise, not ndarray.dot: on the busy Orin each small BLAS call costs ~6 ms of thread start-up.
    for i in rng.integers(0, len(pts), size=(iters, 3)):
        a, b, c = pts[i]
        n = np.cross(b - a, c - a)
        norm = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
        if norm < 1e-9:
            continue
        n = n / norm
        inl = np.abs(X * n[0] + Y * n[1] + Z * n[2] - (n[0] * a[0] + n[1] * a[1] + n[2] * a[2])) < tol_m
        count = int(inl.sum())
        if count > best_n:
            best, best_n = inl, count
    if best is None:
        return None
    centre = pts[best].mean(axis=0)
    n = np.linalg.svd(pts[best] - centre, full_matrices=False)[2][2]
    d = -float(n.dot(centre))
    if d < 0:                          # the camera, at the origin, is on the positive side
        n, d = -n, -d
    return n, d, best_n / float(len(pts))


def depth_blobs(depth_mm, K, exclude, field=None, band_m=0.015, min_height_m=0.02, min_area_m2=9e-4,
                min_width_m=0.02, open_px=5, border_frac=0.4, edge_px=4, min_inliers=0.3, z_min=0.15,
                z_max=1.5):
    """Objects standing on the table outside `exclude` (the known objects, grown): (int labels 1..N or None,
    [info per blob], table info with a count of what each rule "rejected").

    Candidates are pixels more than `band_m` above table_plane, opened by `open_px` (speckle, flying pixels) and
    split into connected components. A component is an object if its footprint is at least `min_area_m2`, its
    90th-percentile height at least `min_height_m`, it is `min_width_m` across somewhere, at most `border_frac` of the
    ring around it lies in `exclude` (else it is a known object's edge), and it is not within `edge_px` of the image
    border or of the depth camera's `field` (depth_coverage.npy; a half-seen object). No table: no blobs."""
    import cv2
    plane = table_plane(depth_mm, K, z_min=z_min, z_max=z_max)
    if plane is None or plane[2] < min_inliers:
        return None, [], {"found": False, "inlier_frac": None if plane is None else round(plane[2], 3)}
    n, d, frac = plane
    z = np.asarray(depth_mm, np.float32) / 1000.0
    h, w = z.shape
    vv, uu = np.mgrid[0:h, 0:w]
    ok = np.isfinite(z) & (z > z_min) & (z < z_max)
    height = (n[0] * (uu - K[0, 2]) / K[0, 0] + n[1] * (vv - K[1, 2]) / K[1, 1] + n[2]) * z + d
    cand = (ok & (height > band_m) & ~exclude).astype(np.uint8)
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1, 2 * open_px + 1))
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k)
    edge = np.zeros((h, w), bool)
    if edge_px > 0:
        edge[:edge_px], edge[-edge_px:], edge[:, :edge_px], edge[:, -edge_px:] = True, True, True, True
    if field is not None and np.shape(field) == (h, w):
        edge |= ~np.asarray(field, bool)
    rejected = {"small": 0, "low": 0, "thin": 0, "hugs_object": 0, "at_edge": 0}
    count, comp, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    inside = cv2.distanceTransform(cand, cv2.DIST_L2, 5)  # px to the nearest non-candidate
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    labels, infos = np.zeros((h, w), np.int32), []
    # Skip components too small to reach min_area_m2 even at z_max (most of them are specks).
    min_px = min_area_m2 / (z_max / K[0, 0]) / (z_max / K[1, 1])
    for c in np.nonzero(stats[1:, cv2.CC_STAT_AREA] >= min_px)[0] + 1:
        x0, y0, bw, bh = (int(v) for v in stats[c, :4])
        x0, y0 = max(x0 - 4, 0), max(y0 - 4, 0)                # room for the ring
        win = (slice(y0, min(y0 + bh + 8, h)), slice(x0, min(x0 + bw + 8, w)))
        blob = comp[win] == c
        zb = z[win][blob]
        area = float(((zb / K[0, 0]) * (zb / K[1, 1])).sum())
        top = float(np.percentile(height[win][blob], 90))
        width = 2.0 * float(inside[win][blob].max()) * float(np.median(zb)) / K[0, 0]
        why = "small" if area < min_area_m2 else "low" if top < min_height_m else "thin" if width < min_width_m else None
        if why is None:
            ring = cv2.dilate(blob.astype(np.uint8), ring_k).astype(bool) & ~blob
            hug = float((ring & exclude[win]).sum()) / max(int(ring.sum()), 1)
            why = "hugs_object" if hug > border_frac else "at_edge" if (ring & edge[win]).any() else None
        if why is not None:
            rejected[why] += 1
            continue
        labels[win][blob] = len(infos) + 1
        infos.append({"area_cm2": round(area * 1e4, 1), "height_mm": round(top * 1000.0, 1),
                      "width_mm": round(width * 1000.0, 1), "pixels": int(blob.sum()), "hug": round(hug, 2)})
    table = {"found": True, "inlier_frac": round(frac, 3), "normal": [round(float(x), 4) for x in n],
             "distance_m": round(d, 4), "rejected": rejected}
    return (labels if infos else None), infos, table


def seg_objects(seg, depth_mm, K, min_area_m2=9e-4):
    """targets=depth's transparent objects: the segmenter's components `seg` (int labels or None) whose footprint
    at the frame's median depth (they read no depth of their own) is at least `min_area_m2` -> (labels 1..N or
    None, {label: True}, specks dropped). A speck as a target would let through a grasp on the table beside it; on
    the rig objects were 13-170 cm2 and specks under 6 cm2."""
    if seg is None:
        return None, {}, 0
    z = np.asarray(depth_mm, np.float32)
    ok = np.isfinite(z) & (z > 150.0)
    zm = float(np.median(z[ok])) / 1000.0 if ok.any() else 0.5
    area = np.bincount(seg.ravel(), minlength=int(seg.max()) + 1)
    present = [k for k in range(1, len(area)) if area[k]]
    keep = [k for k in present if area[k] * (zm / K[0, 0]) * (zm / K[1, 1]) >= min_area_m2]
    if not keep:
        return None, {}, len(present)
    lut = np.zeros(len(area), np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    return lut[seg], {k: True for k in range(1, len(keep) + 1)}, len(present) - len(keep)


def add_blobs(objects, blobs, infos, source="sam3"):
    """`objects` (merge_objects' or seg_objects' result, or empty) plus the depth blobs as further opaque objects:
    (labels, kinds, sources), or () with neither. sources[label] is {"source": source}, or a blob's info with
    "source": "depth"."""
    if objects:
        labels, kinds = objects[0].copy(), dict(objects[1])
    elif blobs is not None:
        labels, kinds = np.zeros(blobs.shape, np.int32), {}
    else:
        return ()
    sources = {k: {"source": source} for k in kinds}
    if blobs is not None:
        base = max(kinds) if kinds else 0
        for i, info in enumerate(infos, start=1):
            labels[(blobs == i) & (labels == 0)] = base + i
            kinds[base + i] = False
            sources[base + i] = dict(info, source="depth")
    return labels, kinds, sources


def grasp_instance(points_cam, K, labels):
    """The object a grasp is on: the label most of `points_cam` (TCP first, then the
    contacts) project onto, ties to the TCP's; 0 when none lands on an object."""
    h, w = labels.shape
    hits = []
    for x, y, z in np.asarray(points_cam, float).reshape(-1, 3):
        if z <= 0:
            hits.append(0)
            continue
        u, v = int(round(K[0, 0] * x / z + K[0, 2])), int(round(K[1, 1] * y / z + K[1, 2]))
        hits.append(int(labels[v, u]) if 0 <= u < w and 0 <= v < h else 0)
    found = [k for k in hits if k]
    if not found:
        return 0
    counts = {k: found.count(k) for k in set(found)}
    top = max(counts.values())
    best = [k for k, n in counts.items() if n == top]
    return hits[0] if hits[0] in best else min(best)


def on_target(points_cam, K, mask):
    """Does any of `points_cam` (camera frame, Nx3) project onto `mask`?"""
    h, w = mask.shape
    for x, y, z in np.asarray(points_cam, float).reshape(-1, 3):
        if z <= 0:
            continue
        u, v = int(round(K[0, 0] * x / z + K[0, 2])), int(round(K[1, 1] * y / z + K[1, 2]))
        if 0 <= u < w and 0 <= v < h and mask[v, u]:
            return True
    return False


def confidence(g):
    """SBG's score, or PEGG-Net's quality."""
    return float(g.get("score", g.get("quality", 0.0)))


def best_index(grasps):
    """Index of the most confident executable grasp, or None."""
    usable = [i for i, g in enumerate(grasps) if g["executable"]]
    return max(usable, key=lambda i: confidence(grasps[i])) if usable else None


class GraspPoseNode(Node):
    """The /live_pipeline/grasp_poses service: grasps -> poses, target check, best grasp, planner scene."""

    def __init__(self):
        super().__init__("grasp_pose_node")
        p = declare(self, live_dir=str(LIVE_DIR), source="auto", base_frame=CAMERA_FRAME,
                    camera_frame=CAMERA_FRAME, approach_offset_mm=100.0, max_aperture_mm=70.0,
                    min_approach_z=0.0, scene_voxel_mm=10.0, tf_timeout=1.0,
                    require_target=True, target_dilate_px=15, target_voxel_px=20, targets="depth",
                    target_mask="seg_masks.npy", seg_min_area_cm2=9.0, graspable_targets=True,
                    transparent_mask="sam3_masks.npy", blob_depth="depth_aligned_mm.npy", depth_completed=True,
                    blob_min_height_mm=20.0, blob_min_area_cm2=9.0, blob_min_width_mm=12.0, blob_exclude_px=25)
        self.live_dir = Path(p["live_dir"])
        self.source, self.base_frame, self.camera_frame = p["source"], p["base_frame"], p["camera_frame"]
        self.approach_offset_m = p["approach_offset_mm"] / 1000.0
        self.max_aperture_m = p["max_aperture_mm"] / 1000.0
        self.min_approach_z = p["min_approach_z"]
        # 10 mm: the NUC checks the table as a plane, so the voxels are objects only and a coarser grid overstates them.
        self.scene_voxel_m = p["scene_voxel_mm"] / 1000.0
        self.tf_timeout = p["tf_timeout"]
        # SBG grasps the whole scene; a pick is for a target, so off-target grasps are not executable.
        self.require_target = bool(p["require_target"])
        # depth | prompt | transparent | none (module docstring). depth never reads ReMake's output, so every depth
        # model under test gets the same targets.
        self.targets = str(p["targets"])
        if self.targets not in ("depth", "prompt", "transparent", "none"):
            raise ValueError(f"targets={self.targets!r}; use depth, prompt, transparent or none")
        self.target_mask = str(p["target_mask"])
        # False under remake:=none (raw depth): nothing was completed, so every remade_frac is 0.
        self.depth_completed = bool(p["depth_completed"])
        self.seg_min_area_m2 = float(p["seg_min_area_cm2"]) / 1e4
        self.last_seg = None
        # targets=prompt: SAM 3's grasp-prompt masks (sam3_masks_all.npy) add the opaque objects.
        self.graspable_targets = bool(p["graspable_targets"])
        # The mask ReMake completed (sam3_masks.npy or seg_masks.npy), so "transparent" means "depth is ReMake's".
        self.transparent_mask = str(p["transparent_mask"])
        # Raw sensor depth, so the blobs do not change with the depth model under test.
        self.blob_depth = str(p["blob_depth"])
        self.blob_min_height_m = float(p["blob_min_height_mm"]) / 1000.0
        self.blob_min_area_m2 = float(p["blob_min_area_cm2"]) / 1e4
        # 12 mm (depth_blobs' own default is 20): the trials' tape roll is ~18 mm across its wall, and at
        # 20 it could never be a target.
        self.blob_min_width_m = float(p["blob_min_width_mm"]) / 1000.0
        self.blob_exclude_px = int(p["blob_exclude_px"])
        self.last_table = None
        self.target_dilate_px = int(p["target_dilate_px"])
        # Voxels within this of a target are marked for the NUC's target_margin; 20 px covers the ~12 px a transparent
        # rim's mask stops short of ReMake's completed edge.
        self.target_voxel_px = int(p["target_voxel_px"])
        self._tf_waited = False
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        q = latched_qos()
        self.pose_pub = self.create_publisher(PoseArray, "/live_pipeline/grasp_poses", q)
        self.best_pub = self.create_publisher(PoseStamped, "/live_pipeline/grasp_pose_camera", q)
        self.width_pub = self.create_publisher(Float64, "/live_pipeline/grasp_width_camera", q)
        self.pregrasp_pub = self.create_publisher(PoseStamped, "/live_pipeline/grasp_pregrasp_best", q)
        self.best_json_pub = self.create_publisher(String, "/live_pipeline/grasp_best", q)
        self.candidates_pub = self.create_publisher(String, "/live_pipeline/grasp_candidates", q)
        self.marker_pub = self.create_publisher(MarkerArray, "/live_pipeline/grasp_pose_markers", q)
        self.create_service(Trigger, "/live_pipeline/grasp_poses", self._handle)

    # ------------------------------------------------------------ inputs

    def _load(self):
        """The detector grasps (peggnet, sbg); with source=auto and both present, only the newer file's."""
        peg_path, sbg_path = self.live_dir / "grasps_peggnet.json", self.live_dir / "grasps_sbg.json"
        peg = json.loads(peg_path.read_text())["grasps"] if peg_path.exists() else []
        sbg = json.loads(sbg_path.read_text())["grasps"] if sbg_path.exists() else []
        if peg and sbg and self.source == "auto":
            if sbg_path.stat().st_mtime >= peg_path.stat().st_mtime:
                peg = []
            else:
                sbg = []
        return peg, sbg

    def _lookup(self):
        """(R, t) camera -> base, or None. Waits once at start-up, then never blocks."""
        if not self.base_frame or self.base_frame == self.camera_frame:
            return None
        wait = 0.0 if self._tf_waited else self.tf_timeout
        self._tf_waited = True
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, rclpy.time.Time(),
                                                 timeout=rclpy.duration.Duration(seconds=wait))
        except Exception:  # noqa: BLE001
            self.get_logger().warning(f"no {self.camera_frame} -> {self.base_frame} transform; "
                                      f"poses stay in the camera frame", once=True)
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return quat_to_rot(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])

    def _depth_name(self):
        """The depth file the detector used (its JSON's depth_source), or None."""
        names = ("grasps_peggnet.json",) if self.source == "peggnet" else ("grasps_sbg.json", "grasps_peggnet.json")
        return next((json.loads((self.live_dir / n).read_text()).get("depth_source")
                     for n in names if (self.live_dir / n).exists()), None)

    def _scene_objects(self):
        """This run's target objects: (labels, kinds, sources), or () when there are none.
        depth: the segmenter's transparent objects plus the depth blobs outside its mask. prompt: SAM 3's objects
        (scene_objects). transparent: the transparency mask's.
        none: no objects, so no grasp is off target and the planner's scene is unlabelled."""
        self.last_table = self.last_seg = None
        if self.targets == "none":
            return ()
        if self.targets == "depth":
            return self._depth_targets()
        objects = scene_objects(self.live_dir, self.graspable_targets and self.targets == "prompt",
                                self.transparent_mask)
        return add_blobs(objects, None, [])

    def _depth_targets(self):
        """targets=depth: seg_objects of target_mask plus the depth blobs outside those objects (grown by
        blob_exclude_px). Each object gets remade_frac, its share inside the mask ReMake completed.

        Only the segmenter's objects (>= seg_min_area) are cut out of the blob region, not the specks it drops: specks
        on an opaque object (the trials' tape roll) sliced it into pieces under min_area, so whether it was a
        target changed from frame to frame."""
        path = self.live_dir / self.target_mask
        if not path.exists():
            raise RuntimeError(f"targets=depth needs {self.target_mask} (segment_trans in the chain, "
                               f"seg_node running)")
        seg = mask_labels(path)
        depth = np.load(self.live_dir / self.blob_depth)
        K = np.load(self.live_dir / "color_intrinsics.npy")
        labels, kinds, specks = seg_objects(seg, depth, K, self.seg_min_area_m2)
        self.last_seg = {"mask": self.target_mask, "objects": len(kinds), "specks_dropped": specks}
        out = self._with_blobs(None if labels is None else (labels, kinds), None if labels is None else labels > 0,
                               depth, source="seg")
        remade = mask_labels(self.live_dir / self.transparent_mask) if self.depth_completed else None
        if out and out[1]:
            n = max(out[1]) + 1
            area = np.bincount(out[0].ravel(), minlength=n)
            hit = (np.zeros(n, np.int64) if remade is None
                   else np.bincount(out[0][remade > 0].ravel(), minlength=n))
            for k in out[1]:
                out[2][k]["remade_frac"] = round(float(hit[k]) / max(int(area[k]), 1), 3)
        return out

    def _with_blobs(self, objects, mask, depth, source="sam3"):
        """`objects` plus the depth blobs of `depth` outside `mask` (bool, or None) grown by
        blob_exclude_px: add_blobs' result. The table fit goes to self.last_table."""
        import cv2
        K = np.load(self.live_dir / "color_intrinsics.npy")
        exclude = np.zeros(depth.shape, np.uint8) if mask is None else mask.astype(np.uint8)
        if self.blob_exclude_px > 0 and exclude.any():
            px = self.blob_exclude_px
            exclude = cv2.dilate(exclude, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1)))
        field = self.live_dir / "depth_coverage.npy"
        field = np.load(field) if field.exists() else None
        blobs, infos, self.last_table = depth_blobs(depth, K, exclude.astype(bool), field,
                                                    min_height_m=self.blob_min_height_m,
                                                    min_area_m2=self.blob_min_area_m2,
                                                    min_width_m=self.blob_min_width_m)
        return add_blobs(objects, blobs, infos, source)

    def _build(self, objects=None):
        """Every detector grasp as a grasp_poses.json entry -> (entries, frame, xform). Each has its grasp and
        pregrasp pose (base frame if TF has it), width, the object it is on, and why it is not executable."""
        peg, sbg = self._load()
        xform = self._lookup()
        frame = self.base_frame if xform else self.camera_frame
        entries = []
        if self.source == "sbg" or (self.source == "auto" and sbg):
            for g in sbg:
                ok = bool(g.get("accepted", True))
                entries.append((None, "sbg", g, ok, "sbg" if ok else "+".join(g.get("reject_reasons") or ["reject"])))
        else:
            for g in peg:
                rec = g.get("recommendation", "peggnet")
                # source=peggnet trusts the quality gate; auto demands recommendation == peggnet.
                ok = bool(g.get("accepted", True)) if self.source == "peggnet" else rec == "peggnet"
                entries.append((g["label"], "peggnet", g, ok, rec))

        if objects is None:
            objects = self._scene_objects()
        mask = target_masks(self.live_dir, self.target_dilate_px, objects=objects) if self.require_target else None
        if mask is None and self.require_target and self.targets == "depth":
            # targets=depth found nothing: nothing is a target (not "no filter", as for a run without masks)
            mask = np.zeros(np.load(self.live_dir / self.blob_depth, mmap_mode="r").shape, bool)
        labels = target_labels(self.live_dir, self.target_dilate_px, objects=objects)
        kinds = objects[1] if objects else {}
        K = np.load(self.live_dir / "color_intrinsics.npy") if (mask is not None or labels is not None) else None
        out = []
        for label, use, g, ok, rec in entries:
            c, R, width = from_sbg(g) if use == "sbg" else from_peggnet(g)
            approach_z = float(R[2, 2])            # in the camera frame, before any transform
            targeted, instance = None, None
            half = 0.5 * (width or 0.0)
            probe = [c, c + half * R[:, 1], c - half * R[:, 1]]
            if mask is not None:
                targeted = on_target(probe, K, mask)
            if labels is not None:
                instance = grasp_instance(probe, K, labels)
            if xform is not None:
                c, R = xform[0] @ c + xform[1], xform[0] @ R
            saturated = bool(g.get("width_saturated", False))
            usual = rec in (None, "peggnet", "sbg")
            if width is None:
                ok, rec = False, "aperture_unknown" if usual else rec
            elif width > self.max_aperture_m or saturated:
                ok, rec = False, ("width_saturated" if saturated else "infeasible_gripper") if usual else rec
            if approach_z < self.min_approach_z:
                ok = False
                rec = "approach_from_unseen_side" if rec in (None, "peggnet", "sbg") else rec
            if targeted is False:
                ok = False
                rec = "not_on_target" if rec in (None, "peggnet", "sbg") else rec
            q = rot_to_quat(R)
            pre = c - R[:, 2] * self.approach_offset_m
            e = {"label": None if label is None else int(label), "source": use, "recommendation": rec,
                 "executable": bool(ok), "frame_id": frame, "transformed": xform is not None,
                 "width_m": None if width is None else float(width), "width_saturated": saturated,
                 "grasp_pose": {"position": c.tolist(), "orientation_xyzw": q.tolist()},
                 "pregrasp_pose": {"position": pre.tolist(), "orientation_xyzw": q.tolist()},
                 "approach_axis": R[:, 2].tolist(), "approach_camera_z": approach_z,
                 "jaw_axis": R[:, 1].tolist(), "on_target": targeted, "instance": instance,
                 "transparent": kinds.get(instance) if instance else None}
            if use == "sbg":
                e["score"], e["rank"] = float(g.get("score", 0.0)), g.get("rank")
            else:
                e["quality"] = float(g.get("quality", 0.0))
            out.append(e)
        return out, frame, xform

    def _objects(self, objects):
        """The run's objects for grasp_poses.json, and their merged labels saved beside it
        (target_objects.npy, undilated) so a replay labels the scene exactly as this run did."""
        if not objects:
            return []
        labels, kinds = objects[0], objects[1]
        sources = objects[2] if len(objects) > 2 else {}
        np.save(self.live_dir / "target_objects.npy", labels.astype(np.uint8 if labels.max() < 256 else np.int32))
        area = np.bincount(labels.ravel(), minlength=max(kinds) + 1)
        return [{"instance": k, "transparent": kinds[k], "pixels": int(area[k]),
                 **{key: v for key, v in sources.get(k, {"source": "sam3"}).items() if key != "pixels"}}
                for k in sorted(kinds)]

    def _obstacles(self, frame, xform, objects=None):
        """Scene voxels of the detector's depth in the candidates' frame, each marked with the target object it lies
        on (labels grown by target_voxel_px); None on any problem."""
        if not self.scene_voxel_m:
            return None
        try:
            src = self._depth_name()
            if not src:
                return None
            K = np.load(self.live_dir / "color_intrinsics.npy")
            pts, voxel = scene_voxels(np.load(self.live_dir / src), K, self.scene_voxel_m)
            labels = (target_labels(self.live_dir, self.target_voxel_px, self.graspable_targets, objects)
                      if self.target_voxel_px > 0 else None)
            flags = inst = None
            if labels is not None:
                z = pts[:, 2]
                u = np.round(K[0, 0] * pts[:, 0] / z + K[0, 2]).astype(int)
                v = np.round(K[1, 1] * pts[:, 1] / z + K[1, 2]).astype(int)
                ok = (z > 0) & (u >= 0) & (u < labels.shape[1]) & (v >= 0) & (v < labels.shape[0])
                inst = np.zeros(len(pts), np.int32)
                inst[ok] = labels[v[ok], u[ok]]
                flags = inst > 0
            if xform is not None:
                pts = pts @ np.asarray(xform[0]).T + xform[1]
            return {"frame_id": frame, "voxel_m": float(voxel), "depth_source": src,
                    "points": np.round(pts, 4).tolist(),
                    # "instance" for a NUC that splits the scene per candidate, "target" for one that does not
                    **({} if flags is None else {"target": flags.astype(int).tolist(),
                                                 "instance": inst.tolist(),
                                                 "target_px": self.target_voxel_px})}
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f"no scene for the planner: {exc}")
            return None

    # ------------------------------------------------------------ outputs

    def _pose_msg(self, pose_dict, frame, stamp_msg):
        msg = PoseStamped()
        msg.header.frame_id, msg.header.stamp = frame, stamp_msg
        x, y, z = pose_dict["position"]
        qx, qy, qz, qw = pose_dict["orientation_xyzw"]
        msg.pose.position, msg.pose.orientation = Point(x=x, y=y, z=z), Quaternion(x=qx, y=qy, z=qz, w=qw)
        return msg

    def _funnel(self, grasps):
        """How many grasps survived each stage so far: SBG's own counts, then this node's."""
        f = {}
        try:
            sbg = json.loads((self.live_dir / "grasps_sbg.json").read_text())
            for key in ("n_raw", "n_after_nms", "n_collision_free", "n_empty", "n_accepted"):
                if sbg.get(key) is not None:
                    f[key[2:]] = sbg[key]
        except (OSError, ValueError):
            pass
        reasons = [g["recommendation"] for g in grasps]
        f.update(unseen_side=reasons.count("approach_from_unseen_side"),
                 not_on_target=reasons.count("not_on_target"),
                 executable=sum(g["executable"] for g in grasps))
        return f

    def _publish(self, grasps, frame, obstacles):
        """Publish the executable poses, the best grasp, the planner's candidates (with the scene) and the markers,
        all with one stamp so the sequencer can match the candidates to the best grasp."""
        now = self.get_clock().now().to_msg()
        pa = PoseArray()
        pa.header.frame_id, pa.header.stamp = frame, now
        pa.poses = [self._pose_msg(g["grasp_pose"], frame, now).pose for g in grasps if g["executable"]]
        self.pose_pub.publish(pa)

        i = best_index(grasps)
        best, pre = PoseStamped(), PoseStamped()
        best.header.stamp = pre.header.stamp = now      # frame_id "" means "nothing executable"
        width, best_json = Float64(data=-1.0), String(data="null")
        if i is not None:
            g = grasps[i]
            best, pre = self._pose_msg(g["grasp_pose"], frame, now), self._pose_msg(g["pregrasp_pose"], frame, now)
            width.data = -1.0 if g["width_m"] is None else float(g["width_m"])
            best_json.data = json.dumps({**g, "index": i})
        self.best_pub.publish(best)
        self.pregrasp_pub.publish(pre)
        self.width_pub.publish(width)
        self.best_json_pub.publish(best_json)
        self.candidates_pub.publish(String(data=json.dumps({
            "frame_id": frame,
            "stamp": now.sec + now.nanosec * 1e-9,     # same stamp as grasp_pose_camera
            "best_index": i,
            "candidates": [{"index": j, "source": g["source"], "width_m": g["width_m"],
                            "confidence": confidence(g), "grasp_pose": g["grasp_pose"],
                            "pregrasp_pose": g["pregrasp_pose"], "instance": g.get("instance"),
                            "transparent": g.get("transparent")}
                           for j, g in enumerate(grasps) if g["executable"]],
            "funnel": self._funnel(grasps),
            **({} if obstacles is None else {"obstacles": obstacles})})))

        arr = MarkerArray()
        for k, g in enumerate(grasps):
            c = np.array(g["grasp_pose"]["position"])
            R = quat_to_rot(*g["grasp_pose"]["orientation_xyzw"])
            alpha = 0.9 if g["executable"] else 0.25
            for j, (axis, rgb, ns) in enumerate(((R[:, 2], (0.2, 0.4, 1.0), "approach"),
                                                 (R[:, 1], (0.2, 1.0, 0.3), "jaw"))):
                length = 0.05 if j == 0 else (g["width_m"] or 0.0) / 2
                tail, head = c - axis * length, (c if j == 0 else c + axis * length)
                m = Marker()
                m.header.frame_id, m.header.stamp = frame, now
                m.ns, m.id, m.type, m.action = f"grasp_{ns}", k, Marker.ARROW, Marker.ADD
                m.scale.x, m.scale.y, m.scale.z = 0.004, 0.008, 0.0
                m.color.r, m.color.g, m.color.b = rgb
                m.color.a = alpha
                m.points = [Point(x=float(tail[0]), y=float(tail[1]), z=float(tail[2])),
                            Point(x=float(head[0]), y=float(head[1]), z=float(head[2]))]
                m.lifetime.sec = 10
                arr.markers.append(m)
            t = Marker()
            t.header.frame_id, t.header.stamp = frame, now
            t.ns, t.id, t.type, t.action = "grasp_pose_text", k, Marker.TEXT_VIEW_FACING, Marker.ADD
            t.pose.position = Point(x=float(c[0]), y=float(c[1]), z=float(c[2]) - 0.04)
            t.pose.orientation.w = 1.0
            t.scale.z = 0.018
            t.color.r = t.color.g = t.color.b = 1.0
            t.color.a = 1.0 if g["executable"] else 0.4
            w = "?" if g["width_m"] is None else f"{g['width_m'] * 1000:.0f}mm"
            t.text = f"{g['label']} {g['source']} {w}" + ("" if g["executable"] else f" [{g['recommendation']}]")
            t.lifetime.sec = 10
            arr.markers.append(t)
        self.marker_pub.publish(arr)

    def _handle(self, request, response):
        """Service: find the targets, build the poses, write grasp_poses.json and the stage record, then publish."""
        t0 = time.time()
        try:
            objects = self._scene_objects()
            grasps, frame, xform = self._build(objects)
            if not grasps:
                raise RuntimeError(f"no grasps built from {self.live_dir}: the detector proposed nothing "
                                   f"(check the SAM 3 grasp prompt / sam3_meta_all.json)")
            best = best_index(grasps)
            found = self._objects(objects)
            payload = {"grasps": grasps, "best_index": best, "best": None if best is None else grasps[best],
                       "objects": found, "table": self.last_table, "seg": self.last_seg,
                       "frame_id": frame, "transformed": xform is not None, "source": self.source,
                       "approach_offset_m": self.approach_offset_m}
            if xform is None and self.base_frame != self.camera_frame:
                payload["warning"] = f"'{self.base_frame}' not in TF; poses are in the camera frame"
            settings = {"source": self.source, "base_frame": self.base_frame, "camera_frame": self.camera_frame,
                        "approach_offset_m": self.approach_offset_m, "max_aperture_m": self.max_aperture_m,
                        "require_target": self.require_target, "target_dilate_px": self.target_dilate_px,
                        "targets": self.targets, "transparent_mask": self.transparent_mask,
                        "depth_completed": self.depth_completed,
                        **({"target_mask": self.target_mask, "seg_min_area_m2": self.seg_min_area_m2}
                           if self.targets == "depth" else
                           {"graspable_targets": self.graspable_targets and self.targets == "prompt"}),
                        "blob_depth": self.blob_depth,
                        "blob_min_height_m": self.blob_min_height_m, "blob_min_area_m2": self.blob_min_area_m2,
                        "blob_min_width_m": self.blob_min_width_m, "blob_exclude_px": self.blob_exclude_px}
            stamp(payload, "grasp_poses", settings)
            (self.live_dir / "grasp_poses.json").write_text(json.dumps(payload, indent=2))
            record_stage(self.live_dir, "grasp_poses", settings, seconds=time.time() - t0,
                         extra={"n_poses": len(grasps), "transformed": xform is not None,
                                "n_executable": sum(g["executable"] for g in grasps),
                                "n_unseen_side": sum(g["recommendation"] == "approach_from_unseen_side"
                                                     for g in grasps),
                                "n_not_on_target": sum(g["recommendation"] == "not_on_target" for g in grasps),
                                "n_objects_transparent": sum(o["transparent"] for o in found),
                                "n_objects_opaque": sum(not o["transparent"] for o in found),
                                "n_objects_depth": sum(o.get("source") == "depth" for o in found),
                                # transparent targets ReMake did not complete (e.g. mask:=sam3 missed them)
                                "n_transparent_not_remade": sum(o["transparent"] and o.get("remade_frac", 1.0) < 0.5
                                                                for o in found),
                                "n_seg_specks": (self.last_seg or {}).get("specks_dropped"),
                                "n_executable_opaque": sum(g["executable"] and g["transparent"] is False
                                                           for g in grasps)})
            self._publish(grasps, frame, self._obstacles(frame, xform, objects))
            response.success = True
            response.message = json.dumps({**payload, "timing_s": round(time.time() - t0, 3)})
            self.get_logger().info(f"{len(grasps)} poses ({sum(g['executable'] for g in grasps)} executable) "
                                   f"in {frame}, best={'none' if best is None else '#%d' % best}")
        except Exception as exc:  # noqa: BLE001
            response.success, response.message = False, f"{type(exc).__name__}: {exc}"
            self.get_logger().error(response.message)
        return response


def main():
    run_node(GraspPoseNode)


if __name__ == "__main__":
    main()
