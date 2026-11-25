"""Scale-Balanced Grasp inference on a metric depth map (no ROS).

Grasp rows use graspnetAPI's 17-column layout:
    [score, width, height, depth, R (3x3 row-major), t (3), object_id]
R columns: x = approach, y = jaw, z = finger height. t is the seed point on the surface.

Published width_m is the OBJECT width: the network's column is 1.2x the opening
(GraspNet clearance) and is clamped at 0.10 m, so it is divided by 1.2 and a clamped
row is flagged width_saturated (a lower bound, never gripper-feasible).
"""
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from . import stubs
from .common import CKPT_DIR, THIRD_PARTY

UPSTREAM = THIRD_PARTY / "sbg"
DEFAULT_CKPT = CKPT_DIR / "scale_balanced_full_model.tar"
GRASP_MAX_WIDTH_M = 0.10
SBG_CLEARANCE = 1.2
SATURATION_TOL_M = 1e-4
COL_SCORE, COL_WIDTH, COL_HEIGHT, COL_DEPTH = 0, 1, 2, 3
COL_ROT, COL_T = slice(4, 13), slice(13, 16)


@dataclass
class SbgConfig:
    """Detector and filter settings; as_dict() goes into the run record."""
    num_points: int = 20000
    num_view: int = 300
    z_min_m: float = 0.15
    z_max_m: float = 1.5
    workspace: tuple = None          # (u0, v0, u1, v1) image fractions, or None
    nms: bool = True
    nms_translation_m: float = 0.03
    nms_rotation_deg: float = 30.0
    collision_check: bool = True
    collision_thresh: float = 0.01
    collision_voxel_m: float = 0.01
    collision_approach_dist_m: float = 0.05
    collision_max_grasps: int = 400
    collision_scene_points: int = 120000
    top_k: int = 50
    score_gate: float = 0.0
    max_aperture_m: float = 0.070    # None = no gripper check
    obs: bool = False                # object balanced sampling is not supported here
    seed: int = 0                    # None = fresh randomness per call
    # Approach cone: keep each seed's approach within this many degrees of approach_axis
    # (camera frame, the direction of travel, e.g. "down"). None = upstream. See _cone_forward.
    approach_cone_deg: float = None
    approach_axis: tuple = (0.0, 0.0, 1.0)
    # Reject a grasp whose closing region holds less than empty_thresh of its volume in points,
    # so the fingers would close on nothing (upstream's return_empty_grasp). Use on completed
    # depth: raw depth has no points on transparent objects.
    empty_check: bool = False
    empty_thresh: float = 0.01

    def as_dict(self):
        d = asdict(self)
        d["workspace"] = list(self.workspace) if self.workspace else None
        return d


def cloud_from_depth(depth_m, K, rgb=None, z_min_m=0.15, z_max_m=1.5, workspace=None):
    """Depth (m) -> camera-frame points (row-major pixel order) and colours in [0, 1]."""
    h, w = depth_m.shape
    z = depth_m.astype(np.float32)
    valid = np.isfinite(z) & (z > z_min_m) & (z < z_max_m)
    if workspace is not None:
        u0, v0, u1, v1 = workspace
        box = np.zeros_like(valid)
        box[int(v0 * h):int(v1 * h), int(u0 * w):int(u1 * w)] = True
        valid &= box
    vv, uu = np.nonzero(valid)
    zz = z[vv, uu]
    pts = np.stack([(uu - K[0, 2]) * zz / K[0, 0], (vv - K[1, 2]) * zz / K[1, 1], zz], 1).astype(np.float32)
    cols = None if rgb is None else rgb[vv, uu].astype(np.float32) / 255.0
    return pts, cols


def sample_points(n_total, n, rng):
    """Upstream rule: without replacement if possible, else all points plus random repeats."""
    if n_total >= n:
        return rng.choice(n_total, n, replace=False)
    return np.concatenate([np.arange(n_total), rng.choice(n_total, n - n_total, replace=True)])


def nms(arr, translation_m=0.03, rotation_deg=30.0):
    """Greedy NMS on score-sorted rows (graspnetAPI rule)."""
    if len(arr) == 0:
        return arr
    arr = arr[np.argsort(-arr[:, COL_SCORE])]
    t = arr[:, COL_T]
    R = arr[:, COL_ROT].reshape(-1, 3, 3)
    keep = np.ones(len(arr), dtype=bool)
    for i in range(len(arr)):
        if not keep[i]:
            continue
        rest = np.nonzero(keep[i + 1:])[0] + i + 1
        if rest.size == 0:
            break
        close = rest[np.linalg.norm(t[rest] - t[i], axis=1) < translation_m]
        if close.size:
            tr = np.einsum("...ij,...ij->...", R[close], R[i][None])
            ang = np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))
            keep[close[ang < rotation_deg]] = False
    return arr[keep]


class _GraspGroupView:
    """The attributes the upstream collision detector reads from a graspnetAPI GraspGroup."""

    def __init__(self, arr):
        self.translations = arr[:, COL_T]
        self.rotation_matrices = arr[:, COL_ROT].reshape(-1, 3, 3)
        self.heights = arr[:, COL_HEIGHT]
        self.depths = arr[:, COL_DEPTH]
        self.widths = arr[:, COL_WIDTH]


# The cone in force for the next forward pass: set by ScaleBalancedGrasp.grasps.
_CONE = {"axis": None, "cos": -2.0}


# SBG's own ApproachNet.forward, kept when _cone_forward replaces it (ScaleBalancedGrasp.__init__).
_UPSTREAM_FORWARD = None


def _cone_forward(self, seed_xyz, seed_features, end_points, record=True):
    """SBG's ApproachNet.forward, unchanged, then each seed's approach re-chosen inside _CONE.

    SBG picks each seed's approach from 300 directions knowing nothing about the arm; on the rig
    its grasps came in 44-61 deg off vertical, which on short objects puts the hand on the table
    or a joint at its limit. A cone about "down" fixes that. Only the choice of view changes;
    angle, depth, width and score are then predicted by upstream as before. With the cone off
    this is upstream's forward and nothing else.
    """
    out = _UPSTREAM_FORWARD(self, seed_xyz, seed_features, end_points, record)
    if record is not False and _CONE["axis"] is not None:
        _choose_views_in_cone(end_points, self.num_view)
    return out


def _choose_views_in_cone(end_points, num_view):
    """Replace the grasp_top_view_* entries with each seed's best view among those in _CONE.

    The views are upstream's (generate_grasp_views; a view's approach direction is -view), and the
    rotation comes from upstream's batch_viewpoint_params_to_matrix with zero in-plane angle, as
    in ApproachNet. If no view lies in the cone, upstream's choice stands.
    """
    from loss_utils import batch_viewpoint_params_to_matrix, generate_grasp_views
    view_score = end_points["view_score"]                          # (batch, seeds, views)
    views = generate_grasp_views(num_view).to(view_score.device)   # (views, 3)
    axis = torch.tensor(_CONE["axis"], dtype=views.dtype, device=views.device)
    allowed = (-views @ axis) >= _CONE["cos"]
    if not bool(allowed.any()):
        return
    score, ind = torch.max(view_score.masked_fill(~allowed, float("-inf")), dim=2)
    xyz = views[ind]                                               # (batch, seeds, 3)
    flat = xyz.reshape(-1, 3)
    rot = batch_viewpoint_params_to_matrix(-flat, flat.new_zeros(flat.shape[0]))
    end_points["grasp_top_view_inds"] = ind
    end_points["grasp_top_view_score"] = score
    end_points["grasp_top_view_xyz"] = xyz
    end_points["grasp_top_view_rot"] = rot.view(*xyz.shape[:2], 3, 3)


def import_upstream(compiled_pointnet2=False):
    """Put the vendored SBG tree on sys.path (as its own scripts do) and install the stubs."""
    for sub in ("models", "utils", "pointnet2", "knn", ""):
        p = str(UPSTREAM / sub) if sub else str(UPSTREAM)
        if p not in sys.path:
            sys.path.insert(0, p)
    stubs.install(UPSTREAM / "pointnet2", compiled_pointnet2=compiled_pointnet2)


class ScaleBalancedGrasp:
    """SBG (GraspNet_MSCQ) loaded once: cloud -> network -> NMS -> collision/empty check -> ranked grasps.

    Patches ApproachNet.forward with _cone_forward at load, and warms up so the first call is fast.
    """

    def __init__(self, checkpoint=DEFAULT_CKPT, config=None, device=None, compiled_pointnet2=False):
        self.cfg = config or SbgConfig()
        if self.cfg.obs:
            raise ValueError("object balanced sampling (obs) is not supported")
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        import_upstream(compiled_pointnet2)
        from graspnet import GraspNet_MSCQ, pred_decode
        from collision_detector import ModelFreeCollisionDetector
        self._pred_decode = pred_decode
        self._collision_cls = ModelFreeCollisionDetector
        import modules                    # upstream models/modules.py, on sys.path now
        global _UPSTREAM_FORWARD
        if modules.ApproachNet.forward is not _cone_forward:
            _UPSTREAM_FORWARD = modules.ApproachNet.forward
            modules.ApproachNet.forward = _cone_forward  # upstream's forward while the cone is off

        # upstream test.py settings
        self.net = GraspNet_MSCQ(input_feature_dim=0, num_view=self.cfg.num_view, num_angle=12,
                                 num_depth=4, cylinder_radius=0.08, hmin=-0.02,
                                 hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False,
                                 obs=False).to(self.device)
        ck = torch.load(str(checkpoint), map_location=self.device, weights_only=False)
        self.net.load_state_dict(ck["model_state_dict"])
        self.net.eval()
        self.checkpoint = checkpoint
        self.epoch = int(ck.get("epoch", -1))
        with torch.no_grad():   # warm-up (also builds the FPS CUDA graphs)
            pts = np.random.default_rng(0).uniform(-0.3, 0.3, (self.cfg.num_points, 3))
            self._forward(pts.astype(np.float32) + np.array([0, 0, 0.6], np.float32))

    def _forward(self, points):
        pc = torch.from_numpy(np.ascontiguousarray(points, np.float32)).unsqueeze(0).to(self.device)
        end_points = self.net({"point_clouds": pc})
        return self._pred_decode(end_points)[0].detach().cpu().numpy().astype(np.float64)

    def grasps(self, depth_m, K, rgb=None, approach_axis=None):
        """Detect grasps on depth_m (m) with intrinsics K. Returns (grasp dicts, info).

        The top_k grasps after NMS are ranked collision-free first, then by score; rejected ones stay
        in the list with reject_reasons. info holds the top rows, the sampled cloud, counts and timings.
        `approach_axis` (camera frame) overrides cfg.approach_axis when the cone is on.
        """
        cfg = self.cfg
        t, t0 = {}, time.time()
        if cfg.approach_cone_deg:
            axis = np.asarray(approach_axis if approach_axis is not None else cfg.approach_axis, float)
            _CONE["axis"] = (axis / np.linalg.norm(axis)).tolist()
            _CONE["cos"] = float(np.cos(np.radians(cfg.approach_cone_deg)))
        else:
            _CONE["axis"], _CONE["cos"] = None, -2.0
        scene, scene_cols = cloud_from_depth(depth_m, K, rgb, cfg.z_min_m, cfg.z_max_m, cfg.workspace)
        if len(scene) < 100:
            raise ValueError(f"only {len(scene)} valid points in {cfg.z_min_m}-{cfg.z_max_m} m")
        rng = np.random.default_rng(cfg.seed)
        idx = sample_points(len(scene), cfg.num_points, rng)
        cloud = scene[idx]
        colors = None if scene_cols is None else scene_cols[idx]
        t["cloud"] = time.time() - t0

        t1 = time.time()
        with torch.no_grad():
            arr = self._forward(cloud)
        t["network"] = time.time() - t1
        n_raw = len(arr)

        arr = arr[np.argsort(-arr[:, COL_SCORE])]
        n_nms = None
        if cfg.nms and len(arr):
            arr = nms(arr, cfg.nms_translation_m, cfg.nms_rotation_deg)
            n_nms = len(arr)

        collision = empty = None
        if cfg.collision_check and len(arr):
            t2 = time.time()
            head = arr[:cfg.collision_max_grasps]
            scene_cd = scene
            if len(scene) > cfg.collision_scene_points:
                scene_cd = scene[rng.choice(len(scene), cfg.collision_scene_points, replace=False)]
            det = self._collision_cls(scene_cd, voxel_size=cfg.collision_voxel_m)
            if cfg.empty_check:
                mask, empty_head = det.detect(_GraspGroupView(head), approach_dist=cfg.collision_approach_dist_m,
                                              collision_thresh=cfg.collision_thresh,
                                              return_empty_grasp=True, empty_thresh=cfg.empty_thresh)
                empty = np.zeros(len(arr), dtype=bool)
                empty[:len(head)] = empty_head
            else:
                mask = det.detect(_GraspGroupView(head), approach_dist=cfg.collision_approach_dist_m,
                                  collision_thresh=cfg.collision_thresh)
            collision = np.ones(len(arr), dtype=bool)     # beyond the checked head: treat as colliding
            collision[:len(head)] = mask
            t["collision"] = time.time() - t2

        if collision is not None:
            order = np.lexsort((-arr[:, COL_SCORE], collision.astype(int)))
            arr, collision = arr[order], collision[order]
            if empty is not None:
                empty = empty[order]
        top = arr[:cfg.top_k]
        top_col = None if collision is None else collision[:cfg.top_k]
        top_empty = None if empty is None else empty[:cfg.top_k]

        out = []
        for i, row in enumerate(top):
            R, tt = row[COL_ROT].reshape(3, 3), row[COL_T]
            width_raw = float(row[COL_WIDTH])
            saturated = width_raw >= GRASP_MAX_WIDTH_M - SATURATION_TOL_M
            width = width_raw / SBG_CLEARANCE
            col = None if top_col is None else bool(top_col[i])
            feasible = (None if cfg.max_aperture_m is None
                        else bool(width <= cfg.max_aperture_m) and not saturated)
            reasons = []
            is_empty = None if top_empty is None else bool(top_empty[i])
            if col:
                reasons.append("collision")
            if is_empty:
                reasons.append("empty")
            if feasible is False:
                reasons.append("width_saturated" if saturated else "infeasible_gripper")
            if row[COL_SCORE] < cfg.score_gate:
                reasons.append("low_score")
            out.append({
                "rank": i,
                "score": float(row[COL_SCORE]),
                "width_m": width,
                "width_raw_m": width_raw,
                "width_saturated": saturated,
                "height_m": float(row[COL_HEIGHT]),
                "depth_m": float(row[COL_DEPTH]),
                "rotation": R.tolist(),
                "translation_m": tt.tolist(),
                "tcp_m": (tt + row[COL_DEPTH] * R[:, 0]).tolist(),
                "approach_axis": R[:, 0].tolist(),
                "jaw_axis": R[:, 1].tolist(),
                "center_px": [float(K[0, 0] * tt[0] / tt[2] + K[0, 2]),
                              float(K[1, 1] * tt[1] / tt[2] + K[1, 2])],
                "collision": col,
                "empty": is_empty,
                "gripper_feasible": feasible,
                "accepted": not reasons,
                "reject_reasons": reasons,
            })
        info = {"array": top, "cloud": cloud, "colors": colors,
                "n_points": int(len(cloud)), "n_scene_points": int(len(scene)),
                "n_raw": int(n_raw), "n_after_nms": n_nms,
                "n_collision_free": None if collision is None else int((~collision).sum()),
                "n_empty": None if top_empty is None else int(top_empty.sum()),
                "approach_cone": None if _CONE["axis"] is None else
                {"deg": cfg.approach_cone_deg, "axis_camera": [round(v, 5) for v in _CONE["axis"]]},
                "timings_s": {k: round(v, 3) for k, v in t.items()}}
        return out, info


# ---------------------------------------------------------------- drawing

DEPTH_BASE_M, TAIL_M = 0.02, 0.04


def gripper_segments(row):
    """Gripper wireframe (pairs of camera-frame points) for one grasp row."""
    R, t = row[COL_ROT].reshape(3, 3), row[COL_T]
    w2, d = row[COL_WIDTH] / 2.0, row[COL_DEPTH]
    local = np.array([
        [[d, -w2, 0], [-DEPTH_BASE_M, -w2, 0]],
        [[d, w2, 0], [-DEPTH_BASE_M, w2, 0]],
        [[-DEPTH_BASE_M, -w2, 0], [-DEPTH_BASE_M, w2, 0]],
        [[-DEPTH_BASE_M, 0, 0], [-DEPTH_BASE_M - TAIL_M, 0, 0]],
    ])
    return local @ R.T + t


def score_colour_bgr(score):
    """red (0) -> yellow -> green (1), graspnetAPI style."""
    s = float(np.clip(score, 0.0, 1.0))
    r = 1.0 if s < 0.5 else 2.0 * (1.0 - s)
    g = 2.0 * s if s < 0.5 else 1.0
    return 0, int(g * 255), int(r * 255)


def _project(p, K):
    p = np.asarray(p, dtype=float)
    z = p[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(z > 1e-6, K[0, 0] * p[..., 0] / z + K[0, 2], np.nan)
        v = np.where(z > 1e-6, K[1, 1] * p[..., 1] / z + K[1, 2], np.nan)
    return np.stack([u, v], axis=-1)


def draw_grasps(img, arr, K, accepted=None):
    """Draw grasp rows onto a BGR image (in place). Rejected ones are thin and grey."""
    import cv2
    n = len(arr)
    labelled = [i for i in range(n) if accepted is None or accepted[i]][:10]
    for i in range(n - 1, -1, -1):
        row = arr[i]
        ok = True if accepted is None else bool(accepted[i])
        colour = score_colour_bgr(row[COL_SCORE]) if ok else (128, 128, 128)
        for a, b in _project(gripper_segments(row), K):
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            cv2.line(img, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))),
                     colour, 2 if ok else 1, cv2.LINE_AA)
        if i in labelled:
            u, v = _project(row[COL_T], K)
            if np.isfinite(u) and np.isfinite(v):
                cv2.putText(img, f"{i}:{row[COL_SCORE]:.2f} {row[COL_WIDTH] * 1000:.0f}mm",
                            (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 0), 1, cv2.LINE_AA)
    n_ok = n if accepted is None else int(np.sum(accepted))
    cv2.putText(img, f"Scale-Balanced-Grasp  top {n}  ({n_ok} accepted)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return img
