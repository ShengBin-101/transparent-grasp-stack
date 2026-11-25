"""Per-frame sensor and geometry measures, from the archived run directories only. Moves nothing.

Imported by conds.py and fig.py. cam_pose(run) is the camera pose in the robot base frame at capture
(capture_tool.json and T_EE_CAM); to_base maps camera points to the base frame.
"""
import json, sys, os, numpy as np
# the installed ee_to_camera (ros/remake_grasp/config/eye_in_hand.yaml, tools/hand_eye/), x y z qx qy qz qw;
# TABLE_Z is the table top in the base frame (m) and SPOT the objects' marked spot, as the NUC planner has them
T_EE_CAM = [0.109234, 0.0087, -0.030235, 0.002532, -0.015076, 0.715291, 0.698659]
TABLE_Z, SPOT = -0.041, np.array([0.50, 0.00])
def R(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def cam_pose(run):
    t = json.load(open(run + "/capture_tool.json"))
    Rb = R(t["orientation_xyzw"]); return Rb @ R(T_EE_CAM[3:]), Rb @ T_EE_CAM[:3] + np.array(t["position"])
def to_base(depth_mm, K, Rc, pc, stride=2):
    d = depth_mm.astype(np.float32)[::stride, ::stride] / 1000
    v, u = np.mgrid[0:depth_mm.shape[0]:stride, 0:depth_mm.shape[1]:stride]
    ok = (d > 0.15) & (d < 1.5)
    P = np.stack([(u[ok] - K[0, 2]) * d[ok] / K[0, 0], (v[ok] - K[1, 2]) * d[ok] / K[1, 1], d[ok]], 1)
    return P @ Rc.T + pc, u[ok], v[ok]
def plane(B, rng=np.random.default_rng(0)):
    best, bn, b0 = 0, None, None
    for _ in range(300):
        a, b, c = B[rng.choice(len(B), 3, replace=False)]
        n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn; k = (np.abs((B - a) @ n) < 0.008).sum()
        if k > best: best, bn, b0 = k, n, a
    if bn[2] < 0: bn = -bn
    z_spot = b0[2] - (bn[0] * (SPOT[0] - b0[0]) + bn[1] * (SPOT[1] - b0[1])) / bn[2]
    return float(np.degrees(np.arccos(bn[2]))), float((z_spot - TABLE_Z) * 1000)
def measure(run):
    Rc, pc = cam_pose(run); ax = Rc[:, 2]
    K = np.load(run + "/color_intrinsics.npy"); raw = np.load(run + "/depth_aligned_mm.npy")
    cov = np.load(run + "/depth_coverage.npy") if os.path.exists(run + "/depth_coverage.npy") else np.ones(raw.shape, bool)
    s = (TABLE_Z - pc[2]) / ax[2]; aim = pc + s * ax
    out = {"run": os.path.basename(run), "tilt": float(np.degrees(np.arccos(-ax[2]))), "cam_up_mm": float((pc[2] - TABLE_Z) * 1000),
           "aim": [float(aim[0]), float(aim[1])], "azimuth": float(np.degrees(np.arctan2(pc[1] - aim[1], pc[0] - aim[0]))),
           "in_field": float(cov.mean()), "holes_in_field": float(((raw == 0) & cov).sum() / max(1, cov.sum()))}
    # where the holes are: each hole pixel's ray onto the table plane, as distance from the spot
    v, u = np.mgrid[0:raw.shape[0]:4, 0:raw.shape[1]:4]
    hole = (raw[v, u] == 0) & cov[v, u]
    rays = np.stack([(u[hole] - K[0, 2]) / K[0, 0], (v[hole] - K[1, 2]) / K[1, 1], np.ones(hole.sum())], 1) @ Rc.T
    t = (TABLE_Z - pc[2]) / rays[:, 2]; hit = pc + rays * t[:, None]
    dist = np.where(t > 0, np.hypot(hit[:, 0] - SPOT[0], hit[:, 1] - SPOT[1]), np.inf)
    n_all = max(1, (cov[v, u]).sum())
    out["holes_within_15cm"] = float((dist < 0.15).sum() / n_all)
    out["holes_15_30cm"] = float(((dist >= 0.15) & (dist < 0.30)).sum() / n_all)
    out["holes_beyond_30cm"] = float((dist >= 0.30).sum() / n_all)
    B, _, _ = to_base(raw, K, Rc, pc)
    out["table_tilt_deg"], out["table_offset_mm"] = plane(B)
    h = (B[:, 2] - TABLE_Z) * 1000; far = np.hypot(B[:, 0] - SPOT[0], B[:, 1] - SPOT[1]) > 0.12
    out["sensor_tall_bg"] = int(((h > 40) & far).sum())
    for name, key in (("remake_depth_mm.npy", "composite_tall_bg"),):
        if os.path.exists(run + "/" + name):
            B2, _, _ = to_base(np.load(run + "/" + name), K, Rc, pc)
            h2 = (B2[:, 2] - TABLE_Z) * 1000; far2 = np.hypot(B2[:, 0] - SPOT[0], B2[:, 1] - SPOT[1]) > 0.12
            out[key] = int(((h2 > 40) & far2).sum()); out[key + "_max_mm"] = float(h2[far2].max()) if far2.any() else 0.0
    return out
if __name__ == "__main__":
    res = []
    for r in sys.argv[2:]:
        try: res.append(measure(r))
        except Exception as e: print("skip", r, e, file=sys.stderr)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
