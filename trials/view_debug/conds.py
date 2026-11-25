"""Tilt and phantom points per paired-replay run and condition (the side-view analysis).

    python3 trials/view_debug/conds.py PAIRED_DIR OUT.json

For each condition directory of each run: the camera's tilt from straight down, the depth points more than
40 mm above the table and 12 cm from the object's spot ("phantom" points), and SBG's and the grasp-pose node's
counts where the replay wrote them. Reads files only; analyse.py makes the tables from its output.
"""
import sys, os, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sensor import cam_pose, to_base, TABLE_Z, SPOT
out = sys.argv[1]; res = []
for run in sorted(glob.glob(out + "/2026*")):
    for cdir in sorted(glob.glob(run + "/*")):
        c = os.path.basename(cdir)
        if c.startswith("_") or not os.path.exists(cdir + "/capture_tool.json"): continue
        Rc, pc = cam_pose(cdir); K = np.load(cdir + "/color_intrinsics.npy")
        tilt = float(np.degrees(np.arccos(-Rc[2, 2])))
        dname = "remake_depth_mm.npy" if os.path.exists(cdir + "/remake_depth_mm.npy") else "depth_aligned_mm.npy"
        B, _, _ = to_base(np.load(cdir + "/" + dname), K, Rc, pc)
        h = (B[:, 2] - TABLE_Z) * 1000; far = np.hypot(B[:, 0] - SPOT[0], B[:, 1] - SPOT[1]) > 0.12
        ph = (h > 40) & far
        row = {"run": os.path.basename(run), "cond": c, "tilt": tilt, "phantom_pts": int(ph.sum()),
               "phantom_near_mm": float(1000 * np.hypot(B[ph, 0] - SPOT[0], B[ph, 1] - SPOT[1]).min()) if ph.any() else None}
        try:
            g = json.load(open(cdir + "/grasps_sbg.json")); row.update(raw=g.get("n_raw"), cfree=g.get("n_collision_free"), acc=g.get("n_accepted"))
        except Exception: pass
        try:
            gp = json.load(open(cdir + "/grasp_poses.json")); gs = gp.get("grasps", gp if isinstance(gp, list) else [])
            row["exe"] = sum(1 for x in gs if x.get("executable"))
        except Exception: row["exe"] = None
        res.append(row)
json.dump(res, open(sys.argv[2], "w"), indent=0)
