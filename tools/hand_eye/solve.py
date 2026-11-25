"""Hand-eye calibration of the wrist L515 from the AprilGrid captures.

    python3 tools/hand_eye/solve.py CAPTURES [--intrinsics factory|calibrated|PATH.npy]

CAPTURES holds NN.png (the colour frame) and NN.json ({"tool": [[x, y, z], [qx, qy, qz, qw]]}, the
arm's O_T_EE when the frame was taken) for each pose, and the intrinsics (README.md). The installed
ee_to_camera is the fit with the factory intrinsics, the ones the pipeline uses (the default).

X = the colour camera's optical frame in the controlled EE frame (fr3_EE, what franka_states
reports as O_T_EE) -- the seven numbers ee_to_camera. The board is fixed on the table, so
    T_base_board = T_base_EE(i) . X . T_cam_board(i)      for every capture i.
1. T_cam_board(i) from the tags (solvePnP + LM, the camera_info distortion).
2. cv2.calibrateHandEye, all five methods.
3. Refinement: X and T_base_board together, minimising the reprojection of every tag corner
   in every image -- the error that actually matters, in pixels.
The CONSISTENCY of T_base_board across captures is the check that needs no ground truth: a
right X puts the fixed board in one place from every pose.
"""
import glob
import json
import math
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board as B  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "nuc", "tools"))
import replay_plan as R  # noqa: E402
from fr3_pose_commander.pose_math import quaternion_from_matrix  # noqa: E402

D = np.array([0.14671, -0.50566, 0.00083, -0.00011, 0.47813])   # /camera/color/camera_info, plumb_bob
LAYOUT = (9, 1, True)


def T_from(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float))[0]
    T[:3, 3] = np.asarray(tvec, float).ravel()
    return T


def spread(Ts):
    """(translation rms mm, max mm, rotation rms deg) of 4x4s about their mean."""
    t = np.array([T[:3, 3] for T in Ts])
    dt = np.linalg.norm(t - t.mean(0), axis=1)
    ref = Ts[0][:3, :3]
    ang = []
    for T in Ts:
        a = np.degrees(np.linalg.norm(cv2.Rodrigues(ref.T.dot(T[:3, :3]))[0]))
        ang.append(a)
    ang = np.array(ang) - np.mean(ang)
    return 1e3 * math.sqrt((dt ** 2).mean()), 1e3 * dt.max(), math.sqrt((ang ** 2).mean())


INTRINSICS = {"factory": "color_intrinsics_factory.npy", "calibrated": "K_calibrated.npy"}


def main(folder, intrinsics="factory"):
    path = os.path.join(folder, INTRINSICS[intrinsics]) if intrinsics in INTRINSICS else intrinsics
    K = np.load(path)
    print("intrinsics: %s" % path)
    samples = []
    for path in sorted(glob.glob(os.path.join(folder, "[0-9][0-9].json"))):
        rec = json.load(open(path))
        img = cv2.imread(path.replace(".json", ".png"))
        dets = B.detect(img)
        if len(dets) < 12:
            print("  %s: only %d tags -- skipped" % (os.path.basename(path), len(dets)))
            continue
        pose = B.solve_pose(dets, K, D, LAYOUT)
        if pose is None or pose[2] > 2.0:
            print("  %s: board pose rms %.2f px -- skipped" % (os.path.basename(path), pose[2] if pose else -1))
            continue
        rv, tv, rms, n = pose
        tool = rec["tool"]
        samples.append({"name": os.path.basename(path)[:2], "T_be": R.mat((tool[0], tool[1])),
                        "T_cb": T_from(rv, tv), "rms": rms, "dets": dets})
    print("%d usable captures (tags per image %s)" % (len(samples), [len(s["dets"]) for s in samples]))

    cfg, ee_cam, _ = R.load_config()
    X_now = R.mat((ee_cam[:3], ee_cam[3:7]))
    Rg = [s["T_be"][:3, :3] for s in samples]
    tg = [s["T_be"][:3, 3] for s in samples]
    Rt = [s["T_cb"][:3, :3] for s in samples]
    tt = [s["T_cb"][:3, 3] for s in samples]
    results = {"current": X_now}
    for name, method in (("tsai", cv2.CALIB_HAND_EYE_TSAI), ("park", cv2.CALIB_HAND_EYE_PARK),
                         ("horaud", cv2.CALIB_HAND_EYE_HORAUD), ("andreff", cv2.CALIB_HAND_EYE_ANDREFF),
                         ("daniilidis", cv2.CALIB_HAND_EYE_DANIILIDIS)):
        Rx, tx = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=method)
        X = np.eye(4); X[:3, :3] = Rx; X[:3, 3] = tx.ravel()
        results[name] = X

    def boards(X):
        return [s["T_be"].dot(X).dot(s["T_cb"]) for s in samples]

    # joint refinement: X and the board's pose in base, on every corner's reprojection
    obj = {i: B.points(i, LAYOUT) for i in range(B.N_TAGS)}

    def unpack(p):
        return T_from(p[:3], p[3:6]), T_from(p[6:9], p[9:12])

    def resid(p):
        X, Z = unpack(p)
        out = []
        for s in samples:
            T_cam_board = np.linalg.inv(s["T_be"].dot(X)).dot(Z)
            rv, tv = cv2.Rodrigues(T_cam_board[:3, :3])[0], T_cam_board[:3, 3]
            for i, c in s["dets"].items():
                proj, _ = cv2.projectPoints(obj[i], rv, tv, K, D)
                out.append((proj.reshape(-1, 2) - c).ravel())
        return np.concatenate(out)

    X0 = results["park"]
    Z0 = boards(X0)[0]
    p0 = np.r_[cv2.Rodrigues(X0[:3, :3])[0].ravel(), X0[:3, 3], cv2.Rodrigues(Z0[:3, :3])[0].ravel(), Z0[:3, 3]]
    fit = least_squares(resid, p0, loss="huber", f_scale=1.0)
    X_ref, Z_ref = unpack(fit.x)
    results["refined"] = X_ref

    def reproj(X):
        # best board pose for this X (so the current calibration is judged fairly)
        Z = boards(X)
        zt = np.mean([z[:3, 3] for z in Z], axis=0)
        zr = cv2.Rodrigues(Z[0][:3, :3])[0].ravel()
        g = least_squares(lambda q: resid(np.r_[cv2.Rodrigues(X[:3, :3])[0].ravel(), X[:3, 3], q]),
                          np.r_[zr, zt], loss="huber", f_scale=1.0)
        r = resid(np.r_[cv2.Rodrigues(X[:3, :3])[0].ravel(), X[:3, 3], g.x]).reshape(-1, 2)
        e = np.linalg.norm(r, axis=1)
        return float(np.sqrt((e ** 2).mean())), float(np.percentile(e, 95))

    print("\n%-11s %-44s %-30s %s" % ("method", "ee_to_camera: t (mm) | rotation vs current (deg)",
                                      "board in base: spread", "reprojection (px)"))
    for name, X in results.items():
        tr, tmax, rr = spread(boards(X))
        drot = np.degrees(np.linalg.norm(cv2.Rodrigues(X_now[:3, :3].T.dot(X[:3, :3]))[0]))
        rms, p95 = reproj(X)
        print("%-11s (%6.1f %6.1f %6.1f) | %5.2f            rms %5.1f mm (max %5.1f), %.2f deg   rms %.2f, 95%% %.2f" % (
            name, *(1e3 * X[:3, 3]), drot, tr, tmax, rr, rms, p95))
    Xn = results["refined"]
    q = quaternion_from_matrix(Xn)
    new = [round(float(v), 6) for v in list(Xn[:3, 3]) + list(q)]
    print("\nrefined ee_to_camera = %s" % new)
    print("current ee_to_camera = %s" % [round(float(v), 6) for v in ee_cam])
    out = os.path.join(folder, "result_%s.json" % (intrinsics if intrinsics in INTRINSICS else "custom"))
    json.dump({"ee_to_camera": new, "n": len(samples), "current": list(ee_cam), "intrinsics": path,
               "board_in_base": [list(map(float, r)) for r in Z_ref[:3]]}, open(out, "w"), indent=1)
    print("written: %s" % out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures")
    ap.add_argument("--intrinsics", default="factory", help="factory | calibrated | a 3x3 .npy")
    a = ap.parse_args()
    main(a.captures, a.intrinsics)
