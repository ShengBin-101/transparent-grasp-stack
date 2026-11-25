"""Leave-one-out: fit ee_to_camera on all board captures but one, predict the one left out.

    python3 tools/hand_eye/loo.py CAPTURES [--against-installed]

The reported figures (4.5 mm median held-out board error, against 8.5 mm for the nominal mounting)
were computed with the recalibrated intrinsics (K_calibrated.npy, D_calibrated.npy), which it reads.
"""
import glob, json, math, os, sys
import numpy as np, cv2
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board as B, solve as S
F = sys.argv[1]
K = np.load(F + "/K_calibrated.npy"); D = np.load(F + "/D_calibrated.npy")
# The comparison: the nominal mounting (what the reported figures compare against), or the installed value.
NOMINAL = [0.105, 0.0, -0.048, 0.0, 0.0, 0.7071067811865476, 0.7071067811865476]
if "--against-installed" in sys.argv:
    cfg, ee_cam, _ = S.R.load_config()
else:
    ee_cam = NOMINAL
X_now = S.R.mat((ee_cam[:3], ee_cam[3:7]))
NOW = "installed" if "--against-installed" in sys.argv else "nominal"
samples = []
for p in sorted(glob.glob(F + "/[0-9][0-9].json")):
    rec = json.load(open(p)); dets = B.detect(cv2.imread(p.replace(".json", ".png")))
    rv, tv, rms, n = B.solve_pose(dets, K, D, S.LAYOUT)
    samples.append({"T_be": S.R.mat((rec["tool"][0], rec["tool"][1])), "T_cb": S.T_from(rv, tv), "dets": dets})
obj = {i: B.points(i, S.LAYOUT) for i in range(B.N_TAGS)}
CENTRE = np.array([0.171, 0.1125, 0, 1])

def resid(p, sub):
    X, Z = S.T_from(p[:3], p[3:6]), S.T_from(p[6:9], p[9:12]); out = []
    for s in sub:
        T = np.linalg.inv(s["T_be"].dot(X)).dot(Z)
        rv, tv = cv2.Rodrigues(T[:3, :3])[0], T[:3, 3]
        for i, c in s["dets"].items():
            pr, _ = cv2.projectPoints(obj[i], rv, tv, K, D); out.append((pr.reshape(-1, 2) - c).ravel())
    return np.concatenate(out)

def fit(sub):
    Rg = [s["T_be"][:3, :3] for s in sub]; tg = [s["T_be"][:3, 3] for s in sub]
    Rt = [s["T_cb"][:3, :3] for s in sub]; tt = [s["T_cb"][:3, 3] for s in sub]
    Rx, tx = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=cv2.CALIB_HAND_EYE_PARK)
    X0 = np.eye(4); X0[:3, :3] = Rx; X0[:3, 3] = tx.ravel(); Z0 = sub[0]["T_be"].dot(X0).dot(sub[0]["T_cb"])
    p0 = np.r_[cv2.Rodrigues(X0[:3, :3])[0].ravel(), X0[:3, 3], cv2.Rodrigues(Z0[:3, :3])[0].ravel(), Z0[:3, 3]]
    f = least_squares(lambda p: resid(p, sub), p0, loss="huber", f_scale=1.0)
    return S.T_from(f.x[:3], f.x[3:6]), S.T_from(f.x[6:9], f.x[9:12])

def board_for(X, sub):
    """The board's pose given X, fitted to `sub` (for the current calibration)."""
    Z = sub[0]["T_be"].dot(X).dot(sub[0]["T_cb"])
    q0 = np.r_[cv2.Rodrigues(Z[:3, :3])[0].ravel(), Z[:3, 3]]
    xr = np.r_[cv2.Rodrigues(X[:3, :3])[0].ravel(), X[:3, 3]]
    f = least_squares(lambda q: resid(np.r_[xr, q], sub), q0, loss="huber", f_scale=1.0)
    return S.T_from(f.x[:3], f.x[3:6])

err_new, err_now, dx = [], [], []
for i in range(len(samples)):
    rest = samples[:i] + samples[i + 1:]
    X, Z = fit(rest)
    s = samples[i]
    pred = s["T_be"].dot(X).dot(s["T_cb"]).dot(CENTRE)[:3]; err_new.append(np.linalg.norm(pred - Z.dot(CENTRE)[:3]))
    Zn = board_for(X_now, rest)
    predn = s["T_be"].dot(X_now).dot(s["T_cb"]).dot(CENTRE)[:3]; err_now.append(np.linalg.norm(predn - Zn.dot(CENTRE)[:3]))
    dx.append(X[:3, 3])
    print("  held out %2d: new %5.1f mm   %s %5.1f mm" % (i, 1e3 * err_new[-1], NOW, 1e3 * err_now[-1]), flush=True)
e1, e0 = np.array(err_new) * 1e3, np.array(err_now) * 1e3
print("\nheld-out board error: new median %.1f mm (rms %.1f, max %.1f) | %s median %.1f mm (rms %.1f, max %.1f)" % (
    np.median(e1), math.sqrt((e1 ** 2).mean()), e1.max(), NOW, np.median(e0), math.sqrt((e0 ** 2).mean()), e0.max()))
print("new wins on %d of %d held-out poses" % ((e1 < e0).sum(), len(e1)))
t = np.array(dx) * 1e3
print("the fitted translation across the 23 fits: x %.1f+-%.1f, y %.1f+-%.1f, z %.1f+-%.1f mm" % (
    t[:, 0].mean(), t[:, 0].std(), t[:, 1].mean(), t[:, 1].std(), t[:, 2].mean(), t[:, 2].std()))
