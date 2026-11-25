"""The calib.io AprilGrid on the table: 9 x 6 tag36h11 tags, 30 mm, 9 mm gaps (39 mm pitch).

    detect(bgr) -> {id: 4x2 image corners}
    points(id, layout) -> 4x3 board-frame corners (m), z = 0
    solve_pose(dets, K, D, layout) -> (rvec, tvec, rms px, n corners)

The LAYOUT (which cell each id is in, which way its corners run) is not assumed: find_layout()
tries every plausible one and keeps the one whose board pose reprojects best."""
import itertools
import numpy as np
import cv2

TAG, GAP = 0.030, 0.009
PITCH = TAG + GAP
N_TAGS = 54


def detect(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
    p = cv2.aruco.DetectorParameters_create()
    p.markerBorderBits = 2                       # AprilGrid tags carry a 2-bit border
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 3, 23, 10
    p.polygonalApproxAccuracyRate, p.errorCorrectionRate = 0.03, 0.6
    corners, ids, _ = cv2.aruco.detectMarkers(g, d, parameters=p)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners) if int(i) < N_TAGS}


def points(tag_id, layout):
    ncols, rot, flip = layout
    r, c = divmod(tag_id, ncols)
    x0, y0 = c * PITCH, r * PITCH
    sq = np.array([[0, 0], [TAG, 0], [TAG, TAG], [0, TAG]], float)       # counter-clockwise in the board frame
    if flip:
        sq = sq[[0, 3, 2, 1]]
    sq = np.roll(sq, rot, axis=0)
    return np.c_[sq + (x0, y0), np.zeros(4)]


def solve_pose(dets, K, D, layout):
    obj = np.concatenate([points(i, layout) for i in sorted(dets)])
    img = np.concatenate([dets[i] for i in sorted(dets)])
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, D, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    rms = float(np.sqrt(((proj.reshape(-1, 2) - img) ** 2).sum(1).mean()))
    return rvec.ravel(), tvec.ravel(), rms, len(img)


def find_layout(dets, K, D):
    best = None
    for layout in itertools.product((9, 6), range(4), (False, True)):
        r = solve_pose(dets, K, D, layout)
        if r is not None and (best is None or r[2] < best[1][2]):
            best = (layout, r)
    return best
