"""Camera poses around the board for hand-eye calibration, each planned from the one before.

    python3 tools/hand_eye/plan_poses.py X Y Z QX QY QZ QW [LIVE_DIR]

The arguments are the tool pose (O_T_EE) the board was seen from, in data/live/rgb.png; writes
poses.json (the plan the calibration captures followed is the one kept here).
"""
import json, math, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "nuc", "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_plan as R
import board as B
from fr3_pose_commander import planning as P, kinematics, viewpoints as V
from fr3_pose_commander.pose_math import quaternion_from_matrix, angle_between
tp = [float(v) for v in sys.argv[1:8]]
cfg, ee_cam, home_q = R.load_config(); limits = P.load_limits(str(R.LIMITS))
home = R.pose_of(kinematics.ee_matrix(home_q))
LIVE = sys.argv[8] if len(sys.argv) > 8 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "live")
K = np.load(os.path.join(LIVE, "color_intrinsics.npy"))
D = np.array([0.14671, -0.50566, 0.00083, -0.00011, 0.47813])
img = cv2.imread(os.path.join(LIVE, "rgb.png"))
dets = B.detect(img); layout, (rv, tv, rms, n) = B.find_layout(dets, K, D)
T_cb = np.eye(4); T_cb[:3, :3] = cv2.Rodrigues(rv)[0]; T_cb[:3, 3] = tv
T_bc = R.mat((tp[:3], tp[3:])).dot(R.mat((ee_cam[:3], ee_cam[3:7])))       # camera in base (current calibration)
T_base_board = T_bc.dot(T_cb)
extent = np.array([8 * B.PITCH + B.TAG, 5 * B.PITCH + B.TAG])
centre = T_base_board.dot([extent[0] / 2, extent[1] / 2, 0, 1])[:3]
board_corners = [T_base_board.dot([x, y, 0, 1])[:3] for x in (0, extent[0]) for y in (0, extent[1])]
print("board centre in base (current calibration): %s, board normal %s" % (np.round(centre, 3), np.round(T_base_board[:3, 2], 3)))
cam_ee = np.linalg.inv(R.mat((ee_cam[:3], ee_cam[3:7])))
def in_view(Tc):
    ok = 0
    for p in board_corners:
        pc = np.linalg.inv(Tc).dot(np.r_[p, 1])[:3]
        if pc[2] <= 0: return False
        u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
        ok += 40 <= u <= 1240 and 40 <= v <= 680
    return ok == 4
cands = []
for h in (0.38, 0.46, 0.54):
    for tilt in (0, 15, 25):
        for az in ([0] if tilt == 0 else range(0, 360, 60)):
            d = h * math.tan(math.radians(tilt))
            cam = centre + np.array([d * math.cos(math.radians(az)), d * math.sin(math.radians(az)), h])
            # the roll that turns the wrist least from home, then an offset for rotation diversity
            rolls = []
            for roll in np.arange(0, 360, 15):
                Tc = V.camera_pose(cam, centre, math.radians(roll)); Te = Tc.dot(cam_ee)
                rolls.append((angle_between(home[1], quaternion_from_matrix(Te)), roll))
            base_roll = min(rolls)[1]
            for off in (-25, 0, 25):
                Tc = V.camera_pose(cam, centre, math.radians(base_roll + off))
                if in_view(Tc):
                    Te = Tc.dot(cam_ee)
                    cands.append({"h": h, "tilt": tilt, "az": az, "roll_off": off, "pose": [list(Te[:3, 3]), list(quaternion_from_matrix(Te))]})
print("%d candidate poses keep the whole board in view" % len(cands))
# pick a varied set: every (h, tilt, az) once, cycling the roll offset, ordered by azimuth sweep per height
chosen, seen = [], set()
for i, c in enumerate(sorted(cands, key=lambda c: (c["h"], c["tilt"], c["az"]))):
    key = (c["h"], c["tilt"], c["az"])
    if key in seen: continue
    want = (-25, 0, 25)[len(chosen) % 3]
    pick = next((x for x in cands if (x["h"], x["tilt"], x["az"]) == key and x["roll_off"] == want), c)
    chosen.append(pick); seen.add(key)
# plan the chain from home, dropping what the planner refuses
q, pose, plan = list(home_q), home, []
for c in chosen:
    goal = (tuple(c["pose"][0]), tuple(c["pose"][1]))
    legs = P.home_legs(pose, goal, cfg)
    r = P.check_legs(q, pose, legs, "nullspace", limits, cfg, home_q)
    if not r["ok"]:
        c["refused"] = r["reasons"][0][:90]; continue
    q = list(r["q_end"]["pulled"]) if "pulled" in r["q_end"] else list(list(r["q_end"].values())[-1])
    pose = goal; plan.append(c)
print("%d poses reachable in a chain from home (of %d)" % (len(plan), len(chosen)))
for c in chosen:
    if "refused" in c: print("  dropped h %.2f tilt %2d az %3d: %s" % (c["h"], c["tilt"], c["az"], c["refused"]))
rot = [angle_between(plan[i]["pose"][1], plan[j]["pose"][1]) for i in range(len(plan)) for j in range(i + 1, len(plan))]
print("rotation diversity between poses: median %.0f deg, max %.0f deg" % (math.degrees(np.median(rot)), math.degrees(max(rot))))
json.dump({"board_layout": layout, "centre": list(centre), "poses": plan}, open("poses.json", "w"), indent=1)
