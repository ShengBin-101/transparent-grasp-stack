"""Tests opaque objects as grasp targets (merged prompt masks; depth blobs beside the segmenter's objects, the
trials' targets=depth) and plan_pick: required.

    docker exec -e RG_RUNS=/opt/remake_grasp/runs remake_grasp python3 /opt/remake_grasp/tests/test_opaque_targets.py

RG_RUNS points the archived-run checks at a runs/ other than this checkout's (a fresh clone has none).
"""
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("RG_RUNS", ROOT / "runs"))
sys.path.insert(0, str(ROOT / "ros/remake_grasp"))
from remake_grasp import grasp_pose_node as G  # noqa: E402
from remake_grasp import sequencer as S  # noqa: E402

fail = []


def ck(name, ok, detail=""):
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  -- %s" % (detail,)))
    if not ok:
        fail.append(name)


# --- merge_objects --------------------------------------------------------------------------
H, W = 480, 640
glass = np.zeros((H, W), np.int32)
glass[200:280, 300:340] = 1                                 # the transparent prompt's glass
allm = np.zeros((H, W), np.int32)
allm[195:285, 296:344] = 1                                  # the grasp prompt's same glass, a little fuller
allm[200:280, 420:480] = 2                                  # an opaque box only the grasp prompt sees
labels, kinds = G.merge_objects(glass, allm)
ck("merge: the glass both prompts found is one object, transparent", labels[240, 320] == 1 and kinds[1] is True)
ck("merge: the grasp prompt's fuller rim joins the glass", labels[197, 320] == 1 and labels[240, 298] == 1)
ck("merge: the box is an object of its own, opaque", labels[240, 450] == 2 and kinds[2] is False, kinds)
ck("merge: nothing else", set(np.unique(labels)) == {0, 1, 2} and len(kinds) == 2)

lab, k = G.merge_objects(glass, None)
ck("merge: no graspable masks -> the transparent labels unchanged, all transparent",
   (lab == glass).all() and k == {1: True})
lab, k = G.merge_objects(None, allm)
ck("merge: no transparent masks -> every graspable object, all opaque", (lab == allm).all() and k == {1: False, 2: False})
ck("merge: neither -> None", G.merge_objects(None, None) is None)
ck("merge: masks of another size are ignored, not mixed", (G.merge_objects(glass, np.ones((10, 10), np.int32))[0] == glass).all())

two = np.zeros((H, W), np.int32)
two[200:280, 300:340], two[200:280, 345:385] = 1, 2         # two glasses side by side
big = np.zeros((H, W), np.int32)
big[200:280, 298:390] = 1                                   # the grasp prompt saw them as one
lab, k = G.merge_objects(two, big)
ck("merge: one graspable mask over two glasses never merges them", lab[240, 320] == 1 and lab[240, 360] == 2 and len(k) == 2, k)
ck("... and joins the one it overlaps most", lab[240, 342] in (1, 2) and k == {1: True, 2: True}, k)

mug = np.zeros((H, W), np.int32)
mug[200:280, 300:340] = 1
parts = np.zeros((H, W), np.int32)
parts[200:280, 300:340], parts[220:260, 340:352] = 1, 2      # the grasp prompt: body, and handle beside it
lab, k = G.merge_objects(mug, parts)
ck("merge: a graspable mask touching no glass stays separate (opaque)", lab[240, 345] == 2 and k[2] is False, k)

partial = np.zeros((H, W), np.int32)
partial[200:280, 330:400] = 1                               # overlaps the glass by 10 of its 40 columns
lab, k = G.merge_objects(glass, partial)
ck("merge: a 25% overlap is not the same object", k == {1: True, 2: False} and lab[240, 320] == 1, k)
ck("merge: where two objects' masks disagree, the transparent mask wins", lab[240, 335] == 1 and lab[240, 350] == 2)

# --- the node: which grasps are executable ---------------------------------------------------
K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
z = 0.5


def cam(u, v):
    return [(u - 320) * z / 600.0, (v - 240) * z / 600.0, z]


def sbg_grasp(u, v, width=0.03):
    R = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)  # x = approach (camera +z), y = jaw (camera +x)
    return {"rotation": R.tolist(), "tcp_m": cam(u, v), "width_m": width, "accepted": True, "score": 0.5}


d = Path(tempfile.mkdtemp())
np.save(d / "color_intrinsics.npy", K)
np.save(d / "sam3_masks.npy", glass)
(d / "grasps_sbg.json").write_text(json.dumps({"grasps": [sbg_grasp(320, 240), sbg_grasp(450, 240), sbg_grasp(600, 60)]}))


def node(live, graspable):
    n = G.GraspPoseNode.__new__(G.GraspPoseNode)
    n.live_dir, n.source, n.base_frame = live, "sbg", G.CAMERA_FRAME
    n.camera_frame, n.max_aperture_m, n.min_approach_z = G.CAMERA_FRAME, 0.07, 0.0
    n.require_target, n.target_dilate_px, n.graspable_targets = True, 15, graspable
    n.approach_offset_m, n.scene_voxel_m, n.target_voxel_px = 0.1, 0.01, 20
    n.blob_min_height_m, n.blob_min_area_m2, n.blob_exclude_px = 0.02, 9e-4, 25
    n.blob_min_width_m = 0.02           # depth_blobs' own default; the node's 12 mm is checked on the warm-up frame
    n.transparent_mask = "sam3_masks.npy"
    n.targets, n.blob_depth = "prompt", "depth_aligned_mm.npy"   # SAM 3's targets: what this file tests
    return n


def verdicts(live, graspable):
    out, _, _ = node(live, graspable)._build()
    return [(g["executable"], g["recommendation"], g["instance"], g["transparent"]) for g in out]


ck("node: transparent mask only -> the glass, as before",
   verdicts(d, True) == [(True, "sbg", 1, True), (False, "not_on_target", 0, None), (False, "not_on_target", 0, None)],
   verdicts(d, True))
np.save(d / "sam3_masks_all.npy", allm)
v = verdicts(d, True)
ck("node: with the grasp prompt's masks the box grasp is executable, on object 2, opaque",
   v == [(True, "sbg", 1, True), (True, "sbg", 2, False), (False, "not_on_target", 0, None)], v)
ck("node: graspable_targets false -> the box is off target again", verdicts(d, False)[1][:2] == (False, "not_on_target"))
n = node(d, True)
objs = n._objects(G.scene_objects(d))
saved = np.load(d / "target_objects.npy")
ck("node: the objects for grasp_poses.json", [(o["instance"], o["transparent"]) for o in objs] == [(1, True), (2, False)], objs)
ck("node: target_objects.npy holds the merged labels", saved.dtype == np.uint8 and (saved == labels).all())

# the scene voxels the NUC splits per candidate: the box's voxels carry the box's label
depth = np.full((H, W), 700.0, np.float32)                  # the table
depth[200:280, 300:340] = 500.0
depth[200:280, 420:480] = 500.0
np.save(d / "remake_depth_mm.npy", depth)
(d / "grasps_sbg.json").write_text(json.dumps({"depth_source": "remake_depth_mm.npy",
                                               "grasps": [sbg_grasp(320, 240), sbg_grasp(450, 240)]}))


class _L:
    def warning(self, *a, **k): pass


n.get_logger = lambda: _L()
ob = n._obstacles(G.CAMERA_FRAME, None)
pts, inst = np.array(ob["points"]), np.array(ob["instance"])
box_voxels = inst[(np.abs(pts[:, 2] - 0.5) < 0.02) & (pts[:, 0] > 0.07)]
ck("obstacles: the box's voxels are labelled 2, not 0 (an obstacle to its own grasp)",
   len(box_voxels) > 0 and (box_voxels == 2).all(), np.unique(box_voxels))
ck("obstacles: the glass keeps label 1", (inst[(np.abs(pts[:, 2] - 0.5) < 0.02) & (pts[:, 0] < 0.04)] == 1).all())

# --- depth blobs: what stands on the table that SAM 3 did not name ---------------------------
yy, xx = np.mgrid[0:H, 0:W]
table = (0.6 + 0.0001 * (yy - 240)).astype(np.float32)      # tilted ~5.7 deg, as a real stand-off
scene = table.copy()
scene[200:280, 420:480] -= 0.050                            # a 60 x 80 mm box, 50 mm tall, no SAM mask
scene[300:400, 50:200] -= 0.005                             # a 5 mm mat: flat, so table
scene[200:280, 300:340] -= 0.080                            # the glass (SAM's), ReMake-completed
scene[160:174, 270:390] -= 0.030                            # a 14 mm sliver along the glass's exclusion (rows >= 175)
rng = np.random.default_rng(1)
speck = rng.random((H, W)) < 0.002
scene[speck] -= 0.04                                        # flying pixels
depth = scene * 1000.0
n_, d_, frac = G.table_plane(depth, K)
slope = 0.0001 * 600 / 0.6                                  # dz/dY: v - 240 = 600 Y / z, z ~ 0.6
true_n = np.array([0.0, -slope, 1.0]) / np.linalg.norm([0.0, -slope, 1.0])
ck("table_plane: the table's normal within 1 deg, most of the image on it",
   np.degrees(np.arccos(abs(n_.dot(true_n)))) < 1.0 and frac > 0.7, (n_, frac))
exclude = G.target_masks(d, 25, objects=G.merge_objects(glass, None))
blobs, infos, tinfo = G.depth_blobs(depth, K, exclude)
ck("depth_blobs: the unnamed box is one blob, ~40 cm2 (60 x 80 px at 0.55 m), ~50 mm tall",
   blobs is not None and len(infos) == 1 and blobs[240, 450] == 1
   and 33 < infos[0]["area_cm2"] < 45 and 45 < infos[0]["height_mm"] < 55, infos)
ck("depth_blobs: not the mat, the glass, the sliver along it, or the specks",
   blobs is not None and blobs[350, 100] == 0 and blobs[240, 320] == 0 and blobs[167, 330] == 0
   and int((blobs > 0).sum()) < 60 * 80 * 1.05, None if blobs is None else int((blobs > 0).sum()))
edge_scene = table.copy()
edge_scene[200:280, 600:640] -= 0.050                       # a box half out of frame, at the right border
eb, ei, et = G.depth_blobs(edge_scene * 1000.0, K, np.zeros((H, W), bool))
ck("depth_blobs: a box cut by the image border is not an object (only partly seen)",
   eb is None and et["rejected"]["at_edge"] == 1, (ei, et.get("rejected")))
field = np.ones((H, W), bool)
field[:, :60] = False                                       # the depth camera sees less than colour
inner = table.copy()
inner[200:280, 60:120] -= 0.050                             # a box at the edge of the depth field
eb, ei, et = G.depth_blobs(inner * 1000.0, K, np.zeros((H, W), bool), field)
ck("depth_blobs: ... nor one cut by the edge of the depth camera's field",
   eb is None and et["rejected"]["at_edge"] == 1, (ei, et.get("rejected")))
ck("depth_blobs: the same box inside the field is one",
   G.depth_blobs(inner * 1000.0, K, np.zeros((H, W), bool))[0] is not None)
ck("depth_blobs: no table (all one far wall) -> none, and says so",
   G.depth_blobs(np.full((H, W), 3000.0, np.float32), K, np.zeros((H, W), bool))[0] is None)
lab, k, src = G.add_blobs(G.merge_objects(glass, None), blobs, infos)
ck("add_blobs: the glass stays 1 (sam3, transparent), the box is 2 (depth, opaque)",
   lab[240, 320] == 1 and lab[240, 450] == 2 and k == {1: True, 2: False}
   and src[1]["source"] == "sam3" and src[2]["source"] == "depth", (k, src))
ck("add_blobs: nothing at all -> ()", G.add_blobs(None, None, []) == ())

db = Path(tempfile.mkdtemp())
np.save(db / "color_intrinsics.npy", K)
np.save(db / "sam3_masks.npy", glass)
np.save(db / "remake_depth_mm.npy", depth)
np.save(db / "depth_aligned_mm.npy", depth)                 # the blobs read the raw sensor depth
(db / "grasps_sbg.json").write_text(json.dumps({"depth_source": "remake_depth_mm.npy", "grasps": [
    sbg_grasp(320, 240), sbg_grasp(450, 240), sbg_grasp(100, 350)]}))
nb = node(db, True)
v = verdicts(db, True)
ck("node, targets=prompt (no depth blobs): only the glass", [x[0] for x in v] == [True, False, False], v)
# targets=depth, the trials' rule: the segmenter's transparent objects plus the depth blobs beside them
np.save(db / "seg_masks.npy", glass)
nb.targets, nb.target_mask, nb.seg_min_area_m2, nb.depth_completed = "depth", "seg_masks.npy", 9e-4, True
out, _, _ = nb._build()
v = [(g["executable"], g["recommendation"], g["instance"], g["transparent"]) for g in out]
ck("node, targets=depth: the box grasp is executable, on object 2, opaque; the mat grasp is not",
   v == [(True, "sbg", 1, True), (True, "sbg", 2, False), (False, "not_on_target", 0, None)], v)
objs = nb._objects(nb._scene_objects())
ck("node: grasp_poses.json says where each object came from",
   [(o["instance"], o["transparent"], o["source"]) for o in objs] == [(1, True, "seg"), (2, False, "depth")]
   and objs[1]["height_mm"] > 45, objs)
ck("node: the table it fitted is recorded", nb.last_table and nb.last_table["found"] and nb.last_table["inlier_frac"] > 0.7,
   nb.last_table)
nb.require_target = True
np.save(db / "seg_masks.npy", np.zeros((H, W), np.int32))
out, _, _ = nb._build()
ck("node: the segmenter found nothing -> the glass (its depth here is whole) and the box are blobs, "
   "and the filter stays on (no mat grasps)",
   [(g["executable"], g["transparent"]) for g in out] == [(True, False), (True, False), (False, None)],
   [(g["executable"], g["recommendation"], g["transparent"]) for g in out])
objs3 = nb._scene_objects()
box = int(objs3[0][240, 450])
ob = nb._obstacles(G.CAMERA_FRAME, None, objs3)
pts, inst = np.array(ob["points"]), np.array(ob["instance"])
ck("obstacles: the box's voxels carry its blob label",
   box > 0 and (inst[(pts[:, 0] > 0.07) & (pts[:, 2] < 0.57)] == box).all() and (inst == box).sum() > 0,
   (box, np.unique(inst)))

# --- archived runs ---------------------------------------------------------------------------


def old_target_labels(live_dir, dilate_px):
    """The earlier target_labels (sam3_masks.npy only), to check runs without graspable masks are unchanged."""
    path = Path(live_dir) / "sam3_masks.npy"
    if not path.exists():
        return None
    m = np.load(path)
    if m.ndim == 3:
        lab = np.zeros(m.shape[1:], np.int32)
        for k in range(m.shape[0]):
            lab[(m[k] > 0) & (lab == 0)] = k + 1
    else:
        lab = m.astype(np.int32)
    if not lab.any():
        return None
    if dilate_px > 0:
        from scipy.ndimage import distance_transform_edt
        dist, (iy, ix) = distance_transform_edt(lab == 0, return_indices=True)
        lab = np.where(dist <= dilate_px, lab[iy, ix], 0).astype(np.int32)
    return lab


runs = sorted(p for p in RUNS.glob("*_auto") if (p / "sam3_masks.npy").exists()) if RUNS.exists() else []
alone = [r for r in runs if not (r / "sam3_masks_all.npy").exists()]
same = [r.name for r in alone
        if ((G.target_labels(r, 15) is None and old_target_labels(r, 15) is None)
            or np.array_equal(G.target_labels(r, 15), old_target_labels(r, 15)))]
if alone:
    ck("archived: every run without graspable masks labels exactly as before (%d runs)" % len(alone),
       len(same) == len(alone), sorted(set(r.name for r in alone) - set(same)))
else:
    print("  skip  archived label checks: no *_auto runs with sam3_masks.npy under %s (set RG_RUNS)" % RUNS)
both = [r for r in runs if (r / "sam3_masks_all.npy").exists()]
for r in both:
    lab, k = G.scene_objects(r)
    t = G.mask_labels(r / "sam3_masks.npy")
    print("       %s: %d transparent, %d graspable masks -> objects %s"
          % (r.name, len(np.unique(t)) - 1, len(np.unique(G.mask_labels(r / "sam3_masks_all.npy"))) - 1,
             {i: ("transparent" if v else "opaque") for i, v in k.items()}))
    ck("archived %s: the transparent objects keep their labels" % r.name,
       all((lab[t == i] == i).all() for i in np.unique(t) if i), r.name)

# --- plan_pick: required ---------------------------------------------------------------------


class Log:
    def info(self, m): pass
    warning = info


class Client:
    def __init__(self, ready=True): self.ready = ready
    def service_is_ready(self): return self.ready


class Pub:
    def publish(self, m): self.sent = m


def sequencer(mode, ready=True):
    q = S.Sequencer.__new__(S.Sequencer)
    q.lock = threading.Lock()
    q.plan_pick, q.planner, q.cand_pub = mode, Client(ready), Pub()
    q.ee_to_camera = q.ee_to_grasp = ((0, 0, 0), (0, 0, 0, 1))
    q.standoff = q.retreat = 0.10
    q.do_retreat, q.max_width, q.settle = True, 0.08, 0.0
    q.camera_frame, q.base_frame = S.CAMERA_FRAME, "fr3_link0"
    q.get_logger = lambda: Log()
    q.candidates = {"frame_id": S.CAMERA_FRAME, "stamp": 5.0, "candidates": [
        {"index": 0, "confidence": 0.5, "width_m": 0.04, "instance": 2, "transparent": False,
         "grasp_pose": {"position": [0, 0, 0.4], "orientation_xyzw": [1, 0, 0, 0]}}],
        # a scene, as the live pipeline always sends one (a required pick without one is refused)
        "obstacles": {"frame_id": S.CAMERA_FRAME, "voxel_m": 0.01, "points": [[0, 0, 0.45]], "instance": [2]}}
    q.record = {}
    # a planner in report mode: success, and nothing chosen
    q._call_raw = lambda c, l: (True, json.dumps({"mode": "report", "chosen": None, "candidates": [],
                                                  "plan_id": json.loads(q.cand_pub.sent.data)["plan_id"]}))
    return q


def outcome(q):
    tool = ((0.4, 0.0, 0.4), (1.0, 0.0, 0.0, 0.0))
    try:
        return q._planned(tool, tool, 5.0)
    except S.SequenceError as exc:
        return "refused: " + str(exc)


ck("required, planner absent: refused", str(outcome(sequencer("required", ready=False))).startswith(
   "refused: plan_pick is required but /live_pipeline/plan_pick is not available"))
r = outcome(sequencer("required"))
ck("required, planner chose nothing (report mode): refused, not the detector's best",
   str(r).startswith("refused: plan_pick is required but the planner chose nothing"), r)
ck("auto, planner chose nothing: the detector's best, as before", outcome(sequencer("auto")) is None)
ck("auto, planner absent: the detector's best, as before", outcome(sequencer("auto", ready=False)) is None)
q = sequencer("required")
outcome(q)
sent = json.loads(q.cand_pub.sent.data)["candidates"]
ck("sequencer: both finger orders carry the object and its kind",
   [(c["instance"], c["transparent"]) for c in sent] == [(2, False), (2, False)], sent)

# targets=none (pipeline:=raw): no filter and no labels, even with masks lying in the directory
dn = Path(tempfile.mkdtemp())
np.save(dn / "color_intrinsics.npy", K)
np.save(dn / "sam3_masks.npy", glass)
np.save(dn / "sam3_masks_all.npy", allm)
np.save(dn / "seg_masks.npy", (glass[0] > 0).astype(np.int32))
np.save(dn / "depth_aligned_mm.npy", depth)
(dn / "grasps_sbg.json").write_text(json.dumps({"depth_source": "depth_aligned_mm.npy",
                                                "grasps": [sbg_grasp(320, 240), sbg_grasp(450, 240), sbg_grasp(600, 60)]}))
nn = node(dn, True)
nn.targets, nn.transparent_mask, nn.depth_completed = "none", "", False
objs = nn._scene_objects()
out, _, _ = nn._build(objs)
ck("targets=none: no objects, every grasp executable (on the glass, the box and the table corner alike), none labelled",
   objs == () and [(g["executable"], g["recommendation"], g["on_target"], g["instance"]) for g in out]
   == [(True, "sbg", None, None)] * 3, (objs, [(g["executable"], g["recommendation"], g["instance"]) for g in out]))
ob = nn._obstacles(G.CAMERA_FRAME, None, objs)
ck("targets=none: the planner's scene is the raw depth's voxels, unlabelled (no target / instance marks)",
   ob is not None and ob["depth_source"] == "depth_aligned_mm.npy" and "instance" not in ob and "target" not in ob,
   None if ob is None else sorted(ob))
ck("targets=none: no target_objects.npy written", nn._objects(objs) == [] and not (dn / "target_objects.npy").exists())

# the trials' tape roll (block 2 warm-up): ~18 mm across its wall, so depth_blobs' 20 mm rule rejects it as
# "thin"; the node's 12 mm (blob_min_width_mm) takes it, with the segmenter's specks excluded as live
run = Path(os.environ.get("RG_RUNS", ROOT / "runs")) / "20261027_051427_auto"
if not (run / "seg_masks.npy").exists():
    print("  skip  %s: not under RG_RUNS (a trial run, in the trial archive)" % run.name)
if (run / "seg_masks.npy").exists():
    import cv2  # noqa: E402
    d, Kr, cov = np.load(run / "depth_aligned_mm.npy"), np.load(run / "color_intrinsics.npy"), np.load(run / "depth_coverage.npy")
    ex = cv2.dilate((np.load(run / "seg_masks.npy") > 0).astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))).astype(bool)
    _, i20, t20 = G.depth_blobs(d, Kr, ex, cov, min_width_m=0.020)
    _, i12, _ = G.depth_blobs(d, Kr, ex, cov, min_width_m=0.012)
    ck("tape roll (warm-up): not a target at the 20 mm width rule, a target at the node's 12 mm",
       not i20 and t20["rejected"]["thin"] > 0 and len(i12) >= 1 and all(i["height_mm"] > 35 for i in i12), (i20, i12))
    src = (ROOT / "ros/remake_grasp/remake_grasp/grasp_pose_node.py").read_text()
    ck("grasp_pose_node: blob_min_width_mm defaults to 12", "blob_min_width_mm=12.0," in src)
# the node itself, on the tape frames: one whole opaque target, its specks no longer cut out (the speck rule)
for stamp in ("20261027_051427_auto", "20261027_053707_auto"):
    run = Path(os.environ.get("RG_RUNS", ROOT / "runs")) / stamp
    if not (run / "seg_masks.npy").exists():
        continue
    tmp = Path(tempfile.mkdtemp())
    for f in ("seg_masks.npy", "depth_aligned_mm.npy", "color_intrinsics.npy", "depth_coverage.npy"):
        (tmp / f).write_bytes((run / f).read_bytes())
    nd = node(tmp, True)
    nd.targets, nd.target_mask, nd.transparent_mask, nd.depth_completed = "depth", "seg_masks.npy", "sam3_masks.npy", True
    nd.seg_min_area_m2, nd.blob_min_width_m, nd.last_seg = 9e-4, 0.012, None
    objs = nd._depth_targets()
    kinds = objs[1] if objs else {}
    ck(f"node, tape frame {stamp[9:15]}: the segmenter's specks dropped, the tape one opaque target",
       len(kinds) == 1 and not any(kinds.values()) and nd.last_seg["specks_dropped"] > 0, (kinds, nd.last_seg))

import yaml  # noqa: E402
cfg = yaml.safe_load((ROOT / "ros/remake_grasp/config/eye_in_hand.yaml").read_text())
ck("eye_in_hand.yaml: plan_pick is required", cfg["sequence"]["plan_pick"] == "required")

print("\nFAILED: " + ", ".join(fail) if fail else "\nall passed")
sys.exit(1 if fail else 0)
