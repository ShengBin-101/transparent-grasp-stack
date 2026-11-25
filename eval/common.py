"""Shared helpers for the TransCG benchmark: the sample list, data IO, saved predictions, metrics.

Copied from the project's training code (its eval/common.py, behind the ladder numbers): the paths changed and
its shape metrics are left out.
Uses numpy / cv2 / PIL / json only, so it imports in every model venv.

Conventions: a sample is one (scene, perspective, camera) of the TransCG test split, keyed
"s{scene:03d}_p{persp:03d}_c{cam}". Depth is float32 metres, 0 = missing. GT comes from the dataset
PNGs at native 1280x720 (D435 1000 units/m, L515 4000 units/m), and predictions are
nearest-upsampled to that size before scoring, the same for every model.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# The only change from the training code: where TransCG and the outputs live.
def _transcg_dir():
    """TransCG root: $RG_TRANSCG, else the repo's datasets/, else /datasets (start.sh's RG_DATASETS mount)."""
    if os.environ.get("RG_TRANSCG"):
        return Path(os.environ["RG_TRANSCG"])
    for d in (ROOT / "datasets" / "transcg" / "transcg", Path("/datasets/transcg/transcg")):
        if (d / "metadata.json").exists():
            return d
    return ROOT / "datasets" / "transcg" / "transcg"


DATA_DIR = _transcg_dir()
SUBSET_FILE = Path(__file__).resolve().parent / "sample_list.json"
ANALYSIS = Path(os.environ.get("RG_EVAL_OUT", str(ROOT / "data" / "eval")))   # predictions, metrics, masks

DEPTH_SCALE = {1: 1000.0, 2: 4000.0}      # PNG units per metre
CAMERA_NAME = {1: "D435", 2: "L515"}
GT_VALID_MIN = 0.01                       # metres; same rule as the repos' zero_mask
REFRACTION_THRESH_M = 0.010               # |raw - gt| above this = refraction pixel
THRESHOLDS = (1.01, 1.03, 1.05, 1.10, 1.25)


# --------------------------------------------------------------------------- #
# Sample list
# --------------------------------------------------------------------------- #
def build_full_test_list():
    """The full test list in the TransCG loaders' order: per test scene, every D435 perspective, then every L515."""
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    out = []
    for scene_id in meta["test"]:
        with open(DATA_DIR / f"scene{scene_id}" / "metadata.json") as f:
            sm = json.load(f)
        for p in sm["D435_valid_perspective_list"]:
            out.append({"scene": scene_id, "perspective": p, "camera": 1,
                        "scene_type": sm["type"], "model_list": sm["model_list"]})
        for p in sm["L515_valid_perspective_list"]:
            out.append({"scene": scene_id, "perspective": p, "camera": 2,
                        "scene_type": sm["type"], "model_list": sm["model_list"]})
    for i, s in enumerate(out):
        s["full_index"] = i
        s["key"] = sample_key(s)
    return out


def sample_key(s):
    return f"s{s['scene']:03d}_p{s['perspective']:03d}_c{s['camera']}"


def make_subset(stride=20):
    """Every stride-th sample of the full test list (how sample_list.json was made)."""
    full = build_full_test_list()
    sub = full[::stride]
    return {"stride": stride, "full_test_samples": len(full), "n": len(sub), "samples": sub}


def load_subset():
    with open(SUBSET_FILE) as f:
        return json.load(f)


def object_names():
    """id -> name from datasets/transcg/transcg/models/<id>-<name>.ply"""
    names = {}
    for p in (DATA_DIR / "models").glob("*.ply"):
        sid, _, name = p.stem.partition("-")
        names[int(sid)] = name
    return names


# --------------------------------------------------------------------------- #
# Raw data IO
# --------------------------------------------------------------------------- #
def sample_dir(s):
    return DATA_DIR / f"scene{s['scene']}" / str(s["perspective"])


def load_sample(s):
    """rgb uint8 HxWx3 (RGB order), raw/gt float32 metres HxW, mask bool HxW."""
    d = sample_dir(s)
    c = s["camera"]
    scale = DEPTH_SCALE[c]
    rgb = np.array(Image.open(d / f"rgb{c}.png").convert("RGB"), dtype=np.uint8)
    raw = np.array(Image.open(d / f"depth{c}.png"), dtype=np.float32) / scale
    gt = np.array(Image.open(d / f"depth{c}-gt.png"), dtype=np.float32) / scale
    mask = np.array(Image.open(d / f"depth{c}-gt-mask.png"), dtype=np.uint8) > 0
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
    return rgb, raw, gt, mask


# --------------------------------------------------------------------------- #
# Predictions on disk: lossless uint16 millimetres at the model's own output size
# --------------------------------------------------------------------------- #
def pred_path(model, key):
    return ANALYSIS / "predictions" / model / f"{key}.png"


def save_pred(model, key, pred_m):
    """Save a depth map in metres as a uint16 mm PNG; returns its path relative to ANALYSIS."""
    p = pred_path(model, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    pred_m = np.nan_to_num(pred_m.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mm = np.clip(np.round(pred_m * 1000.0), 0, 65535).astype(np.uint16)
    cv2.imwrite(str(p), mm)
    return str(p.relative_to(ANALYSIS))


def load_pred(model, key):
    """A saved prediction in metres at the model's output size, or None if there is none."""
    p = pred_path(model, key)
    if not p.exists():
        return None
    mm = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return mm.astype(np.float32) / 1000.0


# --------------------------------------------------------------------------- #
# Metrics (numpy).  All take metres and boolean masks of identical shape.
# --------------------------------------------------------------------------- #
def load_external_mask(mask_dir, key, hw):
    """A segmenter's transparent mask, resized to whatever the model wants.

    Both predict scripts feed the model the dataset's GROUND-TRUTH transparent
    mask by default, which no deployment has. `--mask-dir` swaps in a real one
    so the two model families can be compared on an input a robot could
    actually produce. Nearest-neighbour, because a mask is labels and not a
    signal; a missing file returns None so the caller can count it rather than
    silently score a frame with no mask.
    """
    p = Path(mask_dir) / f"{key}.png"
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != tuple(hw):
        m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def resize_to(pred, hw):
    h, w = hw
    if pred.shape == (h, w):
        return pred
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)


def region_metrics(pred, gt, sel, prefix):
    """RMSE, MAE, relative error and delta-threshold % over the pixels where sel is True, as a flat
    dict keyed <prefix>_<metric> (None for an empty selection)."""
    out = {}
    n = int(sel.sum())
    out[f"{prefix}_px"] = n
    if n == 0:
        for k in ("rmse", "mae", "rel"):
            out[f"{prefix}_{k}"] = None
        for t in THRESHOLDS:
            out[f"{prefix}_d{t:.2f}"] = None
        return out
    p = pred[sel].astype(np.float64)
    g = gt[sel].astype(np.float64)
    err = p - g
    out[f"{prefix}_rmse"] = float(np.sqrt(np.mean(err ** 2)))
    out[f"{prefix}_mae"] = float(np.mean(np.abs(err)))
    out[f"{prefix}_rel"] = float(np.mean(np.abs(err) / (g + 1e-8)))
    ratio = np.maximum(p / (g + 1e-8), g / (p + 1e-8))
    for t in THRESHOLDS:
        out[f"{prefix}_d{t:.2f}"] = float(100.0 * np.mean(ratio < t))
    return out


def input_regions(raw, gt, mask, valid):
    """Split mask & valid into the ReMake paper's three sensor conditions: reflection (no return),
    refraction (|raw - gt| > 10 mm) and normal."""
    m = mask & valid
    raw_valid = raw > GT_VALID_MIN
    reflection = m & ~raw_valid                                     # no return
    refraction = m & raw_valid & (np.abs(raw - gt) > REFRACTION_THRESH_M)
    normal = m & raw_valid & (np.abs(raw - gt) <= REFRACTION_THRESH_M)
    return reflection, refraction, normal


def unified_metrics(pred, gt, raw, mask):
    """All metrics for one prediction at native GT resolution: whole image, mask, background, the
    three sensor regions, plus sanity stats on the prediction."""
    pred = resize_to(pred, gt.shape)
    valid = gt > GT_VALID_MIN
    out = {}
    out.update(region_metrics(pred, gt, valid, "all"))
    out.update(region_metrics(pred, gt, mask & valid, "mask"))
    out.update(region_metrics(pred, gt, valid & ~mask, "bg"))
    refl, refr, norm = input_regions(raw, gt, mask, valid)
    out.update(region_metrics(pred, gt, refl, "reflection"))
    out.update(region_metrics(pred, gt, refr, "refraction"))
    out.update(region_metrics(pred, gt, norm, "normal"))
    # prediction sanity
    out["pred_min"] = float(pred.min())
    out["pred_max"] = float(pred.max())
    out["pred_zero_frac"] = float(np.mean(pred <= GT_VALID_MIN))
    out["pred_zero_frac_in_mask"] = float(np.mean(pred[mask] <= GT_VALID_MIN)) if mask.any() else None
    return out


def native_res_masked_rmse(pred, gt, mask):
    """Masked RMSE with GT nearest-downsampled to the prediction's size, closer to what each
    model repo's own metric code reports. Not used by the benchmark: it defines the "at prediction size"
    figures of the earlier model comparison."""
    h, w = pred.shape
    gt_s = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
    m_s = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    sel = m_s & (gt_s > GT_VALID_MIN)
    if not sel.any():
        return None
    return float(np.sqrt(np.mean((pred[sel].astype(np.float64) - gt_s[sel]) ** 2)))


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #
def append_jsonl(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
