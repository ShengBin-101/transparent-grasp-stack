"""Datasets for the transparency segmenter.

TransCG carries the exact label ReMake consumes -- `depth{c}-gt-mask.png`, one
per frame for both cameras -- on 34,191 training frames. The 8 held-out scenes of
segmenter/val_list.json are removed from training and used for epoch selection; the
frozen 1,177-frame test subset (eval/sample_list.json) is scored by
segmenter/predict.py, never seen in training.

ClearGrasp-real (a different lab, different objects, two cameras) is
EVALUATION ONLY. It is the cross-domain gate in segmenter/README.md: in-domain IoU
on TransCG proves little about the rig.

Data: TransCG as eval/common.py finds it ($RG_TRANSCG, datasets/transcg/transcg, or
/datasets/transcg/transcg); ClearGrasp's test-val release at $RG_CLEARGRASP, else
datasets/cleargrasp/cleargrasp-dataset-test-val, else /datasets/cleargrasp/....
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import common as C  # noqa: E402
from segmenter.augment import brightness_scale, chromatic_jitter, photometric_noise  # noqa: E402
from segmenter.model import INPUT_HW, preprocess  # noqa: E402

VAL_LIST = ROOT / "segmenter" / "val_list.json"


def _cleargrasp_dir():
    if os.environ.get("RG_CLEARGRASP"):
        return Path(os.environ["RG_CLEARGRASP"])
    for d in (ROOT / "datasets" / "cleargrasp" / "cleargrasp-dataset-test-val",
              Path("/datasets/cleargrasp/cleargrasp-dataset-test-val")):
        if d.exists():
            return d
    return ROOT / "datasets" / "cleargrasp" / "cleargrasp-dataset-test-val"


CG_ROOT = _cleargrasp_dir()
DEPTH_SCALE = C.DEPTH_SCALE


def _resize_mask(m, hw):
    return cv2.resize(m.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) > 0


def augment(rgb, mask, valid, rng):
    """Train-time augmentation. Geometric ops hit all three arrays; photometric
    ops hit RGB only. Brightness gain reaches 0.35x because the rig's failing
    capture was at 15/255 (augment.py::brightness_scale)."""
    h, w = mask.shape
    s = rng.uniform(0.6, 1.0)                       # random resized crop, aspect kept
    ch, cw = int(round(h * s)), int(round(w * s))
    y0, x0 = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    rgb, mask, valid = rgb[y0:y0 + ch, x0:x0 + cw], mask[y0:y0 + ch, x0:x0 + cw], valid[y0:y0 + ch, x0:x0 + cw]
    if rng.random() < 0.5:
        rgb, mask, valid = rgb[:, ::-1], mask[:, ::-1], valid[:, ::-1]
    if rng.random() < 0.5:
        rgb, mask, valid = rgb[::-1], mask[::-1], valid[::-1]
    rgb = np.ascontiguousarray(rgb)
    rgb = brightness_scale(rgb, rng, 0.35, 1.4)
    if rng.random() < 0.8:
        rgb = chromatic_jitter(rgb, rng, 0.1)
    if rng.random() < 0.5:
        rgb = photometric_noise(rgb, rng, 0.1)
    return rgb, np.ascontiguousarray(mask), np.ascontiguousarray(valid)


def pack(rgb, mask, valid, key):
    return {"rgb": preprocess(rgb),
            "mask": torch.from_numpy(_resize_mask(mask, INPUT_HW).astype(np.float32)),
            "valid": torch.from_numpy(_resize_mask(valid, INPUT_HW).astype(np.float32)),
            "key": key}


class TransCGSeg(Dataset):
    """split: 'train' (train scenes minus the val scenes), 'val' (the 600-frame
    val list), or 'samples' (an explicit list of eval/common-style samples)."""

    def __init__(self, split="train", samples=None, aug=None, data_dir=None, limit=0):
        self.data_dir = Path(data_dir) if data_dir else C.DATA_DIR
        self.aug = (split == "train") if aug is None else aug
        with open(VAL_LIST) as f:
            vl = json.load(f)
        if split == "train":
            with open(self.data_dir / "metadata.json") as f:
                meta = json.load(f)
            held = set(vl["val_scenes"])
            self.samples = []
            for scene in meta["train"]:
                if scene in held:
                    continue
                with open(self.data_dir / f"scene{scene}" / "metadata.json") as f:
                    sm = json.load(f)
                for cam, key in ((1, "D435_valid_perspective_list"), (2, "L515_valid_perspective_list")):
                    for p in sm[key]:
                        self.samples.append({"scene": scene, "perspective": p, "camera": cam})
        elif split == "val":
            self.samples = vl["samples"]
        elif split == "samples":
            self.samples = list(samples)
        else:
            raise ValueError(split)
        for s in self.samples:
            s.setdefault("key", C.sample_key(s))
        if limit:
            self.samples = self.samples[:limit]

    def __len__(self):
        return len(self.samples)

    def load(self, i):
        s = self.samples[i]
        d = self.data_dir / f"scene{s['scene']}" / str(s["perspective"])
        c = s["camera"]
        rgb = cv2.cvtColor(cv2.imread(str(d / f"rgb{c}.png")), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(d / f"depth{c}-gt-mask.png"), cv2.IMREAD_UNCHANGED) > 0
        raw = cv2.imread(str(d / f"depth{c}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE[c]
        valid = np.nan_to_num(raw) > C.GT_VALID_MIN
        return rgb, mask, valid, s["key"]

    def __getitem__(self, i):
        rgb, mask, valid, key = self.load(i)
        if self.aug:
            rng = np.random.default_rng(np.random.randint(1 << 31))   # torch seeds np in workers
            rgb, mask, valid = augment(rgb, mask, valid, rng)
        return pack(rgb, mask, valid, key)


def _read_exr_depth(path):
    import Imath
    import OpenEXR
    f = OpenEXR.InputFile(str(path))
    dw = f.header()["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    ch = np.frombuffer(f.channel("R", Imath.PixelType(Imath.PixelType.FLOAT)), np.float32)
    return np.nan_to_num(ch.reshape(h, w).copy(), nan=0.0, posinf=0.0, neginf=0.0)


class ClearGraspReal(Dataset):
    """ClearGrasp real-test / real-val, d415 + d435. Evaluation only."""

    def __init__(self, split="real-test", cameras=("d415", "d435"), root=CG_ROOT, need_valid=False, limit=0):
        self.items, self.need_valid = [], need_valid
        for cam in cameras:
            for m in sorted((Path(root) / split / cam).glob("*-mask.png")):
                stem = m.name[:-len("-mask.png")]
                self.items.append((Path(root) / split / cam, stem, f"cg_{split}_{cam}_{stem}"))
        if limit:
            self.items = self.items[:limit]

    def __len__(self):
        return len(self.items)

    def load(self, i):
        d, stem, key = self.items[i]
        rgb = cv2.cvtColor(cv2.imread(str(d / f"{stem}-transparent-rgb-img.jpg")), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(d / f"{stem}-mask.png"), cv2.IMREAD_UNCHANGED) > 0
        if self.need_valid:
            dep = _read_exr_depth(d / f"{stem}-transparent-depth-img.exr")
            if dep.shape != mask.shape:
                dep = cv2.resize(dep, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            valid = dep > C.GT_VALID_MIN
        else:
            valid = np.ones(mask.shape, bool)
        return rgb, mask, valid, key

    def __getitem__(self, i):
        return pack(*self.load(i))
