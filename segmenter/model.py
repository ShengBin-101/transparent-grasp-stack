"""A prompt-free transparency detector: a small head on the frozen
DepthAnything-V2 encoder.

ReMake's fourth input is a per-pixel transparency mask, and it is the largest
lever in the data: 10.5 mm masked RMSE with the ground-truth mask, 30.7 mm at
SAM IoU 0.63, 55.8 mm with none (a ReMake retrained by this project). A promptable model (SAM 3 with a noun
phrase, SAM v1 with a box) segments what the prompt names; this head predicts
transparency from the pixels, so there is nothing to prompt.

Features: patch tokens of DINOv2 blocks 2, 5, 8 and 11 of the DepthAnything-V2
vits encoder (the layers its DPT head reads; no class token), concatenated to
1536 channels on the 37 x 49 token grid. Head: 1x1 conv to 256, two bilinear
x2 + 3x3 conv stages, a 1x1 logit, bilinear to the input size; 1.28 M trainable
parameters. The encoder is frozen. ReMake's relative-depth branch runs the same
encoder, so one pass could serve both, but on the robot they are separate
passes (seg_node and remake_node).

Input: RGB uint8 (RGB order; upstream's loader swaps twice and feeds
DepthAnything BGR), resized straight to 518x686 (H x W, the grid DepthAnything
uses for a 640x480 frame; a 1280x720 frame is squashed ~25 % horizontally, in
training and at deployment alike) and ImageNet-normalised. Output: logits at
the input size.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "remake"))    # the Depth Anything V2 code inside ReMake
from relat_depth_models.depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

INPUT_HW = (518, 686)                      # H, W: multiples of 14, 37 x 49 tokens
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

ENCODERS = {
    "vits": dict(features=64, out_channels=[48, 96, 192, 384]),
    "vitb": dict(features=128, out_channels=[96, 192, 384, 768]),
    "vitl": dict(features=256, out_channels=[256, 512, 1024, 1024]),
}
# Only the Small encoder's weights are Apache-2.0; Depth Anything V2 Base and Large are CC BY-NC 4.0,
# so vitb / vitl have no default here: pass weights=<path>. The released head is vits.
DEFAULT_WEIGHTS = {"vits": REPO / "checkpoints" / "depth_anything_v2_vits.pth"}


def preprocess(rgb):
    """uint8 HxWx3 RGB -> float tensor 3x518x686, ImageNet-normalised."""
    x = cv2.resize(rgb, (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_AREA
                   if rgb.shape[0] > INPUT_HW[0] else cv2.INTER_CUBIC)
    x = (x.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class TransSegHead(nn.Module):
    def __init__(self, encoder="vits", weights="default", use_valid=False, width=256):
        super().__init__()
        self.encoder, self.use_valid = encoder, use_valid
        self.backbone = DepthAnythingV2(encoder=encoder, **ENCODERS[encoder])
        if weights == "default":
            if encoder not in DEFAULT_WEIGHTS:
                raise ValueError(f"no default backbone weights for {encoder}: pass weights=<path>")
            weights = DEFAULT_WEIGHTS[encoder]
        if weights:
            sd = torch.load(str(weights), map_location="cpu", weights_only=False)
            self.backbone.load_state_dict(sd)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.layers = self.backbone.intermediate_layer_idx[encoder]
        dim = self.backbone.pretrained.embed_dim
        in_ch = 4 * dim + (1 if use_valid else 0)
        self.proj = nn.Sequential(nn.Conv2d(in_ch, width, 1), nn.BatchNorm2d(width), nn.ReLU(True))
        self.up1 = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.Conv2d(width, width // 2, 3, padding=1), nn.BatchNorm2d(width // 2), nn.ReLU(True))
        self.out = nn.Conv2d(width // 2, 1, 1)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()            # frozen: BN/dropout state never changes
        return self

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    @torch.no_grad()
    def features(self, x):
        feats = self.backbone.pretrained.get_intermediate_layers(
            x, self.layers, reshape=True, return_class_token=False)
        return torch.cat(feats, dim=1)    # B, 4*dim, H/14, W/14

    def forward(self, x, valid=None):
        f = self.features(x)
        if self.use_valid:
            if valid is None:
                raise ValueError("use_valid=True needs the sensor-validity map")
            v = F.interpolate(valid.float().unsqueeze(1), size=f.shape[-2:], mode="area")
            f = torch.cat([f, v], dim=1)
        y = self.proj(f)
        y = self.up1(F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False))
        y = self.up2(F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False))
        y = self.out(y)
        return F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]


def seg_loss(logits, target, eps=1.0):
    """BCE + soft Dice, both over the whole frame. target float {0,1}."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2))
    dice = 1 - (2 * inter + eps) / (p.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + eps)
    return bce + dice.mean(), {"bce": float(bce.detach()), "dice": float(dice.mean().detach())}


def save_checkpoint(path, model, **meta):
    head = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}
    torch.save({"head_state_dict": head, "encoder": model.encoder, "use_valid": model.use_valid,
                "width": model.proj[0].out_channels, **meta}, str(path))


def load_checkpoint(path, device="cpu", weights="default"):
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    m = TransSegHead(ck["encoder"], weights=weights, use_valid=ck["use_valid"], width=ck["width"])
    missing, unexpected = m.load_state_dict(ck["head_state_dict"], strict=False)
    bad = [k for k in missing if not k.startswith("backbone.")]
    assert not bad and not unexpected, (bad[:3], unexpected[:3])
    return m.to(device).eval(), ck
