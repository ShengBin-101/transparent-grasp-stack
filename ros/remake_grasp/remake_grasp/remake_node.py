"""ReMake depth completion service.

/live_pipeline/depth_remake reads rgb.png, depth_aligned_mm.npy, color_intrinsics.npy, depth_coverage.npy and
the transparency mask (mask_name), and writes remake_depth_mm.npy (network depth inside the mask, and in sensor holes
only with fill_holes, sensor depth elsewhere, nothing outside the depth camera's view), remake_depth_net_mm.npy (raw network output)
and remake_depth_vis.png. Publishes latched /live_pipeline/depth_completed (32FC1 m, NaN = none) and _vis.
Weights and mask are parameters, preset by remake.launch.py (remake:=, mask:=); rel_rgb_fix must match the
checkpoint (True for L_combo_both, False for the released weights).

The relative-depth branch runs its own pass of the DepthAnything-V2 vits encoder. seg_node runs the same encoder
for the mask in a separate pass; nothing is shared between the two nodes.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from . import stubs
from .align import composite_depth
from .common import (CAMERA_FRAME, CKPT_DIR, LIVE_DIR, THIRD_PARTY, colourise, declare, image_msg,
                     latched_qos, record_stage, run_node, save_preview)

NET_W, NET_H = 640, 480
# Input clip (upstream's default): < 0.3 m dropped, > 1.5 m set to 1.5. The training configs used 0-10 m.
DEPTH_MIN_M, DEPTH_MAX_M = 0.3, 1.5


def load_remake(checkpoint, reldepth_checkpoint, device):
    """Load the vendored ReMake (third_party/remake) -> (model, frozen DepthAnythingV2-vits, process_data, epoch)."""
    path = str(THIRD_PARTY / "remake")
    if path not in sys.path:
        sys.path.insert(0, path)
    stubs.install()
    from models.remake import ReMake
    from relat_depth_models import DepthAnythingV2
    from utils.data_preparation import process_data

    model = ReMake.build(lambda_val=1, res=True).to(device)
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    rel = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    rel.load_state_dict(torch.load(str(reldepth_checkpoint), map_location=device))
    for prm in rel.parameters():
        prm.requires_grad = False
    return model, rel.to(device).eval(), process_data, ckpt.get("epoch")


class RemakeNode(Node):
    """Loads ReMake once at start-up and serves /live_pipeline/depth_remake."""

    def __init__(self):
        super().__init__("remake_depth_service")
        # allow_no_mask, composite and fill_out_of_field are debug switches; the trials ran with the defaults
        # (a mask required, ReMake composited onto the sensor depth, nothing filled outside the depth camera's view).
        p = declare(self, live_dir=str(LIVE_DIR), checkpoint=str(CKPT_DIR / "remake_official.tar"),
                    reldepth_checkpoint=str(CKPT_DIR / "depth_anything_v2_vits.pth"),
                    mask_name="sam3_masks.npy", mask_dilate=5, allow_no_mask=False,
                    composite=True, fill_out_of_field=False, frame_id=CAMERA_FRAME, rel_rgb_fix=False,
                    fill_holes=False)
        self.live_dir = Path(p["live_dir"])
        self.mask_name, self.mask_dilate = p["mask_name"], int(p["mask_dilate"])
        self.allow_no_mask, self.composite = p["allow_no_mask"], p["composite"]
        # False: model depth inside the mask only, the sensor's holes elsewhere left empty (align.composite_depth)
        self.fill_holes = bool(p["fill_holes"])
        self.fill_out_of_field, self.frame_id = p["fill_out_of_field"], p["frame_id"]
        self.checkpoint_name, self.rel_rgb_fix = Path(p["checkpoint"]).name, bool(p["rel_rgb_fix"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.time()
        self.model, self.reldepth, self.process_data, self.epoch = load_remake(
            p["checkpoint"], p["reldepth_checkpoint"], self.device)
        self.get_logger().info(f"ReMake ready in {time.time() - t0:.1f}s on {self.device} ({self.checkpoint_name}, "
                               f"epoch {self.epoch}, mask {self.mask_name}, rel_rgb_fix {self.rel_rgb_fix})")
        self.pub_depth = self.create_publisher(Image, "/live_pipeline/depth_completed", latched_qos())
        self.pub_vis = self.create_publisher(Image, "/live_pipeline/depth_completed_vis", latched_qos())
        self.create_service(Trigger, "/live_pipeline/depth_remake", self._handle)

    def _coverage(self, shape):
        """Depth camera's field of view as a bool mask, or None to composite over the whole frame."""
        p = self.live_dir / "depth_coverage.npy"
        if self.fill_out_of_field:
            return None
        if not p.exists():
            self.get_logger().warning("depth_coverage.npy missing: compositing over the whole frame")
            return None
        cov = np.load(p)
        if cov.shape != shape:
            cov = cv2.resize(cov.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return cov.astype(bool)

    def _mask(self, shape):
        """-> (grown mask, tight mask, note) or (None, None, why).

        The network gets the mask grown by mask_dilate (5 px); compositing uses the tight mask so
        the grown ring keeps the sensor's depth. Accepts (H, W) labels or (N, H, W) instances.
        """
        p = self.live_dir / self.mask_name
        if not p.exists():
            return None, None, f"{self.mask_name} not found"
        m = np.load(p)
        mask = (m > 0) if m.ndim == 2 else (m.sum(axis=0) > 0)
        if mask.shape != shape:
            mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        n_inst = int(np.unique(m[m > 0]).size) if m.ndim == 2 else int(m.shape[0])
        tight = mask.copy()
        if self.mask_dilate > 0:
            k = np.ones((2 * self.mask_dilate + 1,) * 2, np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
        return mask, tight, f"{n_inst} instance(s), {mask.mean() * 100:.1f}% of frame"

    def complete(self, bgr, depth_mm, K, mask):
        """Run ReMake at 640x480 with upstream preprocessing; returns completed depth (mm) at input size.

        With rel_rgb_fix the frozen DepthAnything branch gets RGB, as checkpoints trained with
        the project's fix (L_combo_both, training/) expect; upstream converts twice and feeds it BGR, which the
        released checkpoints were trained on. The output typically changes by < 0.1 mm, but a
        checkpoint is run as it was trained.
        """
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2RGB)
        rgb_r = cv2.resize(rgb, (NET_W, NET_H), interpolation=cv2.INTER_LINEAR)
        depth_r = cv2.resize(depth_mm, (NET_W, NET_H), interpolation=cv2.INTER_NEAREST)
        mask_r = cv2.resize(mask.astype(np.uint8), (NET_W, NET_H), interpolation=cv2.INTER_NEAREST)
        Ks = K.copy()
        Ks[0, :] *= NET_W / W
        Ks[1, :] *= NET_H / H
        # Upstream preprocessing, unchanged. camera_type=1 converts mm to m.
        d = self.process_data(rgb_r, depth_r, depth_r.copy(), mask_r, Ks, scene_type="cluttered",
                              camera_type=1, split="test", image_size=(NET_W, NET_H),
                              depth_min=DEPTH_MIN_M, depth_max=DEPTH_MAX_M, depth_norm=1.0,
                              use_aug=False, use_depth_aug=False, no_mask_depth=False,
                              reldepth_model="depthanything")
        if self.rel_rgb_fix:
            # Rebuild the branch input from BGR so img_rel_preprocess's own swap lands on RGB.
            from utils.data_preparation import img_rel_preprocess
            bgr_r = np.ascontiguousarray(rgb_r[:, :, ::-1])
            d["rgb_relat"] = torch.FloatTensor(img_rel_preprocess(cv2.resize(bgr_r, (518, 686))))
        dev = self.device
        with torch.no_grad():
            rel = self.reldepth.forward(d["rgb_relat"].unsqueeze(0).to(dev)).unsqueeze(1)
            rel = F.interpolate(rel, size=(NET_H, NET_W), mode="bilinear", align_corners=False)
            out = self.model(d["rgb"].unsqueeze(0).to(dev), rel, d["depth"].unsqueeze(0).to(dev),
                             torch.from_numpy(mask_r.astype(np.float32)).unsqueeze(0).to(dev))
        pred = cv2.resize(out[0, 0].float().cpu().numpy() * 1000.0, (W, H),
                          interpolation=cv2.INTER_LINEAR).astype(np.float32)
        pred[~np.isfinite(pred)] = 0.0
        return pred

    def _handle(self, request, response):
        """Complete the live frame's depth, composite it onto the sensor depth, save, publish, record.

        Without a mask it fails unless allow_no_mask (a pass-through control run). Warns when the
        input median is outside the training range or the seam outside the mask exceeds 10 mm.
        """
        try:
            t0 = time.time()
            bgr = cv2.imread(str(self.live_dir / "rgb.png"), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"{self.live_dir}/rgb.png")
            H, W = bgr.shape[:2]
            depth_mm = np.nan_to_num(np.load(self.live_dir / "depth_aligned_mm.npy").astype(np.float32))
            K = np.load(self.live_dir / "color_intrinsics.npy").astype(np.float32)
            coverage = self._coverage((H, W))
            mask, tight, note = self._mask((H, W))
            if mask is None:
                if not self.allow_no_mask:
                    raise FileNotFoundError(f"{note}. ReMake only completes depth inside a mask: run "
                                            f"the mask stage first (segment_sam3, or segment_trans for "
                                            f"mask:=seg), or set allow_no_mask")
                mask = tight = np.zeros((H, W), bool)
                note = "none (pass-through control)"
            valid_in = depth_mm > 0
            median_in = float(np.median(depth_mm[valid_in])) if valid_in.any() else 0.0
            if not DEPTH_MIN_M * 1000 <= median_in <= DEPTH_MAX_M * 1000:
                self.get_logger().warning(f"input median {median_in:.0f}mm is outside the 300-1500mm "
                                          f"clip range; it will be clipped")

            refined = self.complete(bgr, depth_mm, K, mask)
            stats = None
            if self.composite:
                np.save(self.live_dir / "remake_depth_net_mm.npy", refined)
                refined, stats = composite_depth(refined, depth_mm, tight, valid_region=coverage,
                                                 fill_holes=self.fill_holes)
                if stats["seam_mm"] is not None and stats["seam_mm"] > 10.0:
                    self.get_logger().warning(f"seam {stats['seam_mm']:.1f}mm outside the mask: "
                                              f"the mask is probably misplaced")
            np.save(self.live_dir / "remake_depth_mm.npy", refined)
            save_preview(self.live_dir / "remake_depth_vis.png", refined,
                         "remake (composited onto the sensor)" if self.composite else "remake (completed)")

            now = self.get_clock().now().to_msg()
            depth_out = refined / 1000.0
            depth_out[refined <= 0] = np.nan
            self.pub_depth.publish(image_msg(depth_out, "32FC1", now, self.frame_id))
            self.pub_vis.publish(image_msg(colourise(refined), "bgr8", now, self.frame_id))

            secs = time.time() - t0
            valid = refined[refined > 0]
            record_stage(self.live_dir, "depth_remake",
                         {"model": "ReMake + DepthAnythingV2 vits (frozen)", "checkpoint": self.checkpoint_name,
                          "checkpoint_epoch": self.epoch, "rel_rgb_fix": self.rel_rgb_fix,
                          "mask_name": self.mask_name, "mask_dilate_px": self.mask_dilate,
                          "input_size": [NET_W, NET_H], "depth_range_m": [DEPTH_MIN_M, DEPTH_MAX_M],
                          "preprocessing": "upstream utils.data_preparation.process_data",
                          "composite": self.composite, "fill_out_of_field": self.fill_out_of_field,
                          "fill_holes": self.fill_holes},
                         seconds=secs,
                         extra={"coverage": float((refined > 0).mean()), "mask_frac": float(mask.mean()),
                                "input_median_mm": median_in, **({"composite": stats} if stats else {})})
            msg = (f"remake ok in {secs:.2f}s, median {np.median(valid):.1f}mm, "
                   f"coverage {(refined > 0).mean():.2f}, mask {note}")
            self.get_logger().info(msg)
            response.success, response.message = True, msg
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"remake failed: {e}")
            response.success, response.message = False, str(e)
        return response


def main():
    run_node(RemakeNode)


if __name__ == "__main__":
    main()
