"""Prompt-free transparency detector service: the mask ReMake completes, without a prompt.

    /live_pipeline/segment_trans   rgb.png -> seg_masks.npy (+ seg_masks.png overlay)
    /live_pipeline/seg_overlay     that overlay, the mask in magenta over the photo (latched bgr8, for Foxglove)

The model (segmenter/) is a 1.28 M-parameter head on the frozen DepthAnything-V2 vits encoder, trained on
TransCG's transparency masks. It labels transparent pixels only (labels are connected components, opaque objects
never appear), so it replaces SAM 3's "transparent object" step. seg_masks.npy follows the sam3_masks.npy contract
(int32 labels, 0 = background); mask:=seg in remake.launch.py / pipeline.launch.py points ReMake and the grasp-pose
node at it, and targets:=depth takes its components as the transparent targets. The encoder pass here is separate
from remake_node's (same weights, not shared). An orange plate with a black foam handle is also marked transparent.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from .common import (CAMERA_FRAME, CKPT_DIR, LIVE_DIR, ROOT, declare, image_msg, latched_qos, record_stage,
                     run_node)


class SegNode(Node):
    """Loads the segmenter once (backbone = the DepthAnything-V2 vits file ReMake uses) and serves segment_trans."""

    def __init__(self):
        super().__init__("seg_trans_service")
        p = declare(self, live_dir=str(LIVE_DIR), checkpoint=str(CKPT_DIR / "seg_vits_rgb_best.pt"),
                    backbone=str(CKPT_DIR / "depth_anything_v2_vits.pth"), mask_name="seg_masks.npy",
                    thresh=0.5, dilate=0, fp16=True, frame_id=CAMERA_FRAME)
        self.live_dir = Path(p["live_dir"])
        self.mask_name, self.thresh, self.dilate = p["mask_name"], float(p["thresh"]), int(p["dilate"])
        self.fp16, self.frame_id = bool(p["fp16"]), p["frame_id"]
        self.checkpoint = Path(p["checkpoint"])
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from segmenter.live_segment import segment
        from segmenter.model import load_checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.time()
        self.model, ck = load_checkpoint(self.checkpoint, self.device, weights=p["backbone"])
        self.segment = segment
        self.meta = {"epoch": ck.get("epoch"), "val_iou": ck.get("val_iou")}
        self.get_logger().info(f"segmenter ready in {time.time() - t0:.1f}s on {self.device} "
                               f"({self.checkpoint.name}, epoch {self.meta['epoch']}, val IoU {self.meta['val_iou']})")
        self.pub_overlay = self.create_publisher(Image, "/live_pipeline/seg_overlay", latched_qos())
        self.create_service(Trigger, "/live_pipeline/segment_trans", self._handle)

    def _handle(self, request, response):
        """Segment rgb.png, save the label map and a magenta overlay, publish the overlay, record the stage.

        Zero components is an answer, not a failure: with no prompt to get wrong, an empty mask
        means nothing transparent.
        """
        try:
            t0 = time.time()
            bgr = cv2.imread(str(self.live_dir / "rgb.png"))
            if bgr is None:
                raise FileNotFoundError(f"{self.live_dir}/rgb.png")
            labels, n, frac = self.segment(self.model, bgr, self.device, self.thresh, self.dilate, self.fp16)
            np.save(self.live_dir / self.mask_name, labels)
            vis = bgr.copy()
            on = labels > 0
            vis[on] = (0.45 * vis[on] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
            cv2.imwrite(str(self.live_dir / Path(self.mask_name).with_suffix(".png")), vis)
            self.pub_overlay.publish(image_msg(vis, "bgr8", self.get_clock().now().to_msg(), self.frame_id))
            secs = time.time() - t0
            record_stage(self.live_dir, "segment_trans",
                         {"model": "transparency head on DepthAnything-V2 vits (frozen)",
                          "checkpoint": self.checkpoint.name, "epoch": self.meta["epoch"],
                          "thresh": self.thresh, "dilate_px": self.dilate, "fp16": self.fp16,
                          "writes": self.mask_name},
                         seconds=secs, extra={"n_components": int(n), "mask_frac": frac})
            msg = f"segment_trans ok in {secs:.2f}s, {n} component(s), {frac * 100:.1f}% of frame -> {self.mask_name}"
            self.get_logger().info(msg)
            response.success, response.message = True, msg
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"segment_trans failed: {e}")
            response.success, response.message = False, str(e)
        return response


def main():
    run_node(SegNode)


if __name__ == "__main__":
    main()
