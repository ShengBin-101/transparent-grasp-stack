"""SAM 3 text-prompted segmentation service.

Services (std_srvs/Trigger):
  /live_pipeline/segment_sam3      rgb.png + the transparency prompt -> sam3_masks.npy  (ReMake's mask in
                                   the BASELINE condition; parameter `prompt`, "transparent object")
  /live_pipeline/segment_sam3_all  rgb.png + the grasp prompt -> sam3_masks_all.npy     (PEGG-Net's input)

Masks are int32 instance labels (0 = background). Zero instances is reported as a failure:
a wrong prompt and an empty scene look the same. The grasp prompt comes from
live_dir/sam3_grasp_prompt.txt when it is there and non-empty, else the parameter `grasp_prompt`.
"""
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from rclpy.node import Node
from std_srvs.srv import Trigger

from .common import LIVE_DIR, THIRD_PARTY, declare, record_stage, run_node

DEFAULT_GRASP_PROMPT = "bottle or cup or box or can or jar or transparent object"


def split_concepts(prompt):
    """'a or b, c' -> ['a', 'b', 'c']. SAM 3 has no 'or', so each concept runs on its own."""
    out = []
    for part in re.split(r",|\bor\b", prompt):
        part = " ".join(part.split())
        if part and part not in out:
            out.append(part)
    return out or [prompt.strip()]


class Sam3Node(Node):
    """Loads SAM 3 once and serves the two segmentation services."""

    def __init__(self):
        super().__init__("sam3_segment_service")
        p = declare(self, live_dir=str(LIVE_DIR), prompt="transparent object", threshold=0.4,
                    hull=False, grasp_prompt=DEFAULT_GRASP_PROMPT, version="sam3")
        self.live_dir = Path(p["live_dir"])
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.version = p["version"]

        # SAM 3 expects an ambient bfloat16 autocast.
        if torch.cuda.is_available():
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if str(THIRD_PARTY / "sam3") not in sys.path:
            sys.path.insert(0, str(THIRD_PARTY / "sam3"))
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
        t0 = time.time()
        ckpt = download_ckpt_from_hf(version=self.version)
        self.processor = Sam3Processor(build_sam3_image_model(checkpoint_path=ckpt))
        self.get_logger().info(f"SAM 3 ready in {time.time() - t0:.1f}s, prompt {self._active_prompt()!r}")

        self.create_service(Trigger, "/live_pipeline/segment_sam3", self._handle)
        self.create_service(Trigger, "/live_pipeline/segment_sam3_all", self._handle_all)

    # ------------------------------------------------------------ prompts

    def _file_or_param(self, filename, param):
        """The prompt in live_dir/filename if non-empty, else the parameter."""
        f = self.live_dir / filename
        if f.exists() and f.read_text().strip():
            text = f.read_text().strip()
            if text != getattr(self, "_file_prompt_seen", None):   # once per new text, not every run
                self._file_prompt_seen = text
                self.get_logger().warning(f"parameter {param} overridden by {f}: {text!r}")
            return text
        return self.get_parameter(param).value

    def _active_prompt(self):
        return self.get_parameter("prompt").value

    # ------------------------------------------------------------ segmentation

    def _segment(self, rgb, prompt, thresh, hull=False):
        """-> (labels int32 HxW, scores, kept indices, per-concept scores).

        Runs each concept of the prompt separately, keeps scores >= thresh, paints the masks into
        one label map with the most confident on top, and drops masks left with < 50 px or < half
        their own area (duplicates across concepts). hull grows each instance to its convex hull.
        """
        # The processor has its own 0.5 floor; lower it so a threshold below 0.5 works.
        self.processor.confidence_threshold = min(0.5, float(thresh))
        state = self.processor.set_image(PILImage.fromarray(rgb))

        def tonp(x):
            return x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

        h, w = rgb.shape[:2]
        all_masks, all_scores, per_concept = [], [], {}
        for concept in split_concepts(prompt):
            out = self.processor.set_text_prompt(state=state, prompt=concept)
            s = tonp(out["scores"]).reshape(-1)
            all_masks.append(tonp(out["masks"]).reshape(len(s), h, w))
            all_scores.append(s)
            per_concept[concept] = [round(float(x), 3) for x in sorted(s, reverse=True) if x >= thresh]
        masks = np.concatenate(all_masks) if all_masks else np.zeros((0, h, w))
        scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
        idx = [i for i in np.argsort(scores)[::-1] if scores[i] >= thresh]

        # Paint weakest first so the most confident mask wins overlaps.
        labels = np.zeros((h, w), np.int32)
        for n, i in reversed(list(enumerate(idx, start=1))):
            labels[masks[i] > 0.5] = n
        keep = [n for n, i in enumerate(idx, start=1)
                if (labels == n).sum() >= 50 and (masks[i] > 0.5).sum() > 0
                and (labels == n).sum() / (masks[i] > 0.5).sum() >= 0.5]
        if len(keep) < len(idx):
            self.get_logger().info(f"{len(idx) - len(keep)} duplicate/occluded mask(s) dropped")
            remap = np.zeros(len(idx) + 1, np.int32)
            for new_n, old_n in enumerate(keep, start=1):
                remap[old_n] = new_n
            labels = remap[labels]
            idx = [idx[n - 1] for n in keep]
        if hull:
            labels = self._hull_fill(labels)
        return labels, scores, idx, per_concept

    @staticmethod
    def _hull_fill(labels):
        """Grow each instance to its convex hull, into background only."""
        out = labels.copy()
        for n in range(1, int(labels.max()) + 1):
            ys, xs = np.nonzero(labels == n)
            if len(xs) < 50:
                continue
            tmp = np.zeros(labels.shape, np.uint8)
            cv2.fillConvexPoly(tmp, cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32)), 1)
            out[(tmp > 0) & (out == 0)] = n
        return out

    @staticmethod
    def _overlay(rgb, labels, n, scores, idx):
        """BGR preview: instances in fixed random colours, each labelled with its score."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lut = np.vstack([[0, 0, 0], np.random.default_rng(0).integers(70, 255, (max(n, 1), 3))]).astype(np.uint8)
        vis = cv2.addWeighted(bgr, 0.55, lut[np.clip(labels, 0, n)], 0.45, 0)
        for k, i in enumerate(idx, start=1):
            ys, xs = np.nonzero(labels == k)
            if len(xs):
                cv2.putText(vis, f"{float(scores[i]):.2f}", (xs.min(), max(ys.min() - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return vis

    def _run(self, prompt, stem, meta_name, stage, response):
        """Segment rgb.png and write <stem>.npy/.png and meta_name; fails on zero instances (after saving)."""
        try:
            t0 = time.time()
            thresh = self.get_parameter("threshold").value
            hull = bool(self.get_parameter("hull").value) and stem == "sam3_masks"
            rgb = np.array(PILImage.open(self.live_dir / "rgb.png").convert("RGB"))
            labels, scores, idx, per_concept = self._segment(rgb, prompt, thresh, hull)
            np.save(self.live_dir / f"{stem}.npy", labels)
            cv2.imwrite(str(self.live_dir / f"{stem}.png"), self._overlay(rgb, labels, len(idx), scores, idx))
            (self.live_dir / meta_name).write_text(json.dumps(
                {"prompt": prompt, "n_instances": len(idx), "threshold": thresh, "hull": hull,
                 "scores": [float(s) for s in scores], "per_concept": per_concept,
                 "inference_s": round(time.time() - t0, 3)}, indent=2))
            record_stage(self.live_dir, stage,
                         {"model": f"SAM 3 ({self.version})", "prompt": prompt, "threshold": thresh,
                          "hull": hull, "concepts": list(per_concept), "writes": f"{stem}.npy"},
                         seconds=time.time() - t0,
                         extra={"n_instances": len(idx),
                                "per_concept": {c: len(v) for c, v in per_concept.items()}})
            if not idx:
                raise RuntimeError(f"SAM 3 returned 0 instances for {prompt!r}")
            msg = f"{stage} ok in {time.time() - t0:.2f}s, {len(idx)} instance(s) for {prompt!r} -> {stem}.npy"
            self.get_logger().info(msg)
            response.success, response.message = True, msg
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"{stage} failed: {e}")
            response.success, response.message = False, str(e)
        return response

    def _handle(self, request, response):
        return self._run(self._active_prompt(), "sam3_masks", "sam3_meta.json", "segment_sam3", response)

    def _handle_all(self, request, response):
        prompt = self._file_or_param("sam3_grasp_prompt.txt", "grasp_prompt")
        return self._run(prompt, "sam3_masks_all", "sam3_meta_all.json", "segment_sam3_all", response)


def main():
    run_node(Sam3Node)


if __name__ == "__main__":
    main()
