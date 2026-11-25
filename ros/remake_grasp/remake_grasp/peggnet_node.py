"""PEGG-Net service: one top-down grasp per object mask on the completed depth.

Service /live_pipeline/grasps_peggnet reads <depth_name>, <mask_name>, color_intrinsics.npy
(and depth_aligned_mm.npy for width measurement) and writes grasps_peggnet.json plus
grasps_peggnet_overlay.png (network crop) and grasps_peggnet_overlay_full.png.
Publishes (latched) grasps_peggnet_overlay, peggnet_quality(_raw) and grasp_markers.

recommendation per grasp:
  peggnet             confident and fits the gripper -> use aperture_m
  infeasible_gripper  the measured width is wider than max_aperture
  aperture_unknown    no usable width
  width_mismatch      the width head disagrees with the measured width (warning only)
  reject              below the quality gate
aperture_m is the OBJECT width (no clearance); None means unknown, never 0.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .common import (CAMERA_FRAME, LIVE_DIR, colourise, declare, image_msg, latched_qos,
                     record_stage, run_node, stamp)
from .peggnet_core import DEFAULT_CKPT, FITTED_OFFSET_SCHEDULE, GraspConfig, PeggNet


def _ap_mm(g):
    a = g.get("aperture_m")
    return "?" if a is None else f"{a * 1000:.0f}mm"


class PeggnetNode(Node):
    """Loads PEGG-Net once and serves /live_pipeline/grasps_peggnet."""

    def __init__(self):
        super().__init__("peggnet_grasp_service")
        p = declare(self, live_dir=str(LIVE_DIR), depth_name="remake_depth_mm.npy",
                    mask_name="sam3_masks_all.npy", measure_depth_name="depth_aligned_mm.npy",
                    checkpoint=str(DEFAULT_CKPT), frame_id=CAMERA_FRAME,
                    selection="per_object", normalisation="meanonly", aperture_source="feature",
                    quality_gate=0.5, aperture_offset_mm=-16.0, offset_schedule=True,
                    feature_width_tol_mm=25.0, min_feature_width_mm=2.0,
                    feature_retry_below_mm=5.0, min_finger_gap_mm=15.0,
                    width_tol_mm=10.0, max_aperture_mm=70.0, depth_readback_px=3)
        self.live_dir = Path(p["live_dir"])
        self.depth_name, self.mask_name = p["depth_name"], p["mask_name"]
        self.measure_depth_name = p["measure_depth_name"]
        self.frame_id = p["frame_id"]
        self.width_tol_m = p["width_tol_mm"] / 1000.0
        self.max_aperture_m = p["max_aperture_mm"] / 1000.0
        self.checkpoint = Path(p["checkpoint"])
        self.cfg = GraspConfig(
            selection=p["selection"], normalisation=p["normalisation"],
            aperture_offset_m=p["aperture_offset_mm"] / 1000.0, quality_gate=p["quality_gate"],
            aperture_source=p["aperture_source"], feature_width_tol_m=p["feature_width_tol_mm"] / 1000.0,
            min_feature_width_m=p["min_feature_width_mm"] / 1000.0,
            feature_retry_below_m=p["feature_retry_below_mm"] / 1000.0,
            min_finger_gap_m=p["min_finger_gap_mm"] / 1000.0, depth_readback_px=p["depth_readback_px"],
            aperture_offset_schedule=FITTED_OFFSET_SCHEDULE if p["offset_schedule"] else ())
        self.engine = PeggNet(self.checkpoint, self.cfg)
        self.get_logger().info(f"PEGG-Net ready ({self.checkpoint.name})")

        self.overlay_pub = self.create_publisher(Image, "/live_pipeline/grasps_peggnet_overlay", latched_qos())
        self.quality_pub = self.create_publisher(Image, "/live_pipeline/peggnet_quality", latched_qos())
        self.quality_raw_pub = self.create_publisher(Image, "/live_pipeline/peggnet_quality_raw", latched_qos())
        self.marker_pub = self.create_publisher(MarkerArray, "/live_pipeline/grasp_markers", latched_qos())
        self.create_service(Trigger, "/live_pipeline/grasps_peggnet", self._handle)

    # ------------------------------------------------------------ inference

    def _measure_depth(self, depth_m):
        """Sensor depth (m) for width measurement, resized to depth_m, or None if absent."""
        path = self.live_dir / self.measure_depth_name if self.measure_depth_name else None
        if path is None or not path.exists():
            return None
        d = np.load(path).astype(np.float32) / 1000.0
        if d.shape != depth_m.shape:
            d = cv2.resize(d, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)
        return d

    def _infer(self, depth_m, masks, K):
        """Run the engine and turn its grasps into service entries with a recommendation each.

        Feasibility uses a measured width (surface, else silhouette), never the prediction.
        """
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        raw, maps = self.engine.grasps(depth_m, masks if self.cfg.selection == "per_object" else None,
                                       fx=fx, measure_depth_m=self._measure_depth(depth_m))
        grasps = []
        for g in raw:
            fr, fc = g["full_rc"]
            z = g["z_m"]
            width_ref = g.get("local_feature_width_m", g.get("local_width_m"))
            feasible = None if width_ref is None else bool(width_ref <= self.max_aperture_m)
            accepted = g["accepted"]
            if g["aperture_m"] is None:
                rec, accepted = "aperture_unknown", False
            elif feasible is False:
                rec = "infeasible_gripper"
            elif accepted:
                rec = "peggnet"
            else:
                rec = "reject"
            predicted = g.get("aperture_predicted_m")
            if (rec == "peggnet" and predicted is not None and width_ref is not None
                    and abs(predicted - width_ref) > self.width_tol_m):
                rec = "width_mismatch"
            rr = int(np.clip(round(fr), 0, depth_m.shape[0] - 1))
            cc = int(np.clip(round(fc), 0, depth_m.shape[1] - 1))
            entry = {
                "label": None if g["label"] is None else int(g["label"]),
                "quality": g["quality"], "accepted": accepted, "recommendation": rec,
                "gripper_feasible": feasible,
                "center_px": [float(fc), float(fr)],
                "center_m": [(cc - cx) / fx * z, (rr - cy) / fy * z, z],
                "angle_rad": g["angle_rad"],
                "aperture_raw_m": g["aperture_raw_m"],
                "aperture_predicted_m": predicted,
                "aperture_source": g["aperture_source"],
                "aperture_m": g["aperture_m"],
                "aperture_undefined": g["aperture_m"] is None,
                "z_m": z, "z_readback_px": g["z_readback_px"], "z_valid_px": g["z_valid_px"],
                "z_centre_measured": g["z_centre_measured"],
            }
            if "local_width_m" in g:
                entry["local_width_m"] = g["local_width_m"]
            if "local_feature_width_m" in g:
                entry["local_feature_width_m"] = g["local_feature_width_m"]
                entry["feature_depth_source"] = g["feature_depth_source"]
            entry["aperture_offset_applied_m"] = g["aperture_offset_applied_m"]
            grasps.append(entry)
        grasps.sort(key=lambda e: e["z_m"])
        return grasps, maps

    # ------------------------------------------------------------ pictures

    @staticmethod
    def _draw(img, grasps, to_px, scale_ap):
        """Grasp bars on img. to_px(g) -> (row, col); scale_ap converts net pixels to img pixels."""
        for g in grasps:
            pr, pc = to_px(g)
            colour = (0, 255, 255) if g["accepted"] else (120, 120, 120)
            if g.get("_ap_px") is not None:
                w = g["_ap_px"] * scale_ap
                dr, dc = np.sin(g["angle_rad"]) * w / 2, -np.cos(g["angle_rad"]) * w / 2
                cv2.line(img, (int(pc + dc), int(pr + dr)), (int(pc - dc), int(pr - dr)), colour, 2)
            cv2.circle(img, (int(pc), int(pr)), 4, (0, 255, 0), -1)
            tag = "grasp" if g["label"] is None else str(g["label"])
            cv2.putText(img, f"{tag}:{_ap_mm(g)}", (int(pc) + 6, int(pr) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        return img

    def _overlay_crop(self, crop, pos, grasps, scale, off):
        """What the network saw: the 480x480 crop with the quality heat in red."""
        valid = crop[crop > 0]
        lo, hi = np.percentile(valid, [2, 98]) if valid.size else (0.0, 1.0)
        grey = np.clip((crop - lo) / max(hi - lo, 1e-6), 0, 1)
        img = (np.dstack([grey] * 3) * 255).astype(np.uint8)
        heat = np.clip(pos, 0, None)
        if heat.max() > 0:
            img[:, :, 2] = np.clip(img[:, :, 2] * 0.5 + heat / heat.max() * 255 * 0.9, 0, 255)
        r0, c0 = off
        return self._draw(img, grasps, lambda g: ((g["center_px"][1] - r0) / scale,
                                                  (g["center_px"][0] - c0) / scale), 1.0)

    def _quality_full(self, pos, scale, off, shape):
        """Quality map placed back in the full frame (NaN outside the network's crop)."""
        r0, c0 = off
        side = int(round(pos.shape[0] * scale))
        full = np.full(shape, np.nan, dtype=np.float32)
        resized = cv2.resize(pos.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
        r1, c1 = min(r0 + side, shape[0]), min(c0 + side, shape[1])
        full[r0:r1, c0:c1] = resized[:r1 - r0, :c1 - c0]
        img = np.zeros(shape + (3,), np.uint8)
        valid = np.isfinite(full)
        if valid.any():
            v = full[valid]
            lo, hi = float(v.min()), float(v.max())
            norm = np.zeros(shape, np.float32)
            norm[valid] = (v - lo) / max(hi - lo, 1e-6)
            col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            img[valid] = col[valid]
            cv2.putText(img, f"peggnet quality  {lo:.3f}-{hi:.3f}  gate {self.cfg.quality_gate}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return full, img

    def _markers(self, grasps):
        """A jaw line and a label per grasp with a known aperture, in the camera frame (5 s lifetime)."""
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        for i, g in enumerate(grasps):
            if g["aperture_m"] is None:
                continue
            ok = g["accepted"]
            cx, cy, cz = g["center_m"]
            half = g["aperture_m"] / 2.0
            dx, dy = -np.cos(g["angle_rad"]) * half, np.sin(g["angle_rad"]) * half
            line = Marker()
            line.header.frame_id, line.header.stamp = self.frame_id, now
            line.ns, line.id, line.type, line.action = "peggnet_grasp", i, Marker.LINE_LIST, Marker.ADD
            line.scale.x = 0.004
            line.color.r, line.color.g, line.color.b, line.color.a = (
                (0.0, 1.0, 1.0, 0.9) if ok else (0.6, 0.6, 0.6, 0.35))
            line.pose.orientation.w = 1.0
            line.points = [Point(x=cx - dx, y=cy - dy, z=cz), Point(x=cx + dx, y=cy + dy, z=cz)]
            line.lifetime.sec = 5
            txt = Marker()
            txt.header.frame_id, txt.header.stamp = self.frame_id, now
            txt.ns, txt.id, txt.type, txt.action = "peggnet_grasp_text", i, Marker.TEXT_VIEW_FACING, Marker.ADD
            txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = cx, cy, cz - 0.03
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.02
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = f"{g['label']}: {_ap_mm(g)} q={g['quality']:.2f}"
            txt.lifetime.sec = 5
            arr.markers += [line, txt]
        self.marker_pub.publish(arr)

    # ------------------------------------------------------------ service

    def _handle(self, request, response):
        """Run PEGG-Net on the live depth and masks, write grasps_peggnet.json and overlays, publish, record."""
        t0 = time.time()
        try:
            dpath, mpath = self.live_dir / self.depth_name, self.live_dir / self.mask_name
            need = [dpath, self.live_dir / "color_intrinsics.npy"]
            if self.cfg.selection == "per_object":
                need.append(mpath)
            for f in need:
                if not f.exists():
                    raise FileNotFoundError(f"missing {f.name}; run the depth and segmentation stages first")
            depth_m = np.load(dpath).astype(np.float32) / 1000.0
            masks = np.load(mpath).astype(np.int32) if mpath.exists() else np.zeros(depth_m.shape, np.int32)
            K = np.load(self.live_dir / "color_intrinsics.npy")
            if masks.shape != depth_m.shape:
                masks = cv2.resize(masks, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)
            if self.cfg.selection == "per_object" and not (masks > 0).any():
                raise RuntimeError(f"{self.mask_name} has no objects in it. SAM 3 found nothing: tune "
                                   f"{self.live_dir}/sam3_grasp_prompt.txt and re-run segment_sam3_all")

            grasps, maps = self._infer(depth_m, masks, K)
            dropped = maps["dropped"]
            counts = {}
            for g in grasps:
                counts[g["recommendation"]] = counts.get(g["recommendation"], 0) + 1
            payload = {
                "grasps": grasps, "recommendation_counts": counts,
                "n_dropped": len(dropped), "dropped": dropped,
                "depth_coverage": float((depth_m > 0).mean()), "depth_source": self.depth_name,
                "aperture_offset_m": self.cfg.aperture_offset_m, "quality_gate": self.cfg.quality_gate,
                "max_aperture_m": self.max_aperture_m, "aperture_source": self.cfg.aperture_source,
                "regime_note": "PEGG-Net is validated top-down only",
                "aperture_note": "aperture_m = object width at the grasp point, no clearance; None = unknown",
            }
            settings = {**self.cfg.as_dict(),
                        "aperture_offset_schedule": [[None if hi == float("inf") else round(hi * 1000, 1),
                                                      round(off * 1000, 1)]
                                                     for hi, off in self.cfg.aperture_offset_schedule],
                        "aperture_offset_schedule_units": "mm",
                        "measure_depth_name": self.measure_depth_name,
                        "max_aperture_m": self.max_aperture_m, "checkpoint": self.checkpoint.name}
            stamp(payload, "peggnet", settings)
            (self.live_dir / "grasps_peggnet.json").write_text(json.dumps(payload, indent=2))

            scale, off, fx = maps["scale"], maps["offset"], K[0, 0]
            for g in grasps:   # aperture in network pixels, for drawing only
                g["_ap_px"] = None if g["aperture_m"] is None else g["aperture_m"] / (scale * g["z_m"] / fx)
            cv2.imwrite(str(self.live_dir / "grasps_peggnet_overlay.png"),
                        self._overlay_crop(maps["crop"], maps["pos"], grasps, scale, off))
            full = self._draw(colourise(depth_m * 1000.0), grasps,
                              lambda g: (g["center_px"][1], g["center_px"][0]), scale)
            cv2.imwrite(str(self.live_dir / "grasps_peggnet_overlay_full.png"), full)
            now = self.get_clock().now().to_msg()
            self.overlay_pub.publish(image_msg(full, "bgr8", now, self.frame_id))
            q_raw, q_img = self._quality_full(maps["pos"], scale, off, depth_m.shape)
            self.quality_pub.publish(image_msg(q_img, "bgr8", now, self.frame_id))
            self.quality_raw_pub.publish(image_msg(q_raw, "32FC1", now, self.frame_id))
            for g in grasps:
                g.pop("_ap_px")
            self._markers(grasps)

            dt = time.time() - t0
            n_recon = sum(not g["z_centre_measured"] for g in grasps)
            record_stage(self.live_dir, "peggnet", settings, seconds=dt,
                         extra={"n_grasps": len(grasps), "n_dropped": len(dropped),
                                "n_z_reconstructed": n_recon,
                                "depth_coverage": float((depth_m > 0).mean()), "recommendations": counts})
            response.success, response.message = True, json.dumps({**payload, "timing_s": round(dt, 3)})
            self.get_logger().info(f"{len(grasps)} grasps ({sum(g['accepted'] for g in grasps)} accepted) "
                                   f"in {dt:.2f}s  " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        except Exception as exc:  # noqa: BLE001
            response.success, response.message = False, f"{type(exc).__name__}: {exc}"
            self.get_logger().error(response.message)
        return response


def main():
    run_node(PeggnetNode)


if __name__ == "__main__":
    main()
