"""Point clouds of the sensor depth and the completed depth, for Foxglove.

Service /live_pipeline/pointcloud publishes (latched, colour optical frame):
  cloud_sensor     depth_aligned_mm.npy, coloured from rgb.png
  cloud_completed  <depth_name>, coloured by `colour`:
                     source  green = sensor kept, magenta = filled by the model
                     rgb     camera colour
                     delta   completed - sensor, blue (nearer) to red (further)
  cloud_delta      only where the two differ by more than delta_min_mm
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger

from .common import CAMERA_FRAME, LIVE_DIR, cloud_msg, declare, latched_qos, record_stage, run_node


def backproject(depth_mm, K, stride=1):
    """-> (points N x 3 in metres, flat pixel index into the full image)."""
    h, w = depth_mm.shape
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    z = np.asarray(depth_mm, np.float32)[::stride, ::stride] / 1000.0
    ok = np.isfinite(z) & (z > 0)
    vv, uu, z = vv[ok], uu[ok], z[ok]
    pts = np.stack([(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z], axis=1)
    return pts.astype(np.float32), vv * w + uu


def _turbo(v01):
    """Values in [0, 1] -> TURBO colours as RGB floats in [0, 1]."""
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1]
    return lut[np.clip(np.asarray(v01) * 255.0, 0, 255).astype(np.uint8)].astype(np.float32) / 255.0


class PointCloudNode(Node):
    def __init__(self):
        super().__init__("pointcloud_node")
        p = declare(self, live_dir=str(LIVE_DIR), depth_name="remake_depth_mm.npy", mask_name="sam3_masks.npy",
                    colour="source", max_points=200000, delta_min_mm=2.0, frame_id=CAMERA_FRAME)
        self.live_dir = Path(p["live_dir"])
        self.depth_name, self.mask_name, self.colour = p["depth_name"], p["mask_name"], p["colour"]
        self.max_points, self.delta_min_mm, self.frame_id = p["max_points"], p["delta_min_mm"], p["frame_id"]
        q = latched_qos()
        self.pub_sensor = self.create_publisher(PointCloud2, "/live_pipeline/cloud_sensor", q)
        self.pub_completed = self.create_publisher(PointCloud2, "/live_pipeline/cloud_completed", q)
        self.pub_delta = self.create_publisher(PointCloud2, "/live_pipeline/cloud_delta", q)
        self.create_service(Trigger, "/live_pipeline/pointcloud", self._handle)

    def _colours(self, mode, idx, rgb, sensor, mask, delta):
        """Per-point colours for `mode` (see the module docstring); delta spans +-50 mm. None = default grey."""
        if mode == "rgb" and rgb is not None:
            return rgb.reshape(-1, 3)[idx]
        if mode == "source":
            measured = sensor[idx] > 0
            if mask is not None:
                measured &= ~mask[idx]
            c = np.empty((len(idx), 3), np.float32)
            c[measured], c[~measured] = (0.15, 0.85, 0.25), (0.95, 0.15, 0.85)
            return c
        if mode == "delta" and delta is not None:
            d = delta[idx]
            c = _turbo(np.clip(d / 100.0 + 0.5, 0, 1))
            c[~np.isfinite(d)] = (0.45, 0.45, 0.45)
            return c
        return None

    def _handle(self, request, response):
        """Publish the three clouds, strided to about max_points, and record delta statistics."""
        try:
            t0 = time.time()
            spath, kpath = self.live_dir / "depth_aligned_mm.npy", self.live_dir / "color_intrinsics.npy"
            if not (spath.exists() and kpath.exists()):
                raise FileNotFoundError("run /live_pipeline/capture first")
            sensor = np.nan_to_num(np.load(spath).astype(np.float32))
            K = np.load(kpath).astype(np.float64)
            cpath = self.live_dir / self.depth_name
            comp = np.nan_to_num(np.load(cpath).astype(np.float32)) if cpath.exists() else None
            bgr = cv2.imread(str(self.live_dir / "rgb.png"))
            rgb = None if bgr is None else bgr[:, :, ::-1].astype(np.float32) / 255.0
            mpath = self.live_dir / self.mask_name
            mask = None
            if self.mask_name and mpath.is_file():          # "" = no mask (remake:=none)
                m = np.load(mpath)
                mask = ((m > 0) if m.ndim == 2 else (m.sum(axis=0) > 0)).reshape(-1)

            n_valid = int((sensor > 0).sum())
            stride = 1 if self.max_points <= 0 or n_valid <= self.max_points else \
                int(np.ceil(np.sqrt(n_valid / float(self.max_points))))
            s_flat = sensor.reshape(-1)
            now = self.get_clock().now().to_msg()
            pts, idx = backproject(sensor, K, stride)
            self.pub_sensor.publish(cloud_msg(pts, self._colours("rgb", idx, rgb, s_flat, mask, None), now, self.frame_id))
            info = {"sensor_points": len(pts), "stride": stride, "completed_points": 0, "delta_points": 0}

            if comp is not None:
                delta = np.full(sensor.size, np.nan, np.float32)
                both = (s_flat > 0) & (comp.reshape(-1) > 0)
                delta[both] = comp.reshape(-1)[both] - s_flat[both]
                pts, idx = backproject(comp, K, stride)
                self.pub_completed.publish(cloud_msg(
                    pts, self._colours(self.colour, idx, rgb, s_flat, mask, delta), now, self.frame_id))
                big = np.isfinite(delta) & (np.abs(delta) > self.delta_min_mm)
                pts_d, idx_d = backproject(np.where(big, comp.reshape(-1), 0.0).reshape(sensor.shape), K, stride)
                self.pub_delta.publish(cloud_msg(
                    pts_d, self._colours("delta", idx_d, rgb, s_flat, mask, delta), now, self.frame_id))
                info.update(completed_points=len(pts), delta_points=len(pts_d),
                            model_only_frac=float(((comp > 0) & ~(sensor > 0)).mean()))
                fin = delta[np.isfinite(delta)]
                if fin.size:
                    info.update(delta_median_mm=float(np.median(fin)),
                                delta_p95_abs_mm=float(np.percentile(np.abs(fin), 95)))
            info["seconds"] = round(time.time() - t0, 3)
            record_stage(self.live_dir, "pointcloud",
                         {"depth_name": self.depth_name, "colour": self.colour, "max_points": self.max_points,
                          "delta_min_mm": self.delta_min_mm, "frame_id": self.frame_id},
                         seconds=time.time() - t0, extra=info)
            response.success, response.message = True, json.dumps(info)
        except Exception as exc:  # noqa: BLE001
            response.success, response.message = False, f"{type(exc).__name__}: {exc}"
            self.get_logger().error(response.message)
        return response


def main():
    run_node(PointCloudNode)


if __name__ == "__main__":
    main()
