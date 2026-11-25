"""Scale-Balanced Grasp service: 6-DoF grasps from the completed depth.

Service /live_pipeline/grasps_sbg reads <depth_name>, color_intrinsics.npy and rgb.png
from the live dir and writes grasps_sbg.json / .npy / _overlay.png.
Publishes (latched) grasps_sbg_overlay, grasp_markers_sbg and sbg_cloud.
With approach_cone_deg set, "down" comes from the live tool pose (approach_axis_source).
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .common import (CAMERA_FRAME, LIVE_DIR, cloud_msg, declare, image_msg, latched_qos,
                     record_stage, run_node, stamp)
from .sbg_core import (DEFAULT_CKPT, ScaleBalancedGrasp, SbgConfig, draw_grasps, gripper_segments,
                       score_colour_bgr)


class SbgNode(Node):
    """Loads SBG once and serves /live_pipeline/grasps_sbg; tracks the tool pose for the approach cone."""

    def __init__(self):
        super().__init__("sbg_grasp_service")
        p = declare(self, live_dir=str(LIVE_DIR), depth_name="remake_depth_mm.npy",
                    checkpoint=str(DEFAULT_CKPT), frame_id=CAMERA_FRAME,
                    num_points=20000, top_k=50, z_min_m=0.15, z_max_m=1.5,
                    nms=True, collision_check=True, collision_thresh=0.01,
                    score_gate=0.0, max_aperture_mm=70.0, seed=0,
                    approach_cone_deg=0.0, approach_axis_source="tool_pose",
                    empty_check=False, empty_thresh=0.01)
        self.live_dir = Path(p["live_dir"])
        self.depth_name = p["depth_name"]
        self.frame_id = p["frame_id"]
        self.cfg = SbgConfig(num_points=p["num_points"], top_k=p["top_k"], z_min_m=p["z_min_m"],
                             z_max_m=p["z_max_m"], nms=p["nms"], collision_check=p["collision_check"],
                             collision_thresh=p["collision_thresh"], score_gate=p["score_gate"],
                             max_aperture_m=p["max_aperture_mm"] / 1000.0 if p["max_aperture_mm"] > 0 else None,
                             seed=None if p["seed"] < 0 else p["seed"],
                             approach_cone_deg=p["approach_cone_deg"] or None,
                             empty_check=bool(p["empty_check"]), empty_thresh=float(p["empty_thresh"]))
        # "Down" for the cone: tool pose + camera mounting ("tool_pose"), else the optical axis,
        # which is right only when the camera looks straight down.
        self.axis_source = p["approach_axis_source"]
        self._tool, self._tool_at = None, 0.0
        self._ee_to_camera = None
        if self.cfg.approach_cone_deg and self.axis_source == "tool_pose":
            import yaml
            from ament_index_python.packages import get_package_share_directory
            cfg_path = Path(get_package_share_directory("remake_grasp")) / "config" / "eye_in_hand.yaml"
            self._ee_to_camera = (yaml.safe_load(cfg_path.read_text()) or {}).get("ee_to_camera")
            self.create_subscription(PoseStamped, "/live_pipeline/tool_pose", self._on_tool, 10)
        if self.cfg.approach_cone_deg:
            self.get_logger().info(f"approach cone {self.cfg.approach_cone_deg:.0f} deg about 'down' "
                                   f"(from {self.axis_source})")
        self.get_logger().info("loading Scale-Balanced-Grasp ...")
        t0 = time.time()
        self.engine = ScaleBalancedGrasp(Path(p["checkpoint"]), self.cfg)
        self.get_logger().info(f"ready in {time.time() - t0:.1f}s (epoch {self.engine.epoch})")

        self.overlay_pub = self.create_publisher(Image, "/live_pipeline/grasps_sbg_overlay", latched_qos())
        self.marker_pub = self.create_publisher(MarkerArray, "/live_pipeline/grasp_markers_sbg", latched_qos())
        self.cloud_pub = self.create_publisher(PointCloud2, "/live_pipeline/sbg_cloud", latched_qos())
        self.create_service(Trigger, "/live_pipeline/grasps_sbg", self._handle)

    def _on_tool(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        self._tool, self._tool_at = ((p.x, p.y, p.z), (o.x, o.y, o.z, o.w)), time.monotonic()

    def _down_in_camera(self):
        """(base -z in the camera frame, "tool_pose") from a tool pose under 2 s old, else (+z, "optical")."""
        if self.axis_source == "tool_pose" and self._ee_to_camera and self._tool is not None \
                and time.monotonic() - self._tool_at < 2.0:
            from . import eye_in_hand as eh
            _, q = eh.compose(self._tool, eh.seven(self._ee_to_camera, "ee_to_camera"))
            R = np.asarray(eh.matrix_from_quat(q))        # camera axes in the base frame
            return R.T.dot([0.0, 0.0, -1.0]), "tool_pose"
        return np.array([0.0, 0.0, 1.0]), "optical"

    def _markers(self, arr, grasps, stamp_msg):
        """Gripper wireframes (score-coloured if accepted, faint grey if not) plus labels for the top 5."""
        out = MarkerArray()
        clear = Marker()
        clear.header.frame_id, clear.header.stamp, clear.action = self.frame_id, stamp_msg, Marker.DELETEALL
        out.markers.append(clear)
        for i, (row, g) in enumerate(zip(arr, grasps)):
            m = Marker()
            m.header.frame_id, m.header.stamp = self.frame_id, stamp_msg
            m.ns, m.id, m.type, m.action = "sbg_grasp", i, Marker.LINE_LIST, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.003 if g["accepted"] else 0.0015
            if g["accepted"]:
                b, gg, r = score_colour_bgr(g["score"])
                m.color.r, m.color.g, m.color.b, m.color.a = r / 255, gg / 255, b / 255, 0.95
            else:
                m.color.r = m.color.g = m.color.b = 0.55
                m.color.a = 0.35
            for a, b in gripper_segments(row):
                m.points += [Point(x=float(a[0]), y=float(a[1]), z=float(a[2])),
                             Point(x=float(b[0]), y=float(b[1]), z=float(b[2]))]
            out.markers.append(m)
            if i < 5 and g["accepted"]:
                t = Marker()
                t.header.frame_id, t.header.stamp = self.frame_id, stamp_msg
                t.ns, t.id, t.type, t.action = "sbg_grasp_text", i, Marker.TEXT_VIEW_FACING, Marker.ADD
                x, y, z = g["translation_m"]
                t.pose.position.x, t.pose.position.y, t.pose.position.z = x, y, z - 0.03
                t.pose.orientation.w = 1.0
                t.scale.z = 0.015
                t.color.r = t.color.g = t.color.b = t.color.a = 1.0
                t.text = f"#{i} {g['score']:.2f} {g['width_m'] * 1000:.0f}mm"
                out.markers.append(t)
        self.marker_pub.publish(out)

    def _handle(self, request, response):
        """Run SBG on the live depth, write grasps_sbg.json/.npy and the overlay, publish, record."""
        t0 = time.time()
        try:
            dpath, kpath = self.live_dir / self.depth_name, self.live_dir / "color_intrinsics.npy"
            for f in (dpath, kpath):
                if not f.exists():
                    raise FileNotFoundError(f"missing {f.name}; run capture and the depth stage first")
            depth_m = np.load(dpath).astype(np.float32) / 1000.0
            K = np.load(kpath)
            bgr = cv2.imread(str(self.live_dir / "rgb.png"))
            axis, axis_from = (self._down_in_camera() if self.cfg.approach_cone_deg else (None, None))
            if self.cfg.approach_cone_deg and axis_from != self.axis_source:
                self.get_logger().warning("approach cone: no live tool pose, so 'down' is taken as "
                                          "the camera's optical axis for this frame")
            grasps, info = self.engine.grasps(depth_m, K, None if bgr is None else bgr[:, :, ::-1],
                                              approach_axis=axis)
            arr = info["array"]
            accepted = np.array([g["accepted"] for g in grasps], dtype=bool)

            payload = {
                "grasps": grasps, "n_grasps": len(grasps), "n_accepted": int(accepted.sum()),
                "n_raw": info["n_raw"], "n_after_nms": info["n_after_nms"],
                "n_collision_free": info["n_collision_free"], "n_empty": info.get("n_empty"),
                "n_points": info["n_points"],
                "n_scene_points": info["n_scene_points"],
                "depth_coverage": float((np.nan_to_num(depth_m) > 0).mean()),
                "depth_source": self.depth_name, "z_min_m": self.cfg.z_min_m, "z_max_m": self.cfg.z_max_m,
                "frame_id": self.frame_id, "stage_timings_s": info["timings_s"],
                "approach_cone": None if not info.get("approach_cone") else
                dict(info["approach_cone"], axis_source=axis_from),
                "convention_note": ("rotation columns: x = approach, y = jaw, z = finger height. "
                                    "tcp_m = translation + depth * approach. width_m is the OBJECT "
                                    "width (network column / 1.2); width_saturated = lower bound."),
            }
            settings = {**self.cfg.as_dict(), "checkpoint": Path(str(self.engine.checkpoint)).name,
                        "epoch": self.engine.epoch, "depth_name": self.depth_name}
            stamp(payload, "sbg", settings)
            (self.live_dir / "grasps_sbg.json").write_text(json.dumps(payload, indent=2))
            np.save(self.live_dir / "grasps_sbg.npy", arr.astype(np.float32))
            canvas = bgr.copy() if bgr is not None else np.zeros(depth_m.shape + (3,), np.uint8)
            overlay = draw_grasps(canvas, arr, K, accepted)
            cv2.imwrite(str(self.live_dir / "grasps_sbg_overlay.png"), overlay)
            now = self.get_clock().now().to_msg()
            self.overlay_pub.publish(image_msg(overlay, "bgr8", now, self.frame_id))
            self.cloud_pub.publish(cloud_msg(info["cloud"], info["colors"], now, self.frame_id))
            self._markers(arr, grasps, now)

            dt = time.time() - t0
            record_stage(self.live_dir, "sbg", settings, seconds=dt,
                         extra={"n_grasps": len(grasps), "n_accepted": int(accepted.sum()),
                                "n_raw": info["n_raw"], "n_points": info["n_points"],
                                "best_score": grasps[0]["score"] if grasps else None})
            brief = {k: v for k, v in payload.items() if k != "grasps"}
            brief["best"] = grasps[0] if grasps else None
            response.success, response.message = True, json.dumps({**brief, "timing_s": round(dt, 3)})
            self.get_logger().info(f"{len(grasps)} grasps ({int(accepted.sum())} accepted) "
                                   f"from {info['n_raw']} raw in {dt:.2f}s")
        except Exception as exc:  # noqa: BLE001
            response.success, response.message = False, f"{type(exc).__name__}: {exc}"
            self.get_logger().error(response.message)
        return response


def main():
    run_node(SbgNode)


if __name__ == "__main__":
    main()
