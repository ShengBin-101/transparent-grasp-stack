"""Capture a synced colour/depth frame and run the pipeline's stages in order, archiving each run to runs/.

  /live_pipeline/capture    align depth to colour and write the frame to the live dir
  /live_pipeline/run_chain  wait for the stages' services, capture, call each stage in `chain`, archive

Chain syntax: "a,b?,c|d". "?" = optional (a failure is recorded and the chain goes on), "a|b" = whichever service
is running. Stage names map to /live_pipeline/<name>.
"""
import json
import os
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import tf2_ros
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger

from .run_export import spawn as spawn_export
from .align import align_depth_to_color, coverage_mask, footprint, quat_to_rot
from .common import (LIVE_DIR, RUNS_DIR, clear_live_dir, code_version, declare, image_to_numpy, new_run,
                     record_stage, run_node, save_preview)

SLOP = 0.1      # s, how far apart colour and depth stamps may be to pair
# Vendor message packages that identify which camera is publishing.
CAMERA_DRIVERS = {"realsense2_camera_msgs": "realsense_l515"}


def archive(live_dir, runs_dir, label):
    """Copy this run's files to runs/<stamp>_<label> and write MANIFEST.txt (stage timings, files). Files more
    than 2 s older than the capture's rgb.png are skipped as left over from another run."""
    live_dir, runs_dir = Path(live_dir), Path(runs_dir)
    dest = runs_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{label}"
    dest.mkdir(parents=True, exist_ok=True)
    marker = live_dir / "rgb.png"
    start = marker.stat().st_mtime - 2.0 if marker.exists() else None
    copied, skipped = [], []
    for p in sorted(live_dir.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        if start is not None and p.stat().st_mtime < start:
            skipped.append(p.name)
            continue
        shutil.copy2(p, dest / p.name)
        copied.append(p.name)
    lines = [f"archived: {time.strftime('%Y-%m-%dT%H:%M:%S')}", f"source:   {live_dir}", ""]
    try:
        rc = json.loads((dest / "run_config.json").read_text())
        lines.append(f"total: {rc.get('total_seconds', '?')}s")   # the stages' own times summed, not wall clock
        for k, v in rc.get("stages", {}).items():
            st = ", ".join(f"{a}={b}" for a, b in list(v.get("settings", {}).items())[:4])
            lines.append(f"  {k:<16} {v.get('seconds', 0):5.2f}s  {st}")
    except (OSError, ValueError) as e:
        lines.append(f"(no run_config.json: {e})")
    lines += ["", f"files ({len(copied)}):"]
    lines += [f"  {f:<34} {(dest / f).stat().st_size / 1024:8.0f} KB" for f in copied]
    if skipped:
        lines += ["", f"skipped ({len(skipped)}), older than this capture:"] + [f"  {f}" for f in skipped]
    (dest / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
    return dest


class Orchestrator(Node):
    """The capture and run_chain services: keeps the latest synced frame and calls the stages in turn."""

    def __init__(self):
        super().__init__("live_pipeline_orchestrator")
        p = declare(self, live_dir=str(LIVE_DIR), runs_dir=str(RUNS_DIR), chain="",
                    chain_timeout=120.0, archive="always", frame_max_age=2.0, pipeline="",
                    ready_timeout=60.0, export=True, tool_max_age=1.0)
        # export: write the run's point clouds (run_export) in the background after archiving.
        # tool_max_age: the arm's pose (/live_pipeline/tool_pose, from the NUC) is recorded with a
        # capture only if one arrived this recently; otherwise capture_tool.json says it was not live.
        self.export = bool(p["export"])
        self.tool_max_age = float(p["tool_max_age"])
        self.tool, self.tool_at = None, 0.0
        self.ready_timeout = float(p["ready_timeout"])
        # the preset (pipeline:= with any MASK / REMAKE / TARGETS override), recorded so runs can be told apart
        self.pipeline = str(p["pipeline"])
        # recorded with every capture: the code this node started with, and the hand-eye in force
        self.code = code_version()
        if self.code.get("dirty"):
            self.get_logger().warning(f"uncommitted changes in {', '.join(self.code['dirty_files'][:5])}"
                                      f"{' ...' if len(self.code['dirty_files']) > 5 else ''}: runs record "
                                      f"diff {self.code['diff_sha1']} on {self.code['commit'][:7]}")
        self.live_dir = Path(p["live_dir"])
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = Path(p["runs_dir"])
        self.chain = [c.strip() for c in p["chain"].split(",") if c.strip()]
        self.chain_timeout = float(p["chain_timeout"])
        self.archive_mode = p["archive"]
        self.frame_max_age = float(p["frame_max_age"])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.lock = threading.Lock()
        self.latest = None       # (colour, depth, colour_info, depth_info)
        self.latest_at = 0.0     # monotonic time of the last matched pair
        self.last_capture = {}

        subs = [Subscriber(self, Image, "/camera/color/image_raw"),
                Subscriber(self, Image, "/camera/depth/image_raw"),
                Subscriber(self, CameraInfo, "/camera/color/camera_info"),
                Subscriber(self, CameraInfo, "/camera/depth/camera_info")]
        self.sync = ApproximateTimeSynchronizer(subs, queue_size=10, slop=SLOP)
        self.sync.registerCallback(self._on_frame)
        self.create_subscription(PoseStamped, "/live_pipeline/tool_pose", self._on_tool, 10)

        # Reentrant group + multithreaded executor: we call other services from inside ours.
        self.group = ReentrantCallbackGroup()
        self.create_service(Trigger, "/live_pipeline/capture", self._on_capture,
                            callback_group=self.group)
        if self.chain:
            self.create_service(Trigger, "/live_pipeline/run_chain", self._on_run_chain,
                                callback_group=self.group)
        self.get_logger().info(f"ready. chain: capture -> {' -> '.join(self.chain) or '(none)'}")

    # ------------------------------------------------------------ capture

    def _on_frame(self, color, depth, cinfo, dinfo):
        with self.lock:
            self.latest = (color, depth, cinfo, dinfo)
            self.latest_at = time.monotonic()

    def _on_tool(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        with self.lock:
            self.tool = ([p.x, p.y, p.z], [o.x, o.y, o.z, o.w])
            self.tool_at = time.monotonic()

    def _save_capture_tool(self):
        """capture_tool.json: the arm's pose (fr3_EE in fr3_link0) at this capture, or why it is not known.
        The same file the sequencer writes at a pick, so replay_plan and run_export read either."""
        with self.lock:
            tool, at = self.tool, self.tool_at
        age = time.monotonic() - at
        if tool is not None and age <= self.tool_max_age:
            rec = {"live": True, "frame_id": "fr3_link0", "position": tool[0], "orientation_xyzw": tool[1],
                   "age_s": round(age, 3), "source": "orchestrator: tool pose at capture"}
        else:
            rec = {"live": False, "why": "no tool pose on /live_pipeline/tool_pose" if tool is None
                   else "the last tool pose arrived %.1fs before the capture" % age}
        (self.live_dir / "capture_tool.json").write_text(json.dumps(rec))

    def _hand_eye(self):
        """The installed eye_in_hand.yaml's ee_to_camera and the file's sha1, as recorded with a capture."""
        import hashlib
        try:
            import yaml
            from ament_index_python.packages import get_package_share_directory
            path = Path(get_package_share_directory("remake_grasp")) / "config" / "eye_in_hand.yaml"
            raw = path.read_bytes()
            return {"ee_to_camera": (yaml.safe_load(raw) or {}).get("ee_to_camera"),
                    "eye_in_hand_sha1": hashlib.sha1(raw).hexdigest()[:12]}
        except Exception as exc:  # noqa: BLE001
            return {"ee_to_camera": None, "eye_in_hand_error": f"{type(exc).__name__}: {exc}"}

    def _identify_camera(self):
        """Which camera is publishing, from the vendor message packages on the graph (else $CAMERA)."""
        found = set()
        for _, types in self.get_topic_names_and_types() + self.get_service_names_and_types():
            found.update(t.split("/", 1)[0] for t in types if t.split("/", 1)[0] in CAMERA_DRIVERS)
        if len(found) == 1:
            pkg = found.pop()
            return {"camera": CAMERA_DRIVERS[pkg], "camera_driver": pkg}
        if len(found) > 1:
            self.get_logger().error("two camera drivers are running at once; stop one")
            return {"camera": "ambiguous", "camera_driver": sorted(found)}
        return {"camera": os.environ.get("CAMERA", "unspecified"), "camera_driver": None}

    def _align_and_save(self):
        """Align the latest depth to the colour image and write rgb.png, depth_aligned_mm.npy, depth_coverage.npy
        and color_intrinsics.npy. Returns an error text, or None; a pair older than frame_max_age is refused."""
        with self.lock:
            latest, at = self.latest, self.latest_at
        if latest is None:
            return "no camera frames received yet -- is the camera driver running?"
        age = time.monotonic() - at
        if age > self.frame_max_age:
            return (f"the last colour/depth pair was matched {age:.1f}s ago (limit "
                    f"{self.frame_max_age:.1f}s). Their stamps may have drifted apart; "
                    f"restart the camera driver.")
        color_msg, depth_msg, cinfo, dinfo = latest
        try:
            tf = self.tf_buffer.lookup_transform(cinfo.header.frame_id, dinfo.header.frame_id,
                                                 rclpy.time.Time(), timeout=Duration(seconds=1.0))
        except Exception as e:  # noqa: BLE001
            return f"tf lookup failed: {e}"

        color = image_to_numpy(color_msg)
        depth = image_to_numpy(depth_msg)
        Kc = np.array(cinfo.k).reshape(3, 3)
        Kd = np.array(dinfo.k).reshape(3, 3)
        q, t = tf.transform.rotation, tf.transform.translation
        R, T = quat_to_rot(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])
        Hc, Wc = color.shape[:2]
        Hd, Wd = depth.shape[:2]

        aligned = align_depth_to_color(depth, Kd, Kc, R, T, (Hc, Wc))
        coverage = coverage_mask(depth, Kd, Kc, R, T, (Hc, Wc))
        sy, sx = footprint(Kc, Kd)

        cv2.imwrite(str(self.live_dir / "rgb.png"), color)
        np.save(self.live_dir / "depth_aligned_mm.npy", aligned)
        np.save(self.live_dir / "depth_coverage.npy", coverage)
        save_preview(self.live_dir / "depth_aligned_vis.png", aligned, "aligned (raw)")
        np.save(self.live_dir / "color_intrinsics.npy", Kc)
        self._save_capture_tool()

        out_of_field = float((~coverage).mean())
        # ~100 on a healthy frame; a near-black frame (1.5 once) silently poisons every mask and ReMake
        rgb_mean = float(color.mean())
        if rgb_mean < 20.0:
            self.get_logger().error(f"the colour image is nearly black (mean {rgb_mean:.1f} of 255): "
                                    f"restart the camera before trusting this run")
        holes = float(((aligned <= 0) & coverage).mean())
        self.get_logger().info(
            f"aligned depth {Wd}x{Hd} -> {Wc}x{Hc}, {sx}x{sy} splat, coverage "
            f"{(aligned > 0).mean():.3f}, out of field {out_of_field * 100:.1f}%, "
            f"holes in field {holes * 100:.2f}%")
        self.last_capture = {"color_size": [int(Wc), int(Hc)], "depth_size": [int(Wd), int(Hd)],
                             "splat": [int(sx), int(sy)], "out_of_field_frac": out_of_field,
                             "holes_in_field_frac": holes, "frame_age_s": round(age, 3),
                             "rgb_mean": round(rgb_mean, 1)}
        return None

    def _capture(self):
        """Clear the live dir, then capture: (error or None, seconds). Clearing first means no stage can read a
        previous run's files, and a failed run leaves no grasp_poses.json for the sequencer."""
        t0 = time.time()
        removed = clear_live_dir(self.live_dir)
        err = self._align_and_save()
        dt = time.time() - t0
        if err is None:
            new_run(self.live_dir)
            record_stage(self.live_dir, "capture",
                         {**self._identify_camera(), "aligner": "zbuffer_forward_warp",
                          "cleared_previous_files": len(removed), **self.last_capture,
                          "pipeline": self.pipeline or None, "code": self.code, **self._hand_eye()},
                         seconds=dt)
        return err, dt

    def _on_capture(self, request, response):
        err, dt = self._capture()
        response.success = err is None
        response.message = err or json.dumps({"live_dir": str(self.live_dir), "seconds": round(dt, 3)})
        if err:
            self.get_logger().warning(f"capture failed: {err}")
        return response

    # ------------------------------------------------------------ chain

    def _call(self, name, timeout):
        """Call /live_pipeline/<name>. Returns (error or None, seconds)."""
        t0 = time.time()
        client = self.create_client(Trigger, f"/live_pipeline/{name}", callback_group=self.group)
        try:
            if not client.wait_for_service(timeout_sec=5.0):
                return f"{name} service not available -- is its node running?", time.time() - t0
            future = client.call_async(Trigger.Request())
            while not future.done():
                if time.time() - t0 > timeout:
                    return f"{name} timed out after {timeout:.0f}s", time.time() - t0
                time.sleep(0.05)
            result = future.result()
            if not result.success:
                return f"{name} failed: {result.message}", time.time() - t0
            self.get_logger().info(f"{name}: {time.time() - t0:.2f}s | {result.message[:300]}")
            return None, time.time() - t0
        finally:
            self.destroy_client(client)

    def _wait_ready(self):
        """Wait up to ready_timeout s in all for every stage's service ("a|b" stages excepted) -> (names still
        missing, s waited). A run right after start-up thus waits for a node still loading (SAM 3 takes ~30-40 s)
        instead of skipping its stage."""
        t0 = time.time()
        missing = []
        for name in (raw.rstrip("?") for raw in self.chain if "|" not in raw):
            client = self.create_client(Trigger, f"/live_pipeline/{name}", callback_group=self.group)
            try:
                if client.wait_for_service(timeout_sec=0.2):
                    continue
                left = self.ready_timeout - (time.time() - t0)
                if left > 0:
                    self.get_logger().info(f"waiting up to {left:.0f}s for {name} (its node may still be loading)")
                if left <= 0 or not client.wait_for_service(timeout_sec=left):
                    missing.append(name)
            finally:
                self.destroy_client(client)
        return missing, time.time() - t0

    def _pick_alternative(self, spec):
        """First service in "a|b" that is running, or None."""
        for name in (n.strip() for n in spec.split("|") if n.strip()):
            client = self.create_client(Trigger, f"/live_pipeline/{name}", callback_group=self.group)
            ok = client.wait_for_service(timeout_sec=1.0)
            self.destroy_client(client)
            if ok:
                return name
        return None

    def _archive(self, label):
        """archive() this run; its path, or None (logged) on failure."""
        try:
            dest = archive(self.live_dir, self.runs_dir, label)
            self.get_logger().info(f"archived to {dest}")
            if self.export and spawn_export(dest) is None:
                self.get_logger().warning(f"point-cloud export of {dest.name} did not start")
            return str(dest)
        except OSError as e:
            self.get_logger().error(f"archive failed: {e}")
            return None

    def _on_run_chain(self, request, response):
        """Service: wait for the stages, capture, call each stage, record the timings, archive. A failed required
        stage stops the chain; an optional one is recorded as skipped."""
        t0 = time.time()
        stages, skipped = {}, {}

        def fail(stage, err):
            self.get_logger().warning(f"run_chain failed at {stage}: {err}")
            reply = {"failed_at": stage, "error": err, "stages": stages,
                     "seconds": round(time.time() - t0, 3)}
            if self.archive_mode == "always":
                reply["archived"] = self._archive(f"failed_{stage.split('|')[0]}")
            response.success, response.message = False, json.dumps(reply)
            return response

        missing, waited = self._wait_ready()       # before the capture, so the frame is fresh
        if waited > 1.0:
            stages["wait_ready"] = round(waited, 3)
        if missing:
            self.get_logger().warning(f"still not available after {waited:.0f}s: {', '.join(missing)}")

        err, dt = self._capture()
        stages["capture"] = round(dt, 3)
        if err:
            return fail("capture", err)

        for raw in self.chain:
            optional = raw.endswith("?")
            spec = raw.rstrip("?")
            name = self._pick_alternative(spec) if "|" in spec else spec
            if name is None:
                err, dt = f"none of {spec.replace('|', ', ')} is running", 0.0
            else:
                err, dt = self._call(name, self.chain_timeout)
            stages[name or spec] = round(dt, 3)
            if err and optional:
                skipped[name or spec] = err
                self.get_logger().warning(f"optional stage {name or spec} skipped: {err}")
                record_stage(self.live_dir, name or spec, {"optional": True, "outcome": "skipped"},
                             seconds=dt, extra={"error": err})
            elif err:
                return fail(name or spec, err)

        # wall_seconds: wall clock from this request to the end of the chain, the optional point cloud included.
        # The time to the grasp pose is wall_seconds minus stage_seconds["pointcloud"] (trials/timings.py).
        total = time.time() - t0
        record_stage(self.live_dir, "run_chain",
                     {"chain": self.chain, "chain_timeout_s": self.chain_timeout},
                     extra={"wall_seconds": round(total, 3), "stage_seconds": dict(stages),
                            "skipped": skipped or None})
        result = {"stages": stages, "seconds": round(total, 3), "live_dir": str(self.live_dir)}
        if skipped:
            result["skipped"] = skipped
        if self.archive_mode in ("success", "always"):
            result["archived"] = self._archive("auto")
        response.success, response.message = True, json.dumps(result)
        self.get_logger().info(f"run_chain complete in {total:.1f}s ("
                               + ", ".join(f"{k}={v:.2f}s" for k, v in stages.items()) + ")")
        return response


def main():
    run_node(Orchestrator, threads=4)


if __name__ == "__main__":
    main()
