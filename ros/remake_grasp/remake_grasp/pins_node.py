"""Pins: poses held fixed in the world, drawn for Foxglove.

Grasps drawn in the camera frame ride along with the camera, so a depth error cannot show. Captured grasps
are pinned in the base frame (fr3_link0) instead: move the arm, and a pin that slides off its object shows
the error.
Publishes /live_pipeline/pins (MarkerArray, 5 Hz) from two sources: every grasp_candidates capture (grasps
and target voxels, put into base with the tool pose at capture time, as the sequencer does; the last
keep_captures stay up), and config/pins.yaml (poses in any TF frame, re-read when the file changes).
With publish_tf it also publishes the arm's TF, which the bridge cannot carry: fr3_link0 -> fr3_EE from
/live_pipeline/tool_pose (Orin clock, as the images), and a static fr3_EE -> camera_link chosen so the camera
driver's own chain puts camera_color_optical_frame at ee_to_camera.

Foxglove: Image panel on /camera/color/image_raw (calibration /camera/color/camera_info), tick
/live_pipeline/pins; or a 3D panel in fr3_link0.
"""
import json
import math
import time
from pathlib import Path

import yaml

from . import eye_in_hand as eh

GRASP_COLOURS = {"best": (1.0, 0.2, 0.2), "other": (1.0, 0.75, 0.1)}
TARGET_COLOUR = (0.2, 0.9, 1.0)
FINGER = 0.045          # m, fingertip (TCP) back to where the finger meets the hand
STEM = 0.04             # m, drawn behind the hand, along -approach


def pose_from_entry(entry):
    """A pins.yaml entry -> ((x, y, z), quaternion). orientation_xyzw, rpy_deg or nothing."""
    pos = tuple(float(v) for v in entry.get("position", (0.0, 0.0, 0.0)))
    if len(pos) != 3:
        raise ValueError(f"pin {entry.get('name')!r}: position needs 3 numbers")
    if "orientation_xyzw" in entry:
        q = eh.normalize(tuple(float(v) for v in entry["orientation_xyzw"]))
    elif "rpy_deg" in entry:
        r, p, y = (math.radians(float(v)) for v in entry["rpy_deg"])
        cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), \
            math.cos(y / 2), math.sin(y / 2)
        q = (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
             cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)
    else:
        q = (0.0, 0.0, 0.0, 1.0)
    return pos, q


def load_pins(path):
    """pins.yaml -> list of dicts with name, frame, pose, shape, width, colour, size."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    out = []
    for i, e in enumerate(data.get("pins") or []):
        shape = str(e.get("shape", "axes"))
        if shape not in ("axes", "gripper", "sphere"):
            raise ValueError(f"pin {e.get('name', i)!r}: shape must be axes, gripper or sphere")
        out.append({"name": str(e.get("name", f"pin{i}")), "frame": str(e.get("frame", "fr3_link0")),
                    "pose": pose_from_entry(e), "shape": shape, "width": float(e.get("width", 0.05)),
                    "colour": tuple(float(c) for c in e.get("colour", (1.0, 0.3, 1.0))),
                    "size": float(e.get("size", 0.05))})
    return out


def pin_gripper_segments(pose, width):
    """Line segments of an open parallel gripper at a grasp pose (z approach, y jaw), in its frame's
    coordinates: two fingers ending at the contacts, the palm, and a stem behind it."""
    def at(x, y, z):
        return eh.compose(pose, ((x, y, z), (0.0, 0.0, 0.0, 1.0)))[0]
    h = width / 2
    return [(at(0, -h, 0), at(0, -h, -FINGER)), (at(0, h, 0), at(0, h, -FINGER)),
            (at(0, -h, -FINGER), at(0, h, -FINGER)), (at(0, 0, -FINGER), at(0, 0, -FINGER - STEM))]


def axes_segments(pose, size):
    """(start, end, rgb) for x red, y green, z blue."""
    o = pose[0]
    return [(o, eh.compose(pose, (tuple(size if i == j else 0.0 for j in range(3)), (0, 0, 0, 1)))[0], c)
            for i, c in enumerate(((1.0, 0.1, 0.1), (0.1, 1.0, 0.1), (0.2, 0.4, 1.0)))]


def camera_link_in_ee(ee_to_camera, link_to_optical):
    """fr3_EE -> camera_link such that fr3_EE -> camera_link -> (driver chain) -> optical = ee_to_camera."""
    return eh.compose(ee_to_camera, eh.inverse(link_to_optical))


def capture_to_base(payload, capture_tool, ee_to_camera, camera_frame):
    """One grasp_candidates payload -> {'grasps': [(index, pose, width, best)], 'target': [points]}
    in base, from the tool pose at capture time."""
    if payload.get("frame_id") != camera_frame:
        raise ValueError(f"grasp_candidates are in {payload.get('frame_id')!r}, not {camera_frame!r}")
    cam = eh.compose(capture_tool, ee_to_camera)
    grasps = []
    for c in payload.get("candidates") or []:
        p = c["grasp_pose"]
        pose = eh.compose(cam, (tuple(p["position"]), eh.normalize(tuple(p["orientation_xyzw"]))))
        grasps.append((c["index"], pose, float(c.get("width_m") or 0.0), c["index"] == payload.get("best_index")))
    target = []
    obs = payload.get("obstacles") or {}
    flags = obs.get("target")
    if flags is not None:
        target = [eh.compose(cam, (tuple(pt), (0, 0, 0, 1)))[0]
                  for pt, f in zip(obs.get("points") or [], flags) if f]
    return {"grasps": grasps, "target": target, "voxel_m": float(obs.get("voxel_m") or 0.01),
            "stamp": payload.get("stamp")}


# ---------------------------------------------------------------- ROS


def main():
    """Run the Pins node. ROS is imported here so the helpers above can be tested without it."""
    from ament_index_python.packages import get_package_share_directory
    import rclpy.time
    import tf2_ros
    from geometry_msgs.msg import Point, PoseStamped, TransformStamped
    from rclpy.node import Node
    from std_msgs.msg import String
    from visualization_msgs.msg import Marker, MarkerArray

    from .common import CAMERA_FRAME, declare, latched_qos, run_node

    share = Path(get_package_share_directory("remake_grasp")) / "config"

    def P(v):
        return Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))

    class Pins(Node):
        """Keeps recent captures and pins.yaml pins in base, republishes them on a timer, and owns the arm TF."""

        def __init__(self):
            super().__init__("pins")
            p = declare(self, config=str(share / "eye_in_hand.yaml"), pins_file=str(share / "pins.yaml"),
                        base_frame="fr3_link0", camera_frame=CAMERA_FRAME, keep_captures=2, rate=5.0,
                        tool_timeout=2.0, show_target=True, publish_tf=True, camera_root="camera_link",
                        ee_frame="fr3_EE")
            self.p = p
            cfg = yaml.safe_load(Path(p["config"]).read_text()) or {}
            self.ee_to_camera = eh.seven(cfg["ee_to_camera"], "ee_to_camera")
            self.tool, self.tool_at = None, 0.0
            self.captures = []                       # newest last
            self.pins, self.pins_mtime, self.pins_error = [], None, None
            self.pub = self.create_publisher(MarkerArray, "/live_pipeline/pins", 10)
            self.create_subscription(PoseStamped, "/live_pipeline/tool_pose", self._on_tool, 10)
            self.create_subscription(String, "/live_pipeline/grasp_candidates", self._on_candidates, latched_qos())
            self.mount = None                        # fr3_EE -> camera_root, once the driver's TF is seen
            if p["publish_tf"]:
                self.tf_buffer = tf2_ros.Buffer()
                self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
                self.tf_pub = tf2_ros.TransformBroadcaster(self)
                self.tf_static_pub = tf2_ros.StaticTransformBroadcaster(self)
            self.create_timer(1.0 / max(0.5, float(p["rate"])), self._tick)
            self.get_logger().info(f"pins in {p['base_frame']}, file {p['pins_file']}, ee_to_camera "
                                   f"{[round(v, 4) for v in list(self.ee_to_camera[0]) + list(self.ee_to_camera[1])]}")

        def _on_tool(self, msg):
            o = msg.pose.orientation
            self.tool = ((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                         eh.normalize((o.x, o.y, o.z, o.w)))
            self.tool_at = time.monotonic()
            if self.p["publish_tf"]:
                self.tf_pub.sendTransform(self._tf(self.p["base_frame"], self.p["ee_frame"], self.tool,
                                                   self.get_clock().now().to_msg()))

        def _tf(self, parent, child, pose, stamp_msg):
            t = TransformStamped()
            t.header.frame_id, t.header.stamp, t.child_frame_id = parent, stamp_msg, child
            (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z) = pose[0]
            (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z,
             t.transform.rotation.w) = pose[1]
            return t

        def _mount(self):
            """Publish fr3_EE -> camera_root once the driver's static chain is known."""
            if self.mount is not None:
                return
            try:
                tf = self.tf_buffer.lookup_transform(self.p["camera_root"], self.p["camera_frame"], rclpy.time.Time())
            except tf2_ros.TransformException:   # the driver's static chain is not up yet: try on the next tick
                return
            t, q = tf.transform.translation, tf.transform.rotation
            self.mount = camera_link_in_ee(self.ee_to_camera, ((t.x, t.y, t.z), eh.normalize((q.x, q.y, q.z, q.w))))
            self.tf_static_pub.sendTransform(self._tf(self.p["ee_frame"], self.p["camera_root"], self.mount,
                                                      self.get_clock().now().to_msg()))
            self.get_logger().info(f"TF: {self.p['ee_frame']} -> {self.p['camera_root']} "
                                   f"{[round(v, 4) for v in list(self.mount[0]) + list(self.mount[1])]}, so "
                                   f"{self.p['camera_frame']} sits at ee_to_camera")

        def _on_candidates(self, msg):
            """Pin one capture's grasps and target voxels in base, using the live tool pose."""
            try:
                payload = json.loads(msg.data)
            except ValueError:
                return
            if not isinstance(payload, dict):
                return
            if self.tool is None or time.monotonic() - self.tool_at > self.p["tool_timeout"]:
                # latched message from before this node started: the capture pose is unknown
                self.get_logger().warning("grasp_candidates arrived without a live tool pose -- not pinned")
                return
            try:
                cap = capture_to_base(payload, self.tool, self.ee_to_camera, self.p["camera_frame"])
            except (ValueError, KeyError, eh.GeometryError) as exc:
                self.get_logger().warning(f"capture not pinned: {exc}")
                return
            self.captures = (self.captures + [cap])[-max(1, int(self.p["keep_captures"])):]
            self.get_logger().info(f"pinned {len(cap['grasps'])} grasps and {len(cap['target'])} target voxels "
                                   f"from the capture at tool {[round(v, 3) for v in self.tool[0]]}")

        def _reload(self):
            """Re-read pins.yaml when its mtime changes; on a parse error keep the previous pins."""
            path = Path(self.p["pins_file"])
            try:
                mtime = path.stat().st_mtime
            except OSError:
                self.pins, self.pins_mtime = [], None
                return
            if mtime == self.pins_mtime:
                return
            self.pins_mtime = mtime
            try:
                self.pins, self.pins_error = load_pins(path), None
                self.get_logger().info(f"{len(self.pins)} pins from {path}: {[q['name'] for q in self.pins]}")
            except Exception as exc:  # noqa: BLE001
                self.pins_error = str(exc)
                self.get_logger().error(f"{path}: {exc} (keeping the previous pins)")

        def _tick(self):
            """Rebuild the whole MarkerArray (DELETEALL first); the newest capture opaque, older ones faint."""
            if self.p["publish_tf"]:
                self._mount()
            self._reload()
            now = self.get_clock().now().to_msg()
            arr = MarkerArray()
            clear = Marker()
            clear.action = Marker.DELETEALL
            arr.markers.append(clear)
            nid = [0]

            def mk(frame, ns, kind, rgb, alpha, scale):
                m = Marker()
                m.header.frame_id, m.header.stamp = frame, now
                m.ns, m.id, m.type, m.action = ns, nid[0], kind, Marker.ADD
                nid[0] += 1
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = scale
                m.color.r, m.color.g, m.color.b = rgb
                m.color.a = alpha
                arr.markers.append(m)
                return m

            base = self.p["base_frame"]
            for age, cap in enumerate(reversed(self.captures)):
                alpha = 1.0 if age == 0 else 0.35
                tag = "latest" if age == 0 else f"previous{age}"
                if self.p["show_target"] and cap["target"]:
                    m = mk(base, f"target_{tag}", Marker.CUBE_LIST, TARGET_COLOUR, 0.5 * alpha, cap["voxel_m"])
                    m.points = [P(v) for v in cap["target"]]
                for idx, pose, width, best in cap["grasps"]:
                    rgb = GRASP_COLOURS["best" if best else "other"]
                    m = mk(base, f"grasps_{tag}", Marker.LINE_LIST, rgb, alpha, 0.003 if best else 0.002)
                    for a, b in pin_gripper_segments(pose, width):
                        m.points += [P(a), P(b)]
                    t = mk(base, f"labels_{tag}", Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0), alpha, 0.015)
                    t.pose.position = P(eh.compose(pose, ((0, 0, -FINGER - STEM - 0.01), (0, 0, 0, 1)))[0])
                    t.text = str(idx)
            for pin in self.pins:
                if pin["shape"] == "sphere":
                    m = mk(pin["frame"], "pins", Marker.SPHERE, pin["colour"], 0.9, pin["size"])
                    m.pose.position = P(pin["pose"][0])
                elif pin["shape"] == "gripper":
                    m = mk(pin["frame"], "pins", Marker.LINE_LIST, pin["colour"], 1.0, 0.003)
                    for a, b in pin_gripper_segments(pin["pose"], pin["width"]):
                        m.points += [P(a), P(b)]
                else:
                    for a, b, rgb in axes_segments(pin["pose"], pin["size"]):
                        m = mk(pin["frame"], "pins", Marker.ARROW, rgb, 1.0, 0.0)
                        m.scale.x, m.scale.y, m.scale.z = 0.004, 0.008, 0.0
                        m.points = [P(a), P(b)]
                t = mk(pin["frame"], "pin_labels", Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0), 1.0, 0.015)
                t.pose.position = P(eh.compose(pin["pose"], ((0, 0, 0.02), (0, 0, 0, 1)))[0])
                t.text = pin["name"]
            self.pub.publish(arr)

    run_node(Pins)


if __name__ == "__main__":
    main()
