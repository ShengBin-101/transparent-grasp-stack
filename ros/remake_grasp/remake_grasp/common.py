"""Shared helpers: paths, the run record (run_config.json), and message builders."""
import array
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("RG_ROOT", "/opt/remake_grasp"))
LIVE_DIR = ROOT / "data" / "live"
RUNS_DIR = ROOT / "runs"
CKPT_DIR = ROOT / "checkpoints"
THIRD_PARTY = ROOT / "third_party"
CAMERA_FRAME = "camera_color_optical_frame"

RUN_FILE = "run_config.json"
SCHEMA = 1
# Files a new capture must not delete (SAM 3's grasp prompt, PEGG-Net's input).
PERSISTED = frozenset({"sam3_grasp_prompt.txt"})


# ---------------------------------------------------------------- run record

def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_json(path, obj, indent=1):
    """Atomic JSON write, world-readable (the container runs as root)."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=indent)
        os.chmod(tmp, 0o666)
        os.replace(tmp, str(path))
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def stamp(payload, stage, settings):
    """Attach the settings that produced an output file."""
    payload["config"] = {"stage": stage, "settings": dict(settings), "at": _now(), "schema": SCHEMA}
    return payload


# The paths whose state a run depends on (the trials' pre-flight checks the same ones).
CODE_PATHS = ("ros", "scripts", "tmuxp", "docker", "third_party", "segmenter")


def code_version(root=ROOT):
    """The repo's commit, and whether the code under CODE_PATHS differs from it: {"commit", "dirty",
    "dirty_files", "diff_sha1"} (diff_sha1 tells two uncommitted states apart), or {"error"}. Call it
    once at start-up: it describes the code the process loaded."""
    import hashlib
    import subprocess
    git = ["git", "-c", "safe.directory=*", "-C", str(root)]
    try:
        commit = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                                check=True).stdout.strip()
        status = subprocess.run(git + ["status", "--porcelain", "--", *CODE_PATHS], capture_output=True,
                                text=True, timeout=10, check=True).stdout.splitlines()
        diff = subprocess.run(git + ["diff", "HEAD", "--", *CODE_PATHS], capture_output=True, timeout=10,
                              check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    files = [line[3:] for line in status]
    return {"commit": commit, "dirty": bool(files), "dirty_files": files[:40],
            "diff_sha1": hashlib.sha1(diff + "\n".join(sorted(files)).encode()).hexdigest()[:12] if files else None}


def new_run(live_dir):
    """Start a fresh run_config.json (best effort) and return it."""
    doc = {"schema": SCHEMA, "host": socket.gethostname(), "live_dir": str(live_dir),
           "started": _now(), "stages": {}}
    try:
        write_json(Path(live_dir) / RUN_FILE, doc)
    except OSError as e:
        _warn_once("run_record", f"cannot write {RUN_FILE} in {live_dir}: {e}")
    return doc


_WARNED = set()


def _warn_once(key, msg):
    """The run record never stops a stage; say once per process when it cannot be written."""
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[remake_grasp] {msg}", file=sys.stderr)


def record_stage(live_dir, stage, settings, seconds=None, extra=None):
    """Add one stage's settings and timing to run_config.json. Never raises."""
    path = Path(live_dir) / RUN_FILE
    try:
        doc = json.loads(path.read_text())
        if doc.get("schema") != SCHEMA:
            doc = {}
    except (OSError, ValueError):
        doc = {}
    if not doc:
        doc = new_run(live_dir)
    entry = {"settings": dict(settings), "at": _now()}
    if seconds is not None:
        entry["seconds"] = round(float(seconds), 3)
    if extra:
        entry.update(extra)
    doc.setdefault("stages", {})[stage] = entry
    # the stages' own times summed: not wall clock (run_chain's wall_seconds is, trials/timings.py)
    doc["total_seconds"] = round(sum(s.get("seconds", 0.0) for s in doc["stages"].values()), 3)
    doc["updated"] = _now()
    try:
        write_json(path, doc)
    except OSError as e:
        _warn_once("run_record", f"cannot write {path}: {e}")
    return doc


def clear_live_dir(live_dir):
    """Delete the previous run's files so no stage can read stale input."""
    removed = []
    for p in sorted(Path(live_dir).iterdir()):
        if p.is_file() and not p.name.startswith(".") and p.name not in PERSISTED:
            try:
                p.unlink()
                removed.append(p.name)
            except OSError as e:
                _warn_once("clear_" + p.name, f"cannot remove the previous run's {p}: {e}")
    return removed


# ---------------------------------------------------------------- images

def colourise(depth_mm, lo_pct=2, hi_pct=98):
    """float mm -> BGR preview (TURBO). Invalid pixels are black."""
    d = np.asarray(depth_mm, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    out = np.zeros(d.shape + (3,), np.uint8)
    if not valid.any():
        return out
    lo, hi = np.percentile(d[valid], [lo_pct, hi_pct])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    norm = np.clip((d - lo) / (hi - lo), 0, 1)
    col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[valid] = col[valid]
    return out


def save_preview(path, depth_mm, label=""):
    """Write a colourised depth PNG captioned with label, coverage and depth range."""
    img = colourise(depth_mm)
    d = np.asarray(depth_mm, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    txt = f"{label}  coverage {valid.mean() * 100:.1f}%"
    if valid.any():
        txt += f"  {d[valid].min():.0f}-{d[valid].max():.0f}mm"
    cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------- ROS messages

def latched_qos(depth=1):
    """Reliable + transient-local, so a viewer that joins late still sees the last result."""
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    return QoSProfile(depth=depth, history=QoSHistoryPolicy.KEEP_LAST,
                      reliability=QoSReliabilityPolicy.RELIABLE,
                      durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


def _msg_bytes(arr):
    """numpy -> array('B') for a uint8[] message field. The generated setter takes array('B') as is; given bytes,
    it checks every byte in Python (~0.7 s for a 1280x720 depth image)."""
    return array.array("B", np.ascontiguousarray(arr).tobytes())


def image_msg(array, encoding, stamp_msg, frame_id):
    """numpy -> sensor_msgs/Image ("32FC1" float depth in metres, "bgr8" preview)."""
    from sensor_msgs.msg import Image
    arr = np.ascontiguousarray(array)
    msg = Image()
    msg.header.stamp = stamp_msg
    msg.header.frame_id = frame_id
    msg.height, msg.width = int(arr.shape[0]), int(arr.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(arr.strides[0])
    msg.data = _msg_bytes(arr)
    return msg


def image_to_numpy(msg):
    """sensor_msgs/Image -> numpy. Colour comes back as BGR uint8, depth as float32."""
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8", "8uc3"):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        img = img.reshape(msg.height, msg.width, 3)
        return np.ascontiguousarray(img[:, :, ::-1] if enc == "rgb8" else img)
    if enc in ("16uc1", "mono16"):
        dt = np.dtype(">u2" if msg.is_bigendian else "<u2")
        img = np.frombuffer(msg.data, dt).reshape(msg.height, msg.step // 2)[:, :msg.width]
        return img.astype(np.float32)
    if enc == "32fc1":
        dt = np.dtype(">f4" if msg.is_bigendian else "<f4")
        return np.frombuffer(msg.data, dt).reshape(msg.height, msg.step // 4)[:, :msg.width].astype(np.float32)
    if enc in ("yuv422", "yuv422_yuy2", "yuyv"):
        yuv = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 2]
        return cv2.cvtColor(yuv.reshape(msg.height, msg.width, 2), cv2.COLOR_YUV2BGR_YUYV)
    raise ValueError(f"unsupported image encoding {msg.encoding!r}")


def cloud_msg(points, colors, stamp_msg, frame_id):
    """N x 3 points (+ N x 3 colours in [0, 1], or None) -> PointCloud2 xyzrgb."""
    from sensor_msgs.msg import PointCloud2, PointField
    points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    n = len(points)
    data = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    if colors is None:
        rgb = np.full(n, 0x00A0A0A0, dtype=np.uint32)
    else:
        c = (np.clip(np.asarray(colors, np.float32), 0, 1) * 255).astype(np.uint32)
        rgb = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
    data["rgb"] = rgb.view(np.float32)
    msg = PointCloud2()
    msg.header.stamp = stamp_msg
    msg.header.frame_id = frame_id
    msg.height, msg.width = 1, n
    msg.fields = [PointField(name=nm, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, nm in enumerate(("x", "y", "z", "rgb"))]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.is_dense = True
    msg.data = _msg_bytes(data)
    return msg


# ---------------------------------------------------------------- node running

def run_node(node_factory, threads=1):
    """rclpy.init, build the node, spin until Ctrl-C, clean up."""
    import rclpy
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
    rclpy.init()
    node = node_factory()
    try:
        if threads > 1:
            ex = MultiThreadedExecutor(num_threads=threads)
            ex.add_node(node)
            ex.spin()
        else:
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def declare(node, **defaults):
    """Declare ROS parameters with defaults and return their current values as a dict."""
    return {k: node.declare_parameter(k, v).value for k, v in defaults.items()}
