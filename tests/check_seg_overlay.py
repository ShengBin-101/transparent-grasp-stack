"""seg_node publishes its overlay: call segment_trans once and check /live_pipeline/seg_overlay against seg_masks.png.

Inside the container, with seg_node running on a live directory that holds an rgb.png:
    tests/run_in_container.sh check_seg_overlay.py <live_dir>
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros" / "remake_grasp"))
from remake_grasp.common import latched_qos  # noqa: E402

live = Path(sys.argv[1] if len(sys.argv) > 1 else "data/live")
rclpy.init()
node = rclpy.create_node("check_seg_overlay")
got = []
node.create_subscription(Image, "/live_pipeline/seg_overlay", got.append, latched_qos())
cli = node.create_client(Trigger, "/live_pipeline/segment_trans")
if not cli.wait_for_service(timeout_sec=120):
    sys.exit("segment_trans not available")
fut = cli.call_async(Trigger.Request())
rclpy.spin_until_future_complete(node, fut, timeout_sec=120)
print("segment_trans:", fut.result().success, fut.result().message)
t0 = time.time()
while not got and time.time() - t0 < 10:
    rclpy.spin_once(node, timeout_sec=0.5)
if not got:
    sys.exit("FAIL: nothing on /live_pipeline/seg_overlay")
m = got[-1]
img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
png = cv2.imread(str(live / "seg_masks.png"))
same = png is not None and img.shape == png.shape and np.array_equal(img, png)
print(f"seg_overlay: {m.width}x{m.height} {m.encoding}, frame {m.header.frame_id}; equals seg_masks.png: {same}")
print("ok" if same and m.encoding == "bgr8" else "FAIL")
sys.exit(0 if same else 1)
