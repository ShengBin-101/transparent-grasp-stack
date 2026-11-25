#!/usr/bin/env python3
"""The pipeline half of scripts/preflight.sh, run inside the container while a session is up.

    python3 /opt/remake_grasp/scripts/preflight_ros.py [--seconds 4]

Listens for a few seconds, then prints ok / WARN / FAIL lines. Exit 0 all ok (warnings allowed),
1 a FAIL, 2 no session running (nothing checked).
"""
import argparse
import os
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.srv import GetParameters
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ros", "remake_grasp"))
from remake_grasp.common import image_to_numpy  # noqa: E402

# config/l515.yaml: depth 640x480 at 30 Hz, colour 1280x720 at 15 Hz. A USB stall drops them far lower. Timed on the
# camera_info topics: a Python subscriber to the 1280x720 images cannot keep up and would read ~11 Hz.
MIN_HZ = {"color": 13.0, "depth": 25.0}
RGB_MEAN = (40.0, 220.0)   # ~100 healthy; 1.5 once (a black frame that poisoned every mask)
MIN_DEPTH_VALID = 0.5      # share of depth pixels with a reading (the raw depth image, not aligned)
ORCHESTRATOR = "/live_pipeline_orchestrator"
fails = []


def line(level, what, text):
    print(f"  {level:<5} {what:<9} {text}")
    if level == "FAIL":
        fails.append(what)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=4.0)
    a = ap.parse_args()
    rclpy.init()
    node = rclpy.create_node("preflight")
    seen = {"color": [], "depth": [], "tool": [], "color_img": [], "depth_img": []}
    last = {}

    def keep(key):
        def cb(msg):
            seen[key].append(time.monotonic())
            last[key] = msg
        return cb

    node.create_subscription(CameraInfo, "/camera/color/camera_info", keep("color"), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, "/camera/depth/camera_info", keep("depth"), qos_profile_sensor_data)
    node.create_subscription(Image, "/camera/color/image_raw", keep("color_img"), qos_profile_sensor_data)
    node.create_subscription(Image, "/camera/depth/image_raw", keep("depth_img"), qos_profile_sensor_data)
    node.create_subscription(PoseStamped, "/live_pipeline/tool_pose", keep("tool"), 10)
    t_end = time.monotonic() + 1.0                       # discovery
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    services = {name for name, _ in node.get_service_names_and_types()}
    if "/live_pipeline/run_chain" not in services:
        print("  WARN  pipeline  no pipeline session running (no /live_pipeline/run_chain): pipeline checks skipped")
        rclpy.shutdown()
        return 2
    for v in seen.values():
        v.clear()
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.seconds:
        rclpy.spin_once(node, timeout_sec=0.05)

    def hz(key):
        ts = seen[key]
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 and ts[-1] > ts[0] else 0.0

    for key in ("color", "depth"):
        f = hz(key)
        if f >= MIN_HZ[key]:
            line("ok", "camera", f"{key} at {f:.0f} Hz")
        else:
            line("FAIL", "camera", f"{key} at {f:.1f} Hz (needs >= {MIN_HZ[key]:.0f}): the USB link stalled or the "
                                   f"module is wedged -- restart the camera pane (initial_reset)")
    if "color_img" in last:
        m = float(image_to_numpy(last["color_img"]).mean())
        lo, hi = RGB_MEAN
        line("ok" if lo <= m <= hi else "FAIL", "camera",
             f"colour image mean {m:.0f} of 255" + ("" if lo <= m <= hi else
                                                    f" (outside {lo:.0f}-{hi:.0f}: lights, lens cap, or a dark-frame fault)"))
    if "depth_img" in last:
        d = image_to_numpy(last["depth_img"])
        valid = float((d > 0).mean())
        line("ok" if valid >= MIN_DEPTH_VALID else "WARN", "camera", f"depth has a reading on {valid * 100:.0f}% of pixels")
    f = hz("tool")
    if f > 1.0:
        line("ok", "arm", f"tool pose from the NUC at {f:.0f} Hz")
    else:
        line("FAIL", "arm", "no tool pose on /live_pipeline/tool_pose: is the bridge, controller and commander up?")
    for srv in ("/live_pipeline/plan_pick", "/live_pipeline/move_to_pose"):
        if srv in services:
            line("ok", "arm", f"{srv} available")
        else:
            line("FAIL", "arm", f"{srv} missing: the NUC commander (run_commander.sh) or the bridge is not up")

    cli = node.create_client(GetParameters, f"{ORCHESTRATOR}/get_parameters")
    if cli.wait_for_service(timeout_sec=2.0):
        fut = cli.call_async(GetParameters.Request(names=["pipeline", "chain"]))
        rclpy.spin_until_future_complete(node, fut, timeout_sec=3.0)
        if fut.done() and fut.result() is not None:
            label, chain = (v.string_value for v in fut.result().values)
            line("ok", "pipeline", f"runs will be labelled '{label}'  (chain: {chain})")
            line("ok", "pipeline", "check that label against the block's condition before the warm-up run")
        else:
            line("WARN", "pipeline", "the orchestrator did not answer for its label")
    else:
        line("WARN", "pipeline", "no orchestrator parameters: is pipeline.launch.py up?")

    # SBG's profile: a PROFILE left set in the shell that started the session changes the approach cone
    cli = node.create_client(GetParameters, "/sbg_grasp_service/get_parameters")
    if cli.wait_for_service(timeout_sec=2.0):
        fut = cli.call_async(GetParameters.Request(names=["approach_cone_deg", "empty_check"]))
        rclpy.spin_until_future_complete(node, fut, timeout_sec=3.0)
        if fut.done() and fut.result() is not None:
            cone, empty = fut.result().values[0].double_value, fut.result().values[1].bool_value
            if cone == 30.0 and empty:
                line("ok", "sbg", "profile picking (30 deg approach cone, empty-grasp check): the trials' setting")
            else:
                line("WARN", "sbg", f"approach cone {cone:g} deg, empty check {empty}: not the trials' picking profile "
                                     "(PROFILE set when the session started?)")
    node.destroy_node()
    rclpy.shutdown()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
