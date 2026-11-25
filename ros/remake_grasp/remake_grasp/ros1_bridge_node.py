"""Run ros1_bridge's parameter_bridge against the NUC's ROS 1 master.

Reads config/ros1_bridge.yaml, writes the topic/service lists to the ROS 1 parameter
server, and starts parameter_bridge. Services are only bridged while the NUC actually
offers them: the master is polled and the bridge restarted when that set changes, so on
the ROS 2 side a missing NUC service shows up as "not available" instead of a hung call.
Uses ROS_MASTER_URI / ROS_IP from the environment.
"""
import os
import signal
import subprocess
import sys
import time
import xmlrpc.client
from pathlib import Path

import yaml

NS = "/remake_grasp_bridge"
CALLER = "/remake_grasp_bridge_launcher"


def log(msg):
    print(f"[ros1_bridge_node] {msg}", flush=True)


def default_config():
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory("remake_grasp")) / "config" / "ros1_bridge.yaml"


def ros1_state(master):
    """(master pid, services registered on it), or (None, None) if it is unreachable."""
    try:
        code, _, (_, _, services) = master.getSystemState(CALLER)
        _, _, pid = master.getPid(CALLER)
        return (pid, {name for name, _ in services}) if code == 1 else (None, None)
    except (OSError, xmlrpc.client.Error):
        return None, None


def stop_proc(proc):
    """SIGINT the bridge, then kill it if it has not exited within 5 s."""
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main():
    """Poll the master every 2 s; restart the bridge when the offered services, the master or the bridge change."""
    args = sys.argv[1:sys.argv.index("--ros-args")] if "--ros-args" in sys.argv else sys.argv[1:]
    cfg_path = Path(args[0]) if args else default_config()
    cfg = yaml.safe_load(cfg_path.read_text())
    from ament_index_python.packages import get_package_prefix
    exe = os.path.join(get_package_prefix("ros1_bridge"), "lib", "ros1_bridge", "parameter_bridge")
    uri = os.environ.get("ROS_MASTER_URI", "http://172.16.0.1:11311")
    master = xmlrpc.client.ServerProxy(uri)
    log(f"ROS 1 master {uri}, ROS_IP={os.environ.get('ROS_IP', '(unset)')}, config {cfg_path}")

    proc, bridged, master_pid = None, None, None
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    last_wait_log = 0.0
    while not stop["now"]:
        pid, available = ros1_state(master)
        if available is None:
            bridged = None                    # a restarted master needs a fresh bridge
            if time.time() - last_wait_log > 10:
                log(f"waiting for the ROS 1 master at {uri} ...")
                last_wait_log = time.time()
            time.sleep(1.0)
            continue
        wanted = [s for s in cfg.get("services_2_to_1", []) if s["service"] in available]
        names = sorted(s["service"] for s in wanted)
        dead = proc is not None and proc.poll() is not None
        if names != bridged or dead or pid != master_pid:
            stop_proc(proc)
            master.setParam(CALLER, f"{NS}/topics", cfg.get("topics", []))
            master.setParam(CALLER, f"{NS}/services_1_to_2", cfg.get("services_1_to_2", []))
            master.setParam(CALLER, f"{NS}/services_2_to_1", wanted)
            log(("bridge exited, restarting. " if dead else "") +
                f"bridging {len(cfg.get('topics', []))} topics and services: {', '.join(names) or '(none yet)'}")
            proc = subprocess.Popen([exe, f"{NS}/topics", f"{NS}/services_1_to_2", f"{NS}/services_2_to_1"])
            bridged, master_pid = names, pid
        time.sleep(2.0)
    stop_proc(proc)


if __name__ == "__main__":
    main()
