"""A stand-in /live_pipeline/<name> Trigger service that comes up after <delay> s (a node still loading).
    python3 tests/fake_trigger.py segment_trans 5"""
import sys
import time

import rclpy
from std_srvs.srv import Trigger

name, delay = sys.argv[1], float(sys.argv[2])
time.sleep(delay)
rclpy.init()
node = rclpy.create_node(f"fake_{name}")


def handle(_req, resp):
    resp.success, resp.message = True, f"fake {name}"
    return resp


node.create_service(Trigger, f"/live_pipeline/{name}", handle)
print(f"fake {name} up after {delay:.0f}s", flush=True)
rclpy.spin(node)
