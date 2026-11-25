#!/usr/bin/env python3
"""Tests that the commander refuses to move when no controller reads its setpoints.

    python3 src/fr3_pose_commander/test/test_require_listener.py

Needs rospy importable (the node imports it); no robot or master, it uses a bare
PoseCommander with a stub publisher. A controller/commander topic mismatch must be refused
with a message naming it, and dry run must be unaffected.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "src"))
spec = importlib.util.spec_from_file_location(
    "pose_commander_node", os.path.join(PKG, "scripts", "pose_commander_node.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
FAILED = []


def check(label, condition, detail=""):
    print("%-4s %s%s" % ("ok" if condition else "FAIL", label, "  (%s)" % detail if detail else ""))
    if not condition:
        FAILED.append(label)


class StubPub(object):
    def __init__(self, n):
        self.n = n
        self.resolved_name = "/cartesian_pose_tracking_controller/target_pose"

    def get_num_connections(self):
        return self.n


def commander(subscribers, dry_run=False):
    c = mod.PoseCommander.__new__(mod.PoseCommander)   # no __init__: no ROS needed
    c.dry_run = dry_run
    c.goal_pub = StubPub(subscribers)
    return c


try:
    commander(0)._require_listener()
    check("no subscriber: refuses", False)
except mod.Refused as exc:
    msg = str(exc)
    check("no subscriber: refuses", True)
    check("the refusal names the topic and the controller/commander pairing",
          "target_pose" in msg and "run_position_controller.sh" in msg and "--position" in msg, msg[:80])

try:
    commander(1)._require_listener()
    check("a subscribed controller: allowed", True)
except mod.Refused as exc:
    check("a subscribed controller: allowed", False, str(exc))

try:
    commander(0, dry_run=True)._require_listener()
    check("dry run with no controller: allowed (it publishes nothing)", True)
except mod.Refused as exc:
    check("dry run with no controller: allowed (it publishes nothing)", False, str(exc))

if FAILED:
    print("\n%d FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("\nall listener checks passed")
