#!/usr/bin/env python3
"""Tests the per-leg log (leg_log.py) and the commander's `_logged` wrapper. No robot, no master.

    python3 src/fr3_pose_commander/test/test_leg_log.py

leg_log.py needs only the standard library; the `_logged` half needs rospy and is skipped
without it. Each line must carry joints and elbow at both ends of a leg, and logging must
never change a leg's result: not on a full disk, a refused leg or a leg that raises.
"""
import datetime
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "src"))
from fr3_pose_commander import leg_log  # noqa: E402

FAILED = []


def check(label, condition, detail=""):
    print("%-4s %s%s" % ("ok" if condition else "FAIL", label,
                         "  (%s)" % detail if detail else ""))
    if not condition:
        FAILED.append(label)


Q = [0.1646, -0.2229, -0.1237, -2.1395, -0.0294, 1.9165, 0.8392]
ELBOW = {"elbow": [-0.1237, -1.0], "elbow_d": [-0.1237, -1.0], "elbow_c": [-0.1237, -1.0]}
STATE = ((0.5, 0.0, 0.4), (1.0, 0.0, 0.0, 0.0), None)

# -- snapshot ----------------------------------------------------------------

s = leg_log.snapshot(STATE, Q, ELBOW, 12.5)
check("snapshot carries q", s["q"] == Q)
check("snapshot carries all three elbow fields",
      s["elbow"] == s["elbow_d"] == s["elbow_c"] == [-0.1237, -1.0])
check("snapshot carries the tool pose", s["position"] == [0.5, 0.0, 0.4]
      and s["quaternion_xyzw"] == [1.0, 0.0, 0.0, 0.0])
empty = leg_log.snapshot(None, None, None, 1.0)
check("missing state, q and elbow come out as None, not an exception",
      empty["position"] is None and empty["q"] is None and empty["elbow_c"] is None)
nan = leg_log.snapshot(STATE, Q[:6] + [float("nan")], ELBOW, 1.0)
check("NaN becomes null, so the line stays valid JSON",
      nan["q"][6] is None and json.loads(json.dumps(nan))["q"][6] is None)
check("an unparsable field is None, not a crash",
      leg_log.snapshot(STATE, ["x"] * 7, ELBOW, 1.0)["q"] is None)

# -- record ------------------------------------------------------------------

r = leg_log.record("guarded", "/t", False, ((0.4, 0.1, 0.2), (0, 0, 0, 1)),
                   s, s, True, "ok")
check("record keeps the goal", r["target"]["position"] == [0.4, 0.1, 0.2])
check("a refused leg with no goal records target null",
      leg_log.record("move", "/t", True, None, s, s, False, "no")["target"] is None)

# -- LegLog ------------------------------------------------------------------

tmp = tempfile.mkdtemp()
try:
    day = datetime.datetime(2026, 9, 26, 12, 0)
    log = leg_log.LegLog(os.path.join(tmp, "logs"), clock=lambda: day)
    path = log.write(r)
    log.write(r)
    check("writes legs-YYYYMMDD.jsonl, creating the directory",
          path == os.path.join(tmp, "logs", "legs-20260926.jsonl"))
    lines = open(path).read().splitlines()
    check("appends one JSON line per record", len(lines) == 2
          and json.loads(lines[1])["kind"] == "guarded")

    blocked = os.path.join(tmp, "file")
    open(blocked, "w").close()
    bad = leg_log.LegLog(os.path.join(blocked, "logs"))      # a file in the way
    check("an unwritable directory returns None instead of raising",
          bad.write(r) is None and bad.last_error is not None, bad.last_error)
    check("disabled writes nothing",
          leg_log.LegLog(os.path.join(tmp, "off"), enabled=False).write(r) is None
          and not os.path.exists(os.path.join(tmp, "off")))
finally:
    shutil.rmtree(tmp)

# -- the commander's wrapper ---------------------------------------------------

try:
    import rospy  # noqa: F401
except ImportError:
    rospy = None
    print("skip the commander's _logged wrapper: rospy is not installed here")

if rospy is not None:
    spec = importlib.util.spec_from_file_location(
        "pose_commander_node", os.path.join(PKG, "scripts", "pose_commander_node.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Log(object):
        enabled = True
        last_error = None

        def __init__(self, fail=False):
            self.entries, self.fail = [], fail

        def write(self, entry):
            if self.fail:
                self.last_error = "OSError: No space left on device"
                return None
            self.entries.append(entry)
            return "/x"

    def commander(log):
        import threading
        c = mod.PoseCommander.__new__(mod.PoseCommander)   # no __init__: no ROS
        c._lock = threading.Lock()
        c._state, c._q, c._elbow = STATE, list(Q), dict(ELBOW)
        c._last_goal = None
        c.command_topic = "/cartesian_pose_tracking_controller/target_pose"
        c.dry_run = True
        c.leg_log = log
        return c

    ok = mod.TriggerResponse(success=True, message="at [..]")

    log = Log()
    c = commander(log)

    def moving_leg():
        c._last_goal = ((0.4, 0.0, 0.3), (1.0, 0.0, 0.0, 0.0))
        c._q = [v + 0.01 for v in Q]       # the arm moved
        return ok

    check("the leg's own response comes back unchanged",
          c._logged("move", moving_leg) is ok)
    e = log.entries[-1]
    check("start q is before the leg, end q after it",
          e["start"]["q"] == Q and math.isclose(e["end"]["q"][0], Q[0] + 0.01))
    check("the line names the controller topic and the goal",
          e["command_topic"].endswith("target_pose") and e["target"] is not None)

    refused = mod.TriggerResponse(success=False, message="z below floor")
    c._logged("move", lambda: refused)
    check("a refused leg is logged with success false and no goal",
          log.entries[-1]["success"] is False and log.entries[-1]["target"] is None)

    def boom():
        raise RuntimeError("controller died")
    try:
        c._logged("home", boom)
        raised = False
    except RuntimeError:
        raised = True
    check("a leg that raises still propagates the exception", raised)
    check("... and is still logged, as 'raised'",
          log.entries[-1]["message"] == "raised" and log.entries[-1]["kind"] == "home")

    full = commander(Log(fail=True))
    check("a full disk costs the line, never the leg",
          full._logged("move", lambda: ok) is ok)

    # Worst case: the write raises and so does the logger (rospy's throttled loggers raise before init_node).
    class Raising(Log):
        def write(self, entry):
            raise RuntimeError("disk on fire")

    real_logwarn = mod.rospy.logwarn

    def raising_logwarn(*_args, **_kwargs):
        raise RuntimeError("logger down too")

    mod.rospy.logwarn = raising_logwarn
    try:
        worst = commander(Raising())
        check("a raising log AND a raising logger still return the leg's result",
              worst._logged("move", lambda: ok) is ok)
    finally:
        mod.rospy.logwarn = real_logwarn

if FAILED:
    print("\n%d FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("\nall passed")
