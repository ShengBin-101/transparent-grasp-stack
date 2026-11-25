"""Tests the sequencer's plan_pick summary, refusals, pick record, weight check and one-pick-per-grasp rule.
No ROS graph, no NUC.

    docker exec remake_grasp python3 /opt/remake_grasp/tests/test_sequencer_summary.py
"""
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros/remake_grasp"))
from remake_grasp import sequencer as S  # noqa: E402

fail = []


def ck(name, ok, detail=""):
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  -- %s" % (detail,)))
    if not ok:
        fail.append(name)


# A real refusal from the rig, sent by an NUC that predates summary_text.
OLD = {"mode": "enforce", "chosen": None, "candidates": [
    {"id": "1:direct", "ok": False, "reasons": ["hand comes within 5 mm of the scene at 45% of the leg"]},
    {"id": "1:flipped", "ok": False, "reasons": ["the leg turns the wrist 122 deg, past the 90 deg per-command limit (limits/rotation_max)"]},
    {"id": "2:direct", "ok": False, "reasons": ["the leg turns the wrist 163 deg, past the 90 deg per-command limit (limits/rotation_max)"]},
    {"id": "10:flipped", "ok": False, "reasons": ["joint 4 reaches -2.722 rad at 23% of the leg, 1.5 deg inside its limit [-2.748, -0.446]"]},
    {"id": "7:direct", "ok": False, "reasons": ["the target is at the edge of reach at 82% of the leg: the arm ... its limit"]}]}
ck("summary from an old reply: counted by kind", S.plan_summary(OLD) ==
   "0 of 5 feasible; 2 wrist turn, 1 arm vs scene, 1 joint limit, 1 edge of reach", S.plan_summary(OLD))
ck("summary: the NUC's own summary_text wins", S.plan_summary({"summary_text": "x"}) == "x")

# run_dir_for: the newest archive written after the grasp.
tmp = Path(tempfile.mkdtemp())
old, new = tmp / "a_auto", tmp / "b_auto"
for d in (old, new):
    d.mkdir()
    (d / "grasp_poses.json").write_text("{}")
now = time.time()
os.utime(old, (now - 600, now - 600))
os.utime(new, (now - 5, now - 5))
ck("run_dir_for: the run archived after the grasp", S.run_dir_for(tmp, now - 20) == new)
ck("run_dir_for: none newer than the grasp -> None", S.run_dir_for(tmp, now + 60) is None)


# _planned with a refusal: the error is the summary, not the JSON; the record gets the reply.
class Log:
    def info(self, m): pass
    warning = info


class Client:
    def service_is_ready(self): return True


class Pub:
    def publish(self, m): self.sent = m


n = S.Sequencer.__new__(S.Sequencer)
n.lock = threading.Lock()
n.plan_pick, n.planner, n.cand_pub = "auto", Client(), Pub()
n.ee_to_camera = n.ee_to_grasp = ((0, 0, 0), (0, 0, 0, 1))
n.standoff = n.retreat = 0.10
n.do_retreat, n.max_width, n.settle = True, 0.08, 0.0
n.camera_frame, n.base_frame = S.CAMERA_FRAME, "fr3_link0"
n.get_logger = lambda: Log()
n.candidates = {"frame_id": S.CAMERA_FRAME, "stamp": 5.0, "candidates": [
    {"index": 0, "confidence": 0.5, "width_m": 0.04,
     "grasp_pose": {"position": [0, 0, 0.4], "orientation_xyzw": [1, 0, 0, 0]}}]}
n.record = {"capture_stamp": 5.0}


def refuse(client, label):
    plan_id = json.loads(n.cand_pub.sent.data)["plan_id"]
    return False, json.dumps(dict(OLD, plan_id=plan_id))


n._call_raw = refuse
tool = ((0.4, 0.0, 0.4), (1.0, 0.0, 0.0, 0.0))
try:
    n._planned(tool, tool, 5.0)
    msg = None
except S.SequenceError as exc:
    msg = str(exc)
ck("enforce refusal: the pick stops with the one-line summary and the funnel",
   msg.startswith("no grasp the arm can do: 0 of 5 feasible; 2 wrist turn, 1 arm vs scene, 1 joint limit, "
                  "1 edge of reach [") and "2 candidates -> 0 feasible -> chose nothing" in msg, msg)
ck("... and the record holds the candidates sent and the whole reply",
   n.record.get("plan_reply", {}).get("mode") == "enforce" and len(n.record.get("candidates_sent", [])) == 2)

# _finish_record writes pick_*.json and capture_tool.json into the run.
n.runs_dir = tmp
n.record = {"capture_stamp": now - 20, "capture_tool": [[0.4, 0, 0.4], [1, 0, 0, 0]], "plan_summary": "x"}


class Resp:
    success, message = False, "no grasp the arm can do: x"


n._finish_record(Resp())
picks = list(new.glob("pick_*.json"))
ct = json.loads((new / "capture_tool.json").read_text()) if (new / "capture_tool.json").exists() else {}
ck("record: pick_*.json in the matching run", len(picks) == 1 and json.loads(picks[0].read_text())["success"] is False)
ck("record: capture_tool.json for replay_plan.py", ct.get("position") == [0.4, 0, 0.4] and ct.get("live") is True, ct)

line = S.funnel_line({"raw": 22, "after_nms": 18, "collision_free": 13, "accepted": 8, "executable": 6,
                     "not_on_target": 2, "unseen_side": 0},
                    {"chosen": "5:direct", "by_confidence": "1:direct",
                     "candidates": [{"id": "5:direct", "ok": True, "utility": 0.061}, {"id": "1:direct", "ok": True,
                                    "utility": 0.05}, {"id": "2:x", "ok": False}]}, 12)
ck("funnel: every stage in one line", line == "SBG 22 raw -> 18 after NMS -> 13 collision-free -> 8 accepted -> "
   "6 executable (2 off target dropped) -> 12 candidates -> 2 feasible -> chose 5:direct (utility 0.061), "
   "by confidence 1:direct", line)

# -- the weight check ----------------------------------------------------------------------
B = {"force": [0.3, -0.2, -2.10]}                      # standoff, fingers open
mug = {"force": [0.3, -0.2, -2.10 + 0.30 * S.G]}      # 0.30 kg carried: z RISES by m*g (measured)
empty = {"force": [0.35, -0.25, -2.18]}               # standing noise
v, line = S.load_verdict(B, mug, "grasp: ... fingers grasp (under mode, predicted 46 mm): HELD at 5 mm")
ck("load: a pinch that lifts 0.30 kg is held", v == "held" and "0.30 kg" in line, line)
v, line = S.load_verdict(B, mug, "grasp: ... MISSED -- fingers closed to 0 mm, nothing between them")
ck("load: fingers closed fully but 0.30 kg carried is a CAGE (the 2026-09-30 handle pick)",
   v == "caged" and "CARRYING 0.30 kg" in line, line)
v, line = S.load_verdict(B, empty, "MISSED -- fingers closed to 1 mm")
ck("load: closed fully and nothing carried is nothing", v == "nothing", line)
v, line = S.load_verdict(B, empty, "HELD at 5 mm")
ck("load: HELD by width but nothing lifted is flagged", v == "not lifted", line)
ck("load: no readings is unknown, not a verdict", S.load_verdict(None, mug)[0] == "unknown")

# _handle_pick measures at the standoff and after the retreat, and reports the load.
n2 = S.Sequencer.__new__(S.Sequencer)
n2.lock = threading.Lock()
n2.busy, n2.return_home, n2.record = False, False, None
n2.get_logger = lambda: Log()
order = []
n2._plan = lambda: ([("standoff", 1, 0.08, False), ("grasp", 2, 0.04, True), ("retreat", 3, None, False)], 2, 3)
n2._leg = lambda name, *a: (order.append(name), name + (": ... MISSED -- fingers closed to 0 mm" if name == "grasp" else ": ok"))[1]
readings = iter([B, mug])
n2._measure_load = lambda: (order.append("measure"), next(readings))[1]
n2._start_record = lambda: setattr(n2, "record", {})
n2._finish_record = lambda r: None


class Resp2:
    success, message = None, ""


r = n2._handle_pick(None, Resp2())
ck("pick: measured after the standoff and after the retreat, not around the grasp",
   order == ["standoff", "measure", "grasp", "retreat", "measure"], order)
ck("pick: the reply and the record carry the verdict", r.success and "caged" in n2.record["load"]["verdict"]
   and "CARRYING" in r.message, r.message)

real = S.load_verdict({"force": [-0.9317, -1.072, -1.2072]}, {"force": [-1.0896, -0.9925, 1.0864]},
                      "HELD at 8 mm")
ck("load: the calibration pick (20260930_021124, mug held) reads 0.23 kg held",
   real[0] == "held" and "0.23 kg" in real[1], real)

# -- one grasp, one pick; a failed run leaves nothing armed; no scene, no pick --------------------
n3 = S.Sequencer.__new__(S.Sequencer)
n3.lock = threading.Lock()
n3.live_dir = Path(tempfile.mkdtemp())
n3.ee_to_camera = n3.ee_to_grasp = ((0, 0, 0), (0, 0, 0, 1))
n3.pose_timeout, n3.max_width, n3.drift_position, n3.drift_angle = 120.0, 0.08, 0.005, 0.02
n3.pick_from, n3.cheaper_jaw, n3.do_retreat, n3.guarded_grasp = "capture", False, True, True
n3.standoff = n3.retreat = 0.10
n3.open_width = 0.078
n3._tool_problem = lambda: None
n3._planned = lambda *a: None


class Clock:
    def now(self):
        return type("T", (), {"nanoseconds": int(6e9)})()


n3.get_clock = lambda: Clock()


def arm():
    n3.grasp, n3.grasp_stamp, n3.width = ((0, 0, 0.4), (1, 0, 0, 0)), 5.0, 0.04
    n3.capture_tool = n3.tool = tool


def plan_error():
    try:
        n3._plan()
        return None
    except S.SequenceError as exc:
        return str(exc)


arm()
(n3.live_dir / "grasp_poses.json").write_text("{}")
first, second = plan_error(), plan_error()
ck("plan: a finished run's grasp plans", first is None, first)
ck("plan: the same grasp cannot be picked twice", "used by one pick" in (second or ""), second)
arm()
(n3.live_dir / "grasp_poses.json").unlink()
err = plan_error()
ck("plan: after a failed run (no grasp_poses.json) the earlier grasp is refused", "did not finish" in (err or ""), err)

n.plan_pick, n.cand_pub, n.record = "required", Pub(), {"capture_stamp": 5.0}
try:
    n._planned(tool, tool, 5.0)
    msg = None
except S.SequenceError as exc:
    msg = str(exc)
ck("required: candidates without a scene are refused before anything is sent",
   msg is not None and "no scene" in msg and not hasattr(n.cand_pub, "sent"), msg)

print("\nFAILED: " + ", ".join(fail) if fail else "\nall passed")
sys.exit(1 if fail else 0)
