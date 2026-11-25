"""One JSON line per commanded leg: the target, and the joints and elbow fields at its start and end.

    nuc/logs/legs-YYYYMMDD.jsonl

The planner predicts each leg's end posture from a model of how the controller resolves the spare joint
(impedance's weak nullspace pull, position control's held elbow_c); these lines are what a prediction is
checked against. Elbow fields: `elbow` measured, `elbow_d` desired, `elbow_c` commanded; [0] is joint 3,
[1] the joint-4 flip sign. Never breaks a motion: every failure (full disk, unwritable directory, missing
state) goes to `last_error`, never raised. No ROS imports.
"""

import datetime
import json
import math
import os


def snapshot(state, q, elbow, stamp_s):
    """The robot as the commander last saw it, as plain JSON-able values.

    `state` is the commander's (position, quaternion, stamp) or None, `q` its
    joint list or None, `elbow` a dict of the three FrankaState elbow arrays or
    None. Anything missing comes out as None rather than raising.
    """
    def floats(values):
        if values is None:
            return None
        try:
            out = [float(v) for v in values]
        except (TypeError, ValueError):
            return None
        return [v if math.isfinite(v) else None for v in out]

    position = quaternion = None
    if state is not None:
        position, quaternion = floats(state[0]), floats(state[1])
    return {
        "t": stamp_s,
        "position": position,
        "quaternion_xyzw": quaternion,
        "q": floats(q),
        "elbow": floats((elbow or {}).get("elbow")),
        "elbow_d": floats((elbow or {}).get("elbow_d")),
        "elbow_c": floats((elbow or {}).get("elbow_c")),
    }


def record(kind, command_topic, dry_run, goal, start, end, success, message,
           planner=None):
    """One log record. `goal` is (position, quaternion) in the base frame, or None when the leg was refused
    before a goal was resolved. `planner` is the planner's verdict, whose predicted end posture beside
    `end.q` is the comparison this log exists for."""
    target = None
    if goal is not None:
        target = {"position": [float(v) for v in goal[0]],
                  "quaternion_xyzw": [float(v) for v in goal[1]]}
    return {
        "kind": kind,                  # move | guarded | home | view
        "command_topic": command_topic,
        "dry_run": bool(dry_run),
        "target": target,
        "start": start,
        "end": end,
        "success": bool(success),
        "message": message,
        "planner": planner,
    }


class LegLog(object):
    """Appends records to legs-YYYYMMDD.jsonl in `directory`."""

    def __init__(self, directory, enabled=True, clock=None):
        self.directory = os.path.expanduser(directory)
        self.enabled = enabled
        self.last_error = None
        self._clock = clock or datetime.datetime.now

    def path(self):
        return os.path.join(self.directory,
                            "legs-%s.jsonl" % self._clock().strftime("%Y%m%d"))

    def write(self, entry):
        """Append one record. Returns the path written, or None."""
        if not self.enabled:
            return None
        try:
            os.makedirs(self.directory, exist_ok=True)
            path = self.path()
            with open(path, "a") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
            self.last_error = None
            return path
        except (OSError, TypeError, ValueError) as exc:
            self.last_error = "%s: %s" % (type(exc).__name__, exc)
            return None
