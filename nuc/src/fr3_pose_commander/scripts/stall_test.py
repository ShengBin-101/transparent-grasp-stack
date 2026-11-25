#!/usr/bin/env python3
"""Stall test: is the standoff miss a steady force (correctable) or a lack of authority?

    rosrun fr3_pose_commander stall_test.py                  # defaults: the A4 standoff
    rosrun fr3_pose_commander stall_test.py --standoff 0.538 -0.012 0.092 --stiff 600

MOVES THE ARM, through the running pose_commander (so its limit checks apply), asking before
each phase. Real picks stopped their legs ~3 cm short of the setpoints; correcting by the
measured error or raising stiffness only helps if the miss is a steady force balanced by K*e.

  A  at the current stiffness K0:
     A0  go_home first, so A1 always measures the full home -> standoff move
     A1  home -> standoff                                       settled error e0
     A2  re-command standoff + accumulated error, twice         e1, e2
     A3  free-space descent of --descent m, commanded plainly   eD
         then the same descent with A2's correction carried     eD_carry
     A4  back to the standoff, then home
  B  at --stiff N/m (translational only; skipped with --no-stiffness):
     B1  home -> standoff                                       e0_stiff
     B2  home, stiffness restored

The fingers are never touched (width -1). Stiffness is always restored on exit, Ctrl-C included.

Before running
  * The commander must be live (dry_run false) with the controller up; A0 then sends the arm
    to the commander's home pose.
  * Remove the object: the descent ends --descent below the standoff, in free space.
  * Stop the Orin's pipeline: it publishes on /live_pipeline/grasp_pose_best, and this script
    refuses to move while another node publishes there.

Reading the result (printed at the end, saved as JSON under ~/stall_tests/):
  |e0_stiff| / |e0| near K0/stiff  -> steady force: correction and stiffness both work
  ratio near 1                     -> no authority in that direction: change posture
  e2 small (a few mm)              -> the standoff correction converges
  eD mostly z                      -> the descent lags on its own
  eD_carry << eD                   -> carrying the correction into the descent is enough
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import rosgraph
import rospy
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from std_srvs.srv import Trigger

from fr3_pose_commander.pose_math import (angle_between, matrix_from_o_t_ee,
                                          normalize_quaternion, quaternion_from_matrix)

POSE_TOPIC = "/live_pipeline/grasp_pose_best"
WIDTH_TOPIC = "/live_pipeline/grasp_width_best"
STATE_TOPIC = "/franka_state_controller/franka_states"
CONTROLLER = "cartesian_impedance_example_controller"
RECONF = "/%s/dynamic_reconfigure_compliance_param_node" % CONTROLLER
MAX_CORRECTION = 0.06     # m; a bigger "error" means something else is wrong -- stop


def qmul(a, b):
    """Hamilton product, [x, y, z, w]."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


class StallTest(object):
    """Runs the phases through pose_commander's topics and services; results go in self.results."""

    def __init__(self, args):
        self.args = args
        self._state = None
        rospy.Subscriber(STATE_TOPIC, FrankaState, self._on_state, queue_size=1)
        self.pose_pub = rospy.Publisher(POSE_TOPIC, PoseStamped, queue_size=1, latch=True)
        self.width_pub = rospy.Publisher(WIDTH_TOPIC, Float64, queue_size=1, latch=True)
        self.move = rospy.ServiceProxy("/live_pipeline/move_to_pose", Trigger)
        self.home = rospy.ServiceProxy("/live_pipeline/go_home", Trigger)
        self.guarded = rospy.ServiceProxy("/live_pipeline/guarded_move_to_pose", Trigger)
        self.results = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args)}

    # -- measurement -----------------------------------------------------------
    def _on_state(self, msg):
        m = matrix_from_o_t_ee(msg.O_T_EE)
        self._state = (m[:3, 3].copy(), np.asarray(quaternion_from_matrix(m), dtype=float),
                       list(msg.q))

    def measure(self, seconds=0.5):
        """Mean settled position and last orientation over `seconds`."""
        pos, t_end = [], time.time() + seconds
        while time.time() < t_end and not rospy.is_shutdown():
            if self._state is not None:
                pos.append(self._state[0])
            time.sleep(0.02)
        if not pos:
            sys.exit("no franka_states -- is the controller running?")
        return np.mean(pos, axis=0), self._state[1]

    # -- safety ------------------------------------------------------------------
    def foreign_publishers(self):
        """(topic, node) for every other node publishing the commander's pose or width topic."""
        me = rospy.get_name()
        pubs, _, _ = rosgraph.Master(me).getSystemState()
        out = []
        for topic, nodes in pubs:
            if topic in (POSE_TOPIC, WIDTH_TOPIC):
                out += [(topic, n) for n in nodes if n != me]
        return out

    def confirm(self, what):
        """Wait for Enter before a phase; with --yes only print it."""
        if self.args.yes:
            print("\n>>> %s" % what)
            return
        try:
            input("\n>>> %s\n    Enter to go, Ctrl-C to abort: " % what)
        except EOFError:
            sys.exit("no terminal to confirm on; pass --yes only if you are watching the arm")

    # -- motion ------------------------------------------------------------------
    def go(self, position, quaternion, label, guarded=False):
        """Command one pose through pose_commander; return the settled error."""
        foreign = self.foreign_publishers()
        if foreign:
            sys.exit("REFUSING: another node publishes the commander's target: %s\n"
                     "Stop the Orin's pipeline (it sends pick waypoints here) and re-run."
                     % foreign)
        msg = PoseStamped()
        msg.header.frame_id = self.args.base_frame
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = position
        (msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w) = quaternion
        self.width_pub.publish(Float64(-1.0))      # leave the fingers alone
        self.pose_pub.publish(msg)
        rospy.sleep(0.5)                           # let the commander take both
        res = self.guarded() if guarded else self.move()
        if not res.success:
            sys.exit("pose_commander refused %s: %s" % (label, res.message))
        rospy.sleep(self.args.dwell)
        p, q = self.measure()
        return p, q, res.message

    def record(self, key, goal_p, goal_q, p, q, note, commanded_p=None):
        """Store and print one phase's error (target - measured, mm) and rotation error (deg)."""
        e = np.asarray(goal_p) - p                  # target - measured: where it fell short
        rot = math.degrees(angle_between(goal_q, q))
        row = {"goal": list(map(float, goal_p)), "measured": list(map(float, p)),
               "commanded": list(map(float, commanded_p if commanded_p is not None else goal_p)),
               "error_mm": [round(float(v) * 1e3, 1) for v in e],
               "error_norm_mm": round(float(np.linalg.norm(e)) * 1e3, 1),
               "rot_error_deg": round(rot, 2), "commander": note}
        self.results[key] = row
        print("    %-10s error (target - measured) x %+6.1f  y %+6.1f  z %+6.1f mm   |e| %5.1f mm   rot %4.1f deg"
              % (key, *row["error_mm"], row["error_norm_mm"], rot))
        return e, rot

    # -- stiffness -----------------------------------------------------------------
    def set_stiffness(self, client, value):
        """Step towards `value` in 100 N/m increments so the arm is never jerked."""
        cur = float(client.get_configuration(timeout=10)["translational_stiffness"])
        step = 100.0 if value > cur else -100.0
        while abs(value - cur) > 1e-6:
            cur = cur + step if abs(value - cur) > 100.0 else value
            client.update_configuration({"translational_stiffness": cur})
            rospy.sleep(0.5)
        rospy.sleep(1.5)     # the controller filters gain changes (~0.2 s time constant)

    # -- the test --------------------------------------------------------------------
    def run(self):
        a = self.args
        rospy.wait_for_service("/live_pipeline/move_to_pose", timeout=10)
        dry = rospy.get_param("/pose_commander/dry_run", None)
        if dry is not False:
            sys.exit("pose_commander dry_run is %r; this test needs the live commander" % dry)
        if self._state is None:
            rospy.sleep(1.0)
        # Start from home so A1 measures the full home -> standoff move.
        self.confirm("A0: go_home first (the pose pose_commander latched at start-up)")
        r = self.home()
        print("    home: %s" % r.message)
        rospy.sleep(self.args.dwell)
        home_p, home_q = self.measure()
        # normalize_quaternion returns a tuple; arithmetic and .copy() need an array.
        q_top = np.asarray(normalize_quaternion(home_q), dtype=float)
        S = np.asarray(a.standoff, dtype=float)
        D = S - np.array([0.0, 0.0, a.descent])
        if D[2] < a.min_z:
            sys.exit("descent would end at z %.3f, below --min-z %.3f" % (D[2], a.min_z))
        print("home  [%.3f %.3f %.3f]   standoff [%.3f %.3f %.3f]   descent end z %.3f"
              % (*home_p, *S, D[2]))
        self.results["home"] = list(map(float, home_p))

        client, k0 = None, None
        if a.stiffness:
            import dynamic_reconfigure.client
            client = dynamic_reconfigure.client.Client(RECONF, timeout=10)
            k0 = float(client.get_configuration(timeout=10)["translational_stiffness"])
            self.results["stiffness_base"] = k0
            print("translational stiffness now %.0f N/m (restored on exit)" % k0)
        else:
            print("no stiffness phase (--no-stiffness: position control has no gains to set)")
        self.results["controller"] = rospy.get_param("/pose_commander/command_topic", "unknown")
        try:
            # ---- A: base stiffness --------------------------------------------
            self.confirm("A1: home -> standoff%s (object REMOVED?)"
                         % (" at %.0f N/m" % k0 if k0 else ""))
            p, q, note = self.go(S, q_top, "A1 standoff")
            e0, _ = self.record("e0", S, q_top, p, q, note)

            corr = np.zeros(3)
            q_cmd = q_top.copy()
            for k in (1, 2):
                corr = corr + (S - p)
                if np.linalg.norm(corr) > MAX_CORRECTION:
                    sys.exit("accumulated correction %.0f mm exceeds %.0f mm; stopping"
                             % (np.linalg.norm(corr) * 1e3, MAX_CORRECTION * 1e3))
                if a.correct_rotation:
                    q_cmd = np.asarray(normalize_quaternion(qmul(qmul(q_top, qconj(q)), q_cmd)),
                                       dtype=float)
                self.confirm("A2.%d: re-command standoff + correction [%+.0f %+.0f %+.0f] mm"
                             % (k, *(corr * 1e3)))
                p, q, note = self.go(S + corr, q_cmd, "A2 correction")
                self.record("e%d" % k, S, q_top, p, q, note, commanded_p=S + corr)
            self.results["correction_mm"] = [round(float(v) * 1e3, 1) for v in corr]

            gd = a.guarded_descent
            self.confirm("A3a: free-space descent %.0f mm, commanded plainly%s"
                         % (a.descent * 1e3, " -- GUARDED (measures the moving floor)" if gd else ""))
            p, q, note = self.go(D, q_top, "A3a descent", guarded=gd)
            self.record("eD", D, q_top, p, q, note)
            self.confirm("A3b: back up to the corrected standoff")
            p, q, note = self.go(S + corr, q_cmd, "A3b back up")
            self.record("eS_again", S, q_top, p, q, note, commanded_p=S + corr)
            self.confirm("A3c: same descent, standoff correction carried over")
            p, q, note = self.go(D + corr, q_cmd, "A3c descent carried", guarded=gd)
            self.record("eD_carry", D, q_top, p, q, note, commanded_p=D + corr)

            self.confirm("A4: back to the standoff, then home")
            self.go(S, q_top, "A4 standoff")
            r = self.home()
            print("    home: %s" % r.message)

            # ---- B: higher stiffness -------------------------------------------
            if client is None:
                return
            self.confirm("B1: raise translational stiffness %.0f -> %.0f N/m (in 100 N/m steps), "
                         "then home -> standoff" % (k0, a.stiff))
            self.set_stiffness(client, a.stiff)
            p, q, note = self.go(S, q_top, "B1 standoff stiff")
            self.record("e0_stiff", S, q_top, p, q, note)
            self.confirm("B2: home, then stiffness back to %.0f" % k0)
            r = self.home()
            print("    home: %s" % r.message)
        finally:
            if client is not None:
                try:
                    self.set_stiffness(client, k0)
                    print("translational stiffness restored to %.0f N/m" % k0)
                except Exception as exc:   # noqa: BLE001
                    print("!!! COULD NOT RESTORE STIFFNESS (%s). Run ./run_stiffness.sh --reset" % exc)
            # In the exit path, so an aborted run still saves what it measured.
            self.verdict()

    def verdict(self):
        """Classify the measured errors, print the verdict and save results as JSON."""
        r, a = self.results, self.args
        n = lambda k: r[k]["error_norm_mm"] if k in r else None   # noqa: E731
        print("\n================ VERDICT ================")
        if n("e0") and n("e0_stiff") is not None and "stiffness_base" in r:
            ratio = n("e0_stiff") / max(n("e0"), 1e-6)
            expect = r["stiffness_base"] / a.stiff
            kind = ("STEADY FORCE -- correction and stiffness both work" if ratio < (1 + expect) / 2
                    else "NO AUTHORITY -- change posture; neither fix helps" if ratio > 0.85
                    else "MIXED -- part force, part authority")
            r["stiffness_ratio"] = round(ratio, 3)
            r["stiffness_verdict"] = kind
            print("standoff error at %.0f vs %.0f N/m: %.1f -> %.1f mm (ratio %.2f; pure force "
                  "balance predicts %.2f)  =>  %s" % (r["stiffness_base"], a.stiff, n("e0"),
                                                      n("e0_stiff"), ratio, expect, kind))
        if n("e2") is not None:
            ok = n("e2") <= 5.0
            r["correction_converges"] = ok
            print("standoff correction: %.1f -> %.1f -> %.1f mm  =>  %s"
                  % (n("e0"), n("e1"), n("e2"), "CONVERGES" if ok else "does not reach 5 mm"))
        if "eD" in r:
            e = np.abs(r["eD"]["error_mm"])
            zfrac = e[2] / max(np.linalg.norm(e), 1e-6)
            r["descent_z_fraction"] = round(float(zfrac), 2)
            print("plain descent: |e| %.1f mm, %.0f%% of it in z  =>  %s"
                  % (n("eD"), 100 * zfrac, "the descent LAGS on its own" if zfrac > 0.7
                     else "mostly inherited from the standoff"))
        import re as _re
        for k in ("eD", "eD_carry"):
            m = _re.search(r"peak ([0-9.]+) N / ([0-9.]+) N\.m", r.get(k, {}).get("commander", ""))
            c = "CONTACT" in r.get(k, {}).get("commander", "")
            if m or c:
                r[k + "_guard"] = {"peak_force_N": float(m.group(1)) if m else None,
                                   "peak_torque_Nm": float(m.group(2)) if m else None,
                                   "tripped": c}
                print("guarded %s: %s" % (k, "TRIPPED in free space -- threshold too tight"
                                          if c else "peak %s N / %s N.m (no trip)"
                                          % (m.group(1), m.group(2))))
        if "eD_carry" in r:
            print("descent with the correction carried: %.1f mm (plain: %.1f mm)"
                  % (n("eD_carry"), n("eD")))
        out = os.path.expanduser("~/stall_tests")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "stall_test_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
        with open(path, "w") as f:
            json.dump(r, f, indent=1)
        print("saved %s" % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standoff", type=float, nargs=3, default=[0.538, -0.012, 0.092],
                    help="base-frame standoff (m); default = attempt 2's standoff")
    ap.add_argument("--descent", type=float, default=0.07, help="free-space descent (m)")
    ap.add_argument("--min-z", type=float, default=0.0,
                    help="refuse a descent ending below this z (table measured at -0.041)")
    ap.add_argument("--stiff", type=float, default=600.0, help="translational stiffness for part B")
    ap.add_argument("--guarded-descent", action="store_true",
                    help="run A3's descents through guarded_move_to_pose and record the guard's "
                         "peak -- the moving noise floor its thresholds must clear")
    ap.add_argument("--no-stiffness", dest="stiffness", action="store_false",
                    help="skip part B -- for fr3_position_control, which has no stiffness to set")
    ap.add_argument("--dwell", type=float, default=1.0, help="s to wait after each move before measuring")
    ap.add_argument("--no-rotation-correction", dest="correct_rotation", action="store_false",
                    help="correct position only in A2")
    ap.add_argument("--base-frame", default="fr3_link0")
    ap.add_argument("--yes", action="store_true", help="skip the per-phase confirmations")
    args = ap.parse_args(rospy.myargv()[1:])
    rospy.init_node("stall_test")
    StallTest(args).run()


if __name__ == "__main__":
    main()
