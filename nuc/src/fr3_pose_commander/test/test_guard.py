#!/usr/bin/env python3
"""Tests the guarded approach's contact decision: wrench deltas, rolling reference, debounce.

    python3 src/fr3_pose_commander/test/test_guard.py

Needs rospy importable (the node imports it); no robot, master or controller. The guard
thresholds the change in O_F_ext_hat_K, not its absolute value (the estimate carries a standing
offset of a few newtons), and watches torque as well as force, because a fingertip glancing off a
rim is mostly a moment.
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
detected, delta = mod.contact_detected, mod.wrench_delta

FORCE, TORQUE = 4.0, 1.0
FAILED = []


def check(label, condition, detail=""):
    print("%-4s %s%s" % ("ok" if condition else "FAIL", label,
                         "  (%s)" % detail if detail else ""))
    if not condition:
        FAILED.append(label)


def w(fx=0.0, fy=0.0, fz=0.0, tx=0.0, ty=0.0, tz=0.0):
    return [fx, fy, fz, tx, ty, tz]


# -- free space ---------------------------------------------------------------
check("no bias, no load, no contact",
      detected(w(), w(), FORCE, TORQUE)[0] is False)

# A standing bias (the estimator's offset, 3.5-3.9 N measured on the rig) must not read as contact.
check("a 3.9 N standing bias with no change is NOT contact",
      detected(w(fz=-3.9), w(fz=-3.9), FORCE, TORQUE)[0] is False,
      "a standing offset with nothing touched")
check("the same bias plus a real 5 N push IS contact",
      detected(w(fz=-3.9 + 5.0), w(fz=-3.9), FORCE, TORQUE)[0] is True)

# -- the force channel --------------------------------------------------------
check("just under the threshold does not fire",
      detected(w(fz=3.9), w(), FORCE, TORQUE)[0] is False)
check("exactly at the threshold fires",
      detected(w(fz=4.0), w(), FORCE, TORQUE)[0] is True)
check("direction does not matter, only magnitude",
      detected(w(fx=-4.5), w(), FORCE, TORQUE)[0] is True)
check("a diagonal push is measured as its magnitude",
      detected(w(fx=3.0, fz=3.0), w(), FORCE, TORQUE)[0] is True,
      "|(3,0,3)| = 4.24 N, over 4.0")

# -- the torque channel, on its own -------------------------------------------
check("a pure moment with negligible force still fires",
      detected(w(fz=0.5, ty=1.2), w(), FORCE, TORQUE)[0] is True,
      "a fingertip catching a rim off-centre")
check("a small moment does not",
      detected(w(ty=0.4), w(), FORCE, TORQUE)[0] is False)

# -- the deltas are reported, not just the verdict -----------------------------
hit, force, torque = detected(w(fz=6.0, tx=2.0), w(fz=1.0), FORCE, TORQUE)
check("the reported force is the change, not the reading",
      abs(force - 5.0) < 1e-9, "6.0 measured - 1.0 baseline = 5.0")
check("the reported torque is the change too", abs(torque - 2.0) < 1e-9)

f, t = delta(w(fz=-3.9), w(fz=-3.9))
check("wrench_delta of a pure bias is zero on both channels",
      abs(f) < 1e-9 and abs(t) < 1e-9)

# -- missing data must not read as 'no contact' -------------------------------
# The node refuses the whole leg when there is no wrench; these pin that predicate.
check("no wrench at all is not a contact verdict",
      detected(None, w(), FORCE, TORQUE) == (False, None, None))
check("no baseline at all is not a contact verdict",
      detected(w(fz=99.0), None, FORCE, TORQUE) == (False, None, None))
check("wrench_delta says None rather than guessing",
      delta(None, w()) == (None, None))

# -- the rolling reference: follows slow free-space drift, not a contact step --
roll = mod.rolling_reference
FB = w(fz=99.0)          # an obviously-wrong fallback, so its use is visible


def ramp(n, per, start=0.0, hz=30.0):
    """n samples rising `per` newtons each, oldest first, 30 Hz."""
    return [(i / hz, w(fz=start + i * per)) for i in range(n)]


check("empty history falls back",
      list(roll([], 1.0, 0.2, 0.5, 5, FB)) == FB)
check("too few samples in the window falls back",
      list(roll(ramp(3, 0.0), 1.0, 0.2, 0.5, 5, FB)) == FB,
      "the first lag+window seconds of every leg")

# A steady drift: the reference trails the present by about (lag + window/2) * rate.
h = ramp(60, 0.01)                     # 0.30 N/s at 30 Hz
now = h[-1][0]
ref = roll(h, now, 0.2, 0.5, 5, FB)
present = h[-1][1][2]
lagged = present - ref[2]
check("a steady drift leaves only the lag, not the whole excursion",
      0.10 < lagged < 0.20,
      "present %.3f N, reference %.3f N, difference %.3f N" % (present, ref[2], lagged))
check("  and that is far below what the same drift gives a FIXED baseline",
      lagged < 0.25 * present,
      "fixed would see %.3f N" % present)

# A step on top of a drift: the reference must NOT follow it.
h2 = ramp(60, 0.01)
step = h2[-1][1][2] + 3.0
h2.append((h2[-1][0] + 1/30.0, w(fz=step)))
ref2 = roll(h2, h2[-1][0], 0.2, 0.5, 5, FB)
check("a step is seen in full -- the reference does not absorb it",
      abs((step - ref2[2]) - 3.0) < 0.25,
      "step 3.00 N, measured %.2f N" % (step - ref2[2]))

# Without the lag the step falls inside the averaging window and partly becomes its own reference.
ref_nolag = roll(h2, h2[-1][0], 0.0, 0.5, 5, FB)
check("WITHOUT the lag the same step reads smaller",
      (step - ref_nolag[2]) < (step - ref2[2]),
      "no lag %.2f N vs lagged %.2f N" % (step - ref_nolag[2], step - ref2[2]))

# End to end, with the real thresholds: drift must not fire, contact must.
ft, tt = 2.0, 0.4
check("0.3 N/s of drift does NOT fire at 2.0 N",
      detected(h[-1][1], roll(h, now, 0.2, 0.5, 5, FB), ft, tt)[0] is False)
check("a 3 N step DOES fire",
      detected(h2[-1][1], ref2, ft, tt)[0] is True)


# -- debounce: per franka_states sample, not per loop tick ----------------------
# The ramp loop runs at 100 Hz and franka_states at ~30 Hz, so the loop sees each
# sample about three times. Feed it exactly that: (sample number, over threshold).
def ticks(samples, debounce=2, repeats=3):
    over, last, fired = 0, None, []
    for n, hit in samples:
        for _ in range(repeats):
            over, last = mod.count_over(over, hit, n, last)
            fired.append(over >= debounce)
    return any(fired)


check("debounce 2: ONE spiky sample, seen on three loop ticks, does not fire",
      ticks([(1, False), (2, True), (3, False), (4, False)]) is False)
check("debounce 2: two samples running over threshold do fire",
      ticks([(1, False), (2, True), (3, True), (4, False)]) is True)
check("debounce 2: spikes in alternate samples never fire",
      ticks([(1, True), (2, False), (3, True), (4, False), (5, True)]) is False)
check("debounce 1: one sample is enough",
      ticks([(1, False), (2, True)], debounce=1) is True)

print()
if FAILED:
    print("%d FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("all guard checks passed")
