#!/usr/bin/env python3
"""Reads the arm's declared payload and the noise floor of its external-wrench estimate.

    rosrun fr3_pose_commander idle_wrench.py            # 5 s
    rosrun fr3_pose_commander idle_wrench.py _seconds:=10

Arm still and touching nothing; franka_control and franka_state_controller are enough.
1. Is the payload declared? m_total is what the robot's gravity compensation uses: the plain
   hand is 0.73 kg, Desk's end-effector-camera profile 1.03 kg. An undeclared 0.3 kg is ~2.9 N,
   which the Cartesian PD (no integral term) turns into a droop of F/K.
2. What is the noise floor? The guard fires on the change in O_F_ext_hat_K from a baseline,
   so its threshold must clear the peak excursion (not the sd: one spike fires it).
A few newtons of standing force is expected when the load is undeclared; read m_total first.
"""
import sys

import numpy as np
import rospy
from franka_msgs.msg import FrankaState


class Collector(object):
    """Collects FrankaState samples: external wrench, peak |dq|, latest q, first mass fields."""

    def __init__(self):
        self.wrench, self.dq, self.masses, self.q = [], [], None, None
        rospy.Subscriber("/franka_state_controller/franka_states",
                         FrankaState, self.on_state, queue_size=200)

    def on_state(self, msg):
        self.wrench.append([float(v) for v in msg.O_F_ext_hat_K])
        self.dq.append(max(abs(float(v)) for v in msg.dq))
        self.q = [float(v) for v in msg.q]
        if self.masses is None:
            self.masses = (float(msg.m_ee), float(msg.m_load), float(msg.m_total),
                           [float(v) for v in msg.F_x_Ctotal])


def main():
    # Before init_node, which blocks until a ROS master answers.
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return
    rospy.init_node("idle_wrench")
    seconds = float(rospy.get_param("~seconds", 5.0))
    c = Collector()
    rospy.loginfo("idle_wrench: collecting %.0fs -- arm STILL and touching NOTHING", seconds)
    rospy.sleep(seconds)
    if not c.wrench:
        print("\nNo franka_states. Is franka_control running?")
        return

    w = np.asarray(c.wrench)
    n, moved = len(w), max(c.dq)
    print("\n%d samples, %.0f Hz" % (n, n / seconds))
    if moved > 0.01:
        print("  !! THE ARM MOVED (max |dq| %.3f rad/s). Every number below is "
              "invalid -- it is measuring motion, not the standing load." % moved)

    m_ee, m_load, m_total, com = c.masses
    print("\n-- what the robot thinks it is carrying ------------------------")
    print("  m_ee    %.3f kg   (from Desk's end-effector config)" % m_ee)
    print("  m_load  %.3f kg   (from set_load; 0 = none applied)" % m_load)
    print("  m_total %.3f kg   centre (%.3f, %.3f, %.3f) m" % (m_total, *com))
    if m_total < 0.85:
        print("  => LOOKS LIKE THE PLAIN HAND. The camera bracket is not declared.")
        print("     Expect ~%.1f N of unmodelled force and ~%.0f mm of droop at 200 N/m."
              % ((1.03 - m_total) * 9.81, (1.03 - m_total) * 9.81 / 200.0 * 1000))
        print("     Fix in Desk (end-effector-camera.json), not with set_load:")
        print("     applying a load restarts the control loop.")
    elif m_total > 0.95:
        print("  => consistent with the camera end-effector profile (1.03 kg).")
    else:
        print("  => between the two known configurations; check Desk.")

    f, t = w[:, :3], w[:, 3:]
    fmag, tmag = np.linalg.norm(f, axis=1), np.linalg.norm(t, axis=1)
    print("\n-- standing external wrench (should be ~0 if the load is right) --")
    print("  force  mean (%+.2f, %+.2f, %+.2f) N   |F| %.2f N" % (*f.mean(0), fmag.mean()))
    print("  torque mean (%+.2f, %+.2f, %+.2f) Nm  |T| %.2f Nm" % (*t.mean(0), tmag.mean()))

    print("\n-- noise, which is what the guard threshold must clear ----------")
    df = np.linalg.norm(f - f.mean(0), axis=1)
    dt = np.linalg.norm(t - t.mean(0), axis=1)
    print("  |dF| about the mean: sd %.3f N   peak %.3f N" % (df.std(), df.max()))
    print("  |dT| about the mean: sd %.3f Nm  peak %.3f Nm" % (dt.std(), dt.max()))
    print("\n  suggested guard thresholds (3x the peak excursion):")
    print("    force_threshold:  %.1f   # N  above baseline" % max(1.0, 3 * df.max()))
    print("    torque_threshold: %.2f   # N.m above baseline" % max(0.2, 3 * dt.max()))
    print("  Put them in config/pose_commander.yaml under `guard:`, then")
    print("  re-run this with the arm gently touched to confirm they fire.")


if __name__ == "__main__":
    main()
