// Self-test for PoseSmoother -- the only code in this package that shapes motion.
//
//   g++-9 -std=c++14 -O2 -I include -I /usr/include/eigen3 \
//       test/test_pose_smoother.cpp -o /tmp/test_pose_smoother && /tmp/test_pose_smoother
//
// Also built by CMake as test_pose_smoother. Against the configured limits, it checks that
// a far step and a rotation step stay within the speed, acceleration and jerk bounds and
// never overshoot; that pose_commander's 12 mm/s, 100 Hz ramp is tracked with the designed
// lag and ends on the target; and that an unchanged target (either quaternion sign) moves nothing.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <fr3_position_control/pose_smoother.h>

using fr3_position_control::ChannelLimits;
using fr3_position_control::PoseSmoother;

namespace {
int failed = 0;
void check(const std::string& label, bool ok, const std::string& detail = "") {
  std::printf("%-4s %s%s%s\n", ok ? "ok" : "FAIL", label.c_str(),
              detail.empty() ? "" : "  (", detail.empty() ? "" : (detail + ")").c_str());
  if (!ok) {
    failed++;
  }
}
std::string fmt(const char* f, double a, double b = 0, double c = 0) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), f, a, b, c);
  return buf;
}

// Must match config/fr3_position_control.yaml.
const ChannelLimits LIN{0.10, 5.0};   // v_max m/s, omega 1/s
const ChannelLimits ANG{0.50, 5.0};   // v_max rad/s, omega 1/s
// Bounds the structure guarantees (see pose_smoother.h), with 2% for discretisation.
const double LIN_A = LIN.v_max * LIN.omega * 1.02;
const double LIN_J = LIN.v_max * LIN.omega * LIN.omega * 1.02;
const double DT = 0.001;

struct Stats {
  double v_max = 0, a_max = 0, j_max = 0;
};

// Run until settled or timeout; track the limits actually used, by finite
// differences of the POSE sequence -- exactly what libfranka checks.
Stats run(PoseSmoother& s, const Eigen::Vector3d& target, const Eigen::Quaterniond& q_target,
          double seconds, std::vector<Eigen::Vector3d>* trace = nullptr) {
  Stats st;
  Eigen::Vector3d p_prev = s.position(), v_prev = Eigen::Vector3d::Zero(),
                  a_prev = Eigen::Vector3d::Zero();
  const int n = static_cast<int>(seconds / DT);
  for (int i = 0; i < n; ++i) {
    s.step(target, q_target, DT);
    Eigen::Vector3d v = (s.position() - p_prev) / DT;
    Eigen::Vector3d a = (v - v_prev) / DT;
    Eigen::Vector3d j = (a - a_prev) / DT;
    st.v_max = std::max(st.v_max, v.norm());
    st.a_max = std::max(st.a_max, a.norm());
    if (i > 1) {
      st.j_max = std::max(st.j_max, j.norm());
    }
    p_prev = s.position();
    v_prev = v;
    a_prev = a;
    if (trace) {
      trace->push_back(s.position());
    }
  }
  return st;
}
}  // namespace

int main() {
  const Eigen::Quaterniond down(0.0, 1.0, 0.0, 0.0);  // w,x,y,z: tool pointing down
  const Eigen::Vector3d home(0.488, 0.015, 0.397);

  {  // 1. nothing to do -> nothing moves
    PoseSmoother s(LIN, ANG);
    s.reset(home, down);
    run(s, home, down, 1.0);
    check("a target equal to the current pose produces no motion",
          (s.position() - home).norm() < 1e-12 &&
              s.orientation().angularDistance(down) < 1e-12);
  }

  {  // 2. a far step (304 mm down, the home -> standoff move as one jump)
    PoseSmoother s(LIN, ANG);
    s.reset(home, down);
    const Eigen::Vector3d goal(0.538, -0.012, 0.092);
    std::vector<Eigen::Vector3d> trace;
    Stats st = run(s, goal, down, 12.0, &trace);
    const double err = (s.position() - goal).norm();
    // overshoot: furthest the trace ever got past the goal along the move
    const Eigen::Vector3d dir = (goal - home).normalized();
    double overshoot = 0;
    for (auto& p : trace) {
      overshoot = std::max(overshoot, (p - goal).dot(dir));
    }
    check("far step: ends on the target", err < 1e-4, fmt("%.4f mm", err * 1e3));
    check("far step: never passes the goal", overshoot <= 1e-9, fmt("%.6f mm", overshoot * 1e3));
    check("far step: speed within limit", st.v_max <= LIN.v_max * 1.001,
          fmt("%.3f m/s", st.v_max));
    check("far step: acceleration within v_max*omega", st.a_max <= LIN_A,
          fmt("%.3f m/s^2", st.a_max));
    check("far step: jerk within v_max*omega^2", st.j_max <= LIN_J, fmt("%.2f m/s^3", st.j_max));
  }

  {  // 3. pose_commander's real input: a 100 Hz ramp at 12 mm/s, 70 mm down
    PoseSmoother s(LIN, ANG);
    const Eigen::Vector3d start(0.538, -0.012, 0.092), goal(0.538, -0.012, 0.022);
    s.reset(start, down);
    const double speed = 0.012, dist = (goal - start).norm(), dur = dist / speed;
    double lag_max = 0;
    Stats st;
    Eigen::Vector3d p_prev = s.position(), v_prev = Eigen::Vector3d::Zero(),
                    a_prev = Eigen::Vector3d::Zero();
    Eigen::Vector3d target = start;
    const int steps = static_cast<int>((dur + 3.0) / DT);
    for (int i = 0; i < steps; ++i) {
      const double t = i * DT;
      if (i % 10 == 0) {  // a new setpoint every 10 ms, like _ramp at rate_hz 100
        const double f = std::min(1.0, t / dur);
        target = start + f * (goal - start);
      }
      s.step(target, down, DT);
      if (t < dur) {
        lag_max = std::max(lag_max, (target - s.position()).norm());
      }
      Eigen::Vector3d v = (s.position() - p_prev) / DT, a = (v - v_prev) / DT;
      st.a_max = std::max(st.a_max, a.norm());
      if (i > 1) {
        st.j_max = std::max(st.j_max, ((a - a_prev) / DT).norm());
      }
      p_prev = s.position();
      v_prev = v;
      a_prev = a;
    }
    const double err = (s.position() - goal).norm();
    check("12 mm/s ramp: ends on the target", err < 1e-4, fmt("%.4f mm", err * 1e3));
    // steady ramp lag = v * 2 / omega = 0.012 * 2 / 5 = 4.8 mm, plus half a 10 ms step
    check("12 mm/s ramp: tracking lag as designed (v*2/omega)", lag_max < 5.5e-3, fmt("%.2f mm", lag_max * 1e3));
    check("12 mm/s ramp: jerk within v_max*omega^2 (100 Hz steps smoothed)", st.j_max <= LIN_J,
          fmt("%.2f m/s^3", st.j_max));
  }

  {  // 4. rotation step: 30 deg about the base z axis
    PoseSmoother s(LIN, ANG);
    s.reset(home, down);
    const Eigen::Quaterniond goal =
        (Eigen::Quaterniond(Eigen::AngleAxisd(30.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ())) * down)
            .normalized();
    double w_max = 0;
    Eigen::Quaterniond s_prev = s.orientation();
    const int n = static_cast<int>(6.0 / DT);
    for (int i = 0; i < n; ++i) {
      s.step(home, goal, DT);
      w_max = std::max(w_max, s_prev.angularDistance(s.orientation()) / DT);
      s_prev = s.orientation();
    }
    const double err = s.orientation().angularDistance(goal);
    check("rotation step: ends on the target orientation", err < 1e-4,
          fmt("%.5f deg", err * 180.0 / M_PI));
    check("rotation step: angular speed within limit", w_max <= ANG.v_max * 1.001,
          fmt("%.3f rad/s", w_max));
    check("rotation step: position untouched", (s.position() - home).norm() < 1e-12);
  }

  {  // 5. the quaternion sign must not matter (q and -q are one orientation)
    PoseSmoother s(LIN, ANG);
    s.reset(home, down);
    Eigen::Quaterniond neg = down;
    neg.coeffs() *= -1.0;
    run(s, home, neg, 1.0);
    check("-q target is the same orientation: no motion",
          s.orientation().angularDistance(down) < 1e-9);
  }

  if (failed) {
    std::printf("\n%d FAILED\n", failed);
    return 1;
  }
  std::printf("\nall pose smoother checks passed\n");
  return 0;
}
