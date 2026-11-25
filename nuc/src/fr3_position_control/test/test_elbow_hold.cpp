// Self-test for planElbow -- which elbow, if any, the pose controller commands.
//
//   g++-9 -std=c++14 -O2 -Wall -Wextra -I include test/test_elbow_hold.cpp
//       -o /tmp/test_elbow_hold && /tmp/test_elbow_hold      (one line)
//
// Also built by CMake as test_elbow_hold. Checks that a valid latched elbow_c is held
// exactly, sign included; that hold_elbow false gives the pose-only command; and that an
// elbow_c libfranka would throw on falls back to pose-only. What the firmware does with the
// elbow needs the arm check in README.md ("The position controller's elbow").
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#include <fr3_position_control/elbow_hold.h>

using fr3_position_control::Elbow;
using fr3_position_control::ElbowPlan;
using fr3_position_control::planElbow;

namespace {
int failed = 0;
void check(const std::string& label, bool ok) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", label.c_str());
  if (!ok) {
    failed++;
  }
}
}  // namespace

int main() {
  // elbow_c as measured on this arm under this controller.
  const Elbow measured{-0.1238, -1.0};

  ElbowPlan p = planElbow(true, measured);
  check("hold + measured elbow_c -> hold", p.hold);
  check("hold + measured elbow_c -> joint 3 angle unchanged", p.elbow[0] == measured[0]);
  check("hold + measured elbow_c -> sign unchanged (-1)", p.elbow[1] == -1.0);

  p = planElbow(true, Elbow{0.25, 1.0});
  check("hold + positive sign -> held as +1", p.hold && p.elbow[0] == 0.25 && p.elbow[1] == 1.0);

  // Joint 3 at exactly 0 rad is a legal elbow; only the SIGN marks "no elbow".
  p = planElbow(true, Elbow{0.0, -1.0});
  check("hold + joint 3 at 0 rad -> still held", p.hold && p.elbow[1] == -1.0);

  p = planElbow(false, measured);
  check("hold_elbow false -> pose only", !p.hold);
  check("hold_elbow false -> elbow left {0,0} (libfranka's 'none')",
        p.elbow[0] == 0.0 && p.elbow[1] == 0.0);

  p = planElbow(true, Elbow{0.0, 0.0});
  check("elbow_c {0,0} -> pose only", !p.hold);
  p = planElbow(true, Elbow{-0.1238, 0.0});
  check("sign 0 -> pose only (libfranka would throw)", !p.hold);
  p = planElbow(true, Elbow{-0.1238, -0.5});
  check("sign -0.5 -> pose only", !p.hold);
  p = planElbow(true, Elbow{std::numeric_limits<double>::quiet_NaN(), -1.0});
  check("NaN angle -> pose only", !p.hold);
  p = planElbow(true, Elbow{std::numeric_limits<double>::infinity(), -1.0});
  check("inf angle -> pose only", !p.hold);
  p = planElbow(true, Elbow{-0.1238, std::numeric_limits<double>::quiet_NaN()});
  check("NaN sign -> pose only", !p.hold);

  check("every plan explains itself", planElbow(true, measured).why[0] != '\0' &&
                                          planElbow(false, measured).why[0] != '\0' &&
                                          planElbow(true, Elbow{0, 0}).why[0] != '\0');

  std::printf("\n%s: %d failed\n", failed == 0 ? "PASS" : "FAIL", failed);
  return failed == 0 ? 0 : 1;
}
