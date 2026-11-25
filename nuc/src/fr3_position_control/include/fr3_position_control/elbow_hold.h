// Decides once, in starting(), which elbow CartesianPoseTrackingController commands.
//
// libfranka's elbow is {joint 3 angle (rad), joint 4 flip sign (+1/-1; -1 on this arm)},
// and {0, 0} means none. Any other sign makes checkElbow() throw inside the 1 kHz loop and
// end the session, so an invalid latched elbow is not sent: the controller falls back to the
// pose-only command. Pure (no ROS or libfranka), so test/test_elbow_hold.cpp pins it.
#pragma once

#include <array>
#include <cmath>

namespace fr3_position_control {

using Elbow = std::array<double, 2>;

struct ElbowPlan {
  bool hold{false};    // true: setCommand(pose, elbow) every cycle; false: setCommand(pose)
  Elbow elbow{};       // what is held (only meaningful when hold)
  const char* why{""}; // one phrase for the start-up log line
};

// Holds `elbow_c` (RobotState::elbow_c at starting(), the elbow the robot already commands),
// so the first cycle is a zero elbow step; falls back to no elbow if it is invalid.
inline ElbowPlan planElbow(bool hold_requested, const Elbow& elbow_c) {
  if (!hold_requested) {
    return {false, Elbow{}, "hold_elbow is false"};
  }
  if (!std::isfinite(elbow_c[0])) {
    return {false, Elbow{}, "latched elbow_c[0] is not finite"};
  }
  if (elbow_c[1] != 1.0 && elbow_c[1] != -1.0) {
    return {false, Elbow{}, "latched elbow_c[1] is not +1 or -1"};
  }
  return {true, elbow_c, "holding the commanded elbow"};
}

}  // namespace fr3_position_control
