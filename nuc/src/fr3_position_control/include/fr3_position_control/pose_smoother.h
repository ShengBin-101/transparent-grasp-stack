// Smooth, overshoot-free tracking of a Cartesian pose target, one 1 kHz step at a time.
// Header-only and free of ROS and libfranka, so test/test_pose_smoother.cpp checks the pose
// sequence sent to the arm without a robot.
#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Geometry>

namespace fr3_position_control {

struct ChannelLimits {
  double v_max;   // m/s or rad/s: rate limiter speed
  double omega;   // 1/s: each smoothing stage's rate
};

// The Cartesian pose interface takes the next pose every millisecond and aborts with a
// *_discontinuity reflex if the poses imply too much velocity, acceleration or jerk; this
// turns pose_commander's 100 Hz ramp, or a single jump, into a jerk-continuous sequence.
// Per channel (rotation on SO(3) via slerp): a rate limiter toward the target at v_max,
// then two first-order lags at rate omega. A monotone input stays monotone, so the command
// never passes the target, and speed <= v_max, accel <= v_max*omega, jerk <= v_max*omega^2;
// a ramp at v is trailed by 2v/omega. libfranka's own rate limiter stays on as a backstop at
// the FR3's far higher limits.
class PoseSmoother {
 public:
  PoseSmoother(const ChannelLimits& linear, const ChannelLimits& angular)
      : lin_(linear), ang_(angular) {}

  void reset(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation) {
    r_p_ = s1_p_ = s2_p_ = position;
    const Eigen::Quaterniond q = orientation.normalized();
    r_q_ = s1_q_ = s2_q_ = q;
  }

  void step(const Eigen::Vector3d& target_position, Eigen::Quaterniond target_orientation,
            double dt) {
    // -- translation -----------------------------------------------------------
    Eigen::Vector3d d = target_position - r_p_;
    const double max_step = lin_.v_max * dt;
    const double n = d.norm();
    if (n > max_step) {
      d *= max_step / n;
    }
    r_p_ += d;
    const double k_lin = std::min(1.0, lin_.omega * dt);
    s1_p_ += k_lin * (r_p_ - s1_p_);
    s2_p_ += k_lin * (s1_p_ - s2_p_);

    // -- rotation, on SO(3), taking the short way -------------------------------
    target_orientation.normalize();
    if (target_orientation.dot(r_q_) < 0.0) {
      target_orientation.coeffs() *= -1.0;
    }
    const double gap = r_q_.angularDistance(target_orientation);
    const double max_turn = ang_.v_max * dt;
    if (gap > 1e-15) {
      r_q_ = r_q_.slerp(std::min(1.0, max_turn / gap), target_orientation).normalized();
    }
    const double k_ang = std::min(1.0, ang_.omega * dt);
    s1_q_ = s1_q_.slerp(k_ang, r_q_).normalized();
    s2_q_ = s2_q_.slerp(k_ang, s1_q_).normalized();
  }

  const Eigen::Vector3d& position() const { return s2_p_; }
  const Eigen::Quaterniond& orientation() const { return s2_q_; }

  // Distance from the commanded pose to the target; zero once the command has arrived.
  double remaining(const Eigen::Vector3d& target_position) const {
    return (target_position - s2_p_).norm();
  }

 private:
  ChannelLimits lin_;
  ChannelLimits ang_;
  Eigen::Vector3d r_p_ = Eigen::Vector3d::Zero(), s1_p_ = Eigen::Vector3d::Zero(),
                  s2_p_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond r_q_ = Eigen::Quaterniond::Identity(), s1_q_ = Eigen::Quaterniond::Identity(),
                     s2_q_ = Eigen::Quaterniond::Identity();
};

}  // namespace fr3_position_control
