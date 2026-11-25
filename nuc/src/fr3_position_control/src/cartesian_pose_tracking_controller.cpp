// CartesianPoseTrackingController; the design notes are in its header.
#include <fr3_position_control/cartesian_pose_tracking_controller.h>

#include <array>
#include <cmath>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <hardware_interface/hardware_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

namespace fr3_position_control {

namespace {

// franka's poses are column-major 4x4 arrays.
void fromArray(const std::array<double, 16>& m, Eigen::Vector3d* p, Eigen::Quaterniond* q) {
  Eigen::Matrix4d t = Eigen::Map<const Eigen::Matrix4d>(m.data());
  *p = t.block<3, 1>(0, 3);
  *q = Eigen::Quaterniond(t.block<3, 3>(0, 0)).normalized();
}

std::array<double, 16> toArray(const Eigen::Vector3d& p, const Eigen::Quaterniond& q) {
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
  t.block<3, 1>(0, 3) = p;
  std::array<double, 16> out{};
  Eigen::Map<Eigen::Matrix4d>(out.data()) = t;
  return out;
}

}  // namespace

bool CartesianPoseTrackingController::init(hardware_interface::RobotHW* robot_hw,
                                           ros::NodeHandle& nh) {
  pose_interface_ = robot_hw->get<franka_hw::FrankaPoseCartesianInterface>();
  if (pose_interface_ == nullptr) {
    ROS_ERROR("CartesianPoseTrackingController: no Cartesian pose interface from hardware");
    return false;
  }
  std::string arm_id;
  if (!nh.getParam("arm_id", arm_id)) {
    ROS_ERROR("CartesianPoseTrackingController: parameter arm_id is missing");
    return false;
  }
  try {
    pose_handle_ = std::make_unique<franka_hw::FrankaCartesianPoseHandle>(
        pose_interface_->getHandle(arm_id + "_robot"));
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM("CartesianPoseTrackingController: no pose handle: " << e.what());
    return false;
  }

  nh.param("linear/max_velocity", linear_.v_max, linear_.v_max);
  nh.param("linear/smoothing_rate", linear_.omega, linear_.omega);
  nh.param("angular/max_velocity", angular_.v_max, angular_.v_max);
  nh.param("angular/smoothing_rate", angular_.omega, angular_.omega);
  nh.param("base_frame", base_frame_, base_frame_);
  nh.param("max_target_jump", max_target_jump_, max_target_jump_);
  nh.param("hold_elbow", hold_elbow_requested_, hold_elbow_requested_);
  if (linear_.v_max <= 0 || linear_.omega <= 0 || angular_.v_max <= 0 || angular_.omega <= 0) {
    ROS_ERROR("CartesianPoseTrackingController: limits must be positive");
    return false;
  }
  smoother_ = std::make_unique<PoseSmoother>(linear_, angular_);

  sub_target_ = nh.subscribe("target_pose", 1, &CartesianPoseTrackingController::targetCallback,
                             this, ros::TransportHints().reliable().tcpNoDelay());
  pub_command_.init(nh, "commanded_pose", 1);

  ROS_INFO("CartesianPoseTrackingController: %s, linear <= %.3f m/s (accel <= %.2f, jerk <= %.2f), "
           "angular <= %.2f rad/s, targets within %.2f m of the command",
           base_frame_.c_str(), linear_.v_max, linear_.v_max * linear_.omega,
           linear_.v_max * linear_.omega * linear_.omega, angular_.v_max, max_target_jump_);
  return true;
}

void CartesianPoseTrackingController::starting(const ros::Time& /*time*/) {
  // Start from the pose the robot already commands (O_T_EE_d), so the first cycle is a
  // zero step, and hold it until a target arrives.
  Eigen::Vector3d p;
  Eigen::Quaterniond q;
  fromArray(pose_handle_->getRobotState().O_T_EE_d, &p, &q);
  smoother_->reset(p, q);
  goal_position_ = p;
  goal_orientation_ = q;

  // Latch the elbow the robot already commands (elbow_c); anything else is a first-cycle
  // step the firmware can refuse as start_elbow_invalid.
  const franka::RobotState& state = pose_handle_->getRobotState();
  const ElbowPlan plan = planElbow(hold_elbow_requested_, state.elbow_c);
  hold_elbow_ = plan.hold;
  elbow_ = plan.elbow;
  // starting() runs twice per start (from the spawner, then on the first 1 kHz cycle, whose
  // latch the motion uses); the second logs only if elbow_c changed, keeping the RT path quiet.
  if (elbow_logged_ && state.elbow_c == logged_elbow_c_) {
    // same latch as already reported
  } else if (hold_elbow_) {
    ROS_INFO("CartesianPoseTrackingController: elbow HELD at joint 3 = %.4f rad, sign %+.0f "
             "(elbow_c; q[2] = %.4f, elbow_d = [%.4f, %+.0f])",
             elbow_[0], elbow_[1], state.q[2], state.elbow_d[0], state.elbow_d[1]);
  } else if (hold_elbow_requested_) {
    // libfranka would reject the latched value (checkElbow throws in the loop): pose only.
    ROS_ERROR("CartesianPoseTrackingController: elbow NOT held (%s): elbow_c = [%.4f, %.4f], "
              "q[2] = %.4f -- pose-only command, firmware chooses joint 3",
              plan.why, state.elbow_c[0], state.elbow_c[1], state.q[2]);
  } else {
    ROS_INFO("CartesianPoseTrackingController: elbow NOT held (%s) -- pose-only command, "
             "firmware chooses joint 3 (elbow_c = [%.4f, %+.0f], q[2] = %.4f)",
             plan.why, state.elbow_c[0], state.elbow_c[1], state.q[2]);
  }
  elbow_logged_ = true;
  logged_elbow_c_ = state.elbow_c;
  {
    std::lock_guard<std::mutex> lock(target_mutex_);
    target_fresh_ = false;
  }
  cycle_ = 0;
}

void CartesianPoseTrackingController::update(const ros::Time& time, const ros::Duration& period) {
  // Pick up a new target without ever blocking the 1 kHz loop.
  {
    std::unique_lock<std::mutex> lock(target_mutex_, std::try_to_lock);
    if (lock.owns_lock() && target_fresh_) {
      const double jump = (target_position_ - smoother_->position()).norm();
      if (jump <= max_target_jump_) {
        goal_position_ = target_position_;
        goal_orientation_ = target_orientation_;
      } else {
        ROS_WARN_THROTTLE(1.0,
                          "CartesianPoseTrackingController: target %.3f m from the command "
                          "(limit %.2f) ignored -- a frame or units bug, not a move",
                          jump, max_target_jump_);
      }
      target_fresh_ = false;
    }
  }

  // franka_hw calls update at 1 kHz; period is 1 ms (0 on the very first cycle).
  const double dt = period.toSec() > 0.0 ? period.toSec() : 0.001;
  smoother_->step(goal_position_, goal_orientation_, dt);
  const std::array<double, 16> pose = toArray(smoother_->position(), smoother_->orientation());
  if (hold_elbow_) {
    pose_handle_->setCommand(pose, elbow_);   // the same elbow every cycle, never re-derived
  } else {
    pose_handle_->setCommand(pose);           // elbow {} = none: firmware chooses joint 3
  }

  if (++cycle_ >= publish_every_ && pub_command_.trylock()) {
    cycle_ = 0;
    auto& m = pub_command_.msg_;
    m.header.stamp = time;
    m.header.frame_id = base_frame_;
    m.pose.position.x = smoother_->position().x();
    m.pose.position.y = smoother_->position().y();
    m.pose.position.z = smoother_->position().z();
    m.pose.orientation.x = smoother_->orientation().x();
    m.pose.orientation.y = smoother_->orientation().y();
    m.pose.orientation.z = smoother_->orientation().z();
    m.pose.orientation.w = smoother_->orientation().w();
    pub_command_.unlockAndPublish();
  }
}

void CartesianPoseTrackingController::targetCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
  // Only base-frame targets are accepted; another or an empty frame_id is refused, not misread.
  if (msg->header.frame_id != base_frame_) {
    ROS_WARN_THROTTLE(1.0,
                      "CartesianPoseTrackingController: target in frame '%s', expected '%s' -- "
                      "ignored", msg->header.frame_id.c_str(), base_frame_.c_str());
    return;
  }
  Eigen::Quaterniond q(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y,
                       msg->pose.orientation.z);
  if (!std::isfinite(q.norm()) || q.norm() < 1e-6) {
    ROS_WARN_THROTTLE(1.0, "CartesianPoseTrackingController: invalid orientation -- ignored");
    return;
  }
  Eigen::Vector3d p(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
  if (!p.allFinite()) {
    ROS_WARN_THROTTLE(1.0, "CartesianPoseTrackingController: non-finite position -- ignored");
    return;
  }
  std::lock_guard<std::mutex> lock(target_mutex_);
  target_position_ = p;
  target_orientation_ = q.normalized();
  target_fresh_ = true;
}

}  // namespace fr3_position_control

PLUGINLIB_EXPORT_CLASS(fr3_position_control::CartesianPoseTrackingController,
                       controller_interface::ControllerBase)
