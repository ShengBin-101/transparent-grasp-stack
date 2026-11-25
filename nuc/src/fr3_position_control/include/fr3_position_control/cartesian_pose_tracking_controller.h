// A Cartesian pose controller that tracks a commanded target.
//
// Takes targets on ~target_pose (PoseStamped in the base frame, the same message as the
// impedance controller's equilibrium_pose, so pose_commander drives either by topic name)
// and moves the commanded pose toward them at 1 kHz through PoseSmoother. The robot's stiff
// internal joint impedance tracks that pose to about a millimetre, where a soft impedance
// controller stalls short on joint friction. Contact is resisted, not yielded to:
// pose_commander's guard is the soft stop and libfranka's collision reflex the hard one.
#pragma once

#include <memory>
#include <mutex>
#include <string>

#include <Eigen/Geometry>
#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <geometry_msgs/PoseStamped.h>
#include <hardware_interface/robot_hw.h>
#include <realtime_tools/realtime_publisher.h>
#include <ros/node_handle.h>
#include <ros/time.h>

#include <fr3_position_control/elbow_hold.h>
#include <fr3_position_control/pose_smoother.h>

namespace fr3_position_control {

class CartesianPoseTrackingController
    : public controller_interface::MultiInterfaceController<franka_hw::FrankaPoseCartesianInterface,
                                                            franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time& time) override;
  void update(const ros::Time& time, const ros::Duration& period) override;

 private:
  void targetCallback(const geometry_msgs::PoseStampedConstPtr& msg);

  franka_hw::FrankaPoseCartesianInterface* pose_interface_{nullptr};
  std::unique_ptr<franka_hw::FrankaCartesianPoseHandle> pose_handle_;

  std::unique_ptr<PoseSmoother> smoother_;
  ChannelLimits linear_{0.10, 5.0};
  ChannelLimits angular_{0.50, 5.0};
  std::string base_frame_{"fr3_link0"};
  double max_target_jump_{0.60};   // m from the current command; farther is refused

  // hold_elbow: latch the elbow the robot already commands (elbow_c: joint 3 angle, joint 4
  // sign) in starting() and send it every cycle, so joint 3 stays where it started, as
  // pose_commander's "elbow_locked" planner assumes. elbow_c, not q[2], so the first cycle is
  // a zero step. The cost: joint 3 no longer helps reach, so a pose the other six joints
  // cannot make stops with a reflex (the planner refuses such legs first). false = pose only.
  bool hold_elbow_requested_{true};
  bool hold_elbow_{false};
  Elbow elbow_{};
  bool elbow_logged_{false};   // starting() logs the latch once, again only if it changed
  Elbow logged_elbow_c_{};

  // Written by the subscriber thread; update() only try_locks, so the 1 kHz loop never blocks.
  std::mutex target_mutex_;
  Eigen::Vector3d target_position_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond target_orientation_{Eigen::Quaterniond::Identity()};
  bool target_fresh_{false};

  // What update() is steering to (a copy, so the RT loop owns it).
  Eigen::Vector3d goal_position_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond goal_orientation_{Eigen::Quaterniond::Identity()};

  ros::Subscriber sub_target_;
  realtime_tools::RealtimePublisher<geometry_msgs::PoseStamped> pub_command_;
  int publish_every_{20};   // cycles between ~commanded_pose messages (50 Hz)
  int cycle_{0};
};

}  // namespace fr3_position_control
