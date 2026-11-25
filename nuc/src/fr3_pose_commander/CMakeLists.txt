cmake_minimum_required(VERSION 3.0.2)
project(fr3_pose_commander)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS
  scripts/pose_commander_node.py
  scripts/publish_camera_tf.py
  scripts/publish_tool_pose.py
  scripts/serve_description.py
  scripts/mock_joint_states.py
  scripts/mock_franka_states.py
  scripts/watch_tracking.py
  scripts/idle_wrench.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
