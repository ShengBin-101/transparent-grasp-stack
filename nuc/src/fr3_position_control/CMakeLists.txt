cmake_minimum_required(VERSION 3.4)
project(fr3_position_control)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  controller_interface
  hardware_interface
  franka_hw
  pluginlib
  realtime_tools
  geometry_msgs
)
find_package(Eigen3 REQUIRED)
# Same resolution as franka_example_controllers; build.sh points Franka_DIR at the
# workspace's local libfranka 0.10.0.
find_package(Franka 0.9.0 QUIET)
if(NOT Franka_FOUND)
  find_package(Franka 0.8.0 REQUIRED)
endif()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES fr3_position_control
  CATKIN_DEPENDS roscpp controller_interface hardware_interface franka_hw pluginlib
                 realtime_tools geometry_msgs
  DEPENDS Franka
)

add_library(fr3_position_control src/cartesian_pose_tracking_controller.cpp)
add_dependencies(fr3_position_control ${catkin_EXPORTED_TARGETS})
target_include_directories(fr3_position_control PUBLIC include)
target_include_directories(fr3_position_control SYSTEM PUBLIC
  ${EIGEN3_INCLUDE_DIRS} ${Franka_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(fr3_position_control PUBLIC ${Franka_LIBRARIES} ${catkin_LIBRARIES})

# The smoother's self-test: plain executable, no robot, no ROS.
#   devel/lib/fr3_position_control/test_pose_smoother
add_executable(test_pose_smoother test/test_pose_smoother.cpp)
target_include_directories(test_pose_smoother PRIVATE include)
target_include_directories(test_pose_smoother SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIRS})

# The elbow-latch self-test: plain executable, no robot, no ROS.
#   devel/lib/fr3_position_control/test_elbow_hold
add_executable(test_elbow_hold test/test_elbow_hold.cpp)
target_include_directories(test_elbow_hold PRIVATE include)

install(TARGETS fr3_position_control
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(FILES fr3_position_control_plugin.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
