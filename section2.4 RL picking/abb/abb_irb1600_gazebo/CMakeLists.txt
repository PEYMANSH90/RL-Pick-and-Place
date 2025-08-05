cmake_minimum_required(VERSION 3.8)
project(abb_irb1600_gazebo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(ros_gz REQUIRED)
find_package(ros_gz_sim REQUIRED)
find_package(gz_ros2_control REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  worlds
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install Python scripts
install(PROGRAMS
  scripts/arm_gripper_loop_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package() 