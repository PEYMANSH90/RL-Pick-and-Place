#!/bin/bash

# ABB IRB1600 Gazebo Simulation Launch Script
# This script launches the ABB IRB1600 robot in Gazebo with ROS 2 Control

echo "Launching ABB IRB1600 robot in Gazebo..."

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Source workspace
source ~/ros2_ws/install/setup.bash

# Launch the robot
ros2 launch abb_irb1600_gazebo abb_irb1600_gazebo.launch.py

echo "Robot launched successfully!" 