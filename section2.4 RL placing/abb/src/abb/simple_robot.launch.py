#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the path to the URDF file
    urdf_file_path = os.path.join(
        get_package_share_directory('abb_irb1600_support'),
        'urdf',
        'irb1600_6_12.urdf'
    )
    
    # Read the URDF file
    with open(urdf_file_path, 'r') as file:
        robot_description_content = file.read()
    
    # Robot state publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content
        }]
    )
    
    return LaunchDescription([
        robot_state_publisher_node
    ]) 