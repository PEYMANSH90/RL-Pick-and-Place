#!/usr/bin/env python3
"""
Launch file for Torque Compensation System
==========================================

Launches the torque compensation node and load simulator for testing.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'use_load_simulator',
            default_value='true',
            description='Whether to use load simulator or real sensors'
        ),
        
        # Torque Compensation Node
        Node(
            package='abb_section2_4',  # Replace with your package name
            executable='torque_compensation_node.py',
            name='torque_compensation_node',
            output='screen',
            parameters=[{
                'control_frequency': 10.0,  # Hz
                'gear_ratio': 120.0,
                'dt': 0.16,
            }]
        ),
        
        # Load Simulator Node (conditional)
        Node(
            package='abb_section2_4',  # Replace with your package name
            executable='load_simulator_node.py',
            name='load_simulator_node',
            output='screen',
            condition=LaunchConfiguration('use_load_simulator'),
            parameters=[{
                'max_force': 50.0,  # N
                'max_torque': 25.0,  # N⋅m
                'load_frequency': 0.5,  # Hz
            }]
        ),
    ]) 