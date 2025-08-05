#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from robot_state_publisher import RobotStatePublisher
import os

def main():
    rclpy.init()
    
    # Read the URDF file
    urdf_path = os.path.join(os.path.dirname(__file__), 'abb_irb1600_support/urdf/irb1600_6_12.urdf')
    print(f"Loading URDF from: {urdf_path}")
    
    with open(urdf_path, 'r') as file:
        robot_description = file.read()
    
    print("URDF loaded successfully!")
    
    # Create robot state publisher node
    node = Node('robot_state_publisher')
    node.declare_parameter('robot_description', robot_description)
    
    # Create robot state publisher
    robot_state_publisher = RobotStatePublisher(node)
    
    print("Robot State Publisher started successfully!")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 