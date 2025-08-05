#!/usr/bin/env python3
"""
Load Simulator Node for ABB IRB1600 Robot
=========================================

This node simulates varying loads on the end effector to test
the torque compensation system.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
import numpy as np
import math

class LoadSimulatorNode(Node):
    """ROS2 node for simulating end effector loads"""
    
    def __init__(self):
        super().__init__('load_simulator_node')
        
        # Publisher for simulated load
        self.load_pub = self.create_publisher(WrenchStamped, '/ee_load', 10)
        
        # Load simulation parameters
        self.time = 0.0
        self.load_frequency = 0.5  # Hz
        self.max_force = 50.0  # N
        self.max_torque = 25.0  # N⋅m
        
        # Load pattern (similar to training)
        self.load_pattern = [10, 75, -25, 40, 10, -10, -30, 5, 18, 10, 18, 17]
        self.pattern_index = 0
        
        # Timer for load updates (10 Hz)
        self.load_timer = self.create_timer(0.1, self.publish_load)
        
        self.get_logger().info("🔧 Load Simulator Node initialized!")
        self.get_logger().info(f"📊 Max Force: {self.max_force} N, Max Torque: {self.max_torque} N⋅m")
    
    def publish_load(self):
        """Publish simulated load on end effector"""
        # Create load message
        load_msg = WrenchStamped()
        load_msg.header.stamp = self.get_clock().now().to_msg()
        load_msg.header.frame_id = "tool0"  # End effector frame
        
        # Simulate varying load (similar to training pattern)
        pattern_value = self.load_pattern[self.pattern_index]
        normalized_load = pattern_value / 75.0  # Normalize to [-1, 1] range
        
        # Apply load in Z direction (vertical force)
        load_msg.wrench.force.x = 0.0
        load_msg.wrench.force.y = 0.0
        load_msg.wrench.force.z = normalized_load * self.max_force
        
        # Apply torque around Z axis
        load_msg.wrench.torque.x = 0.0
        load_msg.wrench.torque.y = 0.0
        load_msg.wrench.torque.z = normalized_load * self.max_torque
        
        # Publish load
        self.load_pub.publish(load_msg)
        
        # Update pattern index (change every 2 seconds)
        if int(self.time * 10) % 20 == 0:  # Every 2 seconds (20 * 0.1s)
            self.pattern_index = (self.pattern_index + 1) % len(self.load_pattern)
        
        # Log load changes
        if int(self.time * 10) % 50 == 0:  # Log every 5 seconds
            force_magnitude = np.linalg.norm([load_msg.wrench.force.x, load_msg.wrench.force.y, load_msg.wrench.force.z])
            torque_magnitude = np.linalg.norm([load_msg.wrench.torque.x, load_msg.wrench.torque.y, load_msg.wrench.torque.z])
            self.get_logger().info(f"📊 Load: Force={force_magnitude:.1f}N, Torque={torque_magnitude:.1f}N⋅m")
        
        self.time += 0.1

def main(args=None):
    rclpy.init(args=args)
    node = LoadSimulatorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 