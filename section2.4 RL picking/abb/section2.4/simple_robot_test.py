#!/usr/bin/env python3
"""
Simple Robot Movement Test
==========================

This script directly publishes torque commands to make the robot move
so you can see it working in Gazebo.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import time

class SimpleRobotTest(Node):
    """Simple test to make robot move"""
    
    def __init__(self):
        super().__init__('simple_robot_test')
        
        # Publisher for torque commands
        self.torque_pub = self.create_publisher(Float64MultiArray, '/effort_controllers/commands', 10)
        
        # Timer for movement (1 Hz)
        self.movement_timer = self.create_timer(1.0, self.send_movement_command)
        
        self.time = 0.0
        self.get_logger().info("🤖 Simple Robot Test initialized!")
        self.get_logger().info("🎯 Robot should start moving in Gazebo!")
    
    def send_movement_command(self):
        """Send simple movement commands"""
        # Create sinusoidal movement pattern
        base_torque = 10.0  # Base torque magnitude
        frequency = 0.5  # Hz
        
        # Create torques for each joint
        torques = []
        for i in range(6):
            # Different phase for each joint
            phase = i * np.pi / 3
            torque = base_torque * np.sin(2 * np.pi * frequency * self.time + phase)
            torques.append(torque)
        
        # Create and publish message
        msg = Float64MultiArray()
        msg.data = torques
        
        self.torque_pub.publish(msg)
        
        # Log movement
        self.get_logger().info(f"📊 Time: {self.time:.1f}s, Torques: {[f'{t:.1f}' for t in torques]}")
        
        self.time += 1.0

def main(args=None):
    rclpy.init(args=args)
    test_node = SimpleRobotTest()
    
    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        pass
    finally:
        test_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 