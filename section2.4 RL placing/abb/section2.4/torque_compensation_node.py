#!/usr/bin/env python3
"""
Torque Compensation Node for ABB IRB1600 Robot
==============================================

This node loads the trained TD3 model and applies torque compensation
to maintain robot position under varying end effector loads.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
import numpy as np
import os
import sys

# Add the path to load the trained model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the trained model
from stable_baselines3 import TD3

class TorqueCompensationNode(Node):
    """ROS2 node for torque compensation using trained TD3 model"""
    
    def __init__(self):
        super().__init__('torque_compensation_node')
        
        # Load the trained TD3 model
        self.model_path = "/home/peyman/ros2_ws/results/curriculum_level_2_model.pkl"
        if os.path.exists(self.model_path):
            self.model = TD3.load(self.model_path)
            self.get_logger().info(f"✅ Loaded trained model from: {self.model_path}")
        else:
            self.get_logger().error(f"❌ Model not found at: {self.model_path}")
            return
        
        # Robot parameters
        self.gear_ratio = 120.0
        self.dt = 0.16  # 160ms timestep as per training
        
        # Joint names for ABB IRB1600
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.num_joints = len(self.joint_names)
        
        # Current state
        self.current_joint_positions = np.zeros(self.num_joints)
        self.current_joint_velocities = np.zeros(self.num_joints)
        self.current_joint_efforts = np.zeros(self.num_joints)
        self.target_positions = np.zeros(self.num_joints)  # Initial positions
        self.ee_load = np.zeros(6)  # [fx, fy, fz, tx, ty, tz]
        
        # Control variables
        self.integral_error = np.zeros(self.num_joints)
        self.prev_error = np.zeros(self.num_joints)
        self.prev_prev_error = np.zeros(self.num_joints)
        
        # Publishers and Subscribers
        self.torque_pub = self.create_publisher(Float64MultiArray, '/effort_controllers/commands', 10)
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.ee_load_sub = self.create_subscription(
            WrenchStamped, '/ee_load', self.ee_load_callback, 10
        )
        
        # Control timer (10 Hz = 100ms, close to training 160ms)
        self.control_timer = self.create_timer(0.1, self.apply_torque_compensation)
        
        self.get_logger().info("🚀 Torque Compensation Node initialized!")
        self.get_logger().info(f"📊 Controlling {self.num_joints} joints: {self.joint_names}")
    
    def joint_state_callback(self, msg):
        """Update current joint states"""
        for i, joint_name in enumerate(self.joint_names):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                self.current_joint_positions[i] = msg.position[idx]
                self.current_joint_velocities[i] = msg.velocity[idx]
                self.current_joint_efforts[i] = msg.effort[idx]
    
    def ee_load_callback(self, msg):
        """Update end effector load"""
        self.ee_load[0] = msg.wrench.force.x
        self.ee_load[1] = msg.wrench.force.y
        self.ee_load[2] = msg.wrench.force.z
        self.ee_load[3] = msg.wrench.torque.x
        self.ee_load[4] = msg.wrench.torque.y
        self.ee_load[5] = msg.wrench.torque.z
    
    def calculate_joint_loads(self):
        """Calculate joint loads from end effector load using Jacobian"""
        # Simplified: assume load is distributed equally across joints
        # In practice, you'd use the robot's Jacobian matrix
        joint_loads = np.zeros(self.num_joints)
        
        # Convert EE force to joint torques (simplified)
        # For now, distribute the force magnitude across joints
        force_magnitude = np.linalg.norm(self.ee_load[:3])  # Force magnitude
        torque_magnitude = np.linalg.norm(self.ee_load[3:])  # Torque magnitude
        
        # Distribute load across joints (simplified approach)
        for i in range(self.num_joints):
            joint_loads[i] = (force_magnitude + torque_magnitude) / self.num_joints
        
        return joint_loads
    
    def get_observation(self, joint_idx):
        """Create observation vector for the TD3 model (same as training)"""
        # Calculate error for this joint
        joint_error = self.target_positions[joint_idx] - self.current_joint_positions[joint_idx]
        
        # Update control variables
        self.integral_error[joint_idx] += joint_error * self.dt
        derivative_error = (joint_error - self.prev_error[joint_idx]) / self.dt
        
        # Create observation (same format as training)
        obs = np.array([
            joint_error / 100.0,  # Normalize error
            self.prev_error[joint_idx] / 100.0,  # Normalize previous error
            self.integral_error[joint_idx] / 100.0,  # Normalize integral error
            derivative_error / 100.0,  # Normalize derivative error
            self.current_joint_efforts[joint_idx],  # Current motor torque
            self.calculate_joint_loads()[joint_idx],  # Joint load
        ], dtype=np.float32)
        
        # Update previous error
        self.prev_prev_error[joint_idx] = self.prev_error[joint_idx]
        self.prev_error[joint_idx] = joint_error
        
        return obs
    
    def apply_torque_compensation(self):
        """Apply torque compensation using the trained TD3 model"""
        try:
            # Calculate joint loads from EE load
            joint_loads = self.calculate_joint_loads()
            
            # Get compensated torques for each joint
            compensated_torques = np.zeros(self.num_joints)
            
            for joint_idx in range(self.num_joints):
                # Get observation for this joint
                obs = self.get_observation(joint_idx)
                
                # Get action from trained model
                action, _ = self.model.predict(obs, deterministic=True)
                motor_torque = float(action[0])  # T* from model
                
                # Apply gear ratio to get joint torque
                joint_torque = motor_torque * self.gear_ratio
                compensated_torques[joint_idx] = joint_torque
            
            # Publish torques to robot
            torque_msg = Float64MultiArray()
            torque_msg.data = compensated_torques.tolist()
            self.torque_pub.publish(torque_msg)
            
            # Log performance
            if self.get_clock().now().nanoseconds % 1000000000 < 100000000:  # Log every ~1 second
                max_error = np.max(np.abs(self.target_positions - self.current_joint_positions))
                max_load = np.max(np.abs(joint_loads))
                self.get_logger().info(f"📊 Max Error: {max_error:.3f} rad, Max Load: {max_load:.2f} N⋅m")
        
        except Exception as e:
            self.get_logger().error(f"❌ Error in torque compensation: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TorqueCompensationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 