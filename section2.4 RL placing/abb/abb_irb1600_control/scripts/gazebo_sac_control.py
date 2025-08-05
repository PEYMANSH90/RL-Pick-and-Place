#!/usr/bin/env python3
"""
Gazebo SAC Control for ABB IRB1600 Robot
========================================

This script integrates the trained SAC agent with Gazebo simulation
to control the ABB IRB1600 robot for torque-based control.

Features:
- Real-time joint state monitoring
- SAC agent integration
- Torque control via effort controllers
- Performance visualization
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from control_msgs.msg import JointControllerCommand
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import os
import json
import time
from datetime import datetime
import threading
from collections import deque

class GazeboSACController(Node):
    """ROS2 node for SAC-based torque control in Gazebo"""
    
    def __init__(self):
        super().__init__('gazebo_sac_controller')
        
        # Initialize SAC agent
        self.model_path = self.find_latest_model()
        if self.model_path:
            self.model = SAC.load(self.model_path)
            self.get_logger().info(f"Loaded SAC model from: {self.model_path}")
        else:
            self.get_logger().error("No SAC model found! Please train the model first.")
            return
        
        # Robot parameters
        self.joint_names = [
            'joint_1', 'joint_2', 'joint_3', 
            'joint_4', 'joint_5', 'joint_6'
        ]
        
        # Control parameters
        self.control_frequency = 6.25  # Hz (160ms step duration)
        self.control_period = 1.0 / self.control_frequency
        
        # State tracking
        self.current_joint_positions = np.zeros(6)
        self.current_joint_velocities = np.zeros(6)
        self.current_joint_efforts = np.zeros(6)
        self.target_torques = np.zeros(6)
        self.applied_torques = np.zeros(6)
        
        # Performance tracking
        self.torque_errors = []
        self.power_consumption = []
        self.time_stamps = []
        self.episode_reward = 0.0
        self.step_count = 0
        
        # System parameters (from paper)
        self.U = 48  # Voltage
        self.psi = 1.6  # Motor constant
        
        # Initialize ROS2 publishers and subscribers
        self.setup_ros_communication()
        
        # Start control loop
        self.control_timer = self.create_timer(self.control_period, self.control_callback)
        
        # Performance monitoring
        self.performance_timer = self.create_timer(1.0, self.performance_callback)
        
        self.get_logger().info("Gazebo SAC Controller initialized successfully!")
    
    def find_latest_model(self):
        """Find the most recent SAC model file"""
        # Look in the current directory and results subdirectory
        search_paths = [".", "results", "../results", "../../results"]
        
        for path in search_paths:
            if os.path.exists(path):
                model_files = [f for f in os.listdir(path) if f.startswith("SAC_model_") and f.endswith(".pkl")]
                if model_files:
                    # Sort by modification time and get the latest
                    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse=True)
                    return os.path.join(path, model_files[0])
        
        return None
    
    def setup_ros_communication(self):
        """Setup ROS2 publishers and subscribers"""
        
        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Publishers for each joint effort controller
        self.effort_publishers = {}
        for joint_name in self.joint_names:
            topic_name = f'/irb1600/{joint_name}_effort_controller/commands'
            self.effort_publishers[joint_name] = self.create_publisher(
                JointControllerCommand,
                topic_name,
                10
            )
        
        # Performance data publisher
        self.performance_pub = self.create_publisher(
            Float64MultiArray,
            '/sac_performance',
            10
        )
        
        self.get_logger().info("ROS2 communication setup completed")
    
    def joint_state_callback(self, msg):
        """Callback for joint state updates"""
        # Extract joint data
        for i, joint_name in enumerate(self.joint_names):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                self.current_joint_positions[i] = msg.position[idx]
                self.current_joint_velocities[i] = msg.velocity[idx]
                self.current_joint_efforts[i] = msg.effort[idx]
    
    def create_observation(self):
        """Create observation vector for SAC agent"""
        # For now, focus on joint 1 (you can extend to all joints)
        joint_idx = 0
        
        # Calculate error (simplified - you can implement more sophisticated load estimation)
        target_load = self.generate_target_load()
        current_torque = self.current_joint_efforts[joint_idx]
        error = target_load - current_torque
        
        # Create observation: [error, integral_error, derivative_error, current_torque, load_torque, prev_T_star]
        # For now, use simplified values
        integral_error = 0.0  # You can implement proper integration
        derivative_error = 0.0  # You can implement proper differentiation
        prev_T_star = self.applied_torques[joint_idx] if self.step_count > 0 else 0.0
        
        obs = np.array([
            error, integral_error, derivative_error, 
            current_torque, target_load, prev_T_star
        ], dtype=np.float32)
        
        return obs, target_load
    
    def generate_target_load(self):
        """Generate target load profile (simplified version)"""
        # Use the load profile from the paper
        user_loads = [10, 75, -25, 40, 10, -10, -30, 5, 18, 10, 18, 17]
        
        # Calculate which load to use based on current time
        time_index = int(time.time() * self.control_frequency) % len(user_loads)
        return user_loads[time_index]
    
    def control_callback(self):
        """Main control loop callback"""
        try:
            # Create observation for SAC agent
            obs, target_load = self.create_observation()
            
            # Get action from SAC agent
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Apply action to joint 1 (you can extend to all joints)
            joint_idx = 0
            joint_name = self.joint_names[joint_idx]
            
            # Scale action to reasonable torque range
            torque_command = float(action[0]) * 10.0  # Scale factor
            
            # Publish torque command
            effort_msg = JointControllerCommand()
            effort_msg.data = torque_command
            self.effort_publishers[joint_name].publish(effort_msg)
            
            # Update tracking variables
            self.target_torques[joint_idx] = target_load
            self.applied_torques[joint_idx] = torque_command
            
            # Calculate performance metrics
            error = target_load - self.current_joint_efforts[joint_idx]
            power = np.abs(self.U * self.current_joint_efforts[joint_idx] / self.psi)
            
            # Store performance data
            self.torque_errors.append(error)
            self.power_consumption.append(power)
            self.time_stamps.append(time.time())
            
            # Calculate reward (same as training)
            error_penalty = error**2
            action_penalty = 0.01 * torque_command**2
            reward = -(error_penalty + action_penalty)
            self.episode_reward += reward
            
            self.step_count += 1
            
            # Log every 50 steps
            if self.step_count % 50 == 0:
                self.get_logger().info(
                    f"Step {self.step_count}: Error={error:.2f}, "
                    f"Power={power:.2f}W, Reward={reward:.2f}"
                )
                
        except Exception as e:
            self.get_logger().error(f"Error in control callback: {str(e)}")
    
    def performance_callback(self):
        """Callback for performance monitoring and publishing"""
        if len(self.torque_errors) > 0:
            # Calculate performance metrics
            rmse = np.sqrt(np.mean(np.array(self.torque_errors)**2))
            avg_power = np.mean(self.power_consumption)
            
            # Publish performance data
            performance_msg = Float64MultiArray()
            performance_msg.data = [
                rmse, avg_power, self.episode_reward, 
                float(self.step_count)
            ]
            self.performance_pub.publish(performance_msg)
    
    def plot_performance(self):
        """Plot performance data"""
        if len(self.torque_errors) == 0:
            self.get_logger().warn("No performance data to plot")
            return
        
        # Convert to numpy arrays
        errors = np.array(self.torque_errors)
        power = np.array(self.power_consumption)
        times = np.array(self.time_stamps) - self.time_stamps[0]
        
        # Create performance plot
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Torque Error
        plt.subplot(2, 2, 1)
        plt.plot(times, errors, 'r', linewidth=1)
        plt.title('Torque Tracking Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Error')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Power Consumption
        plt.subplot(2, 2, 2)
        plt.plot(times, power, 'b', linewidth=1)
        plt.title('Power Consumption')
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Applied Torques
        plt.subplot(2, 2, 3)
        applied_torques = np.array(self.applied_torques[:len(times)])
        plt.plot(times, applied_torques, 'g', linewidth=1)
        plt.title('Applied Torques')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Performance Summary
        plt.subplot(2, 2, 4)
        metrics = ['RMSE', 'Avg Power', 'Total Reward', 'Steps']
        values = [
            np.sqrt(np.mean(errors**2)),
            np.mean(power),
            self.episode_reward,
            self.step_count
        ]
        plt.bar(metrics, values, color=['red', 'blue', 'green', 'orange'])
        plt.title('Performance Summary')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('gazebo_sac_performance.png')
        plt.show()
        
        # Save performance data
        performance_data = {
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "avg_power": float(np.mean(power)),
            "total_reward": float(self.episode_reward),
            "total_steps": int(self.step_count),
            "max_error": float(np.max(np.abs(errors))),
            "min_error": float(np.min(np.abs(errors)))
        }
        
        with open('gazebo_sac_performance.json', 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        self.get_logger().info("Performance data saved to files")

def main(args=None):
    rclpy.init(args=args)
    
    controller = GazeboSACController()
    
    try:
        # Run the controller
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("Shutting down SAC controller...")
        controller.plot_performance()
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 