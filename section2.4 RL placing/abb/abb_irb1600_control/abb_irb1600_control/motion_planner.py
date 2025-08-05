#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_py.core import MoveItPy
from moveit_py.planning_interface import MoveItPy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class MotionPlanner(Node):
    def __init__(self):
        super().__init__('motion_planner')
        
        # Initialize MoveItPy
        self.moveit = MoveItPy(node_name="motion_planner")
        self.arm = self.moveit.get_planning_component("manipulator")
        
        # Target position
        self.target_position = [1.0, 0.0, 1.0]  # x, y, z
        
        # Subscribe to joint states for torque calculation
        self.joint_states_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )
        
        # Store joint states
        self.current_joint_states = None
        
        # Get robot model for torque calculations
        self.robot_model = self.moveit.get_robot_model()
        
        # Find all possible joint configurations
        self.find_joint_configurations()
        
    def joint_states_callback(self, msg):
        self.current_joint_states = msg
        
    def calculate_torque(self, joint_config):
        """Calculate required torque for a given joint configuration."""
        # This is a simplified torque calculation
        # In reality, you would need to consider:
        # - Link masses and inertias
        # - Gravity compensation
        # - Friction
        # - Dynamic effects
        
        # For now, we'll use a simple model based on joint positions
        torques = []
        for i, angle in enumerate(joint_config):
            # Simple gravity compensation model
            # Each joint needs more torque the further it is from vertical
            gravity_torque = abs(math.sin(angle)) * 10.0  # 10.0 is a scaling factor
            torques.append(gravity_torque)
            
        return sum(torques)  # Return total torque
        
    def find_joint_configurations(self):
        """Find all possible joint configurations for the target position."""
        # Create a pose for the target position
        target_pose = Pose()
        target_pose.position.x = self.target_position[0]
        target_pose.position.y = self.target_position[1]
        target_pose.position.z = self.target_position[2]
        
        # Try different orientations
        roll = np.random.uniform(-math.pi, math.pi)
        pitch = np.random.uniform(-math.pi, math.pi)
        yaw = np.random.uniform(-math.pi, math.pi)
        
        # Convert to quaternion
        rot = R.from_euler('xyz', [roll, pitch, yaw])
        quat = rot.as_quat()
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]
        
        # Set the target pose
        self.arm.set_pose_target(target_pose)
        
        # Get all possible solutions
        solutions = []
        for i in range(8):  # Try different seed configurations
            # Set random seed configuration
            seed_state = self.arm.get_random_joint_values()
            self.arm.set_start_state(seed_state)
            
            # Plan to the target
            plan = self.arm.plan()
            if plan:
                # Get the joint values at the end of the plan
                joint_values = plan.trajectory.joint_trajectory.points[-1].positions
                solutions.append(joint_values)
        
        # Calculate torques for each solution
        torques = [self.calculate_torque(sol) for sol in solutions]
        
        # Find the solution with minimum torque
        if torques:
            min_torque_idx = np.argmin(torques)
            min_torque_solution = solutions[min_torque_idx]
            min_torque = torques[min_torque_idx]
            
            self.get_logger().info(f"Found {len(solutions)} solutions")
            self.get_logger().info(f"Minimum torque solution: {min_torque_solution}")
            self.get_logger().info(f"Minimum torque: {min_torque}")
            
            # Execute the minimum torque solution
            self.arm.set_joint_value_target(min_torque_solution)
            self.arm.execute()
        else:
            self.get_logger().error("No solutions found!")

def main(args=None):
    rclpy.init(args=args)
    motion_planner = MotionPlanner()
    rclpy.spin(motion_planner)
    motion_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 