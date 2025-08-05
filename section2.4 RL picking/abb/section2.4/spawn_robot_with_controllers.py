#!/usr/bin/env python3
"""
Spawn ABB IRB1600 Robot with Effort Controllers
===============================================

This script spawns the robot in Gazebo with proper effort controllers
to accept torque commands from the TD3 model.
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from ros_gz_interfaces.srv import SpawnEntity
import os

class RobotSpawner(Node):
    """Node to spawn robot with controllers"""
    
    def __init__(self):
        super().__init__('robot_spawner')
        
        # Wait for Gazebo to be ready
        self.get_logger().info("🔄 Waiting for Gazebo to be ready...")
        rclpy.spin_once(self, timeout_sec=2.0)
        
        # Spawn robot
        self.spawn_robot()
        
    def spawn_robot(self):
        """Spawn the ABB IRB1600 robot with effort controllers"""
        try:
            # Get the URDF file path
            urdf_path = os.path.expanduser("~/ros2_ws/src/abb/abb_irb1600_support/urdf/irb1600_6_12.urdf")
            
            if not os.path.exists(urdf_path):
                self.get_logger().error(f"❌ URDF file not found: {urdf_path}")
                return
            
            # Read the URDF file
            with open(urdf_path, 'r') as f:
                urdf_content = f.read()
            
            # Add effort controllers to the URDF
            urdf_with_controllers = self.add_controllers_to_urdf(urdf_content)
            
            # Create spawn service client
            spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
            
            # Wait for service
            while not spawn_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("⏳ Waiting for spawn service...")
            
            # Create spawn request
            request = SpawnEntity.Request()
            request.name = "abb_irb1600"
            request.xml = urdf_with_controllers
            
            # Spawn robot
            future = spawn_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            
            if future.result().success:
                self.get_logger().info("✅ Robot spawned successfully with effort controllers!")
                self.get_logger().info("🎯 Robot is ready to accept torque commands!")
            else:
                self.get_logger().error("❌ Failed to spawn robot")
                
        except Exception as e:
            self.get_logger().error(f"❌ Error spawning robot: {e}")
    
    def add_controllers_to_urdf(self, urdf_content):
        """Add effort controllers to the URDF"""
        # Add controller configuration
        controller_config = """
  <!-- Effort Controllers -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/</robotNamespace>
      <controlPeriod>0.001</controlPeriod>
      <robotParam>robot_description</robotParam>
    </plugin>
  </gazebo>
  
  <!-- Joint Controllers -->
  <gazebo>
    <plugin name="joint_state_controller" filename="libgazebo_ros_joint_state_publisher.so">
      <jointName>joint_1, joint_2, joint_3, joint_4, joint_5, joint_6</jointName>
      <updateRate>50</updateRate>
    </plugin>
  </gazebo>
  
  <!-- Effort Controllers for each joint -->
  <gazebo>
    <plugin name="joint_1_effort_controller" filename="libgazebo_ros_force.so">
      <bodyName>link_1</bodyName>
      <topicName>/effort_controllers/joint_1</topicName>
      <maxForce>1000</maxForce>
    </plugin>
  </gazebo>
  
  <gazebo>
    <plugin name="joint_2_effort_controller" filename="libgazebo_ros_force.so">
      <bodyName>link_2</bodyName>
      <topicName>/effort_controllers/joint_2</topicName>
      <maxForce>1000</maxForce>
    </plugin>
  </gazebo>
  
  <gazebo>
    <plugin name="joint_3_effort_controller" filename="libgazebo_ros_force.so">
      <bodyName>link_3</bodyName>
      <topicName>/effort_controllers/joint_3</topicName>
      <maxForce>1000</maxForce>
    </plugin>
  </gazebo>
  
  <gazebo>
    <plugin name="joint_4_effort_controller" filename="libgazebo_ros_force.so">
      <bodyName>link_4</bodyName>
      <topicName>/effort_controllers/joint_4</topicName>
      <maxForce>1000</maxForce>
    </plugin>
  </gazebo>
  
  <gazebo>
    <plugin name="joint_5_effort_controller" filename="libgazebo_ros_force.so">
      <bodyName>link_5</bodyName>
      <topicName>/effort_controllers/joint_5</topicName>
      <maxForce>1000</maxForce>
    </plugin>
  </gazebo>
  
  <gazebo>
    <plugin name="joint_6_effort_controller" filename="libgazebo_ros_force.so">
      <bodyName>link_6</bodyName>
      <topicName>/effort_controllers/joint_6</topicName>
      <maxForce>1000</maxForce>
    </plugin>
  </gazebo>
</robot>
"""
        
        # Insert controller config before closing robot tag
        if '</robot>' in urdf_content:
            urdf_with_controllers = urdf_content.replace('</robot>', controller_config)
        else:
            urdf_with_controllers = urdf_content + controller_config
        
        return urdf_with_controllers

def main(args=None):
    rclpy.init(args=args)
    spawner = RobotSpawner()
    
    try:
        rclpy.spin(spawner)
    except KeyboardInterrupt:
        pass
    finally:
        spawner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 