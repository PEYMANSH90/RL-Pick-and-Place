import math
import csv
import os
import PyKDL as kdl
from kdl_parser_py.urdf import treeFromFile
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time
import numpy as np
from threading import Thread

# --- CONFIGURABLE PARAMETERS ---
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'abb_irb1600_support/urdf/irb1600_6_12.urdf')
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'torque_results.csv')

# Target position
TARGET_POS = [1.0, 0.0, 1.0]

# Step size for orientation changes (in degrees)
STEP_SIZE = 1.0

# Movement parameters
STEPS_PER_MOVE = 100  # More steps = smoother motion
MOVE_TIME = 0.5  # Time for each move in seconds

# Joint limits (in degrees)
JOINT_LIMITS = {
    'joint_1': (-180, 180),
    'joint_2': (-90, 150),
    'joint_3': (-180, 75),
    'joint_4': (-400, 400),
    'joint_5': (-120, 120),
    'joint_6': (-400, 400)
}

# --- UTILITY FUNCTIONS ---
def deg2rad(deg):
    return deg * math.pi / 180.0

def euler_to_kdl_frame(x, y, z, roll, pitch, yaw):
    rot = kdl.Rotation.RPY(roll, pitch, yaw)
    pos = kdl.Vector(x, y, z)
    return kdl.Frame(rot, pos)

def get_neighbor_orientations(roll, pitch, yaw):
    """Get 6 neighboring orientations by changing one angle at a time."""
    return [
        (roll + STEP_SIZE, pitch, yaw),
        (roll - STEP_SIZE, pitch, yaw),
        (roll, pitch + STEP_SIZE, yaw),
        (roll, pitch - STEP_SIZE, yaw),
        (roll, pitch, yaw + STEP_SIZE),
        (roll, pitch, yaw - STEP_SIZE)
    ]

# --- LOAD URDF AND BUILD KDL CHAIN ---
def load_kdl_chain(urdf_path, base_link, ee_link):
    ok, tree = treeFromFile(urdf_path)
    if not ok:
        raise RuntimeError("Failed to parse URDF into KDL tree")
    chain = tree.getChain(base_link, ee_link)
    return chain

# --- IK SOLVER ---
def get_ik_solution(chain, ik_solver, target_frame):
    q_init = kdl.JntArray(chain.getNrOfJoints())
    q_out = kdl.JntArray(chain.getNrOfJoints())
    ret = ik_solver.CartToJnt(q_init, target_frame, q_out)
    if ret >= 0:
        return [q_out]
    else:
        return []

# --- GRAVITY TORQUE CALCULATION ---
def compute_gravity_torque(chain, q, gravity_vector=kdl.Vector(0, 0, -9.81)):
    n_joints = chain.getNrOfJoints()
    torques = kdl.JntArray(n_joints)
    id_solver = kdl.ChainIdSolver_RNE(chain, gravity_vector)
    q_dot = kdl.JntArray(n_joints)
    q_dotdot = kdl.JntArray(n_joints)
    for i in range(n_joints):
        q_dot[i] = 0.0
        q_dotdot[i] = 0.0
    f_ext = [kdl.Wrench() for _ in range(chain.getNrOfSegments())]
    id_solver.CartToJnt(q, q_dot, q_dotdot, f_ext, torques)
    return [torques[i] for i in range(n_joints)]

class JointPublisher(Node):
    def __init__(self):
        super().__init__('joint_publisher')
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.current_positions = [0.0] * len(self.joint_names)

    def publish_joints(self, joint_angles):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [math.radians(angle) for angle in joint_angles]
        self.publisher.publish(msg)
        self.current_positions = joint_angles

def evaluate_orientation(chain, ik_solver, roll, pitch, yaw, joint_publisher):
    """Evaluate an orientation and return the minimum torque configuration."""
    target_frame = euler_to_kdl_frame(*TARGET_POS, deg2rad(roll), deg2rad(pitch), deg2rad(yaw))
    ik_solutions = get_ik_solution(chain, ik_solver, target_frame)
    
    if not ik_solutions:
        return None
    
    min_torque = float('inf')
    best_config = None
    
    # Find best solution
    for q in ik_solutions:
        joint_angles = [math.degrees(q[i]) for i in range(q.rows())]
        torques = compute_gravity_torque(chain, q)
        total_torque = sum(abs(t) for t in torques)
        if total_torque < min_torque:
            min_torque = total_torque
            best_config = {
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw,
                'joint_angles': joint_angles,
                'total_torque': total_torque
            }
    
    # Set position and wait
    if best_config:
        joint_publisher.publish_joints(best_config['joint_angles'])
        time.sleep(1.0)  # Stay at each position for 1 second
    
    return best_config

def main():
    # Initialize ROS2
    rclpy.init()
    joint_publisher = JointPublisher()
    
    print('Loading URDF and building KDL chain...')
    print(f'URDF path: {URDF_PATH}')
    base_link = 'base_link'
    ee_link = 'tool0'
    chain = load_kdl_chain(URDF_PATH, base_link, ee_link)
    ik_solver = kdl.ChainIkSolverPos_LMA(chain)

    # Initial orientation
    current_roll = 30.0
    current_pitch = 30.0
    current_yaw = 30.0
    
    visited_orientations = set()
    results = []
    
    while True:
        current_orientation = (current_roll, current_pitch, current_yaw)
        if current_orientation in visited_orientations:
            break
            
        visited_orientations.add(current_orientation)
        print(f"\nEvaluating orientation: {current_orientation}")
        
        # Evaluate current orientation
        current_config = evaluate_orientation(chain, ik_solver, current_roll, current_pitch, current_yaw, joint_publisher)
        if current_config:
            results.append(current_config)
            print(f"Current orientation torque: {current_config['total_torque']}")
        
        # Evaluate neighboring orientations
        neighbors = get_neighbor_orientations(current_roll, current_pitch, current_yaw)
        best_neighbor = None
        min_neighbor_torque = float('inf')
        
        for roll, pitch, yaw in neighbors:
            if (roll, pitch, yaw) in visited_orientations:
                continue
                
            config = evaluate_orientation(chain, ik_solver, roll, pitch, yaw, joint_publisher)
            if config and config['total_torque'] < min_neighbor_torque:
                min_neighbor_torque = config['total_torque']
                best_neighbor = (roll, pitch, yaw)
                results.append(config)
        
        if best_neighbor is None or min_neighbor_torque >= current_config['total_torque']:
            print("Reached local minimum")
            break
            
        # Move to the best neighbor
        current_roll, current_pitch, current_yaw = best_neighbor
        print(f"Moving to orientation: {best_neighbor} with torque: {min_neighbor_torque}")

    # Find the global minimum
    if results:
        best = min(results, key=lambda x: x['total_torque'])
        print('\nBest configuration found:')
        print('Orientation (deg):', best['roll'], best['pitch'], best['yaw'])
        print('Joint angles (deg):', best['joint_angles'])
        print('Total torque:', best['total_torque'])
        
        # Move to the best configuration smoothly
        joint_publisher.publish_joints(best['joint_angles'])
    else:
        print('No valid configurations found.')

    # Write to CSV
    with open(CSV_PATH, 'w', newline='') as csvfile:
        fieldnames = ['roll', 'pitch', 'yaw', 'joint_angles', 'total_torque']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f'Results written to {CSV_PATH}')

    # Cleanup ROS2
    joint_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 