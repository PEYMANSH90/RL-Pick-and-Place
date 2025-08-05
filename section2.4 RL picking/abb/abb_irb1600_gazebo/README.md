# ABB IRB1600 Gazebo Simulation Package

This package provides Gazebo simulation for the ABB IRB1600 6-axis robotic arm using ROS 2 Control.

## Features

- Full 6-DOF ABB IRB1600 robot simulation in Gazebo
- ROS 2 Control integration for joint trajectory control
- Gripper simulation using joint_6
- RViz visualization
- Python controller for automated pick-and-place operations

## Prerequisites

- ROS 2 Humble (or later)
- Gazebo
- ROS 2 Control packages:
  - `ros2_control`
  - `ros2_controllers`
  - `gazebo_ros2_control`
  - `joint_trajectory_controller`
  - `gripper_action_controller`

## Installation

1. Make sure you have the required dependencies:
```bash
sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-gazebo-ros2-control ros-humble-joint-trajectory-controller ros-humble-gripper-action-controller
```

2. Build the workspace:
```bash
cd ~/ros2_ws
colcon build --packages-select abb_irb1600_gazebo
source install/setup.bash
```

## Usage

### Quick Launch

Use the provided bash script for quick launching:
```bash
cd abb_irb1600_support/gazebo
./launch_robot.sh
```

### Manual Launch

Or launch manually using ROS 2:
```bash
ros2 launch abb_irb1600_gazebo abb_irb1600_gazebo.launch.py
```

### Launch Options

You can customize the launch with various parameters:

```bash
# Launch without RViz
ros2 launch abb_irb1600_gazebo abb_irb1600_gazebo.launch.py launch_rviz:=false

# Use a different world
ros2 launch abb_irb1600_gazebo abb_irb1600_gazebo.launch.py world:=my_custom_world.world

# Launch only the controllers (without Gazebo)
ros2 launch abb_irb1600_gazebo load_ros2_controllers.launch.py
```

## Controlling the Robot

### Using the Python Controller

Run the automated pick-and-place controller:
```bash
ros2 run abb_irb1600_gazebo arm_gripper_loop_controller.py
```

### Manual Control

You can also control the robot manually using ROS 2 topics:

```bash
# Send joint trajectory commands
ros2 topic pub /arm_controller/follow_joint_trajectory trajectory_msgs/msg/JointTrajectory

# Control gripper
ros2 topic pub /gripper_action_controller/gripper_cmd control_msgs/msg/GripperCommand
```

## File Structure

```
gazebo/
├── config/
│   └── ros2_controllers.yaml          # Controller configuration
├── launch/
│   ├── abb_irb1600_gazebo.launch.py  # Main launch file
│   └── load_ros2_controllers.launch.py # Controller-only launch
├── scripts/
│   └── arm_gripper_loop_controller.py # Python controller
├── urdf/
│   ├── gazebo_sim_ros2_control.urdf.xacro
│   └── abb_irb1600_6_12_ros2_control.urdf.xacro
├── worlds/
│   └── empty.world                    # Gazebo world
├── launch_robot.sh                    # Quick launch script
├── package.xml
├── CMakeLists.txt
└── README.md
```

## Troubleshooting

### Common Issues

1. **Controllers not loading**: Make sure all ROS 2 Control packages are installed
2. **URDF parsing errors**: Check that the URDF file paths are correct
3. **Gazebo not starting**: Ensure Gazebo is properly installed and configured

### Debug Commands

```bash
# Check if controllers are loaded
ros2 control list_controllers

# Check joint states
ros2 topic echo /joint_states

# Check robot state
ros2 topic echo /robot_description
```

## Customization

### Modifying Joint Positions

Edit the target and home positions in `scripts/arm_gripper_loop_controller.py`:

```python
self.target_pos = [0.5, -0.3, 0.2, 0.0, 0.5, 0.0]  # Target position
self.home_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]    # Home position
```

### Adding Custom Worlds

Place your custom world files in the `worlds/` directory and reference them in the launch command.

### Modifying Controller Parameters

Edit `config/ros2_controllers.yaml` to adjust controller settings like update rates, constraints, etc.

## License

This package is provided as-is for educational and research purposes. 