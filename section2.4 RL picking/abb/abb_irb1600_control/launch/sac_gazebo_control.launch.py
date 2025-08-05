#!/usr/bin/env python3
"""
Launch file for SAC-based torque control with ABB IRB1600 in Gazebo
==================================================================

This launch file starts:
1. Gazebo simulation with ABB IRB1600 robot
2. SAC controller node
3. Required ROS2 bridges and controllers
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.conditions import IfCondition

def generate_launch_description():
    # Declare the launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_gui = LaunchConfiguration('use_gui')
    use_sac_control = LaunchConfiguration('use_sac_control')
    
    # Get the path to the URDF file
    urdf_path = PathJoinSubstitution([
        FindPackageShare('abb_irb1600_support'),
        'urdf',
        'irb1600_6_12.xacro'
    ])

    # Robot State Publisher Node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': ParameterValue(
                Command([
                    FindExecutable(name='xacro'),
                    ' ',
                    urdf_path
                ]),
                value_type=str
            )
        }]
    )

    # Joint State Publisher Node (for initial joint states)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'publish_default_positions': True,
            'publish_default_velocities': True,
            'publish_default_efforts': True
        }]
    )

    # Gazebo launch using gz
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '--headless-rendering'],
        output='screen'
    )

    # Spawn the robot in Gazebo
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', '/robot_description',
            '-name', 'irb1600',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.0'
        ],
        output='screen'
    )

    # Bridge for joint states
    joint_state_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model',
            '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock'
        ],
        output='screen'
    )

    # SAC Controller Node
    sac_controller = Node(
        package='abb_irb1600_control',
        executable='gazebo_sac_control.py',
        name='sac_controller',
        output='screen',
        condition=IfCondition(use_sac_control),
        parameters=[{
            'use_sim_time': use_sim_time,
            'control_frequency': 6.25,  # Hz (160ms step duration)
            'joint_names': ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        }]
    )

    # Controller spawner for effort controllers
    controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_1_effort_controller',
            'joint_2_effort_controller', 
            'joint_3_effort_controller',
            'joint_4_effort_controller',
            'joint_5_effort_controller',
            'joint_6_effort_controller'
        ],
        output='screen',
        condition=IfCondition(use_sac_control)
    )

    # Performance monitor node
    performance_monitor = Node(
        package='rqt_plot',
        executable='rqt_plot',
        arguments=['/sac_performance/data[0]:/sac_performance/data[1]'],
        output='screen',
        condition=IfCondition(use_sac_control)
    )

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'use_gui',
            default_value='false',
            description='Flag to enable Gazebo GUI'
        ),
        DeclareLaunchArgument(
            'use_sac_control',
            default_value='true',
            description='Flag to enable SAC-based torque control'
        ),
        
        # Core nodes
        robot_state_publisher,
        joint_state_publisher,
        gazebo,
        spawn_entity,
        joint_state_bridge,
        
        # SAC control nodes
        sac_controller,
        controller_spawner,
        performance_monitor
    ]) 