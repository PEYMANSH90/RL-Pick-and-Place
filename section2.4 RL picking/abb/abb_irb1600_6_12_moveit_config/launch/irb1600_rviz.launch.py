from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_gui = LaunchConfiguration('use_gui')

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

    # Joint State Publisher Node
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    # Joint State Publisher GUI Node
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
        condition=IfCondition(use_gui),
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    # RViz Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }],
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('abb_irb1600_6_12_moveit_config'),
            'rviz',
            'moveit.rviz'
        ])]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'use_gui',
            default_value='true',
            description='Flag to enable joint_state_publisher_gui'
        ),
        robot_state_publisher,
        joint_state_publisher,
        joint_state_publisher_gui,
        rviz_node
    ]) 