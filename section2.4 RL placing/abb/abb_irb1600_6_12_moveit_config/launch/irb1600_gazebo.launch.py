from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.conditions import IfCondition
from kdl_parser_py.urdf import treeFromFile

def generate_launch_description():
    # Declare the launch arguments
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
        gazebo,
        spawn_entity,
        joint_state_bridge
    ])

def load_kdl_chain(urdf_path, base_link, ee_link):
    ok, tree = treeFromFile(urdf_path)
    if not ok:
        raise RuntimeError("Failed to parse URDF into KDL tree")
    chain = tree.getChain(base_link, ee_link)
    return chain 