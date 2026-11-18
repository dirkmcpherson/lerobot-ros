"""Gen3 Lite + 2F gripper in headless Gazebo with the stock kortex controllers.

A slim equivalent of kortex_bringup's kortex_sim_control.launch.py for
``robot_type:=gen3_lite dof:=6 gripper:=gen3_lite_2f``: same xacro, same controllers
yaml, same physics engine, but no GUI (``gz sim -s``), no RViz and no camera bridge, and
it needs only kortex_description (not kortex_bringup / kortex_driver).

    ros2 launch docker/gazebo/gen3_lite_sim.launch.py
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

CONTROLLERS = PathJoinSubstitution(
    [FindPackageShare("kortex_description"), "arms/gen3_lite/6dof/config", "ros2_controllers.yaml"]
)


def spawner(*args):
    return Node(package="controller_manager", executable="spawner", arguments=[*args, "-c", "/controller_manager"])


def generate_launch_description():
    robot_description = ParameterValue(
        Command(
            [
                FindExecutable(name="xacro"), " ",
                PathJoinSubstitution([FindPackageShare("kortex_description"), "robots", "kinova.urdf.xacro"]),
                " robot_ip:=xxx.yyy.zzz.www name:=gen3_lite arm:=gen3_lite dof:=6 vision:=false",
                " gripper:=gen3_lite_2f sim_gazebo:=true simulation_controllers:=", CONTROLLERS,
            ]
        ),
        value_type=str,
    )
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]),
                launch_arguments={
                    "gz_args": " -r -s -v 2 empty.sdf --physics-engine gz-physics-bullet-featherstone-plugin"
                }.items(),
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_description, "use_sim_time": True}],
            ),
            Node(
                package="ros_gz_sim",
                executable="create",
                arguments=["-topic", "robot_description", "-name", "gen3_lite", "-z", "0.3"],
            ),
            spawner("joint_state_broadcaster"),
            spawner("joint_trajectory_controller"),
            spawner("twist_controller", "--inactive"),
            spawner("gen3_lite_2f_gripper_controller"),
        ]
    )
