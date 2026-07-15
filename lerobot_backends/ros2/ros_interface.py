# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import threading
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand, ParallelGripperCommand
from lerobot.utils.errors import DeviceNotConnectedError
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import Executor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .config import ActionType, GripperActionType, ROS2InterfaceConfig
# NOTE: MoveIt2Servo is imported lazily inside connect() (see CARTESIAN_VELOCITY
# branch) so backends that don't use cartesian-velocity control (e.g. the
# joint-trajectory Kinova Gen3 Lite) don't require moveit_msgs to be installed.

logger = logging.getLogger(__name__)


class ROS2Interface:
    """Class to interface with a MoveIt2 manipulator.

    This class supports both JointGroupPositionController and JointTrajectoryController
    from ros2_control for arm control, depending on the configuration:

    - ActionType.JOINT_POSITION:
      Uses JointGroupPositionController.
      Publishes Float64MultiArray messages to '/position_controller/commands'

    - ActionType.JOINT_TRAJECTORY:
      Uses JointTrajectoryController.
      Publishes JointTrajectory messages to '/arm_controller/joint_trajectory'

    The gripper control also supports both trajectory and action-based control
    via the gripper_action_type configuration option.
    """

    def __init__(self, config: ROS2InterfaceConfig, action_type: ActionType, use_sim_time: bool = False):
        self.config = config
        self.action_type = action_type
        self._use_sim_time = use_sim_time
        self.robot_node: Node | None = None
        self.pos_cmd_pub: Publisher | None = None
        self.traj_cmd_pub: Publisher | None = None
        self.traj_action_client: ActionClient | None = None
        self.gripper_action_client: ActionClient | None = None
        self.gripper_traj_pub: Publisher | None = None
        self.executor: Executor | None = None
        self.moveit2_servo: MoveIt2Servo | None = None
        self.executor_thread: threading.Thread | None = None
        self.is_connected = False
        self._last_joint_state: dict[str, dict[str, float]] | None = None

    def connect(self) -> None:
        if not rclpy.ok():
            rclpy.init()

        from rclpy.parameter import Parameter as RclpyParameter
        self.robot_node = Node(
            "moveit2_interface_node",
            namespace=self.config.namespace,
            parameter_overrides=[
                RclpyParameter("use_sim_time", RclpyParameter.Type.BOOL, self._use_sim_time),
            ],
        )
        if self.action_type == ActionType.JOINT_POSITION:
            self.pos_cmd_pub = self.robot_node.create_publisher(
                Float64MultiArray, self.config.position_topic, 10
            )
        elif self.action_type == ActionType.JOINT_TRAJECTORY:
            if self.config.use_trajectory_action:
                # Prefer the FollowJointTrajectory ACTION over the command topic.
                # On some setups (notably ROS 2 Lyrical) the JTC command topic
                # silently drops one-shot trajectories, whereas the action interface
                # delivers reliably and reports an explicit result code.
                action_name = self.config.arm_topic.rsplit("/", 1)[0] + "/follow_joint_trajectory"
                self.traj_action_client = ActionClient(
                    self.robot_node,
                    FollowJointTrajectory,
                    action_name,
                    callback_group=ReentrantCallbackGroup(),
                )
            else:
                self.traj_cmd_pub = self.robot_node.create_publisher(
                    JointTrajectory, self.config.arm_topic, 10
                )
        elif self.action_type == ActionType.CARTESIAN_VELOCITY:
            from .moveit_servo import MoveIt2Servo  # requires moveit_msgs (apt: ros-<distro>-moveit-msgs)
            self.moveit2_servo = MoveIt2Servo(
                node=self.robot_node,
                frame_id=self.config.base_link,
                callback_group=ReentrantCallbackGroup(),
            )

        if self.config.gripper_joint_name:
            if self.config.gripper_action_type == GripperActionType.TRAJECTORY:
                self.gripper_traj_pub = self.robot_node.create_publisher(
                    JointTrajectory, self.config.gripper_traj_topic, 10
                )
            elif self.config.gripper_action_type == GripperActionType.PARALLEL_ACTION:
                self.gripper_action_client = ActionClient(
                    self.robot_node,
                    ParallelGripperCommand,
                    self.config.gripper_topic,
                    callback_group=ReentrantCallbackGroup(),
                )
                # Goal command is a sensor_msgs/JointState; the joint name is fixed
                # for the life of the client, only the position array changes per call.
                self._goal_msg = ParallelGripperCommand.Goal()
                self._goal_msg.command.name = [self.config.gripper_joint_name]
            else:
                self.gripper_action_client = ActionClient(
                    self.robot_node,
                    GripperCommand,
                    self.config.gripper_topic,
                    callback_group=ReentrantCallbackGroup(),
                )
                self._goal_msg = GripperCommand.Goal()

        self.joint_state_sub = self.robot_node.create_subscription(
            JointState,
            "joint_states",
            self._joint_state_callback,
            10,
        )

        # Create and start the executor in a separate thread
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.robot_node)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        time.sleep(3)  # Give some time to connect to services and receive messages

        if self.traj_action_client is not None:
            if not self.traj_action_client.wait_for_server(timeout_sec=5.0):
                logger.warning(
                    "Joint-trajectory action server not available after 5s. "
                    "Joint commands may be dropped."
                )
            else:
                logger.info("Joint-trajectory action server connected.")

        if self.gripper_action_client is not None:
            if not self.gripper_action_client.wait_for_server(timeout_sec=5.0):
                logger.warning(
                    "Gripper action server not available after 5s. "
                    "Gripper commands will be skipped."
                )
            else:
                logger.info("Gripper action server connected.")

        self.is_connected = True

    def send_joint_position_command(
        self, joint_positions: list[float], unnormalize: bool = True, time_from_start_sec: float = 1.0
    ) -> None:
        """
        Send a command to the robot's joints.
        Args:
            joint_positions (list[float]): The target positions for the joints.
            unnormalize (bool): Whether to unnormalize the joint positions based on the robot's configuration.
        """
        if not self.robot_node:
            raise DeviceNotConnectedError("ROS2Interface is not connected. You need to call `connect()`.")

        if unnormalize:
            if self.config.min_joint_positions is None or self.config.max_joint_positions is None:
                raise ValueError(
                    "Joint position normalization requires min and max joint positions to be set."
                )
            joint_positions = [
                min(max(pos, min_pos), max_pos)
                for pos, min_pos, max_pos in zip(
                    joint_positions,
                    self.config.min_joint_positions,
                    self.config.max_joint_positions,
                    strict=True,
                )
            ]

        if len(joint_positions) != len(self.config.arm_joint_names):
            raise ValueError(
                f"Expected {len(self.config.arm_joint_names)} joint positions, but got {len(joint_positions)}."
            )

        if self.action_type == ActionType.JOINT_TRAJECTORY:
            point = JointTrajectoryPoint()
            point.positions = joint_positions
            sec = int(time_from_start_sec)
            nanosec = int((time_from_start_sec - sec) * 1e9)
            point.time_from_start = Duration(sec=sec, nanosec=nanosec)

            if self.traj_action_client is not None:
                # Reliable path: send as a FollowJointTrajectory goal. Fire-and-forget
                # (send_goal_async) so the control loop never blocks; a new goal preempts
                # the previous one, which is the desired behavior for streaming waypoints.
                if not self.traj_action_client.server_is_ready():
                    logger.warning("Joint-trajectory action server not ready, skipping command.")
                    return
                goal = FollowJointTrajectory.Goal()
                goal.trajectory.joint_names = self.config.arm_joint_names
                goal.trajectory.points = [point]
                self.traj_action_client.send_goal_async(goal)
            elif self.traj_cmd_pub is not None:
                msg = JointTrajectory()
                msg.joint_names = self.config.arm_joint_names
                msg.points = [point]
                self.traj_cmd_pub.publish(msg)
            else:
                raise DeviceNotConnectedError("Trajectory command interface is not initialized.")
        else:
            if self.pos_cmd_pub is None:
                raise DeviceNotConnectedError("Position command publisher is not initialized.")
            msg = Float64MultiArray()
            msg.data = joint_positions
            self.pos_cmd_pub.publish(msg)

    def servo(self, linear, angular, normalize: bool = True) -> None:
        if not self.moveit2_servo:
            raise DeviceNotConnectedError("ROS2Interface is not connected. You need to call `connect()`.")

        if normalize:
            linear = [v * self.config.max_linear_velocity for v in linear]
            angular = [v * self.config.max_angular_velocity for v in angular]
        self.moveit2_servo.servo(linear=linear, angular=angular)

    def send_gripper_command(self, position: float, unnormalize: bool = True) -> bool:
        """
        Send a command to the gripper to move to a specific position.
        Args:
            position (float): The target position for the gripper (0=open, 1=closed).
        Returns:
            bool: True if the command was sent successfully, False otherwise.
        """
        if not self.robot_node:
            raise DeviceNotConnectedError("ROS2Interface is not connected. You need to call `connect()`.")

        open_pos = self.config.gripper_open_position
        closed_pos = self.config.gripper_close_position

        if unnormalize:
            # Map normalized position (0=open, 1=closed) to actual gripper joint position
            gripper_goal = open_pos + position * (closed_pos - open_pos)
        else:
            gripper_goal = position

        # Clamp to valid range regardless of source — sim physics can drift below the
        # open limit, and passing an out-of-range goal causes the action to stall forever.
        lo, hi = min(open_pos, closed_pos), max(open_pos, closed_pos)
        gripper_goal = max(lo, min(hi, float(gripper_goal)))

        if self.config.gripper_action_type == GripperActionType.TRAJECTORY:
            if self.gripper_traj_pub is None:
                raise DeviceNotConnectedError("Gripper command publisher is not initialized.")
            msg = JointTrajectory()
            msg.joint_names = [self.config.gripper_joint_name]
            point = JointTrajectoryPoint()
            point.positions = [gripper_goal]
            msg.points = [point]
            self.gripper_traj_pub.publish(msg)
            return True
        else:
            if not self.gripper_action_client:
                raise DeviceNotConnectedError("Gripper action client is not initialized.")

            if not self.gripper_action_client.server_is_ready():
                logger.warning("Gripper action server not available, skipping command.")
                return False

            # send_goal_async: fire-and-forget so the control loop is never blocked
            # waiting for the gripper to physically reach its target.
            if self.config.gripper_action_type == GripperActionType.PARALLEL_ACTION:
                # ParallelGripperCommand goal is a JointState; command.name was set
                # at connect() time, here we set the per-call target position.
                self._goal_msg.command.position = [gripper_goal]
            else:
                self._goal_msg.command.position = gripper_goal
            self.gripper_action_client.send_goal_async(self._goal_msg)
            return True

    @property
    def joint_state(self) -> dict[str, dict[str, float]] | None:
        """Get the last received joint state."""
        return self._last_joint_state

    def _joint_state_callback(self, msg: "JointState") -> None:
        self._last_joint_state = self._last_joint_state or {}
        positions = {}
        velocities = {}
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        for joint_name in self.config.arm_joint_names:
            idx = name_to_index.get(joint_name)
            if idx is None:
                raise ValueError(f"Joint '{joint_name}' not found in joint state.")
            positions[joint_name] = msg.position[idx]
            velocities[joint_name] = msg.velocity[idx]

        if self.config.gripper_joint_name:
            idx = name_to_index.get(self.config.gripper_joint_name)
            if idx is None:
                raise ValueError(
                    f"Gripper joint '{self.config.gripper_joint_name}' not found in joint state."
                )
            positions[self.config.gripper_joint_name] = msg.position[idx]
            velocities[self.config.gripper_joint_name] = msg.velocity[idx]

        self._last_joint_state["position"] = positions
        self._last_joint_state["velocity"] = velocities

    def disconnect(self):
        if self.joint_state_sub:
            self.joint_state_sub.destroy()
            self.joint_state_sub = None
        if self.pos_cmd_pub:
            self.pos_cmd_pub.destroy()
            self.pos_cmd_pub = None
        if self.traj_cmd_pub:
            self.traj_cmd_pub.destroy()
            self.traj_cmd_pub = None
        if self.gripper_action_client:
            self.gripper_action_client.destroy()
            self.gripper_action_client = None
        if self.gripper_traj_pub:
            self.gripper_traj_pub.destroy()
            self.gripper_traj_pub = None
        if self.robot_node:
            self.robot_node.destroy_node()
            self.robot_node = None
        if self.moveit2_servo:
            self.moveit2_servo = None

        if self.executor:
            self.executor.shutdown()
            self.executor = None
        if self.executor_thread:
            self.executor_thread.join()
            self.executor_thread = None

        self.is_connected = False
