"""Stand-in for the kortex driver: /joint_states at 100 Hz plus FollowJointTrajectory and
GripperCommand action servers that jump to each goal. Usage: fake_driver.py <seconds>"""

import sys
import time

import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

START = [-0.255, -0.607, 2.202, 0.249, -1.469, -1.227]  # tool at (0.30, 0.16, 0.22)


class Driver(Node):
    def __init__(self):
        super().__init__("fake_kortex")
        self.joints = list(START)
        self.gripper = 0.0
        self.goals = 0
        self.pub = self.create_publisher(JointState, "joint_states", 10)
        self.create_timer(0.01, self.publish_state)
        ActionServer(
            self, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory", self.on_trajectory
        )
        ActionServer(self, GripperCommand, "/gen3_lite_2f_gripper_controller/gripper_cmd", self.on_gripper)

    def publish_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"joint_{i}" for i in range(1, 7)] + ["right_finger_bottom_joint"]
        msg.position = [*self.joints, self.gripper]
        msg.velocity = [0.0] * 7
        self.pub.publish(msg)

    def on_trajectory(self, goal_handle):
        self.joints = list(goal_handle.request.trajectory.points[-1].positions)
        self.goals += 1
        goal_handle.succeed()
        return FollowJointTrajectory.Result()

    def on_gripper(self, goal_handle):
        self.gripper = goal_handle.request.command.position
        goal_handle.succeed()
        return GripperCommand.Result()


def main():
    lifetime = float(sys.argv[1]) if len(sys.argv) > 1 else 22.0
    rclpy.init()
    node = Driver()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    start = time.time()
    while time.time() - start < lifetime:
        executor.spin_once(timeout_sec=0.05)
    print(f"fake driver exiting after {node.goals} trajectory goals", flush=True)


if __name__ == "__main__":
    main()
