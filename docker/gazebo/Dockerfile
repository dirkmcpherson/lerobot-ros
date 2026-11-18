# Gen3 Lite in headless Gazebo (ROS 2 Jazzy, Gazebo Harmonic, gz_ros2_control) for testing
# the HIL-SERL adapter without an arm. See README.md in this directory.
#
#   docker build --network host -t lerobot-ros-gazebo:jazzy docker/gazebo
#   (add --build-arg BASE=<a local ROS Jazzy image> to reuse layers you already have)
ARG BASE=ros:jazzy
FROM ${BASE}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-control-msgs ros-jazzy-rmw-cyclonedds-cpp ros-jazzy-ros2launch ros-jazzy-ros2controlcli \
      ros-jazzy-kortex-description ros-jazzy-kinova-gen3-lite-moveit-config \
      ros-jazzy-gz-ros2-control ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge \
      ros-jazzy-controller-manager ros-jazzy-joint-state-broadcaster \
      ros-jazzy-joint-trajectory-controller ros-jazzy-gripper-controllers \
      ros-jazzy-picknik-twist-controller ros-jazzy-picknik-reset-fault-controller \
      ros-jazzy-robot-state-publisher ros-jazzy-xacro \
    && rm -rf /var/lib/apt/lists/*

# kortex_description 0.2.6 (apt): kortex_robot.xacro passes `gripper` to the Gen3 Lite's
# load_arm macro, which does not declare it, so the Lite fails to expand
# ('Invalid parameter "gripper"'). Declare it; the Lite macro does not use it.
RUN f=/opt/ros/jazzy/share/kortex_description/arms/gen3_lite/6dof/urdf/gen3_lite_macro.xacro \
    && sed -i 's/^    gripper_joint_name\r\?$/    gripper:=gen3_lite_2f\n    gripper_joint_name/' "$f" \
    && grep -q 'gripper:=gen3_lite_2f' "$f" \
    && bash -c 'source /opt/ros/jazzy/setup.bash && xacro /opt/ros/jazzy/share/kortex_description/robots/gen3_lite_gen3_lite_2f.xacro sim_gazebo:=true > /dev/null'

WORKDIR /work
CMD ["bash"]
