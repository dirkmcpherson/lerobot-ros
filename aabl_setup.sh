
# Bring up gazebo control for a simualted kinova arm WITHOUT a gripper
# ros2 launch kortex_bringup kortex_sim_control.launch.py   use_sim_time:=true   launch_rviz:=false
# ros2 launch kortex_bringup kortex_sim_control.launch.py   use_sim_time:=true   launch_rviz:=false robot_type:=gen3_lite dof:=6 gripper:=gen3_lite_2f
# ros2 launch kortex_bringup kortex_sim_control.launch.py   use_sim_time:=true   launch_rviz:=false robot_type:=gen3 dof:=7 gripper:=robotiq_2f_85

# There's an existing issue with grippers in gazebo that will be fixed from within gazebo later. Not worth getting into it for simulated robots, just smoke test here and when you want to grip things, do it on the robot.
ros2 launch kortex_bringup kortex_sim_control.launch.py   use_sim_time:=true   launch_rviz:=false robot_type:=gen3 dof:=7


# We modified the launch file to hardcode in some task cubes to mess with. Here's a command to copy that file into "forked_kortex" in case our work gets blown away
cp ~/workspace/ros2_kortex_ws/src/ros2_kortex/kortex_bringup/launch/kortex_sim_control.launch.py ./forked_cortex
cp ~/workspace/ros2_kortex_ws/src/ros2_kortex/kortex_description/arms/gen3_lite/6dof/urdf/kortex.ros2_control.xacro ./forked_cortex
cp ~/workspace/ros2_kortex_ws/src/ros2_kortex/kortex_description/grippers/gen3_lite_2f/urdf/gen3_lite_2f_macro.xacro ./forked_cortex

# here's how you build just the bringup package with colcon so your urdf changes get put in build
colcon build --packages-select kortex_bringup --symlink-install

# EXAMPLE POINT TASK TRAINED WITH DIFFUSION
# generate a bunch of pointing data into the lerobot format
python record_kinova_data.py

# train diffusion on the recorded lerobot data
bash train_kinova_reach.sh

# evaluate the trained policy
python ./eval_kinova_reach.py

## for teleop collection, you need to move the urdf file locally (or specify its location). There's notes about this in user_input spacemouse.py

## Compile the xacro into urdf:
xacro ~/workspace/ros2_kortex_ws/src/ros2_kortex/kortex_description/robots/gen3_lite_gen3_lite_2f.xacro > /tmp/gen3_lite.urdf
python record_kinova_data_teleoperated.py --input spacemouse --urdf /tmp/gen3_lite.urdf


########################################################################
# REAL Gen3 Lite on ROS 2 Lyrical (Ubuntu 26.04). Full notes + the "why":
#   docs/LYRICAL_KINOVA.md
########################################################################

# Session env (Lyrical needs Python 3.14 to import rclpy + torch together):
conda activate lerobot-ros-314
source /opt/ros/lyrical/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# HOME THE ARM FIRST (Kinova web app at http://192.168.1.10). If a joint is outside
# its URDF soft limit, joint_trajectory_controller won't activate, which also stalls
# the gripper (see docs §7.1).

# Bring up the real arm:
ros2 launch kortex_bringup gen3_lite.launch.py robot_ip:=192.168.1.10 launch_rviz:=false

# Gripper test (Lyrical gripper action is ParallelGripperCommand, goal is a JointState):
ros2 action send_goal /gen3_lite_2f_gripper_controller/gripper_cmd \
  control_msgs/action/ParallelGripperCommand \
  "{command: {name: [right_finger_bottom_joint], position: [0.5]}}"   # 0=open, 0.85=closed

# Cartesian EEF move (twist is in the TOOL frame -> closed-loop world-Z servo):
python real_eef_move_z.py
