
# Bring up gazebo control for a simualted kinova arm WITHOUT a gripper
# ros2 launch kortex_bringup kortex_sim_control.launch.py   use_sim_time:=true   launch_rviz:=false
ros2 launch kortex_bringup kortex_sim_control.launch.py   use_sim_time:=true   launch_rviz:=false robot_type:=gen3_lite dof:=6 gripper:=gen3_lite_2f


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
