
# Bring up gazebo control for a simualted kinova arm WITHOUT a gripper
ros2 launch kortex_bringup kortex_sim_control.launch.py   use_sim_time:=true   launch_rviz:=false

# EXAMPLE POINT TASK TRAINED WITH DIFFUSION
# generate a bunch of pointing data into the lerobot format
python record_kinova_data.py

# train diffusion on the recorded lerobot data
bash train_kinova_reach.sh

# evaluate the trained policy
python ./eval_kinova_reach.py




## for teleop collection, you need to move the urdf file locally (or specify its location). There's notes about this in user_input spacemouse.py
TODO