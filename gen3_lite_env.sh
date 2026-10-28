# Source me (`. gen3_lite_env.sh`) in every terminal used for the Gen3 Lite on Jazzy:
# driver, lerobot-record / -train / -rollout.
#
# Starts from a clean ROS environment because other workspaces (e.g. ~/gen3_ws from
# gen3-kinesthetic-teaching-ros2) carry their own kortex_description, which shadows
# ~/workspace/ros2_kortex_ws's and breaks gen3_lite.launch.py and the teleop URDF.
for _ in 1 2 3 4; do conda deactivate 2>/dev/null || true; done
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
source /opt/ros/jazzy/setup.bash
source "${KORTEX_WS:-$HOME/workspace/ros2_kortex_ws}/install/setup.bash"
# Fast DDS (the Jazzy default) has hung controller_manager services on this setup.
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
conda activate "${LEROBOT_ENV:-lerobot-ros-06}"
