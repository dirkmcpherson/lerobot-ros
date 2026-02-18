
import time
import logging
from dataclasses import dataclass, field

from lerobot_robot_ros.config import ROS2Config, ROS2InterfaceConfig, ActionType, GripperActionType
from lerobot_robot_ros.robot import ROS2Robot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KinovaGen3Config(ROS2Config):
    action_type: ActionType = ActionType.JOINT_TRAJECTORY
    
    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            # Kinova Gen3 standard joint names
            arm_joint_names=[
                "joint_1", "joint_2", "joint_3", "joint_4", 
                "joint_5", "joint_6", "joint_7"
            ],
            # Robotiq 2F-85 standard joint name
            # Temporarily disabled because it's missing from /joint_states in the sim
            gripper_joint_name=None, 
            
            # Topics based on ros2_kortex
            namespace="",  # Assuming root namespace, adjust if needed
            
            # Joint limits for normalization (extracted from gen3_macro.xacro)
            # Continuous joints (1, 3, 5, 7) set to +/- 2pi
            min_joint_positions=[-6.2832, -2.24, -6.2832, -2.57, -6.2832, -2.09, -6.2832],
            max_joint_positions=[6.2832, 2.24, 6.2832, 2.57, 6.2832, 2.09, 6.2832],
            
            # Gripper configuration
            gripper_open_position=0.0,
            gripper_close_position=0.8, # Approximate closed value for 2F-85
            gripper_action_type=GripperActionType.ACTION,
        )
    )

def main():
    logger.info("Initializing Kinova Gen3 Robot...")
    
    # Initialize configuration
    config = KinovaGen3Config()
    
    # Create robot instance
    robot = ROS2Robot(config)
    
    try:
        logger.info("Connecting to robot...")
        robot.connect()
        logger.info("Successfully connected!")
        
        # Give it a moment to receive first messages
        time.sleep(2.0)
        
        # Wait for joint state
        logger.info("Waiting for first joint state...")
        max_retries = 10
        for i in range(max_retries):
            if robot.ros2_interface.joint_state is not None:
                break
            logger.info(f"Waiting for joint state... ({i+1}/{max_retries})")
            time.sleep(1.0)
            
        if robot.ros2_interface.joint_state is None:
             raise TimeoutError("Timed out waiting for joint state from ROS 2.")
             
        # Read observation
        logger.info("Reading initial observation...")
        obs = robot.get_observation()
        logger.info(f"Observation keys: {obs.keys()}")
        
        # Print initial joint positions
        joint_positions = {k: v for k, v in obs.items() if "joint" in k and ".pos" in k}
        logger.info(f"Initial Joint Positions: {joint_positions}")
        
        # Simple test action: maintain current position
        # This verifies that we can construct and send a valid action packet
        logger.info("Sending hold position action...")
        action = {}
        
        # Populate action with current positions (hold)
        for joint in config.ros2_interface.arm_joint_names:
            key = f"{joint}.pos"
            if key in obs:
               action[key] = obs[key]
            else:
               logger.warning(f"Could not find {key} in observation, setting to 0.0")
               action[key] = 0.0
               
        # Set gripper to current position
        gripper_pos_key = f"{config.ros2_interface.gripper_joint_name}.pos"
        if gripper_pos_key in obs:
            # Map back to 0-1 range roughly if needed, or just pass through for now if logic supports it
            # The send_action logic expects "gripper.pos" in range [0, 1] usually
            # But the robot class maps it. Let's send a safe 'open' command for test.
            action["gripper.pos"] = 0.0 
        else:
             action["gripper.pos"] = 0.0

        # Send action
        robot.send_action(action)
        logger.info("Action sent successfully.")
        
        time.sleep(1.0)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        logger.info("Disconnecting...")
        robot.disconnect()
        logger.info("Done.")

if __name__ == "__main__":
    main()
