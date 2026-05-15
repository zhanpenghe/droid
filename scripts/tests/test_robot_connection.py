from droid.robot_env import RobotEnv
import numpy as np

print("Connecting to robot...")
env = RobotEnv(do_reset=False)
print("Connected!")

state, timestamps = env.get_state()
print("Robot state keys:", list(state.keys()))
print("Joint positions:", state.get("joint_positions"))
print("EE pose:", state.get("cartesian_position"))
print("Robot connection OK.")
