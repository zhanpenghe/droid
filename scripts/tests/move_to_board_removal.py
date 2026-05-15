"""Move robot to a convenient pose for removing the Charuco board."""
import numpy as np
from droid.robot_env import RobotEnv

env = RobotEnv(do_reset=False, launch=False)
joints = np.array([np.pi/4, -1/5*np.pi, 0, -2/3*np.pi, 0, 7/12*np.pi, np.pi/3])
print("Moving to board removal pose...")
env._robot.update_joints(joints, velocity=False, blocking=True)
print("Done. Remove the board now.")
