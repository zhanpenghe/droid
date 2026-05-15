"""
Calibrate a third-person camera without an Oculus controller.
Mount the Charuco board on the gripper, then run this script.
The robot will automatically move through the calibration trajectory.
"""

import argparse
import os
import time
import numpy as np
import cv2
from droid.robot_env import RobotEnv
from droid.trajectory_utils.misc import calibrate_camera


class AutoController:
    """Simple controller that auto-starts calibration without Oculus."""

    def __init__(self, auto_start_delay=3.0):
        self._start_time = time.time()
        self._auto_start_delay = auto_start_delay
        self._started = False

    def reset_state(self):
        self._start_time = time.time()
        self._started = False

    def get_info(self):
        elapsed = time.time() - self._start_time
        if not self._started and elapsed > self._auto_start_delay:
            self._started = True
            print("Auto-starting calibration...")
            return {"movement_enabled": True, "success": True, "failure": False}
        # movement_enabled=False keeps the robot in joint-position hold mode while waiting
        return {"movement_enabled": False, "success": False, "failure": False}

    def forward(self, obs):
        # Return zero cartesian velocity + open gripper
        return np.zeros(7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", type=str, required=True,
                        help="Serial number of the camera to calibrate (e.g. 38178251)")
    parser.add_argument("--delay", type=float, default=5.0,
                        help="Seconds to wait before auto-starting (default: 5)")
    args = parser.parse_args()

    print(f"Calibrating camera: {args.camera_id}")
    print(f"Make sure the Charuco board is mounted on the gripper.")
    print(f"Starting in {args.delay} seconds...")

    # Per-camera config: reset joints and trajectory params
    camera_configs = {
        "38178251": dict(
            reset_joints=np.array([np.pi/4, -1/5*np.pi, 0, -2/3*np.pi, 0, 7/12*np.pi, np.pi/3]),
            pos_scale=0.07,
            angle_scale=0.15,
        ),
        "33409691": dict(
            reset_joints=np.array([-np.pi/4, -1/5*np.pi, 0, -7*np.pi/12, 0, 7/12*np.pi, -np.pi/2]),
            pos_scale=0.07,
            angle_scale=0.15,
        ),
    }

    if args.camera_id not in camera_configs:
        print(f"No config for camera {args.camera_id}. Add it to camera_configs.")
        exit(1)

    cfg = camera_configs[args.camera_id]

    env = RobotEnv(do_reset=False, launch=False)
    controller = AutoController(auto_start_delay=args.delay)

    env.reset_joints = cfg["reset_joints"]
    env._robot.update_joints(env.reset_joints, velocity=False, blocking=True)

    # Capture preview images before calibration to verify board visibility
    save_dir = os.path.expanduser("~/Desktop/calib_preview")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving preview images to {save_dir} ...")
    for i in range(5):
        cam_obs, _ = env.read_cameras()
        for full_cam_id, img in cam_obs["image"].items():
            if args.camera_id in full_cam_id:
                filepath = os.path.join(save_dir, f"preview_{i}_{full_cam_id}.png")
                cv2.imwrite(filepath, img)
        time.sleep(0.5)
    print("Preview images saved. Check them before proceeding.")

    success = calibrate_camera(env, args.camera_id, controller, reset_robot=False,
                               wait_for_controller=True,
                               pos_scale=cfg["pos_scale"], angle_scale=cfg["angle_scale"])

    if success:
        print(f"Calibration successful for camera {args.camera_id}!")
    else:
        print("Calibration failed or was aborted.")
