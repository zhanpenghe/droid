"""
Capture depth + RGB images from both ZED cameras and save with extrinsics.
The robot resets to its default pose before capture.

Output folder: ~/Desktop/depth_capture/<timestamp>/
  rgb_<cam_id>_left/right.png
  depth_<cam_id>_left/right.npy   (float32, meters, NaN = invalid)
  intrinsics.json
  extrinsics.json
"""

import os
import json
import time
import datetime
import numpy as np
import cv2

from droid.robot_env import RobotEnv
from droid.calibration.calibration_utils import load_calibration_info
from droid.misc.parameters import varied_camera_1_id, varied_camera_2_id

CAMERA_IDS = [varied_camera_1_id, varied_camera_2_id]

if __name__ == "__main__":
    # ── Reset robot and open cameras with depth enabled ──────────────────────
    print("Resetting robot...")
    env = RobotEnv(do_reset=True, launch=False,
                   camera_kwargs={"varied_camera": {"image": True, "depth": True}})
    print("Robot at home pose.")

    # Re-issue joint position command so the controller actively holds during capture
    env._robot.update_joints(env.reset_joints, velocity=False, blocking=True)

    # ── Output folder ────────────────────────────────────────────────────────
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.expanduser(f"~/Desktop/depth_capture/{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving to {out_dir}")

    # ── Save extrinsics from calibration ─────────────────────────────────────
    calib = load_calibration_info()
    extrinsics_to_save = {k: v for k, v in calib.items()
                          if any(cid in k for cid in CAMERA_IDS)}
    with open(os.path.join(out_dir, "extrinsics.json"), "w") as f:
        json.dump(extrinsics_to_save, f, indent=2)

    # ── Read cameras (capture immediately while joint controller is active) ──
    env.camera_reader.set_trajectory_mode()
    cam_obs, _ = env.read_cameras()

    # ── Save intrinsics ───────────────────────────────────────────────────────
    intrinsics = {}
    for cam in env.camera_reader.camera_dict.values():
        for full_id, info in cam.get_intrinsics().items():
            intrinsics[full_id] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                                   for k, v in info.items()}
    with open(os.path.join(out_dir, "intrinsics.json"), "w") as f:
        json.dump(intrinsics, f, indent=2)

    # ── Save RGB and depth images ─────────────────────────────────────────────
    for full_id, img in cam_obs.get("image", {}).items():
        if not any(cid in full_id for cid in CAMERA_IDS):
            continue
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(os.path.join(out_dir, f"rgb_{full_id}.png"), img)

    for full_id, depth in cam_obs.get("depth", {}).items():
        if not any(cid in full_id for cid in CAMERA_IDS):
            continue
        depth = depth.astype(np.float32)
        np.save(os.path.join(out_dir, f"depth_{full_id}.npy"), depth)
        valid = np.isfinite(depth)
        if valid.any():
            print(f"  {full_id}: valid={valid.mean()*100:.1f}%  "
                  f"range=[{np.nanmin(depth):.2f}, {np.nanmax(depth):.2f}] m")
        else:
            print(f"  {full_id}: no valid depth pixels")

    print(f"\nDone. Files in {out_dir}:")
    for fn in sorted(os.listdir(out_dir)):
        print(f"  {fn}")
