# Camera Calibration Procedure

## Prerequisites
- Charuco board mounted on the gripper
- Both ZED cameras connected via USB 3.0
- Laptop connected to robot network (172.16.0.1)

## Step 1: Hardware Setup
1. Power on the Franka robot
2. Connect all cables (robot ethernet, ZED cameras via USB 3.0)
3. Open Franka Desk in browser: `http://172.16.0.2`
4. Log in and **unlock the robot** (click the lock icon)
5. Enable **FCI mode** (press the physical button on the control box until the light turns blue)

## Step 2: NUC — Start Servers (in order)

SSH into the NUC (`172.16.0.3`, user: `viscam`), then run each in a separate terminal:

```bash
# Terminal 1 — robot controller (wait for "Connected." before proceeding)
bash ~/Research/droid/scripts/server/launch_robot.sh

# Terminal 2 — zerorpc server
bash ~/Research/droid/scripts/server/launch_server.sh

# Terminal 3 — gripper server
bash ~/Research/droid/scripts/server/launch_gripper.sh
```

> **Order matters:** start `launch_robot.sh` first and wait for `Connected.` before starting the others.

## Step 3: Laptop — Connect to Robot

From the laptop, connect the zerorpc server to polymetis:

```bash
/home/zhanpeng/.local/bin/micromamba run -p Y/envs/robot python -c "
from droid.misc.server_interface import ServerInterface
s = ServerInterface(ip_address='172.16.0.3', launch=False)
s.launch_robot()
print('Done.')
"
```

## Step 4: Attach Charuco Board

Mount the Charuco board firmly on the gripper before running calibration.

## Step 5: Calibrate Each Camera

Run calibration for each camera. The robot will move to the start position, pause, then automatically run the calibration trajectory.

```bash
# Left camera
/home/zhanpeng/.local/bin/micromamba run -p Y/envs/robot python scripts/tests/calibrate_no_controller.py --camera_id 38178251

# Right camera
/home/zhanpeng/.local/bin/micromamba run -p Y/envs/robot python scripts/tests/calibrate_no_controller.py --camera_id 33409691
```

Results are saved to `droid/calibration/calibration_info.json`.

## Step 6: Remove Charuco Board

Move the robot to a convenient pose for board removal:

```bash
/home/zhanpeng/.local/bin/micromamba run -p Y/envs/robot python scripts/tests/move_to_board_removal.py
```

## Step 7: Verify Calibration with a Depth Capture

After calibrating, capture a test frame to confirm the extrinsics look correct before running the full pipeline.

```bash
micromamba run -p Y/envs/robot python scripts/tests/capture_depth.py
```

This resets the robot to home pose, captures RGB + ZED depth from both cameras, and saves everything (images, depth `.npy`, `intrinsics.json`, `extrinsics.json`) to `~/Desktop/depth_capture/<timestamp>/`.

---

## Fast-FoundationStereo Depth Pipeline

Fast-FoundationStereo replaces the ZED's built-in depth with a stereo neural network that produces denser, more accurate point clouds. The model weights (`20-30-48/model_best_bp2_serialize.pth`) live at `~/Desktop/3d_policy/20-30-48/`.

### Offline — process a saved capture

Run on any timestamped capture directory that contains `rgb_<id>_left/right.png` for both cameras:

```bash
micromamba run -p Y/envs/robot python droid/pc_utils/merge_fast_fs_depth_to_pc.py \
    --capture_dir ~/Desktop/depth_capture/<timestamp> \
    --baseline 0.12 \
    --min_depth 0.1 \
    --max_depth 2.0 \
    --valid_iters 8
```

Output: `pointcloud_fast_fs_merged.ply` inside the capture directory (~1.5 M points, ~700 ms on first timed frame after warmup).

| Flag | Default | Notes |
|---|---|---|
| `--capture_dir` | script dir | directory with `rgb_*.png`, `intrinsics.json`, `extrinsics.json` |
| `--baseline` | `0.12` | stereo baseline in metres (ZED 2/2i = 0.12, ZED Mini = 0.063) |
| `--min_depth` | `0.1` | clip near depth (m) |
| `--max_depth` | `2.0` | clip far depth (m) |
| `--valid_iters` | `8` | refinement iterations — reduce to 4 for ~25 % speedup |

### Live — stream point clouds from cameras

Resets the robot to home, then runs Fast-FoundationStereo continuously and prints per-frame timing:

```bash
micromamba run -p Y/envs/robot python scripts/tests/live_fast_fs_pc.py \
    --n_frames 0 \
    --save_every 0 \
    --valid_iters 8
```

Press **Ctrl-C** to stop — the last point cloud is saved automatically to `~/Desktop/depth_capture/live_<timestamp>/`.

| Flag | Default | Notes |
|---|---|---|
| `--n_frames` | `0` | frames to run; `0` = run until Ctrl-C |
| `--save_every` | `0` | save a PLY every N frames; `0` = only save on exit |
| `--valid_iters` | `8` | same as offline |

### Using the processor in your own code

```python
from droid.pc_utils.fast_fs_processor import FastFSStereoProcessor
from droid.misc.parameters import varied_camera_1_id, varied_camera_2_id

processor = FastFSStereoProcessor()          # loads model once
processor.warmup()                           # compiles CUDA kernels

# inside your observation loop:
cam_obs, _ = env.read_cameras()
points, colors, timing = processor.process(
    cam_obs, intrinsics, extrinsics,
    camera_ids=[varied_camera_1_id, varied_camera_2_id],
)
# points: (N, 3) float32 world-frame XYZ
# colors: (N, 3) uint8 RGB
```

`intrinsics` comes from `cam.get_intrinsics()` and `extrinsics` from `load_calibration_info()` — both are already available inside `RobotEnv`.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `'FrankaRobot' has no attribute '_robot'` | Run Step 3 (`launch_robot()` call) again |
| `'GripperInterface' has no attribute 'metadata'` | Gripper server not running — start `launch_gripper.sh` on NUC |
| `Automatic error recovery failed` | Robot is locked — go back to Step 1 and unlock via Franka Desk |
| `CAMERA NOT DETECTED` | ZED cameras on wrong USB port — plug into USB 3.0 directly (not hub) |
| Robot sags/drops during calibration | Normal if controller not active — script now holds position via `wait_for_controller=True` |
