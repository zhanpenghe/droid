"""
Live Fast-FoundationStereo point cloud capture from ZED cameras.

Opens both ZED cameras, runs Fast-FoundationStereo on every frame, and prints
timing. Press Ctrl-C to stop; the last point cloud is saved to ~/Desktop/depth_capture/.

Usage:
  python live_fast_fs_pc.py [--n_frames N] [--save_every N] [--baseline 0.12]
                             [--min_depth 0.1] [--max_depth 2.0] [--valid_iters 8]
"""

import argparse
import datetime
import json
import os
import signal
import sys
import time

import numpy as np

from droid.robot_env import RobotEnv
from droid.calibration.calibration_utils import load_calibration_info
from droid.misc.parameters import varied_camera_1_id, varied_camera_2_id
from droid.pc_utils.fast_fs_processor import FastFSStereoProcessor

CAMERA_IDS = [varied_camera_1_id, varied_camera_2_id]


def save_ply(filename, points, colors):
    N = len(points)
    dtype = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32),
                      ("red", np.uint8), ("green", np.uint8), ("blue", np.uint8)])
    data = np.empty(N, dtype=dtype)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {N}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(filename, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_frames", type=int, default=0,
                        help="Number of frames to process (0 = run until Ctrl-C)")
    parser.add_argument("--save_every", type=int, default=0,
                        help="Save a PLY every N frames (0 = only save on exit)")
    parser.add_argument("--baseline", type=float, default=0.12)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=2.0)
    parser.add_argument("--valid_iters", type=int, default=8)
    args = parser.parse_args()

    # ── Output dir ──────────────────────────────────────────────────────────
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.expanduser(f"~/Desktop/depth_capture/live_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ── Robot + cameras ─────────────────────────────────────────────────────
    print("Resetting robot to home position ...")
    env = RobotEnv(do_reset=True, launch=False,
                   camera_kwargs={"varied_camera": {"image": True, "depth": False}})
    print("Robot at home pose.")

    # Re-issue joint command so the controller actively holds during collection
    env._robot.update_joints(env.reset_joints, velocity=False, blocking=True)

    camera_wrapper = env.camera_reader
    camera_wrapper.set_trajectory_mode()

    # ── Intrinsics ──────────────────────────────────────────────────────────
    intrinsics = {}
    for cam in camera_wrapper.camera_dict.values():
        for full_id, info in cam.get_intrinsics().items():
            intrinsics[full_id] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                                   for k, v in info.items()}
    with open(os.path.join(out_dir, "intrinsics.json"), "w") as f:
        json.dump(intrinsics, f, indent=2)

    # ── Extrinsics ──────────────────────────────────────────────────────────
    calib = load_calibration_info()
    extrinsics = {k: v for k, v in calib.items()
                  if any(cid in k for cid in CAMERA_IDS)}
    with open(os.path.join(out_dir, "extrinsics.json"), "w") as f:
        json.dump(extrinsics, f, indent=2)

    # ── Model ───────────────────────────────────────────────────────────────
    processor = FastFSStereoProcessor(
        baseline=args.baseline,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        valid_iters=args.valid_iters,
    )

    # Warmup using actual image shape from first frame
    cam_obs, _ = env.read_cameras()
    sample_img = cam_obs["image"][f"{CAMERA_IDS[0]}_left"]
    processor.warmup(img_shape=sample_img.shape[:2])

    # ── Loop ────────────────────────────────────────────────────────────────
    last_points, last_colors = None, None
    frame = 0

    def handle_exit(sig, frame_):
        print("\nInterrupted — saving last point cloud ...")
        if last_points is not None:
            out_path = os.path.join(out_dir, "pointcloud_last.ply")
            save_ply(out_path, last_points, last_colors)
            print(f"Saved {len(last_points):,} points → {out_path}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)

    print(f"\nRunning {'indefinitely' if args.n_frames == 0 else args.n_frames} frames. "
          "Press Ctrl-C to stop.\n")

    capture_times, process_times = [], []

    while args.n_frames == 0 or frame < args.n_frames:
        t_cap_start = time.perf_counter()
        cam_obs, _ = env.read_cameras()
        t_capture = (time.perf_counter() - t_cap_start) * 1000

        t_proc_start = time.perf_counter()
        points, colors, timing = processor.process(cam_obs, intrinsics, extrinsics, CAMERA_IDS)
        t_process = (time.perf_counter() - t_proc_start) * 1000

        capture_times.append(t_capture)
        process_times.append(t_process)

        parts = [f"[{cid}] {t['inference_ms']:.0f}+{t['filter_ms']:.0f}+{t['unproject_ms']:.0f} ms "
                 f"→ {t['n_points']:,} pts"
                 for cid, t in timing.items()]
        print(f"frame {frame:04d} | capture {t_capture:.0f} ms | process {t_process:.0f} ms | "
              f"total {t_capture + t_process:.0f} ms | " + " | ".join(parts))

        last_points, last_colors = points, colors

        if args.save_every > 0 and (frame + 1) % args.save_every == 0:
            out_path = os.path.join(out_dir, f"pointcloud_{frame:04d}.ply")
            save_ply(out_path, points, colors)
            print(f"  → saved {out_path}")

        frame += 1

    # ── Summary ─────────────────────────────────────────────────────────────
    if capture_times:
        def stats(vals):
            return f"avg {sum(vals)/len(vals):.0f} ms  min {min(vals):.0f} ms  max {max(vals):.0f} ms"
        totals = [c + p for c, p in zip(capture_times, process_times)]
        print(f"\n{'─'*60}")
        print(f"  frames     : {len(capture_times)}")
        print(f"  capture    : {stats(capture_times)}")
        print(f"  processing : {stats(process_times)}")
        print(f"  total      : {stats(totals)}")
        print(f"{'─'*60}")

    # ── Final save ──────────────────────────────────────────────────────────
    if last_points is not None:
        out_path = os.path.join(out_dir, "pointcloud_final.ply")
        save_ply(out_path, last_points, last_colors)
        print(f"Saved {len(last_points):,} points → {out_path}")


if __name__ == "__main__":
    main()
