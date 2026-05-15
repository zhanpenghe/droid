"""
Run Fast-FoundationStereo on both ZED stereo pairs, merge into a world-frame
point cloud, and report processing time (excluding model load and file I/O).

Usage:
  python merge_fast_fs_depth_to_pc.py [--capture_dir PATH] [--baseline 0.12]
                                       [--min_depth 0.1] [--max_depth 2.0]
                                       [--valid_iters 8]
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# pc_utils -> droid (pkg) -> droid (repo) -> 3d_policy
FAST_FS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "Fast-FoundationStereo"))
sys.path.insert(0, FAST_FS_DIR)

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE

CAMERAS = ["33409691", "38178251"]
WEIGHTS_DIR = os.path.abspath(os.path.join(FAST_FS_DIR, "..", "20-30-48"))


def load_model(weights_dir):
    model_path = os.path.join(weights_dir, "model_best_bp2_serialize.pth")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    return model.cuda().eval()


def run_inference(model, img_left, img_right, valid_iters=8):
    t0 = torch.as_tensor(img_left).cuda().float()[None].permute(0, 3, 1, 2)
    t1 = torch.as_tensor(img_right).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(t0.shape, divis_by=32, force_square=False)
    t0, t1 = padder.pad(t0, t1)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE):
        disp = model.forward(t0, t1, iters=valid_iters, test_mode=True,
                             optimize_build_volume="pytorch1")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    disp = padder.unpad(disp.float()).cpu().numpy().reshape(img_left.shape[:2]).clip(0, None)
    del t0, t1
    torch.cuda.empty_cache()
    return disp, elapsed


def disp_to_depth(disp, fx, baseline):
    H, W = disp.shape
    xx = np.tile(np.arange(W), (H, 1))
    disp[xx - disp < 0] = np.inf
    return fx * baseline / disp


def filter_depth(depth, median_ksize=5, max_diff=0.05, erode_iters=2):
    """Remove isolated depth pixels using a median filter + mask erosion."""
    depth_f = depth.copy()
    valid = np.isfinite(depth_f)
    depth_f[~valid] = 0
    smoothed = cv2.medianBlur(depth_f.astype(np.float32), median_ksize)
    depth_f[valid & (np.abs(depth_f - smoothed) > max_diff)] = np.inf
    valid2 = np.isfinite(depth_f).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    valid2 = cv2.erode(valid2, kernel, iterations=erode_iters)
    depth_f[valid2 == 0] = np.inf
    return depth_f


def depth_to_world_points(depth, rgb, K, extrinsic, min_depth, max_depth):
    H, W = depth.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    d = depth[valid]
    p_cam = np.stack([(uu[valid] - K[0, 2]) / K[0, 0] * d,
                      (vv[valid] - K[1, 2]) / K[1, 1] * d,
                      d], axis=1)
    R = Rotation.from_euler("xyz", extrinsic[3:6]).as_matrix()
    t = np.array(extrinsic[:3])
    p_world = (R @ p_cam.T).T + t
    colors = rgb[valid].astype(np.uint8)
    return p_world.astype(np.float32), colors


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture_dir", default=SCRIPT_DIR)
    parser.add_argument("--baseline", type=float, default=0.12,
                        help="Stereo baseline in meters (ZED 2/2i: 0.12, ZED Mini: 0.063)")
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=2.0)
    parser.add_argument("--valid_iters", type=int, default=8)
    args = parser.parse_args()

    capture_dir = os.path.abspath(args.capture_dir)

    with open(os.path.join(capture_dir, "intrinsics.json")) as f:
        intrinsics = json.load(f)
    with open(os.path.join(capture_dir, "extrinsics.json")) as f:
        extrinsics = json.load(f)

    # ── Load model (not timed) ──────────────────────────────────────
    print("Loading Fast-FoundationStereo model...")
    model = load_model(WEIGHTS_DIR)

    # ── Load images (not timed) ─────────────────────────────────────
    imgs = {}
    for cam_id in CAMERAS:
        for side in ("left", "right"):
            key = f"{cam_id}_{side}"
            imgs[key] = cv2.cvtColor(
                cv2.imread(os.path.join(capture_dir, f"rgb_{key}.png")),
                cv2.COLOR_BGR2RGB)

    # ── Warmup (first run compiles kernels) ─────────────────────────
    print("Warming up (first run compiles CUDA kernels)...")
    _ = run_inference(model, imgs[f"{CAMERAS[0]}_left"], imgs[f"{CAMERAS[0]}_right"],
                      args.valid_iters)

    # ── Processing (timed) ──────────────────────────────────────────
    print("Running timed inference...")
    all_pts, all_cols = [], []
    t_total = 0.0

    for cam_id in CAMERAS:
        view_id = f"{cam_id}_left"
        K = np.array(intrinsics[view_id]["cameraMatrix"])
        ext = extrinsics[view_id]

        disp, t_inf = run_inference(model, imgs[f"{cam_id}_left"], imgs[f"{cam_id}_right"],
                                    args.valid_iters)
        depth_raw = disp_to_depth(disp, K[0, 0], args.baseline)

        t_start = time.perf_counter()
        depth = filter_depth(depth_raw)
        t_filter = time.perf_counter() - t_start

        t_start = time.perf_counter()
        pts, cols = depth_to_world_points(depth, imgs[f"{cam_id}_left"], K, ext,
                                          args.min_depth, args.max_depth)
        t_unproject = time.perf_counter() - t_start

        n_removed = (np.isfinite(depth_raw) & (depth_raw > args.min_depth) &
                     (depth_raw < args.max_depth)).sum() - len(pts)
        print(f"  [{cam_id}] inference: {t_inf*1000:.0f} ms  "
              f"filter: {t_filter*1000:.0f} ms  "
              f"unproject: {t_unproject*1000:.0f} ms  "
              f"points: {len(pts):,} ({n_removed:,} removed)")

        all_pts.append(pts)
        all_cols.append(cols)
        t_total += t_inf + t_filter + t_unproject

    merged_pts = np.concatenate(all_pts)
    merged_cols = np.concatenate(all_cols)

    print(f"\n  Total processing time: {t_total*1000:.0f} ms")

    # ── Save (not timed) ────────────────────────────────────────────
    out_path = os.path.join(capture_dir, "pointcloud_fast_fs_merged.ply")
    save_ply(out_path, merged_pts, merged_cols)
    print(f"  Saved {len(merged_pts):,} points → {out_path}")
