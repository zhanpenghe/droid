"""
Run FoundationStereo on both ZED stereo pairs, merge into a world-frame
point cloud, and report processing time (excluding model load and file I/O).

Usage:
  python merge_fs_depth_to_pc.py [--capture_dir PATH] [--baseline 0.12]
                                  [--min_depth 0.1] [--max_depth 2.0]
                                  [--z_far 3.0] [--valid_iters 32]
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import open3d as o3d
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(REPO_DIR, "FoundationStereo"))

from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder

CAMERAS = ["33409691", "38178251"]


def load_model(ckpt_path):
    cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt_path), "cfg.yaml"))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    model = FoundationStereo(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.cuda().eval()


def run_inference(model, img_left, img_right, valid_iters=32):
    t0 = torch.as_tensor(img_left).cuda().float()[None].permute(0, 3, 1, 2)
    t1 = torch.as_tensor(img_right).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(t0.shape, divis_by=32, force_square=False)
    t0, t1 = padder.pad(t0, t1)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        disp = model.forward(t0, t1, iters=valid_iters, test_mode=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    disp = padder.unpad(disp.float()).cpu().numpy().reshape(img_left.shape[:2])
    del t0, t1
    torch.cuda.empty_cache()
    return disp, elapsed


def disp_to_depth(disp, fx, baseline):
    H, W = disp.shape
    xx = np.tile(np.arange(W), (H, 1))
    disp[xx - disp < 0] = np.inf
    return fx * baseline / disp


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
    colors = rgb[valid][:, ::-1].astype(np.uint8)
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
    parser.add_argument("--z_far", type=float, default=3.0)
    parser.add_argument("--valid_iters", type=int, default=32)
    args = parser.parse_args()

    capture_dir = os.path.abspath(args.capture_dir)
    ckpt_path = os.path.join(REPO_DIR, "model_best_bp2-001.pth")

    with open(os.path.join(capture_dir, "intrinsics.json")) as f:
        intrinsics = json.load(f)
    with open(os.path.join(capture_dir, "extrinsics.json")) as f:
        extrinsics = json.load(f)

    # ── Load model (not timed) ──────────────────────────────────────
    print("Loading model...")
    model = load_model(ckpt_path)

    # ── Load images (not timed) ─────────────────────────────────────
    imgs = {}
    for cam_id in CAMERAS:
        for side in ("left", "right"):
            key = f"{cam_id}_{side}"
            path = os.path.join(capture_dir, f"rgb_{key}.png")
            imgs[key] = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    # ── Processing (timed) ──────────────────────────────────────────
    all_pts, all_cols = [], []
    t_inference_total = 0.0

    for cam_id in CAMERAS:
        view_id = f"{cam_id}_left"
        K = np.array(intrinsics[view_id]["cameraMatrix"])
        ext = extrinsics[view_id]

        disp, t_inf = run_inference(model, imgs[f"{cam_id}_left"], imgs[f"{cam_id}_right"],
                                    args.valid_iters)
        depth = disp_to_depth(disp, K[0, 0], args.baseline)
        t_inference_total += t_inf

        t_start = time.perf_counter()
        pts, cols = depth_to_world_points(depth, imgs[f"{cam_id}_left"], K, ext,
                                          args.min_depth, args.max_depth)
        t_unproject = time.perf_counter() - t_start

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)

        t_start = time.perf_counter()
        pcd_clean, _ = pcd.remove_radius_outlier(nb_points=30, radius=0.05)
        t_outlier = time.perf_counter() - t_start

        n_removed = len(pts) - len(pcd_clean.points)
        print(f"  [{cam_id}] inference: {t_inf*1000:.0f} ms  "
              f"unproject: {t_unproject*1000:.0f} ms  "
              f"outlier: {t_outlier*1000:.0f} ms  "
              f"points: {len(pcd_clean.points):,} ({n_removed:,} removed)")

        all_pts.append(np.asarray(pcd_clean.points).astype(np.float32))
        all_cols.append((np.asarray(pcd_clean.colors) * 255).astype(np.uint8))
        t_inference_total += t_outlier

    merged_pts = np.concatenate(all_pts)
    merged_cols = np.concatenate(all_cols)

    print(f"\n  Total processing time: {t_inference_total*1000:.0f} ms")

    # ── Save (not timed) ────────────────────────────────────────────
    clean_pts = np.asarray(pcd_clean.points).astype(np.float32)
    clean_cols = (np.asarray(pcd_clean.colors) * 255).astype(np.uint8)
    out_path = os.path.join(capture_dir, "pointcloud_fs_merged.ply")
    save_ply(out_path, clean_pts, clean_cols)
    print(f"  Saved {len(clean_pts):,} points → {out_path}")
