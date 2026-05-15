"""
Merge depth images from two ZED cameras into a single world-frame pointcloud.

Usage:
  python merge_depth_to_pc.py --capture_dir ~/Desktop/depth_capture/20260514_153442
  python merge_depth_to_pc.py  # uses latest capture folder automatically

Output: <capture_dir>/pointcloud.ply  (binary PLY, opens in MeshLab/CloudCompare)
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def depth_to_points(depth, rgb, intrinsics, extrinsic_pose,
                    min_depth=0.1, max_depth=5.0):
    """Unproject a depth image to world-frame 3D points with color.

    depth:          HxW float32, meters (NaN/inf = invalid)
    rgb:            HxWx3 or HxWx4 BGR/BGRA image
    intrinsics:     dict with fx, fy, cx, cy
    extrinsic_pose: [x, y, z, rx, ry, rz] camera-to-base transform (Euler XYZ)
    Returns: (Nx3 float32 world points, Nx3 uint8 RGB colors)
    """
    H, W = depth.shape
    K = np.array(intrinsics["cameraMatrix"])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))

    valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    d = depth[valid]
    u_v = uu[valid]
    v_v = vv[valid]

    # Unproject to camera frame
    X = (u_v - cx) / fx * d
    Y = (v_v - cy) / fy * d
    Z = d
    p_cam = np.stack([X, Y, Z], axis=1)  # Nx3

    # Transform to world/base frame: p_world = R @ p_cam + t
    t = np.array(extrinsic_pose[:3], dtype=np.float64)
    Rmat = Rotation.from_euler("xyz", extrinsic_pose[3:6]).as_matrix()
    p_world = (Rmat @ p_cam.T).T + t

    # Extract colors (drop alpha, convert BGR→RGB)
    bgr = rgb[:, :, :3] if rgb.shape[2] == 4 else rgb
    colors = bgr[valid][:, ::-1].astype(np.uint8)

    return p_world.astype(np.float32), colors


def save_ply(filename, points, colors):
    """Save colored pointcloud as binary little-endian PLY."""
    N = len(points)
    dtype = np.dtype([
        ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ("red", np.uint8), ("green", np.uint8), ("blue", np.uint8),
    ])
    data = np.empty(N, dtype=dtype)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(filename, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    print(f"Saved {N:,} points → {filename}")


def latest_capture_dir():
    base = os.path.expanduser("~/Desktop/depth_capture")
    dirs = sorted(p for p in glob.glob(os.path.join(base, "*")) if os.path.isdir(p))
    if not dirs:
        raise FileNotFoundError(f"No capture folders found in {base}")
    return dirs[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture_dir", type=str, default=None,
                        help="Path to capture folder (default: latest)")
    parser.add_argument("--min_depth", type=float, default=0.1,
                        help="Minimum valid depth in meters (default: 0.1)")
    parser.add_argument("--max_depth", type=float, default=2.0,
                        help="Maximum valid depth in meters (default: 2.0)")
    args = parser.parse_args()

    capture_dir = os.path.expanduser(args.capture_dir) if args.capture_dir else latest_capture_dir()
    print(f"Using capture folder: {capture_dir}")

    with open(os.path.join(capture_dir, "intrinsics.json")) as f:
        intrinsics = json.load(f)

    with open(os.path.join(capture_dir, "extrinsics.json")) as f:
        extrinsics = json.load(f)

    all_points = []
    all_colors = []
    view_ids = []

    depth_files = sorted(glob.glob(os.path.join(capture_dir, "depth_*.npy")))
    if not depth_files:
        raise FileNotFoundError("No depth_*.npy files found in capture folder")

    for depth_path in depth_files:
        fname = os.path.basename(depth_path)           # depth_38178251_left.npy
        view_id = fname[len("depth_"):-len(".npy")]    # 38178251_left

        if view_id not in intrinsics:
            print(f"  Skipping {view_id}: no intrinsics")
            continue
        if view_id not in extrinsics:
            print(f"  Skipping {view_id}: not in calibration (no extrinsic)")
            continue

        rgb_path = os.path.join(capture_dir, f"rgb_{view_id}.png")
        if not os.path.exists(rgb_path):
            print(f"  Skipping {view_id}: no RGB image")
            continue

        depth = np.load(depth_path)
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)

        intr = intrinsics[view_id]
        ext = extrinsics[view_id]["pose"] if isinstance(extrinsics[view_id], dict) else extrinsics[view_id]

        pts, cols = depth_to_points(depth, rgb, intr, ext,
                                    min_depth=args.min_depth,
                                    max_depth=args.max_depth)
        print(f"  {view_id}: {len(pts):,} valid points")
        all_points.append(pts)
        all_colors.append(cols)
        view_ids.append(view_id)

    if not all_points:
        print("No points to save.")
    else:
        import open3d as o3d  # noqa: E402

        def make_pcd(points, colors):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
            return pcd

        # Save and show each individual view
        for view_id, pts, cols in zip(view_ids, all_points, all_colors):
            out_path = os.path.join(capture_dir, f"pointcloud_{view_id}.ply")
            save_ply(out_path, pts, cols)
            print(f"Showing {view_id} — close window to continue.")
            o3d.visualization.draw_geometries([make_pcd(pts, cols)],
                                              window_name=f"View: {view_id}")

        # Save and show merged
        merged_points = np.concatenate(all_points, axis=0)
        merged_colors = np.concatenate(all_colors, axis=0)
        out_path = os.path.join(capture_dir, "pointcloud_merged.ply")
        save_ply(out_path, merged_points, merged_colors)
        print("Showing merged — close window to exit.")
        o3d.visualization.draw_geometries([make_pcd(merged_points, merged_colors)],
                                          window_name="Merged Pointcloud")
