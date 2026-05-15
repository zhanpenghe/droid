"""
FastFSStereoProcessor: live Fast-FoundationStereo depth estimation from ZED stereo pairs.

Designed to accept cam_obs directly from droid's read_cameras() and return a
merged world-frame point cloud without any disk I/O.
"""

import os
import sys
import time

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FAST_FS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "Fast-FoundationStereo"))
sys.path.insert(0, FAST_FS_DIR)

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE

DEFAULT_WEIGHTS_DIR = os.path.abspath(os.path.join(FAST_FS_DIR, "..", "20-30-48"))


class FastFSStereoProcessor:
    """
    Wraps Fast-FoundationStereo for live stereo depth estimation.

    Load once at init, call process() on every new camera observation.
    """

    def __init__(
        self,
        weights_dir=DEFAULT_WEIGHTS_DIR,
        valid_iters=8,
        baseline=0.12,
        min_depth=0.1,
        max_depth=2.0,
    ):
        self.valid_iters = valid_iters
        self.baseline = baseline
        self.min_depth = min_depth
        self.max_depth = max_depth

        model_path = os.path.join(weights_dir, "model_best_bp2_serialize.pth")
        print(f"Loading Fast-FoundationStereo from {model_path} ...")
        self.model = torch.load(model_path, map_location="cpu", weights_only=False).cuda().eval()
        print("Model loaded.")

    def warmup(self, img_shape=(720, 1280)):
        """Trigger CUDA kernel compilation with a dummy forward pass."""
        print("Warming up (first run compiles CUDA kernels) ...")
        dummy = np.zeros((*img_shape, 3), dtype=np.uint8)
        self._run_inference(dummy, dummy)
        print("Warmup done.")

    # ── internal helpers ────────────────────────────────────────────────────

    def _run_inference(self, img_left_rgb, img_right_rgb):
        t0 = torch.as_tensor(img_left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        t1 = torch.as_tensor(img_right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            disp = self.model.forward(t0, t1, iters=self.valid_iters, test_mode=True,
                                      optimize_build_volume="pytorch1")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        disp = padder.unpad(disp.float()).cpu().numpy().reshape(img_left_rgb.shape[:2]).clip(0, None)
        del t0, t1
        torch.cuda.empty_cache()
        return disp, elapsed

    @staticmethod
    def _to_rgb(img):
        """Accept BGRA (ZED default) or BGR and return RGB uint8."""
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _disp_to_depth(disp, fx, baseline):
        H, W = disp.shape
        xx = np.tile(np.arange(W), (H, 1))
        disp = disp.copy()
        disp[xx - disp < 0] = np.inf
        return fx * baseline / disp

    @staticmethod
    def _filter_depth(depth, median_ksize=5, max_diff=0.05, erode_iters=2):
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

    @staticmethod
    def _depth_to_world_points(depth, rgb, K, extrinsic, min_depth, max_depth):
        H, W = depth.shape
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
        d = depth[valid]
        p_cam = np.stack([(uu[valid] - K[0, 2]) / K[0, 0] * d,
                          (vv[valid] - K[1, 2]) / K[1, 1] * d,
                          d], axis=1)
        R_mat = Rotation.from_euler("xyz", extrinsic[3:6]).as_matrix()
        t = np.array(extrinsic[:3])
        p_world = (R_mat @ p_cam.T).T + t
        colors = rgb[valid].astype(np.uint8)
        return p_world.astype(np.float32), colors

    # ── public API ──────────────────────────────────────────────────────────

    def process(self, cam_obs, intrinsics, extrinsics, camera_ids):
        """
        Run Fast-FoundationStereo on a live camera observation.

        Args:
            cam_obs:     output of read_cameras()[0]; expects
                         cam_obs["image"]["<cam_id>_left"] and ["<cam_id>_right"]
            intrinsics:  dict keyed by "<cam_id>_left" / "_right", each entry has
                         "cameraMatrix" (3x3 list) — from cam.get_intrinsics()
            extrinsics:  dict keyed by "<cam_id>_left", value is
                         [tx, ty, tz, rx, ry, rz] — from load_calibration_info()
            camera_ids:  list of camera serial-number strings,
                         e.g. ["38178251", "33409691"]

        Returns:
            points  np.ndarray (N, 3) float32 — merged world-frame XYZ
            colors  np.ndarray (N, 3) uint8  — RGB colors
            timing  dict per camera_id with inference/filter/unproject ms
        """
        all_pts, all_cols = [], []
        timing = {}

        for cam_id in camera_ids:
            img_left = self._to_rgb(cam_obs["image"][f"{cam_id}_left"])
            img_right = self._to_rgb(cam_obs["image"][f"{cam_id}_right"])

            K = np.array(intrinsics[f"{cam_id}_left"]["cameraMatrix"])
            ext = extrinsics[f"{cam_id}_left"]

            disp, t_inf = self._run_inference(img_left, img_right)
            depth_raw = self._disp_to_depth(disp, K[0, 0], self.baseline)

            t0 = time.perf_counter()
            depth = self._filter_depth(depth_raw)
            t_filter = time.perf_counter() - t0

            t0 = time.perf_counter()
            pts, cols = self._depth_to_world_points(
                depth, img_left, K, ext, self.min_depth, self.max_depth
            )
            t_unproject = time.perf_counter() - t0

            n_removed = int(
                (np.isfinite(depth_raw) & (depth_raw > self.min_depth) &
                 (depth_raw < self.max_depth)).sum() - len(pts)
            )
            timing[cam_id] = {
                "inference_ms": t_inf * 1000,
                "filter_ms": t_filter * 1000,
                "unproject_ms": t_unproject * 1000,
                "n_points": len(pts),
                "n_removed": n_removed,
            }
            all_pts.append(pts)
            all_cols.append(cols)

        points = np.concatenate(all_pts)
        colors = np.concatenate(all_cols)
        return points, colors, timing
