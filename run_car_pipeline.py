#!/usr/bin/env python3
"""
End-to-end car reconstruction pipeline.

Stages:
  1. Per-frame Pi3X inference with car masks and multimodal conditioning
  2. Stitch frames via Umeyama full-pose alignment to trajectory
  3. Filter by reprojection consensus (ratio-based mask check)
  4. Crop to estimated bounding box with per-side padding

Usage:
  python run_car_pipeline.py
  python run_car_pipeline.py --skip_inference
  python run_car_pipeline.py --cameras at_cam_01 at_cam_02 at_cam_03 at_cam_07 at_cam_08 at_cam_09
  python run_car_pipeline.py --frame_start 40 --frame_end 80 --frame_skip 2
  python run_car_pipeline.py --reproj_ratio 0.7 --pad_front 0.05 --pad_rear 0.20
"""

import os
import sys
import math
import time
import argparse
import yaml
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from safetensors.torch import load_file
from plyfile import PlyData, PlyElement
import open3d as o3d

from pi3.models.pi3x import Pi3X
from pi3.utils.basic import write_ply
from pi3.utils.geometry import depth_edge

# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULT_SCAN_DIR = "/isilon/Automotive/RnD/elad.e/mpsfm_data/2024-09-27T16-31-13.000Z_d8350e36-4628-4c39-8164-05521c599810"
DEFAULT_TRAJECTORY = "/isilon/Automotive/RnD/elad.e/uv-3d/sessions/demo_room_test/run_2026-03-18T09-29-58-448462/02_trajectory/trajectory.npz"
DEFAULT_OUTPUT_DIR = "/isilon/Automotive/RnD/elad.e/pi3/output_pipeline"
DEFAULT_CAMERAS = ["at_cam_01", "at_cam_02", "at_cam_03", "at_cam_07", "at_cam_08", "at_cam_09"]

PIXEL_LIMIT = 255000
ALIGNMENT_ERR_THRESHOLD = 0.3


# ── calibration & trajectory ─────────────────────────────────────────────────

def parse_calibration(scan_dir, extrinsics_path=None):
    """Parse extrinsics.yaml → dict[cam_name] → {c2w, K, resolution, dist_coeffs}.

    Args:
        scan_dir: Directory containing camera image sub-folders. Used as fallback
            for extrinsics.yaml when extrinsics_path is None.
        extrinsics_path: Explicit path to extrinsics.yaml. Allows the calibration
            file to live in a different directory from the raw frames (e.g. the
            UV-3D calibration directory vs. the images directory).
    """
    if extrinsics_path is None:
        extrinsics_path = os.path.join(scan_dir, "extrinsics.yaml")
    with open(extrinsics_path) as f:
        ext_data = yaml.safe_load(f)
    cam_order_keys = sorted(
        [k for k in ext_data if k.startswith("cam")],
        key=lambda x: int(x.replace("cam", ""))
    )
    cam_info = {}
    abs_pose = np.eye(4)
    for key in cam_order_keys:
        entry = ext_data[key]
        name = entry["cam_name"]
        if "T_cn_cnm1" in entry:
            abs_pose = abs_pose @ np.linalg.inv(np.array(entry["T_cn_cnm1"]))
        intr = entry["intrinsics"]
        K = np.array([[intr[0], 0, intr[2]],
                      [0, intr[1], intr[3]],
                      [0, 0, 1]], dtype=np.float64)
        dist = np.array(entry.get("distortion_coeffs", [0, 0, 0, 0]), dtype=np.float64)
        cam_info[name] = {
            "c2w": abs_pose.copy(),
            "K": K,
            "resolution": entry["resolution"],
            "dist_coeffs": dist,
        }
    return cam_info


def load_trajectory(trajectory_path):
    traj = np.load(trajectory_path, allow_pickle=True)
    return traj["pose_cache_cameras"], traj["pose_cache_frames"], traj["pose_cache_values"]


def select_frames_by_displacement(trajectory_path, frame_start, frame_end, min_step_m=0.15):
    """
    Select frames so that each consecutive pair is separated by at least
    `min_step_m` of actual car travel (fused odometry).

    This handles variable car speed, stops mid-scan, and acceleration naturally:
    frames where the car barely moved are skipped; frames where it moved a lot
    are included at the right density.

    Uses `fused_distances_values` from the trajectory .npz.
    Falls back to camera-pose cumulative distance if fused distances are absent.

    Returns (selected_frames, step_distances) where step_distances[i] is the
    odometry gap between selected_frames[i] and selected_frames[i+1].
    """
    traj = np.load(trajectory_path, allow_pickle=True)

    if "fused_distances_values" in traj and "fused_distances_frames" in traj:
        fd_frames = traj["fused_distances_frames"].astype(int)
        fd_vals   = traj["fused_distances_values"].astype(float)
        mask = (fd_frames >= frame_start) & (fd_frames <= frame_end)
        fd_frames = fd_frames[mask]
        fd_vals   = fd_vals[mask]
    else:
        # Fallback: cumulative camera translation
        cameras = traj["pose_cache_cameras"]
        frames  = traj["pose_cache_frames"].astype(int)
        values  = traj["pose_cache_values"]
        cam0 = cameras[0]
        sel = (cameras == cam0) & (frames >= frame_start) & (frames <= frame_end)
        order = np.argsort(frames[sel])
        fd_frames = frames[sel][order]
        positions = values[sel][order][:, :3, 3]
        fd_vals = np.concatenate([[0.0],
                                  np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))])

    # Greedy walk: keep a frame once at least min_step_m has accumulated
    selected = [fd_frames[0]]
    last_dist = fd_vals[0]
    for i in range(1, len(fd_frames)):
        if fd_vals[i] - last_dist >= min_step_m:
            selected.append(fd_frames[i])
            last_dist = fd_vals[i]

    # Per-step distances for reporting
    frame_to_dist = dict(zip(fd_frames, fd_vals))
    step_dists = [frame_to_dist[selected[i+1]] - frame_to_dist[selected[i]]
                  for i in range(len(selected) - 1)]

    return selected, step_dists


# ── image / mask loading ─────────────────────────────────────────────────────

def load_images(cameras, frame_idx, scan_dir):
    frame_name = f"frame_{frame_idx:04d}.png"
    sources, labels = [], []
    for cam in cameras:
        path = os.path.join(scan_dir, cam, frame_name)
        if not os.path.exists(path):
            continue
        sources.append(Image.open(path).convert("RGB"))
        labels.append(cam)
    if not sources:
        return None, [], None
    W, H = sources[0].size
    scale = math.sqrt(PIXEL_LIMIT / (W * H))
    Wt, Ht = W * scale, H * scale
    k, m = round(Wt / 14), round(Ht / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > Wt / Ht:
            k -= 1
        else:
            m -= 1
    TW, TH = max(1, k) * 14, max(1, m) * 14
    to_tensor = transforms.ToTensor()
    tensors = [to_tensor(img.resize((TW, TH), Image.Resampling.LANCZOS)) for img in sources]
    return torch.stack(tensors), labels, (TW / W, TH / H, TW, TH)


def build_condition_tensors(labels, cam_info, traj_poses, scale_info, device):
    scale_x, scale_y, TW, TH = scale_info
    N = len(labels)
    poses = np.zeros((N, 4, 4), dtype=np.float64)
    Ks = np.zeros((N, 3, 3), dtype=np.float64)
    for i, cam in enumerate(labels):
        if cam in traj_poses:
            poses[i] = traj_poses[cam]
        elif cam in cam_info:
            poses[i] = cam_info[cam]["c2w"]
        if cam in cam_info:
            K = cam_info[cam]["K"].copy()
            K[0, 0] *= scale_x; K[0, 2] *= scale_x
            K[1, 1] *= scale_y; K[1, 2] *= scale_y
            Ks[i] = K
    return (torch.from_numpy(poses).float().unsqueeze(0).to(device),
            torch.from_numpy(Ks).float().unsqueeze(0).to(device))


def resolve_mask_path(masks_dir, cam, frame_name):
    """Return the mask file path for a given camera and frame.

    Tries two naming conventions:
      1. frame_NNNN_mask.png  — raw-scan layout (default)
      2. frame_NNNN.png       — UV-3D segmentation stage layout

    Returns None if neither exists.
    """
    name1 = frame_name.replace(".png", "_mask.png")
    p1 = os.path.join(masks_dir, cam, name1)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(masks_dir, cam, frame_name)
    if os.path.exists(p2):
        return p2
    return None


def load_car_masks(labels, frame_name, H, W, masks_dir):
    masks = []
    for cam in labels:
        mask_path = resolve_mask_path(masks_dir, cam, frame_name)
        if mask_path is not None:
            m = Image.open(mask_path).convert("L").resize((W, H), Image.Resampling.NEAREST)
            masks.append(np.array(m) > 128)
        else:
            masks.append(np.ones((H, W), dtype=bool))
    return np.stack(masks)


# ── alignment ────────────────────────────────────────────────────────────────

def umeyama_alignment(src_pts, tgt_pts):
    src = np.array(src_pts, dtype=np.float64)
    tgt = np.array(tgt_pts, dtype=np.float64)
    n = src.shape[0]
    sm, tm = src.mean(0), tgt.mean(0)
    sc, tc = src - sm, tgt - tm
    sv = np.sum(sc ** 2) / n
    H = (sc.T @ tc) / n
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    s = np.trace(np.diag(D) @ S) / sv
    t = tm - s * R @ sm
    return s, R, t


def transform_points(pts, s, R, t):
    return (s * (R @ pts.T).T) + t


def align_full_poses(pi3_poses, traj_poses, cameras):
    src, tgt = [], []
    off = 0.5
    for i, cam in enumerate(cameras):
        if cam not in traj_poses:
            continue
        p, tr = pi3_poses[i], traj_poses[cam]
        src.append(p[:3, 3]); tgt.append(tr[:3, 3])
        for ax in range(3):
            src.append(p[:3, 3] + off * p[:3, ax])
            tgt.append(tr[:3, 3] + off * tr[:3, ax])
    return umeyama_alignment(np.array(src), np.array(tgt))


# ── reprojection filter ──────────────────────────────────────────────────────

def project_points(xyz_world, w2c, K, dist, W, H):
    """Project world points into a camera. Returns (u, v, valid_mask)."""
    N = xyz_world.shape[0]
    pts_cam = (w2c[:3, :3] @ xyz_world.T).T + w2c[:3, 3]
    z = pts_cam[:, 2]
    valid = z > 0.1

    x_norm = pts_cam[:, 0] / np.maximum(z, 1e-8)
    y_norm = pts_cam[:, 1] / np.maximum(z, 1e-8)
    r2 = x_norm ** 2 + y_norm ** 2
    k1, k2, p1, p2 = dist[0], dist[1], dist[2], dist[3]
    radial = 1 + k1 * r2 + k2 * r2 ** 2
    x_dist = x_norm * radial + 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm ** 2)
    y_dist = y_norm * radial + p1 * (r2 + 2 * y_norm ** 2) + 2 * p2 * x_norm * y_norm

    u = K[0, 0] * x_dist + K[0, 2]
    v = K[1, 1] * y_dist + K[1, 2]

    valid &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, valid


def reprojection_filter(xyz, rgb, cam_info, traj_cams, traj_frames, traj_poses,
                        cameras, frames, masks_dir, ratio_threshold, batch_size=500000):
    """Keep points where car_hits / visible_views >= ratio_threshold."""
    N = xyz.shape[0]
    car_count = np.zeros(N, dtype=np.int32)
    visible_count = np.zeros(N, dtype=np.int32)

    views = []
    for cam in cameras:
        for frame_idx in frames:
            sel = (traj_cams == cam) & (traj_frames == frame_idx)
            if not sel.any():
                continue
            c2w = traj_poses[sel][0]
            w2c = np.linalg.inv(c2w)
            info = cam_info[cam]
            mask_path = resolve_mask_path(masks_dir, cam, f"frame_{frame_idx:04d}.png")
            if mask_path is None:
                continue
            views.append((cam, frame_idx, w2c, info["K"], info["dist_coeffs"], mask_path))

    print(f"  Reprojecting into {len(views)} camera views...")

    for vi, (cam, fidx, w2c, K, dist, mask_path) in enumerate(views):
        mask = np.array(Image.open(mask_path).convert("L")) > 128
        H_img, W_img = mask.shape

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            u, v, valid = project_points(xyz[start:end], w2c, K, dist, W_img, H_img)
            visible_count[start:end] += valid.astype(np.int32)
            ui = u[valid].astype(np.int32)
            vi_arr = v[valid].astype(np.int32)
            on_car = mask[vi_arr, ui]
            idx = np.where(valid)[0] + start
            car_count[idx[on_car]] += 1

        if (vi + 1) % 25 == 0:
            print(f"    {vi + 1}/{len(views)} views processed")

    print(f"    {len(views)}/{len(views)} views processed")

    denom = np.maximum(visible_count, 1)
    ratio = car_count.astype(np.float32) / denom
    keep = ratio >= ratio_threshold

    n_kept = keep.sum()
    print(f"  Reprojection filter (ratio >= {ratio_threshold:.0%}): "
          f"kept {n_kept:,} / {N:,} ({100 * n_kept / N:.1f}%)")
    return xyz[keep], rgb[keep]


# ── bounding box estimation ──────────────────────────────────────────────────

def estimate_bbox(cam_info, traj_cams, traj_frames, traj_poses, xyz_points, masks_dir):
    """Estimate car bounding box from masks, calibration, and trajectory."""

    # WIDTH from cam_04 (right side) and cam_06 (left side) overhead cameras.
    # cam_04: rightmost masked pixel across all frames -> ray to floor -> Z_min (right edge)
    # cam_06: leftmost masked pixel across all frames -> ray to floor -> Z_max (left edge)
    H5 = cam_info["at_cam_05"]["resolution"][1]
    center_row = H5 // 2
    floor_y_est = np.percentile(xyz_points[:, 1], 99)

    z_right, z_left = None, None
    for cam_name, direction in [("at_cam_04", "right"), ("at_cam_06", "left")]:
        K = cam_info[cam_name]["K"]
        fx, cx = K[0, 0], K[0, 2]

        extreme_col, extreme_frame = None, None
        for f in range(0, 120):
            mp = resolve_mask_path(masks_dir, cam_name, f"frame_{f:04d}.png")
            if mp is None:
                continue
            m = np.array(Image.open(mp).convert("L")) > 128
            cols = np.where(m.any(axis=0))[0]
            if len(cols) == 0:
                continue
            if direction == "right":
                c = cols.max()
                if extreme_col is None or c > extreme_col:
                    extreme_col, extreme_frame = c, f
            else:
                c = cols.min()
                if extreme_col is None or c < extreme_col:
                    extreme_col, extreme_frame = c, f

        if extreme_col is not None:
            sel = (traj_cams == cam_name) & (traj_frames == extreme_frame)
            if sel.any():
                c2w = traj_poses[sel][0]
                cam_pos = c2w[:3, 3]
                x_norm = (extreme_col - cx) / fx
                ray_cam = np.array([x_norm, 0.0, 1.0])
                ray_world = c2w[:3, :3] @ ray_cam
                t = (floor_y_est - cam_pos[1]) / ray_world[1]
                hit_z = cam_pos[2] + t * ray_world[2]
                if direction == "right":
                    z_right = hit_z
                else:
                    z_left = hit_z

    if z_right is not None and z_left is not None:
        car_width = z_left - z_right
        print(f"  Width from cam_04/cam_06 rays: Z_right={z_right:.3f}, Z_left={z_left:.3f}")
    else:
        car_width = 1.83

    # FRONT / REAR (X axis) from cam_05 (overhead) mask ray-casting.
    #
    # cam_05 looks straight down at the car.  Each frame the mask outline
    # of the car is visible.  We iterate every frame that has mask pixels,
    # find the extreme rows (min row = leading edge, max row = trailing edge
    # in image space), ray-cast each extreme pixel through the floor plane
    # and record the resulting world X coordinate.  The overall min/max world
    # X across all frames are the actual front and rear bumper positions.
    #
    # This mirrors the cam_04 / cam_06 width estimation exactly.
    cam5_K = cam_info["at_cam_05"]["K"]
    fx5, fy5 = cam5_K[0, 0], cam5_K[1, 1]
    cx5, cy5 = cam5_K[0, 2], cam5_K[1, 2]

    x_front, x_rear = None, None   # world X of front / rear bumper

    for f in range(0, 120):
        mp = resolve_mask_path(masks_dir, "at_cam_05", f"frame_{f:04d}.png")
        if mp is None:
            continue
        m = np.array(Image.open(mp).convert("L")) > 128
        rows = np.where(m.any(axis=1))[0]
        if len(rows) == 0:
            continue

        sel = (traj_cams == "at_cam_05") & (traj_frames == f)
        if not sel.any():
            continue
        c2w = traj_poses[sel][0]
        cam_pos = c2w[:3, 3]

        for row, label in [(rows.min(), "leading"), (rows.max(), "trailing")]:
            cols_at_row = np.where(m[row, :])[0]
            col = int(np.median(cols_at_row))
            x_n = (col  - cx5) / fx5
            y_n = (row  - cy5) / fy5
            ray_cam   = np.array([x_n, y_n, 1.0])
            ray_world = c2w[:3, :3] @ ray_cam
            # intersect with floor plane (world Y = floor_y_est)
            if abs(ray_world[1]) < 1e-6:
                continue
            t = (floor_y_est - cam_pos[1]) / ray_world[1]
            if t < 0:
                continue
            wx = cam_pos[0] + t * ray_world[0]
            if label == "leading":
                if x_front is None or wx > x_front:
                    x_front = wx
            else:
                if x_rear is None or wx < x_rear:
                    x_rear = wx

    if x_front is not None and x_rear is not None:
        car_length = x_front - x_rear
        print(f"  Front/rear from cam_05 rays: X_front={x_front:.3f}, X_rear={x_rear:.3f}, "
              f"length={car_length:.2f} m")
    else:
        # Fallback: camera displacement (original method)
        x_front, x_rear = None, None
        first_frame, last_frame = None, None
        for f in range(0, 120):
            mp = resolve_mask_path(masks_dir, "at_cam_05", f"frame_{f:04d}.png")
            if mp is None:
                continue
            m = np.array(Image.open(mp).convert("L")) > 128
            if m[center_row, :].any():
                if first_frame is None:
                    first_frame = f
                last_frame = f
        if first_frame is not None and last_frame is not None:
            sel_f = (traj_cams == "at_cam_05") & (traj_frames == first_frame)
            sel_l = (traj_cams == "at_cam_05") & (traj_frames == last_frame)
            if sel_f.any() and sel_l.any():
                pos_first = traj_poses[sel_f][0][:3, 3]
                pos_last  = traj_poses[sel_l][0][:3, 3]
                car_length = float(np.linalg.norm(pos_last - pos_first))
            else:
                car_length = 3.40
        else:
            car_length = 3.40
        print(f"  Front/rear fallback (camera displacement): length={car_length:.2f} m")

    # BOTTOM (floor Y) from cam_01 side camera.
    # cam_01 Y-axis maps directly to world +Y (floor direction).
    # Scan all cam_01 masks and find the pixel with the MAX row (bottommost
    # in image = most positive world Y = bottom of car).
    # Ray-cast through that pixel to the car's Z center plane to get world Y.
    y_bottom = None
    cam1_K = cam_info["at_cam_01"]["K"]
    fy1, cy1 = cam1_K[1, 1], cam1_K[1, 2]
    fx1, cx1 = cam1_K[0, 0], cam1_K[0, 2]
    z_center = np.median(xyz_points[:, 2])

    cam1_H = cam_info["at_cam_01"]["resolution"][1]
    max_row, max_row_col, max_row_frame = None, None, None
    for f in range(0, 120):
        mp = resolve_mask_path(masks_dir, "at_cam_01", f"frame_{f:04d}.png")
        if mp is None:
            continue
        m = np.array(Image.open(mp).convert("L")) > 128
        rows = np.where(m.any(axis=1))[0]
        if len(rows) == 0:
            continue
        r = rows.max()
        if r >= cam1_H - 200:
            continue
        if max_row is None or r > max_row:
            cols_at_r = np.where(m[r, :])[0]
            max_row = r
            max_row_col = int(np.median(cols_at_r))
            max_row_frame = f

    if max_row is not None:
        sel_c1 = (traj_cams == "at_cam_01") & (traj_frames == max_row_frame)
        if sel_c1.any():
            c2w_c1 = traj_poses[sel_c1][0]
            cam1_pos = c2w_c1[:3, 3]
            x_norm = (max_row_col - cx1) / fx1
            y_norm = (max_row - cy1) / fy1
            ray_cam = np.array([x_norm, y_norm, 1.0])
            ray_world = c2w_c1[:3, :3] @ ray_cam
            t = (z_center - cam1_pos[2]) / ray_world[2]
            y_bottom = cam1_pos[1] + t * ray_world[1]
            print(f"  Bottom from cam_01 ray: max_row={max_row}, frame={max_row_frame}, Y_bottom={y_bottom:.3f}")

    # HEIGHT: derive from point cloud roof and measured bottom
    roof_y_est = np.percentile(xyz_points[:, 1], 1)
    if y_bottom is not None:
        car_height = y_bottom - roof_y_est
    else:
        car_height = 1.44

    print(f"  Estimated car dimensions:")
    print(f"    Length: {car_length:.2f} m  Width: {car_width:.2f} m  Height: {car_height:.2f} m")

    result = {"length": car_length, "width": car_width, "height": car_height}
    if z_right is not None:
        result["z_right"] = z_right
    if z_left is not None:
        result["z_left"] = z_left
    if y_bottom is not None:
        result["y_bottom"] = y_bottom
    if x_front is not None:
        result["x_front"] = x_front
    if x_rear is not None:
        result["x_rear"] = x_rear
    return result


def apply_bbox(xyz, rgb, dims, xyz_points, pad):
    """Apply bounding box filter with asymmetric per-side padding.

    Uses the estimated car dimensions (from masks, trajectory, calibration)
    centered on the point cloud median, then adds padding on each side.

    The car coordinate system (from trajectory):
      X = travel direction, front of car = less negative X (X_max)
      Y = vertical, roof = more negative Y (Y_min)
      Z = lateral, left side = more positive Z (Z_max)
    """
    p5 = np.percentile(xyz_points, 5, axis=0)
    p95 = np.percentile(xyz_points, 95, axis=0)
    center = (p5 + p95) / 2

    if "x_front" in dims and "x_rear" in dims:
        # Use ray-cast absolute positions (accurate)
        car_front_x = dims["x_front"]
        car_rear_x  = dims["x_rear"]
    else:
        # Fall back to center ± half length
        car_rear_x  = center[0] - dims["length"] / 2
        car_front_x = center[0] + dims["length"] / 2
    if "y_bottom" in dims:
        floor_y = dims["y_bottom"]
        roof_y = floor_y - dims["height"]
    else:
        roof_y = center[1] - dims["height"] / 2
        floor_y = center[1] + dims["height"] / 2

    if "z_right" in dims and "z_left" in dims:
        z_lo = dims["z_right"]
        z_hi = dims["z_left"]
    else:
        z_lo = center[2] - dims["width"] / 2
        z_hi = center[2] + dims["width"] / 2

    x_min = car_rear_x - pad["rear"]
    x_max = car_front_x + pad["front"]
    y_min = roof_y - pad["top"]
    y_max = floor_y + pad["bottom"]
    z_min_f = z_lo - pad["right"]
    z_max_f = z_hi + pad["left"]

    print(f"  Car center (midpoint of p5/p95): ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    print(f"  Car bbox from estimated dims:")
    print(f"    X: [{car_rear_x:.3f}, {car_front_x:.3f}] (length={dims['length']:.2f}m)")
    print(f"    Y: [{roof_y:.3f}, {floor_y:.3f}] (height={dims['height']:.2f}m)")
    print(f"    Z: [{z_lo:.3f}, {z_hi:.3f}] (width={dims['width']:.2f}m)")
    print(f"  Bounding Box (with padding):")
    print(f"    X (length): [{x_min:.3f}, {x_max:.3f}] = {x_max - x_min:.3f} m  "
          f"(rear pad={pad['rear']:.2f}, front pad={pad['front']:.2f})")
    print(f"    Y (height): [{y_min:.3f}, {y_max:.3f}] = {y_max - y_min:.3f} m  "
          f"(top pad={pad['top']:.2f}, bottom pad={pad['bottom']:.2f})")
    print(f"    Z (width):  [{z_min_f:.3f}, {z_max_f:.3f}] = {z_max_f - z_min_f:.3f} m  "
          f"(right pad={pad['right']:.2f}, left pad={pad['left']:.2f})")

    inside = ((xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) &
              (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max) &
              (xyz[:, 2] >= z_min_f) & (xyz[:, 2] <= z_max_f))

    n_in = inside.sum()
    print(f"  Kept {n_in:,} / {len(xyz):,} ({100 * n_in / len(xyz):.1f}%)")

    bbox_limits = {
        "x_min": float(x_min), "x_max": float(x_max),
        "y_min": float(y_min), "y_max": float(y_max),
        "z_min": float(z_min_f), "z_max": float(z_max_f),
    }
    return xyz[inside], rgb[inside], bbox_limits


# ── PLY I/O ──────────────────────────────────────────────────────────────────

def write_merged_ply(xyz, rgb, path):
    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.uint8)
    normals = np.zeros_like(xyz)
    dt = [("x", "f4"), ("y", "f4"), ("z", "f4"),
          ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
          ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    el = np.empty(xyz.shape[0], dtype=dt)
    el[:] = list(map(tuple, np.concatenate((xyz, normals, rgb), axis=1)))
    PlyData([PlyElement.describe(el, "vertex")]).write(path)


def write_bbox_viz_ply(xyz, rgb, bbox, path):
    """Write PLY with bounding box edges drawn in red."""
    corners = np.array([
        [bbox["x_min"], bbox["y_min"], bbox["z_min"]],
        [bbox["x_max"], bbox["y_min"], bbox["z_min"]],
        [bbox["x_max"], bbox["y_max"], bbox["z_min"]],
        [bbox["x_min"], bbox["y_max"], bbox["z_min"]],
        [bbox["x_min"], bbox["y_min"], bbox["z_max"]],
        [bbox["x_max"], bbox["y_min"], bbox["z_max"]],
        [bbox["x_max"], bbox["y_max"], bbox["z_max"]],
        [bbox["x_min"], bbox["y_max"], bbox["z_max"]],
    ], dtype=np.float32)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    edge_pts = []
    for a, b in edges:
        for t in np.linspace(0, 1, 200):
            edge_pts.append(corners[a] * (1 - t) + corners[b] * t)
    edge_pts = np.array(edge_pts, dtype=np.float32)
    edge_rgb = np.tile(np.array([255, 0, 0], dtype=np.uint8), (len(edge_pts), 1))

    all_xyz = np.concatenate([xyz.astype(np.float32), edge_pts], axis=0)
    all_rgb = np.concatenate([rgb.astype(np.uint8), edge_rgb], axis=0)
    write_merged_ply(all_xyz, all_rgb, path)


# ── Stage 5: Poisson mesh reconstruction ─────────────────────────────────────

def point_cloud_to_mesh(
    input_path: str,
    output_path: str,
    depth: int = 9,
    density_quantile: float = 0.01,
    normal_radius: float = 0.1,
    normal_nn: int = 30,
    orient_k: int = 30,
    reestimate_normals: bool = True,
    voxel_size: float | None = None,
) -> str:
    """Convert a point cloud PLY to a triangle mesh via Poisson reconstruction.

    Returns the output path on success.
    """
    from pathlib import Path as _Path
    in_p  = _Path(input_path)
    out_p = _Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    _banner("STAGE 5: Poisson mesh reconstruction")

    # 5a. Load
    print(f"  [5a] Loading point cloud: {in_p.name} …")
    t0  = time.time()
    pcd = o3d.io.read_point_cloud(str(in_p))
    n_pts = len(pcd.points)
    print(f"       {n_pts:,} points  ({time.time()-t0:.1f}s)")
    if n_pts == 0:
        raise ValueError("Point cloud is empty — cannot build mesh.")

    # 5b. Normals
    need = reestimate_normals or not pcd.has_normals()
    if need:
        vox = voxel_size
        if vox is None and n_pts > 2_000_000:
            vox = 0.02          # 2 cm — fast enough for large clouds
            print(f"  [5b] Large cloud ({n_pts:,} pts) — downsampling to {vox} m voxels …")
        if vox is not None:
            pcd = pcd.voxel_down_sample(vox)
            print(f"       Downsampled to {len(pcd.points):,} pts")
        print(f"  [5b] Estimating & orienting normals …")
        t0 = time.time()
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=normal_nn
            )
        )
        pcd.orient_normals_consistent_tangent_plane(orient_k)
        print(f"       Done  ({time.time()-t0:.1f}s)")
    else:
        print("  [5b] Normals already present — skipping estimation.")

    # 5c. Poisson reconstruction
    print(f"  [5c] Poisson reconstruction (depth={depth}) …")
    t0 = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.1, linear_fit=False
    )
    print(f"       Initial: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.triangles):,} triangles  ({time.time()-t0:.1f}s)")
    if len(mesh.vertices) == 0:
        raise RuntimeError(
            "Poisson returned an empty mesh. Try --mesh-reestimate-normals "
            "or a coarser --mesh-voxel-size."
        )

    # 5d. Density filtering
    print(f"  [5d] Removing low-density vertices (quantile={density_quantile}) …")
    threshold = np.quantile(densities, density_quantile)
    mesh.remove_vertices_by_mask(np.asarray(densities) < threshold)
    print(f"       After filter: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.triangles):,} triangles")

    # 5e. Save
    print(f"  [5e] Saving → {out_p.name} …")
    t0 = time.time()
    o3d.io.write_triangle_mesh(str(out_p), mesh)
    mb = out_p.stat().st_size / 1e6
    print(f"       Saved  ({mb:.1f} MB, {time.time()-t0:.1f}s)")

    return str(out_p)


def _banner(title: str) -> None:
    """Print a stage separator banner."""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


# ── main pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end car reconstruction: Pi3X → stitch → reproj filter → bbox crop")

    parser.add_argument("--scan_dir", type=str, default=DEFAULT_SCAN_DIR)
    parser.add_argument(
        "--extrinsics", type=str, default=None,
        help="Explicit path to extrinsics.yaml. "
             "Default: {scan_dir}/extrinsics.yaml. "
             "Use when calibration lives separately from images (e.g. UV-3D sessions)."
    )
    parser.add_argument(
        "--masks_dir", type=str, default=None,
        help="Directory containing per-camera car silhouette masks. "
             "Accepts frame_NNNN_mask.png (raw scan) or frame_NNNN.png (UV-3D segmentation). "
             "Default: {scan_dir}/masks"
    )
    parser.add_argument("--trajectory", type=str, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    parser.add_argument("--frame_start", type=int, default=40)
    parser.add_argument("--frame_end", type=int, default=80)
    parser.add_argument("--frame_skip", type=int, default=None,
                        help="Fixed frame stride (overrides displacement-based selection). "
                             "Use only when you want a uniform skip regardless of car speed.")
    parser.add_argument("--min_step_m", type=float, default=0.15,
                        help="Minimum car displacement (metres) between consecutive processed frames "
                             "(default: 0.15 m = 15 cm). Ignored when --frame_skip is given.")
    parser.add_argument("--alignment_threshold", type=float, default=ALIGNMENT_ERR_THRESHOLD)

    parser.add_argument("--reproj_ratio", type=float, default=0.7,
                        help="Reprojection filter: min car-mask hit ratio to keep a point")
    parser.add_argument("--skip_reproj", action="store_true", default=True,
                        help="Skip the reprojection filter stage (disabled by default)")
    parser.add_argument("--enable_reproj", action="store_true",
                        help="Enable the reprojection filter stage")

    parser.add_argument("--pad_front", type=float, default=0.15, help="Bbox padding: front (meters)")
    parser.add_argument("--pad_rear", type=float, default=0.10, help="Bbox padding: rear (meters)")
    parser.add_argument("--pad_top", type=float, default=0.10, help="Bbox padding: roof (meters)")
    parser.add_argument("--pad_bottom", type=float, default=0.0, help="Bbox padding: bottom (meters)")
    parser.add_argument("--pad_left", type=float, default=0.05, help="Bbox padding: left side (meters)")
    parser.add_argument("--pad_right", type=float, default=0.05, help="Bbox padding: right side (meters)")

    parser.add_argument("--skip_inference", action="store_true",
                        help="Reuse existing per-frame PLYs, skip Pi3X inference")

    # Stage 5 – mesh generation
    parser.add_argument("--skip_mesh", action="store_true",
                        help="Skip Stage 5: Poisson mesh reconstruction")
    parser.add_argument("--mesh_depth", type=int, default=9,
                        help="Poisson octree depth (default 9)")
    parser.add_argument("--mesh_density_quantile", type=float, default=0.01,
                        help="Remove mesh vertices below this density quantile (default 0.01)")
    parser.add_argument("--mesh_voxel_size", type=float, default=None,
                        help="Voxel size (m) for downsampling before normal estimation. "
                             "Auto-set to 0.02 for clouds >2M pts.")

    args = parser.parse_args()

    # Resolve masks_dir: explicit arg → {scan_dir}/masks (default)
    if args.masks_dir is None:
        args.masks_dir = os.path.join(args.scan_dir, "masks")

    t_start = time.time()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    if args.frame_skip is not None:
        # Manual override: uniform stride, ignore odometry
        frames = list(range(args.frame_start, args.frame_end + 1, args.frame_skip))
        frame_note = f"fixed skip={args.frame_skip} (manual override)"
        step_summary = None
    else:
        # Displacement-based selection: next frame only after ≥ min_step_m of travel
        frames, step_dists = select_frames_by_displacement(
            args.trajectory, args.frame_start, args.frame_end, args.min_step_m
        )
        mean_cm = (sum(step_dists) / len(step_dists) * 100) if step_dists else 0
        frame_note = (f"displacement-based (min {args.min_step_m*100:.0f} cm/step), "
                      f"avg {mean_cm:.1f} cm/step")
        step_summary = (f"    min={min(step_dists)*100:.1f} cm  "
                        f"max={max(step_dists)*100:.1f} cm  "
                        f"mean={mean_cm:.1f} cm") if step_dists else None

    if args.enable_reproj:
        args.skip_reproj = False

    print("=" * 70)
    print("  Car Reconstruction Pipeline")
    print("=" * 70)
    print(f"  Cameras : {args.cameras}")
    print(f"  Frames  : {args.frame_start}–{args.frame_end}  →  {len(frames)} selected  [{frame_note}]")
    if step_summary:
        print(f"  Step gaps:{step_summary}")
    print(f"  Output  : {args.output_dir}")
    print(f"  Stages  : 1-inference({'skip' if args.skip_inference else 'run'})  "
          f"3-reproj({'skip' if args.skip_reproj else f'ratio>={args.reproj_ratio:.0%}'})  "
          f"5-mesh({'skip' if args.skip_mesh else f'depth={args.mesh_depth}'})")
    print(f"  Bbox pad: front={args.pad_front:.2f} rear={args.pad_rear:.2f} "
          f"top={args.pad_top:.2f} bottom={args.pad_bottom:.2f} "
          f"left={args.pad_left:.2f} right={args.pad_right:.2f}")
    print("=" * 70)

    # ── Setup ────────────────────────────────────────────────────────────────

    print("\nParsing calibration...")
    cam_info = parse_calibration(args.scan_dir, extrinsics_path=args.extrinsics)

    print("Loading trajectory...")
    traj_cams, traj_frames, traj_poses = load_trajectory(args.trajectory)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Stage 1: Per-frame Pi3X inference ────────────────────────────────────

    _banner(f"STAGE 1 / 5  ▸  Per-frame Pi3X inference  ({len(frames)} frames)")

    all_pi3_poses = {}

    if args.skip_inference:
        print("  [skip] Reusing existing per-frame PLYs")
        for frame_idx in frames:
            fk = f"frame_{frame_idx:04d}"
            ply_path = os.path.join(args.output_dir, f"{fk}_caronly.ply")
            if os.path.exists(ply_path):
                all_pi3_poses[fk] = None
        print(f"  Found {len(all_pi3_poses)} existing per-frame PLYs")
    else:
        print("  Loading Pi3X model …")
        t_model = time.time()
        model = Pi3X.from_pretrained("yyfz233/Pi3X", use_multimodal=True).eval()
        model = model.to(device)
        print(f"  Model loaded  ({time.time()-t_model:.1f}s)")

        n_frames = len(frames)
        for fi, frame_idx in enumerate(frames):
            fk = f"frame_{frame_idx:04d}"
            t_frame = time.time()
            print(f"  [{fi+1:>3}/{n_frames}] {fk} …", end="", flush=True)

            imgs, labels, scale_info = load_images(args.cameras, frame_idx, args.scan_dir)
            if imgs is None or len(labels) < 2:
                print(" SKIP (not enough images)")
                continue

            ftraj = {}
            for cam in labels:
                mask = (traj_cams == cam) & (traj_frames == frame_idx)
                if mask.any():
                    ftraj[cam] = traj_poses[mask][0]

            poses_t, Ks_t = build_condition_tensors(labels, cam_info, ftraj, scale_info, device)
            imgs_gpu = imgs.to(device)
            torch.cuda.empty_cache()

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    res = model(imgs_gpu[None], poses=poses_t, intrinsics=Ks_t)

            conf_mask = torch.sigmoid(res["conf"][..., 0]) > 0.1
            non_edge = ~depth_edge(res["local_points"][..., 2], rtol=0.03)
            conf_mask = torch.logical_and(conf_mask, non_edge)[0]

            TH, TW = imgs.shape[2], imgs.shape[3]
            fn = f"frame_{frame_idx:04d}.png"
            car_masks = torch.from_numpy(
                load_car_masks(labels, fn, TH, TW, args.masks_dir)
            ).to(conf_mask.device)
            combined = torch.logical_and(conf_mask, car_masks)

            ply_path = os.path.join(args.output_dir, f"{fk}_caronly.ply")
            pts = res["points"][0][combined].cpu()
            cols = imgs.permute(0, 2, 3, 1).to(combined.device)[combined].cpu()
            write_ply(pts, cols, ply_path)

            poses_np = res["camera_poses"][0].cpu().numpy()
            all_pi3_poses[fk] = (poses_np, labels)
            n_pts = combined.sum().item()
            print(f"  {n_pts:>7,} pts  ({time.time()-t_frame:.1f}s)")
            del res, imgs_gpu
            torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()
        print(f"\n  Stage 1 complete: {len(all_pi3_poses)} frames inferred")

    # ── Stage 2: Stitch with Umeyama alignment ──────────────────────────────

    _banner("STAGE 2 / 5  ▸  Stitch frames (Umeyama full-pose alignment)")

    all_xyz, all_rgb = [], []
    n_passed, n_total = 0, 0

    pose_model = None
    frame_plys = [f for f in frames
                  if os.path.exists(os.path.join(args.output_dir, f"frame_{f:04d}_caronly.ply"))]
    n_stitch = len(frame_plys)
    print(f"  Stitching {n_stitch} available frame PLYs …")

    for fi, frame_idx in enumerate(frames):
        fk = f"frame_{frame_idx:04d}"
        ply_path = os.path.join(args.output_dir, f"{fk}_caronly.ply")
        if not os.path.exists(ply_path):
            continue

        ply = PlyData.read(ply_path)
        v = ply["vertex"]
        xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
        rgb = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.uint8)

        if all_pi3_poses.get(fk) is not None:
            pi3_poses_np, labels = all_pi3_poses[fk]
        else:
            imgs, labels, scale_info = load_images(args.cameras, frame_idx, args.scan_dir)
            if imgs is None or len(labels) < 2:
                continue
            if pose_model is None:
                print("  Loading Pi3X model for pose estimation …")
                pose_model = Pi3X.from_pretrained("yyfz233/Pi3X", use_multimodal=True).eval()
                pose_model = pose_model.to(device)

            ftraj = {}
            for cam in labels:
                sel = (traj_cams == cam) & (traj_frames == frame_idx)
                if sel.any():
                    ftraj[cam] = traj_poses[sel][0]

            poses_t, Ks_t = build_condition_tensors(labels, cam_info, ftraj, scale_info, device)
            imgs_gpu = imgs.to(device)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    res = pose_model(imgs_gpu[None], poses=poses_t, intrinsics=Ks_t)
            pi3_poses_np = res["camera_poses"][0].cpu().numpy()
            del res, imgs_gpu
            torch.cuda.empty_cache()

        ftraj = {}
        for cam in labels:
            sel = (traj_cams == cam) & (traj_frames == frame_idx)
            if sel.any():
                ftraj[cam] = traj_poses[sel][0]
        if len(ftraj) < 2:
            continue

        s, R, t = align_full_poses(pi3_poses_np, ftraj, labels)

        pi3_pos = np.array([pi3_poses_np[labels.index(c), :3, 3] for c in labels if c in ftraj])
        tgt_pos = np.array([ftraj[c][:3, 3] for c in labels if c in ftraj])
        aligned = transform_points(pi3_pos, s, R, t)
        err = float(np.mean(np.linalg.norm(aligned - tgt_pos, axis=1)))

        n_total += 1
        passed = err < args.alignment_threshold
        if passed:
            xyz_w = transform_points(xyz, s, R, t)
            all_xyz.append(xyz_w)
            all_rgb.append(rgb)
            n_passed += 1

        status = "✓ OK  " if passed else "✗ SKIP"
        print(f"  [{n_total:>3}/{n_stitch}] {fk}: {len(xyz):>7,} pts | "
              f"s={s:.4f} | err={err:.4f} m | {status}")

    if pose_model is not None:
        del pose_model
        torch.cuda.empty_cache()

    if not all_xyz:
        print("ERROR: No frames passed alignment filter!")
        sys.exit(1)

    merged_xyz = np.concatenate(all_xyz, axis=0)
    merged_rgb = np.concatenate(all_rgb, axis=0)
    print(f"\n  Stage 2 complete: {len(merged_xyz):,} points  "
          f"({n_passed}/{n_total} frames passed, "
          f"{n_total - n_passed} skipped — err > {args.alignment_threshold} m)")

    stitched_path = os.path.join(args.output_dir, "stitched_raw.ply")
    write_merged_ply(merged_xyz, merged_rgb, stitched_path)
    print(f"  Saved: stitched_raw.ply  ({os.path.getsize(stitched_path) / 1024 ** 2:.1f} MB)")

    # ── Stage 3: Reprojection filter ─────────────────────────────────────────

    if not args.skip_reproj:
        _banner(f"STAGE 3 / 5  ▸  Reprojection filter  (ratio ≥ {args.reproj_ratio:.0%})")

        merged_xyz, merged_rgb = reprojection_filter(
            merged_xyz, merged_rgb, cam_info, traj_cams, traj_frames, traj_poses,
            args.cameras, frames, args.masks_dir, args.reproj_ratio)

        reproj_path = os.path.join(args.output_dir, f"stitched_reproj_{int(args.reproj_ratio * 100)}.ply")
        write_merged_ply(merged_xyz, merged_rgb, reproj_path)
        print(f"  Saved: stitched_reproj_{int(args.reproj_ratio * 100)}.ply  "
              f"({os.path.getsize(reproj_path) / 1024 ** 2:.1f} MB)")
    else:
        print(f"\n  [skip] Stage 3 — reprojection filter skipped")

    # ── Stage 4: Bounding box estimation and crop ────────────────────────────

    _banner("STAGE 4 / 5  ▸  Bounding box estimation and crop")

    dims = estimate_bbox(cam_info, traj_cams, traj_frames, traj_poses,
                         merged_xyz, args.masks_dir)

    pad = {
        "front": args.pad_front, "rear": args.pad_rear,
        "top": args.pad_top, "bottom": args.pad_bottom,
        "left": args.pad_left, "right": args.pad_right,
    }

    final_xyz, final_rgb, bbox = apply_bbox(merged_xyz, merged_rgb, dims, merged_xyz, pad)

    # ── Write Stage 4 outputs ─────────────────────────────────────────────────

    out_path = os.path.join(args.output_dir, "car_final.ply")
    write_merged_ply(final_xyz, final_rgb, out_path)
    print(f"\n  Saved: car_final.ply  "
          f"({os.path.getsize(out_path) / 1024 ** 2:.1f} MB, {len(final_xyz):,} pts)")

    viz_path = os.path.join(args.output_dir, "car_final_viz.ply")
    write_bbox_viz_ply(final_xyz, final_rgb, bbox, viz_path)
    print(f"  Saved: car_final_viz.ply  ({os.path.getsize(viz_path) / 1024 ** 2:.1f} MB)")

    # ── Stage 5: Poisson mesh reconstruction ─────────────────────────────────

    mesh_path = None
    if not args.skip_mesh:
        mesh_out = os.path.join(args.output_dir, "car_final_mesh.ply")
        mesh_path = point_cloud_to_mesh(
            input_path=out_path,
            output_path=mesh_out,
            depth=args.mesh_depth,
            density_quantile=args.mesh_density_quantile,
            voxel_size=args.mesh_voxel_size,
            reestimate_normals=True,
        )
    else:
        print(f"\n  [skip] Stage 5 — mesh reconstruction skipped")

    # ── Summary ───────────────────────────────────────────────────────────────

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  DONE  ({elapsed:.0f}s total)")
    print(f"{'─' * 70}")
    print(f"  Point cloud : car_final.ply  "
          f"({os.path.getsize(out_path) / 1024 ** 2:.1f} MB, {len(final_xyz):,} pts)")
    if mesh_path:
        print(f"  Mesh        : car_final_mesh.ply  "
              f"({os.path.getsize(mesh_path) / 1024 ** 2:.1f} MB)")
    print(f"  Dimensions  : L={dims['length']:.2f} m  "
          f"W={dims['width']:.2f} m  H={dims['height']:.2f} m")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
