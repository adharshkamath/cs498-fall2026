"""Task 1: rasterize one LiDAR sweep into a four-channel BEV tensor.

Run directly with: python task1_bev.py
"""

import numpy as np

from hw3_utils import (
    DATA_ROOT, GRID_H, GRID_W, OUTPUT_ROOT, RESOLUTION, Z_MAX, Z_MIN,
    load_debug_frames, metric_to_grid, save_bev_boxes, save_bev_channels, write_json,
)


def rasterize_points(points: np.ndarray) -> np.ndarray:
    """Return `[4,400,352]` float32 BEV values in `[0,1]`.

    Input `points` has shape `[N,4]` with columns `(x,y,z,intensity)` in the
    LiDAR frame. The four output channels are max height, mean height, mean
    intensity, and log-density. Empty cells must be zero.

    Use `metric_to_grid`, a flat index `row * GRID_W + column`,
    `np.bincount` for counts/sums, and `np.maximum.at` for maximum height.
    Do not loop over BEV cells.
    """
    # ------------------------ Task 1 Modify your code below ---------------------
    roi_x, roi_y, roi_z = (0, 70.4), (-40, 40), (-3, 1)
    # Filter points within the region of interest
    mask = (
        (points[:, 0] >= roi_x[0]) & (points[:, 0] <= roi_x[1]) &
        (points[:, 1] >= roi_y[0]) & (points[:, 1] <= roi_y[1]) &
        (points[:, 2] >= roi_z[0]) & (points[:, 2] <= roi_z[1])
    )
    filtered_points = points[mask]

    grid_indices = metric_to_grid(x=filtered_points[:, 0], y=filtered_points[:, 1])
    rows, cols, _ = grid_indices
    grid_indices = rows * GRID_W + cols
    heights = filtered_points[:, 2].copy()
    heights_normalized = np.clip((heights - Z_MIN) / (Z_MAX - Z_MIN), 0, 1)


    max_height = np.zeros(GRID_H * GRID_W, dtype=np.float32)
    np.maximum.at(max_height, grid_indices, heights_normalized)

    sum_height = np.zeros(GRID_H * GRID_W, dtype=np.float32)
    np.add.at(sum_height, grid_indices, heights_normalized)
    mean_height = np.zeros(GRID_H * GRID_W, dtype=np.float32)
    counts = np.bincount(grid_indices, minlength=GRID_H * GRID_W)
    mean_height[counts > 0] = sum_height[counts > 0] / counts[counts > 0]
    mean_height = np.clip(mean_height, 0, 1)

    intensities = filtered_points[:, 3].copy()
    intensities_normalized = np.clip(intensities, 0, 1)
    sum_intensity = np.zeros(GRID_H * GRID_W, dtype=np.float32)
    np.add.at(sum_intensity, grid_indices, intensities_normalized)
    mean_intensity = np.zeros(GRID_H * GRID_W, dtype=np.float32)
    mean_intensity[counts > 0] = sum_intensity[counts > 0] / counts[counts > 0]
    mean_intensity = np.clip(mean_intensity, 0, 1)

    log_density = np.zeros(GRID_H * GRID_W, dtype=np.float32)
    log_density[counts > 0] = np.log(1 + counts[counts > 0]) / np.log(65)
    log_density = np.clip(log_density, 0, 1)

    bev = np.stack([max_height, mean_height, mean_intensity, log_density], axis=0)
    bev = bev.reshape(4, GRID_H, GRID_W)
    return bev 
    # ---------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK ----------------


def main():
    points, boxes, frame_ids = load_debug_frames(DATA_ROOT / "debug_frames.npz")
    output = OUTPUT_ROOT / "task1"
    bev = rasterize_points(points[0])

    save_bev_channels(bev, output / "bev_channels.png")
    save_bev_boxes(bev, boxes[0], output / "bev_boxes.png", title=f"KITTI frame {frame_ids[0]}")

    occupied = np.max(bev, axis=0) > 0
    metrics = {
        "shape": list(bev.shape),
        "resolution_m": RESOLUTION,
        "occupied_cells": int(occupied.sum()),
        "occupancy_fraction": float(occupied.mean()),
        "channel_min": bev.reshape(4, -1).min(axis=1).tolist(),
        "channel_max": bev.reshape(4, -1).max(axis=1).tolist(),
    }
    write_json(output / "metrics.json", metrics)
    print("Checkpoint 1: bev_channels.png, bev_boxes.png, and metrics.json")
    return metrics


if __name__ == "__main__":
    print("Task 1 complete:", main())
