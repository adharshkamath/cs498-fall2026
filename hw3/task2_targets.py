"""Task 2: encode metric 3D boxes as CenterPoint supervision targets.

Run directly with: python task2_targets.py
"""

import numpy as np

from hw3_utils import (
    DATA_ROOT, MAX_OBJECTS, NUM_CLASSES, OUT_H, OUT_W, OUTPUT_RESOLUTION,
    OUTPUT_ROOT, draw_gaussian, gaussian_radius, load_debug_frames,
    metric_to_output_grid, output_grid_to_metric, save_target_debug, write_json,
)
from task1_bev import rasterize_points


def encode_targets(boxes: np.ndarray) -> dict[str, np.ndarray]:
    """Encode `[M,7]` boxes into one dense heatmap and sparse regressions.

    Each box is `[x,y,z,length,width,height,yaw]`. Draw one Gaussian on the
    `[1,100,88]` center heatmap. At the integer center cell, store the flat
    index, sub-cell `(du,dv)`, center z, log-dimensions, and `(sin(yaw),
    cos(yaw))`. The mask marks valid rows among `MAX_OBJECTS` slots.
    """
    # ------------------------ Task 2A Modify your code below --------------------

    # Assuming we are expected to return 4 objects
    heatmap = np.zeros((1, 100, 88), dtype=np.float32)
    indices = np.zeros((MAX_OBJECTS), dtype=np.int64)
    regressions = np.zeros((MAX_OBJECTS, 8), dtype=np.float32)
    mask = np.zeros((MAX_OBJECTS,), dtype=np.bool_)

    for idx, box in enumerate(boxes):
        u, v, _ = metric_to_output_grid(box[0], box[1])
        # u_center = (box[0] - x_min) / OUTPUT_RESOLUTION
        # v_center = OUT_H - ((box[1] - y_min) / OUTPUT_RESOLUTION)
        center_int = np.array([np.floor(u), np.floor(v)]).astype(np.int64)
        radius = gaussian_radius(box[5] / OUTPUT_RESOLUTION, box[4] / OUTPUT_RESOLUTION)
        radius = max(0, np.floor(radius))
        draw_gaussian(heatmap[0], center_int, radius)

        indices[idx] = center_int[1] * OUT_W + center_int[0]
        du = u - center_int[0]
        dv = v - center_int[1]
        z = box[2]
        log_dims = np.log(box[3:6])
        sin_yaw = np.sin(box[6])
        cos_yaw = np.cos(box[6])
        regressions[idx] = np.array([du, dv, z, *log_dims, sin_yaw, cos_yaw])
        mask[idx] = True

    return {
        "heatmap": heatmap,
        "indices": indices,
        "regressions": regressions,
        "mask": mask,
    }
    # ---------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK ----------------


def decode_targets(targets: dict[str, np.ndarray]) -> np.ndarray:
    """Invert `encode_targets` and return decoded metric boxes `[M,7]`."""
    # ------------------------ Task 2B Modify your code below --------------------
    heatmap = targets["heatmap"]
    indices = targets["indices"]
    regressions = targets["regressions"]
    mask = targets["mask"]

    decoded_boxes = []
    for idx in range(MAX_OBJECTS):
        if not mask[idx]:
            continue
        index = indices[idx]
        u_center = index % OUT_W
        v_center = index // OUT_W
        du, dv, z, log_length, log_width, log_height, sin_yaw, cos_yaw = regressions[idx]
        u_center += du
        v_center += dv
        x, y = output_grid_to_metric(u_center, v_center)
        length = np.exp(log_length)
        width = np.exp(log_width)
        height = np.exp(log_height)
        yaw = np.arctan2(sin_yaw, cos_yaw)
        decoded_boxes.append([x, y, z, length, width, height, yaw])

    return np.array(decoded_boxes, dtype=np.float32)


    # ---------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK ----------------


def main():
    points, boxes, frame_ids = load_debug_frames(DATA_ROOT / "debug_frames.npz")
    output = OUTPUT_ROOT / "task2"
    targets = encode_targets(boxes[0])
    decoded = decode_targets(targets)
    bev = rasterize_points(points[0])
    save_target_debug(targets["heatmap"], bev, boxes[0], decoded, output / "target_roundtrip.png")

    count = min(len(boxes[0]), len(decoded))
    center_error = (
        np.linalg.norm(boxes[0][:count, :2] - decoded[:count, :2], axis=1)
        if count else np.array([], dtype=np.float32)
    )
    metrics = {
        "frame_id": str(frame_ids[0]),
        "heatmap_shape": list(targets["heatmap"].shape),
        "heatmap_max": float(targets["heatmap"].max()),
        "ground_truth_boxes": int(len(boxes[0])),
        "encoded_boxes": int(targets["mask"].sum()),
        "decoded_boxes": int(len(decoded)),
        "mean_center_error_m": float(center_error.mean()) if len(center_error) else None,
    }
    write_json(output / "metrics.json", metrics)
    print("Checkpoint 2: target_roundtrip.png and metrics.json")
    return metrics


if __name__ == "__main__":
    print("Task 2 complete:", main())
