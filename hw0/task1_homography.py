"""HW0 Task 1: homography and virtual-ad insertion.

Work from Checkpoint 1A through 1D, then run:

    python task1_homography.py

The starter remains runnable while you replace each placeholder.
"""

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from hw0_utils import (
    COURT_UV,
    COURT_XY,
    float_image,
    project_planar,
    read_video_frame,
    save_rgb,
    warp_rgba,
    write_json,
)


# The UV locations below are provided image pixels in frame 0 of
# data/tennis/match005.mp4. You derive only their matching metric XY locations.
TENNIS_POINT_NAMES = (
    "A: near left doubles corner",
    "B: near right doubles corner",
    "C: far left doubles corner",
    "D: far right doubles corner",
    "E: near left singles corner",
    "F: near right singles corner",
    "G: far left singles corner",
    "H: far right singles corner",
)
TENNIS_UV = np.array(
    [
        [240.0, 845.0],   # A
        [1679.0, 845.0],  # B
        [514.0, 343.0],   # C
        [1408.0, 343.0],  # D
        [429.0, 843.0],   # E
        [1491.0, 843.0],  # F
        [627.0, 343.0],   # G
        [1294.0, 343.0],  # H
    ],
    dtype=float,
)


# ------------------------ Task 1: Modify your code below ------------------------

# Replace each placeholder [0.0, 0.0] with the metric tennis-court XY point
# corresponding to the provided UV row with the same letter. Use meters, the
# origin and axes defined in the handout, and keep the A--H order unchanged.
TENNIS_XY = np.array(
    [
        [-5.48, 0.0],
        [5.48, 0.0],
        [-5.48, 23.77],
        [5.48, 23.77],
        [-4.11, 0.0],
        [4.11, 0.0],
        [-4.11, 23.77],
        [4.11, 23.77],
    ],
    dtype=float,
)


def estimate_homography(xy: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Return H such that [u, v, 1]^T is proportional to H [x, y, 1]^T.

    Replace the identity placeholder with normalized DLT:
      1. normalize XY and UV separately so each centroid is zero and the mean
         distance from the origin is sqrt(2);
      2. build the 2N x 9 homogeneous design matrix;
      3. use SVD to take its right-null-space vector;
      4. reshape, denormalize, and choose a stable matrix scale.
    """
    xy_centroid = np.mean(xy, axis=0)
    xy_normalized = xy - xy_centroid
    xy_scaling = np.mean(np.linalg.norm(xy_normalized, axis=1)) / np.sqrt(2)
    xy_scaled = xy_normalized / xy_scaling
    uv_centroid = np.mean(uv, axis=0)
    uv_normalized = uv - uv_centroid
    uv_scaling = np.mean(np.linalg.norm(uv_normalized, axis=1)) / np.sqrt(2)
    uv_scaled = uv_normalized / uv_scaling

    design_matrix = []
    for (x, y), (u, v) in zip(xy_scaled, uv_scaled):
        design_matrix.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        design_matrix.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    design_matrix = np.array(design_matrix)

    _, _, vh = np.linalg.svd(design_matrix)
    H = vh[-1].reshape(3, 3)

    normalization_matrix = np.array([[1 / xy_scaling, 0, -xy_centroid[0] / xy_scaling],
                                     [0, 1 / xy_scaling, -xy_centroid[1] / xy_scaling],
                                     [0, 0, 1]])
    denormalization_matrix = np.array([[uv_scaling, 0, uv_centroid[0]],
                                       [0, uv_scaling, uv_centroid[1]],
                                       [0, 0, 1]])
    H = denormalization_matrix @ H @ normalization_matrix
    H = H / H[2, 2]
    return H



def logo_to_image_homography(
    court_to_image: np.ndarray,
    logo_shape: tuple[int, ...],
    lower_left_xy: tuple[float, float],
    size_xy: tuple[float, float],
) -> np.ndarray:
    """Compose logo pixels -> metric court rectangle -> image pixels.

    Map logo pixels into the requested metric court rectangle, then apply
    court_to_image. Include the vertical flip because logo pixel y points down
    while court y points up.
    """
    logo_height, logo_width = logo_shape[:2]
    logo_corners = np.array(
        [
            [0, logo_height - 1], [logo_width - 1, logo_height - 1],
            [logo_width - 1, 0], [0, 0],
        ],
        dtype=float,
    )
    court_corners = np.array(
        [
            lower_left_xy, [lower_left_xy[0] + size_xy[0], lower_left_xy[1]],
            [lower_left_xy[0] + size_xy[0], lower_left_xy[1] + size_xy[1]],
            [lower_left_xy[0], lower_left_xy[1] + size_xy[1]],
        ],
        dtype=float,
    )
    logo_to_court = estimate_homography(logo_corners, court_corners)
    del logo_shape, lower_left_xy, size_xy
    return court_to_image @ logo_to_court


def alpha_blend(foreground_rgba: np.ndarray, background_rgb: np.ndarray) -> np.ndarray:
    """Composite an RGBA foreground over an RGB background.

    Replace the placeholder with the per-pixel alpha equation from the
    handout. Alpha is the final channel of foreground_rgba and must blend all
    three foreground RGB channels with the matching background pixel.
    """
    alpha = foreground_rgba[..., 3:4]
    blended_rgb = alpha * foreground_rgba[..., :3] + (1 - alpha) * background_rgb
    del foreground_rgba
    return blended_rgb


# ------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK --------------------


def main() -> dict[str, float | int | None]:
    """Run the four Task 1 checkpoints in the same order as the handout."""
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs/task1"
    output_dir.mkdir(parents=True, exist_ok=True)
    court = float_image(root / "data/court/court.png")[..., :3]
    logo = float_image(root / "data/court/logo.png")

    # Checkpoint 1A: estimate the basketball court homography and inspect its fit.
    court_to_image = estimate_homography(COURT_XY, COURT_UV)
    basketball_predicted_uv = project_planar(COURT_XY, court_to_image)
    basketball_error = np.linalg.norm(basketball_predicted_uv - COURT_UV, axis=1)
    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.imshow(court)
    axis.scatter(COURT_UV[:, 0], COURT_UV[:, 1], c="#e84a27", marker="x", s=60, label="given")
    axis.scatter(
        basketball_predicted_uv[:, 0],
        basketball_predicted_uv[:, 1],
        facecolors="none",
        edgecolors="#00b7ff",
        s=80,
        label="projected",
    )
    axis.legend(loc="lower left")
    axis.set_title("Checkpoint 1A: basketball homography fit")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "task1a_basketball_homography.png", dpi=150)
    plt.close(figure)
    print(f"Checkpoint 1A saved; max corner error = {np.max(basketball_error):.3f} px")

    # Checkpoint 1B: place the 6 m x 3 m logo and alpha-composite it onto the court.
    logo_to_image = logo_to_image_homography(
        court_to_image,
        logo.shape,
        lower_left_xy=(23.0, 2.5),
        size_xy=(6.0, 3.0),
    )
    warped_logo = warp_rgba(logo, logo_to_image, court.shape[:2])
    basketball_insertion = alpha_blend(warped_logo, court)
    save_rgb(output_dir / "task1b_basketball_insertion.png", basketball_insertion)
    print("Checkpoint 1B saved: task1b_basketball_insertion.png")

    # Checkpoint 1C: reuse the same three functions on the tennis frame.
    if TENNIS_XY.shape != TENNIS_UV.shape or TENNIS_XY.shape[1] != 2:
        raise ValueError("TENNIS_XY and TENNIS_UV must have the same N x 2 shape")
    tennis_frame = read_video_frame(root / "data/tennis/match005.mp4", 0)
    tennis_ready = len(np.unique(TENNIS_XY, axis=0)) >= 4 and np.linalg.matrix_rank(
        TENNIS_XY - TENNIS_XY.mean(axis=0)
    ) == 2
    if tennis_ready:
        tennis_to_image = estimate_homography(TENNIS_XY, TENNIS_UV)
        tennis_logo_to_image = logo_to_image_homography(
            tennis_to_image,
            logo.shape,
            lower_left_xy=(-3.0, 0.5),
            size_xy=(6.0, 3.0),
        )
        tennis_warp = warp_rgba(logo, tennis_logo_to_image, tennis_frame.shape[:2])
        tennis_insertion = alpha_blend(tennis_warp, tennis_frame.astype(float) / 255.0)
        tennis_predicted_uv = project_planar(TENNIS_XY, tennis_to_image)
        tennis_error = np.linalg.norm(tennis_predicted_uv - TENNIS_UV, axis=1)
    else:
        # Keep the full script runnable while the student is working on 1A/1B.
        tennis_to_image = np.eye(3)
        tennis_insertion = tennis_frame.astype(float) / 255.0
        tennis_predicted_uv = np.empty((0, 2), dtype=float)
        tennis_error = np.empty(0, dtype=float)
    save_rgb(output_dir / "task1c_tennis_insertion.png", tennis_insertion)

    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.imshow(tennis_frame)
    axis.scatter(TENNIS_UV[:, 0], TENNIS_UV[:, 1], c="#ff5a5f", s=48)
    for name, (u, v) in zip("ABCDEFGH", TENNIS_UV, strict=True):
        axis.text(u + 8, v - 8, name, color="white", weight="bold")
    axis.set_title("Checkpoint 1C: provided tennis UV points A-H")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "task1c_tennis_correspondences.png", dpi=150)
    plt.close(figure)

    alignment = tennis_frame.copy()
    court_edges = ((0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4))
    for first, second in court_edges:
        cv2.line(
            alignment,
            tuple(np.rint(TENNIS_UV[first]).astype(int)),
            tuple(np.rint(TENNIS_UV[second]).astype(int)),
            (0, 255, 80),
            2,
        )
        if tennis_ready:
            cv2.line(
                alignment,
                tuple(np.rint(tennis_predicted_uv[first]).astype(int)),
                tuple(np.rint(tennis_predicted_uv[second]).astype(int)),
                (255, 95, 35),
                1,
            )
    save_rgb(output_dir / "task1c_tennis_alignment.png", alignment)
    status = "complete" if tennis_ready else "placeholder; fill TENNIS_XY to enable the estimate"
    print(f"Checkpoint 1C saved ({status}): correspondences, insertion, and alignment")

    # Checkpoint 1D: collect the quantitative results used in the report.
    metrics: dict[str, float | int | None] = {
        "basketball_corner_max_error_px": float(np.max(basketball_error)),
        "tennis_correspondence_count": int(len(TENNIS_XY)),
        "tennis_8_anchor_reprojection_rmse_px": (
            float(np.sqrt(np.mean(tennis_error**2))) if tennis_ready else None
        ),
    }
    write_json(output_dir / "task1d_metrics.json", metrics)
    np.savez(
        output_dir / "task1_results.npz",
        basketball_homography=court_to_image,
        tennis_homography=tennis_to_image,
    )
    print("Checkpoint 1D saved: task1d_metrics.json")
    return metrics


if __name__ == "__main__":
    print("Task 1 complete:", main())
