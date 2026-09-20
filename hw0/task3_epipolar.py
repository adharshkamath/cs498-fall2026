"""HW0 Task 3: two-view geometry and epipolar lines.

Work from Checkpoint 3A through 3C, then run:

    python task3_epipolar.py
"""

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from hw0_utils import plot_two_views, symmetric_epipolar_distance, write_json


# ------------------------ Task 3: Modify your code below ------------------------

def normalize_points(points):
    points_mean = np.mean(points, axis=0)
    shifted_points = points - points_mean
    points_scaling = np.sqrt(2) / np.mean(np.linalg.norm(shifted_points, axis=1))
    normalized_points = shifted_points * points_scaling
    return normalized_points, points_mean, points_scaling

def estimate_fundamental_matrix(matches: np.ndarray) -> np.ndarray:
    """Estimate a rank-two F from rows (u1, v1, u2, v2), N >= 8.

    Replace the runnable placeholder with the normalized eight-point method:
      1. normalize the points in each view independently;
      2. build the N x 9 design matrix from (u1, v1, u2, v2);
      3. solve its right null space with SVD;
      4. enforce rank two by zeroing the smallest singular value; and
      5. denormalize and choose a stable scale.
    """
    normalized_1, points_mean_1, points_scaling_1 = normalize_points(matches[:, :2])
    normalized_2, points_mean_2, points_scaling_2 = normalize_points(matches[:, 2:])
    solnmatrix = []
    for i in range(len(matches)):
        u1, v1 = normalized_1[i]
        u2, v2 = normalized_2[i]
        solnmatrix.append([u1 * u2, u2 * v1, u2, v2 * u1, v1 * v2, v2, u1, v1, 1])
    solnmatrix = np.array(solnmatrix)
    _, _, vh = np.linalg.svd(solnmatrix)
    fund_m = vh[-1].reshape(3, 3)
    u, s, vh = np.linalg.svd(fund_m)
    s[-1] = 0
    fund_m = u @ np.diag(s) @ vh
    dm1 = np.array([[points_scaling_1, 0, -points_scaling_1 * points_mean_1[0]],
                                      [0, points_scaling_1, -points_scaling_1 * points_mean_1[1]],
                                      [0, 0, 1]])
    dm2 = np.array([[points_scaling_2, 0, -points_scaling_2 * points_mean_2[0]],
                                      [0, points_scaling_2, -points_scaling_2 * points_mean_2[1]],
                                      [0, 0, 1]])
    fund_m = dm2.T @ fund_m @ dm1
    return fund_m / fund_m[2, 2]



# ------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK --------------------


def main() -> dict[str, float | int]:
    """Run the three Task 3 checkpoints in the same order as the handout."""
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs/task3"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_matches = np.load(root / "data/epipolar/all_good_matches.npy")
    eight_matches = np.load(root / "data/epipolar/eight_good_matches.npy")
    images = [
        iio.imread(root / "data/epipolar/images/0000.png"),
        iio.imread(root / "data/epipolar/images/0005.png"),
    ]

    # Checkpoint 3A: estimate F from the supplied eight correspondences.
    fundamental = estimate_fundamental_matrix(eight_matches)
    np.savetxt(
        output_dir / "task3a_fundamental_matrix.txt",
        fundamental,
        fmt="%.10e",
        header="Estimated fundamental matrix F",
    )
    print(f"Checkpoint 3A saved; rank(F) = {np.linalg.matrix_rank(fundamental, tol=1e-8)}")

    # Checkpoint 3B: visualize matches first, then the epipolar lines from F.
    plot_two_views(images, all_matches, output_dir / "task3b_correspondences.png")
    plot_two_views(images, all_matches, output_dir / "task3b_epipolar_lines.png", fundamental)
    print("Checkpoint 3B saved: correspondences and epipolar lines")

    # Checkpoint 3C: evaluate F on all 200 correspondences.
    errors = symmetric_epipolar_distance(all_matches, fundamental)
    metrics: dict[str, float | int] = {
        "fundamental_rank": int(np.linalg.matrix_rank(fundamental, tol=1e-8)),
        "mean_symmetric_epipolar_error_px": float(np.mean(errors)),
        "median_symmetric_epipolar_error_px": float(np.median(errors)),
    }
    write_json(output_dir / "task3c_metrics.json", metrics)
    np.savez(output_dir / "task3_results.npz", fundamental=fundamental)
    print("Checkpoint 3C saved: task3c_metrics.json")
    return metrics


if __name__ == "__main__":
    print("Task 3 complete:", main())
