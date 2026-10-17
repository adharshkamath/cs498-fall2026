"""HW1 Task 2: robust rigid fitting with RANSAC.

Run directly:

    python task2_ransac.py
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np

from hw1_utils import (
    project_rigid_transform,
    rotation_error_degrees,
    save_ransac_plot,
    to_homogeneous,
    transform_points,
    write_json,
)
from task1_alignment import fit_rigid_svd


# ------------------------ Task 2: Modify your code below ------------------------

def ransac_rigid(
    source: np.ndarray,
    target: np.ndarray,
    threshold: float,
    max_trials: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return T_source_to_target and its final inlier mask.

    In each trial, sample three distinct pairs, skip nearly collinear samples,
    fit with Task 1A, and retain the hypothesis with the most inliers. Finally,
    refit once using all inliers from the best hypothesis. Score a hypothesis
    after mapping the homogeneous [4, N] source columns into the target frame.
    Treat a cross-product norm below 1e-8 in either triple as degenerate.
    Keep the first hypothesis on ties; return identity/all-false if none has
    three inliers. Recompute the final mask after refitting (residual <= threshold).
    """

    three_point_tuples = np.empty((max_trials, 3), dtype=int)
    for i in range(max_trials):
        sample_indices = rng.choice(source.shape[1], size=3, replace=False)
        three_point_tuples[i] = sample_indices
    collinear = np.zeros(max_trials, dtype=bool)
    for i in range(max_trials):
        p1 = source[:3, three_point_tuples[i][0]]
        p2 = source[:3, three_point_tuples[i][1]]
        p3 = source[:3, three_point_tuples[i][2]]
        if np.linalg.norm(np.cross(p2 - p1, p3 - p1)) < 1e-8:
            collinear[i] = True

        p1 = target[:3, three_point_tuples[i][0]]
        p2 = target[:3, three_point_tuples[i][1]]
        p3 = target[:3, three_point_tuples[i][2]]
        if np.linalg.norm(np.cross(p2 - p1, p3 - p1)) < 1e-8:
            collinear[i] = True

    samples = three_point_tuples[~collinear]
    if len(samples) == 0:
        return np.eye(4), np.zeros(source.shape[1], dtype=bool)
    best_inliers = np.zeros(source.shape[1], dtype=bool)
    for i in range(samples.shape[0]):
        sample_source = source[:, samples[i]]
        sample_target = target[:, samples[i]]
        transformed_source = transform_points(fit_rigid_svd(sample_source, sample_target), source)
        residuals = np.linalg.norm(transformed_source[:3] - target[:3], axis=0)
        inliers = residuals <= threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers.sum() < 3:
        return np.eye(4), np.zeros(source.shape[1], dtype=bool)

    final_source = source[:, best_inliers]
    final_target = target[:, best_inliers]
    final_transform = fit_rigid_svd(final_source, final_target)
    final_transformed_source = transform_points(final_transform, source)
    final_residuals = np.linalg.norm(final_transformed_source[:3] - target[:3], axis=0)
    final_inliers = final_residuals <= threshold
    return final_transform, final_inliers


# ------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK --------------------


def main(render: bool = True) -> dict[str, float | int | str | None]:
    """Corrupt real pairs, compare least squares with RANSAC, and save outputs."""
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs/task2"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(588)
    with np.load(root / "data/registration_demo.npz") as data:
        source = to_homogeneous(data["task1_source"].astype(np.float64))
        target = to_homogeneous(data["task1_target"].astype(np.float64))
        full_target = to_homogeneous(data["target"].astype(np.float64))
        reference = project_rigid_transform(data["reference_transform"])

    # Corrupt 35% of the supplied real correspondences.
    corrupted = rng.choice(
        source.shape[1], int(0.35 * source.shape[1]), replace=False
    )
    replacement = rng.choice(full_target.shape[1], len(corrupted), replace=False)
    target[:, corrupted] = full_target[:, replacement]

    # Checkpoint 2A: compare an all-pairs fit with robust fitting.
    least_squares = fit_rigid_svd(source, target)
    threshold = 0.025
    robust, inliers = ransac_rigid(source, target, threshold, 500, rng)
    aligned = transform_points(robust, source)
    residuals = np.linalg.norm(aligned[:3] - target[:3], axis=0)
    np.savetxt(output_dir / "ransac_transform.txt", robust, fmt="%.8f")
    print(f"Checkpoint 2A: RANSAC retained {inliers.sum()}/{len(inliers)} pairs")

    # Checkpoint 2B: visualize and report the retained consensus.
    save_ransac_plot(
        output_dir / "ransac_alignment.png", aligned, inliers, residuals, threshold
    )
    metrics = {
        "status": "complete" if inliers.sum() >= 3 else "starter_placeholder",
        "inlier_count": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "inlier_rmse_m": (
            float(np.sqrt(np.mean(residuals[inliers] ** 2))) if np.any(inliers) else None
        ),
        "least_squares_rotation_error_deg": rotation_error_degrees(
            least_squares, reference
        ),
        "ransac_rotation_error_deg": rotation_error_degrees(robust, reference),
        "ransac_translation_error_m": float(
            np.linalg.norm(robust[:3, 3] - reference[:3, 3])
        ),
    }
    write_json(output_dir / "ransac_metrics.json", metrics)
    print("Checkpoint 2B saved: ransac_alignment.png and ransac_metrics.json")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-rendering", action="store_true")
    print("Task 2 complete:", main(render=not parser.parse_args().skip_rendering))
