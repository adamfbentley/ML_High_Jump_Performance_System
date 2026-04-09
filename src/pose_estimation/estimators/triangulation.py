"""Multi-view triangulation for 3D pose reconstruction.

Given 2D landmarks from two or more camera views with known calibration,
reconstructs 3D joint positions via Direct Linear Transform (DLT).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraCalibration:
    """Intrinsic and extrinsic parameters for a single camera view."""

    intrinsic: np.ndarray  # (3, 3) camera matrix
    extrinsic: np.ndarray  # (3, 4) [R|t] world-to-camera
    distortion: np.ndarray | None = None  # distortion coefficients

    @property
    def projection_matrix(self) -> np.ndarray:
        """Full (3, 4) projection matrix P = K @ [R|t]."""
        return self.intrinsic @ self.extrinsic


def triangulate_point(
    point_2d_views: list[np.ndarray],
    projection_matrices: list[np.ndarray],
) -> np.ndarray:
    """Triangulate a single 3D point from multiple 2D observations.

    Uses the Direct Linear Transform (DLT) method.

    Args:
        point_2d_views: List of (2,) arrays, one per camera view.
        projection_matrices: List of (3, 4) projection matrices.

    Returns:
        (3,) array of the reconstructed 3D point.
    """
    n_views = len(point_2d_views)
    if n_views < 2:
        raise ValueError("Need at least 2 views for triangulation")

    A = np.zeros((2 * n_views, 4))
    for i, (pt, P) in enumerate(zip(point_2d_views, projection_matrices)):
        x, y = pt[0], pt[1]
        A[2 * i] = x * P[2] - P[0]
        A[2 * i + 1] = y * P[2] - P[1]

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def triangulate_landmarks(
    landmarks_per_view: list[np.ndarray],
    calibrations: list[CameraCalibration],
    min_confidence: float = 0.5,
) -> np.ndarray:
    """Triangulate all landmarks from multiple synchronized camera views.

    Args:
        landmarks_per_view: List of (N_landmarks, 3) arrays [x, y, confidence].
        calibrations: Camera calibration for each view.
        min_confidence: Minimum visibility to include a view for triangulation.

    Returns:
        (N_landmarks, 3) array of 3D positions. NaN for failed landmarks.
    """
    n_landmarks = landmarks_per_view[0].shape[0]
    proj_matrices = [c.projection_matrix for c in calibrations]
    points_3d = np.full((n_landmarks, 3), np.nan)

    for j in range(n_landmarks):
        valid_views = []
        valid_projs = []
        for view_lm, P in zip(landmarks_per_view, proj_matrices):
            if view_lm[j, 2] >= min_confidence:
                valid_views.append(view_lm[j, :2])
                valid_projs.append(P)
        if len(valid_views) >= 2:
            points_3d[j] = triangulate_point(valid_views, valid_projs)

    return points_3d
