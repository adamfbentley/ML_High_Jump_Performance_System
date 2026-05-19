"""Tests for hand-labelled scene-anchor JSON conversion."""

from __future__ import annotations

import numpy as np

from scripts.evaluate_calibration_truth import _labels_to_scene_anchors
from src.pose_estimation.scene_calibration import (
    estimate_scene_scale_mpp,
    fit_per_frame_homography,
    homography_valid_mask,
)


def test_partial_hand_labels_preserve_pairwise_scale_without_full_homography():
    payload = {
        "video_stem": "partial_fixture",
        "n_frames": 5,
        "bar_height_m": 1.80,
        "labels": [
            {
                "frame": 0,
                "points": {
                    "left_base": [10.0, 100.0],
                    "right_base": [50.0, 100.0],
                },
            },
            {
                "frame": 2,
                "points": {
                    "left_base": [20.0, 100.0],
                    "right_base": [60.0, 100.0],
                    "left_top": [20.0, 60.0],
                    "right_top": [60.0, 60.0],
                },
            },
            {
                "frame": 4,
                "points": {
                    "left_base": [30.0, 100.0],
                    "right_base": [70.0, 100.0],
                },
            },
        ],
    }

    anchors = _labels_to_scene_anchors(payload)
    scale_mpp = estimate_scene_scale_mpp(anchors)
    homographies = fit_per_frame_homography(anchors)
    valid_homography = homography_valid_mask(homographies, anchors.confidence)

    assert anchors.confidence.tolist() == [0.5, 0.5, 1.0, 0.5, 0.5]
    np.testing.assert_allclose(scale_mpp, np.full(5, 4.02 / 40.0))
    assert valid_homography.tolist() == [False, False, True, False, False]


def test_legacy_complete_hand_labels_still_interpolate_all_anchors():
    payload = {
        "video_stem": "complete_fixture",
        "n_frames": 3,
        "bar_height_m": 1.80,
        "labels": [
            {
                "frame": 0,
                "points": {
                    "left_base": [10.0, 100.0],
                    "right_base": [50.0, 100.0],
                    "left_top": [10.0, 60.0],
                    "right_top": [50.0, 60.0],
                },
            },
            {
                "frame": 2,
                "points": {
                    "left_base": [20.0, 100.0],
                    "right_base": [60.0, 100.0],
                    "left_top": [20.0, 60.0],
                    "right_top": [60.0, 60.0],
                },
            },
        ],
    }

    anchors = _labels_to_scene_anchors(payload)
    homographies = fit_per_frame_homography(anchors)
    valid_homography = homography_valid_mask(homographies, anchors.confidence)

    np.testing.assert_allclose(anchors.confidence, np.ones(3))
    assert valid_homography.tolist() == [True, True, True]
