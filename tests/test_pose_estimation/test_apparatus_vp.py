"""Tests for the VP-constrained apparatus line fit (seeded by pose priors)."""

from __future__ import annotations

import numpy as np
import pytest

from src.pose_estimation.apparatus_vp import fit_apparatus_lines


def _synthetic_apparatus_crop():
    cv2 = pytest.importorskip("cv2")
    w, h = 600, 400
    img = np.full((h, w, 3), 130, np.uint8)
    # Two standards (sep 300 px) -> 4.02 m => scale 74.6 px/m.
    x_left, x_right = 150, 450
    y_bar, y_padtop, y_ground = 190, 268, 320
    pale = (210, 210, 210)
    dark = (45, 45, 45)
    cv2.line(img, (x_left, 170), (x_left, 335), pale, 3, cv2.LINE_AA)
    cv2.line(img, (x_right, 170), (x_right, 335), pale, 3, cv2.LINE_AA)
    cv2.line(img, (x_left - 6, y_bar), (x_right + 6, y_bar), dark, 3, cv2.LINE_AA)       # bar
    cv2.line(img, (x_left + 4, y_padtop), (x_right - 4, y_padtop), (235, 235, 235), 3, cv2.LINE_AA)  # pad top
    cv2.line(img, (x_left + 2, y_ground), (x_right - 2, y_ground), (235, 235, 235), 3, cv2.LINE_AA)  # ground
    scale = 300.0 / 4.02
    return img, dict(x_left=x_left, x_right=x_right, y_bar=y_bar, y_ground=y_ground,
                     bar_x=(x_left + x_right) / 2.0, scale=scale)


def test_vp_fit_recovers_corners_from_pose_seeds():
    img, gt = _synthetic_apparatus_crop()
    fit = fit_apparatus_lines(
        img,
        bar_x_px=gt["bar_x"],
        bar_y_px=gt["y_bar"],
        ground_y_px=gt["y_ground"],
        scale_px_per_m=gt["scale"],
        upright_separation_m=4.02,
    )
    assert fit is not None
    assert fit.left_top_px[0] == pytest.approx(gt["x_left"], abs=14)
    assert fit.left_top_px[1] == pytest.approx(gt["y_bar"], abs=14)
    assert fit.right_top_px[0] == pytest.approx(gt["x_right"], abs=14)
    assert fit.left_base_px[1] == pytest.approx(gt["y_ground"], abs=14)
    assert fit.right_base_px[1] == pytest.approx(gt["y_ground"], abs=14)
    # Supporting segments were actually found for the standards and bar.
    assert fit.n_support["left_standard"] >= 1
    assert fit.n_support["right_standard"] >= 1
    assert fit.n_support["bar"] >= 1


def test_vp_fit_falls_back_to_seeds_on_blank_crop():
    blank = np.full((400, 600, 3), 128, np.uint8)
    fit = fit_apparatus_lines(
        blank, bar_x_px=300, bar_y_px=190, ground_y_px=320, scale_px_per_m=74.6,
        upright_separation_m=4.02,
    )
    assert fit is not None
    # No segments: corners fall back to the seeded standard-x / line-y.
    half = (4.02 / 2.0) * 74.6
    assert fit.left_base_px[0] == pytest.approx(300 - half, abs=2)
    assert fit.right_base_px[0] == pytest.approx(300 + half, abs=2)
    assert fit.left_base_px[1] == pytest.approx(320, abs=2)


def test_vp_fit_rejects_degenerate_crop():
    tiny = np.zeros((8, 8, 3), np.uint8)
    assert fit_apparatus_lines(tiny, bar_x_px=4, bar_y_px=2, ground_y_px=6,
                               scale_px_per_m=10.0) is None
