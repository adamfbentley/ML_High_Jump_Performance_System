"""Tests for base-post-guided stable takeoff anchor detection."""

from __future__ import annotations

import numpy as np
import pytest
from scripts.detect_stable_takeoff_anchors import (
    BarScanConfig,
    LineCandidate,
    _centerline_from_upright_edge_pair,
    _fit_local_red_bar_line,
    _line_from_points,
    _x_at_y,
    detect_bar_from_base_posts,
)


def _draw_sloped_line(
    frame: np.ndarray,
    *,
    left_x: int,
    right_x: int,
    y_center: float,
    x_mid: float,
    slope: float,
    colour: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2 = pytest.importorskip("cv2")
    left_y = y_center + slope * (left_x - x_mid)
    right_y = y_center + slope * (right_x - x_mid)
    cv2.line(
        frame,
        (int(round(left_x)), int(round(left_y))),
        (int(round(right_x)), int(round(right_y))),
        colour,
        thickness,
        cv2.LINE_AA,
    )


def test_detect_bar_from_base_posts_skips_pad_and_noisy_background():
    pytest.importorskip("cv2")
    frame = np.full((520, 760, 3), 42, dtype=np.uint8)
    left_base = np.array([180.0, 440.0])
    right_base = np.array([610.0, 500.0])
    slope = float((right_base[1] - left_base[1]) / (right_base[0] - left_base[0]))
    x_mid = float((left_base[0] + right_base[0]) / 2.0)

    # Landing-pad top: lower and red, so it should be detected as the first
    # horizontal-ish structure but not selected as the crossbar.
    _draw_sloped_line(
        frame,
        left_x=180,
        right_x=610,
        y_center=365.0,
        x_mid=x_mid,
        slope=slope,
        colour=(25, 35, 215),
        thickness=7,
    )

    # Background distractors: straight-ish, but not red-supported across the
    # full post span.
    _draw_sloped_line(
        frame,
        left_x=250,
        right_x=480,
        y_center=295.0,
        x_mid=x_mid,
        slope=slope,
        colour=(215, 215, 215),
        thickness=4,
    )
    _draw_sloped_line(
        frame,
        left_x=330,
        right_x=610,
        y_center=255.0,
        x_mid=x_mid,
        slope=slope,
        colour=(80, 130, 160),
        thickness=4,
    )

    # Actual crossbar: next consistent red horizontal line above the pad.
    _draw_sloped_line(
        frame,
        left_x=180,
        right_x=610,
        y_center=240.0,
        x_mid=x_mid,
        slope=slope,
        colour=(20, 30, 230),
        thickness=5,
    )

    detection = detect_bar_from_base_posts(
        frame,
        left_base_px=left_base,
        right_base_px=right_base,
        config=BarScanConfig(
            scan_step_px=2,
            band_radius_px=3,
            sweep_samples=120,
            sweep_bins=12,
            min_pad_to_bar_gap_px=65.0,
        ),
    )

    assert detection.pad_top is not None
    assert detection.pad_top.y_center_px == pytest.approx(365.0, abs=10.0)
    assert detection.selected_bar.y_center_px == pytest.approx(240.0, abs=10.0)

    left_top = np.asarray(detection.points_px["left_top"], dtype=float)
    right_top = np.asarray(detection.points_px["right_top"], dtype=float)
    detected_slope = float((right_top[1] - left_top[1]) / (right_top[0] - left_top[0]))
    assert detected_slope == pytest.approx(slope, abs=0.015)


def test_bar_refinement_uses_local_red_line_not_base_slope():
    pytest.importorskip("cv2")
    frame = np.full((520, 760, 3), 42, dtype=np.uint8)
    left_base = np.array([180.0, 440.0])
    right_base = np.array([610.0, 500.0])
    base_slope = float((right_base[1] - left_base[1]) / (right_base[0] - left_base[0]))
    x_mid = float((left_base[0] + right_base[0]) / 2.0)
    true_bar_slope = -0.045
    true_y_center = 240.0

    _draw_sloped_line(
        frame,
        left_x=210,
        right_x=580,
        y_center=true_y_center,
        x_mid=x_mid,
        slope=true_bar_slope,
        colour=(20, 30, 230),
        thickness=5,
    )
    initial = LineCandidate(
        y_center_px=true_y_center,
        left_px=np.array(
            [left_base[0], true_y_center + base_slope * (left_base[0] - x_mid)]
        ),
        right_px=np.array(
            [right_base[0], true_y_center + base_slope * (right_base[0] - x_mid)]
        ),
        score=1.0,
        coverage=1.0,
        red_coverage=1.0,
    )

    refined = _fit_local_red_bar_line(
        frame,
        left_base_px=left_base,
        right_base_px=right_base,
        initial_bar=initial,
        config=BarScanConfig(refine_bar_window_px=55.0),
    )

    assert refined is not None
    refined_bar = refined.selected_bar
    refined_slope = float((refined_bar.right_px[1] - refined_bar.left_px[1]) / 430.0)
    assert refined_slope == pytest.approx(true_bar_slope, abs=0.02)
    assert refined_slope != pytest.approx(base_slope, abs=0.05)


def test_bar_refinement_places_top_at_post_intersection():
    cv2 = pytest.importorskip("cv2")
    frame = np.full((520, 760, 3), 42, dtype=np.uint8)
    true_left_base = np.array([180.0, 440.0])
    true_right_base = np.array([610.0, 500.0])
    left_base = np.array([172.0, 440.0])
    right_base = np.array([618.0, 500.0])
    x_mid = float((true_left_base[0] + true_right_base[0]) / 2.0)
    true_bar_slope = -0.045
    true_y_center = 240.0

    left_post_top = np.array([192.0, 200.0])
    right_post_top = np.array([600.0, 200.0])
    cv2.line(
        frame,
        tuple(true_left_base.astype(int)),
        tuple(left_post_top.astype(int)),
        (225, 225, 225),
        4,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        tuple(true_right_base.astype(int)),
        tuple(right_post_top.astype(int)),
        (225, 225, 225),
        4,
        cv2.LINE_AA,
    )
    _draw_sloped_line(
        frame,
        left_x=205,
        right_x=590,
        y_center=true_y_center,
        x_mid=x_mid,
        slope=true_bar_slope,
        colour=(20, 30, 230),
        thickness=5,
    )
    initial = LineCandidate(
        y_center_px=true_y_center,
        left_px=np.array([left_base[0], true_y_center]),
        right_px=np.array([right_base[0], true_y_center]),
        score=1.0,
        coverage=1.0,
        red_coverage=1.0,
    )

    refined = _fit_local_red_bar_line(
        frame,
        left_base_px=left_base,
        right_base_px=right_base,
        initial_bar=initial,
        config=BarScanConfig(refine_bar_window_px=65.0),
    )

    assert refined is not None
    refined_bar = refined.selected_bar
    # The synthetic upright leans, so the true bar/post intersections are not
    # at the base x coordinates.  A regression to base-x projection fails here.
    assert refined_bar.left_px[0] > left_base[0] + 4.0
    assert refined_bar.right_px[0] < right_base[0] - 4.0
    refined_slope = float(
        (refined_bar.right_px[1] - refined_bar.left_px[1])
        / (refined_bar.right_px[0] - refined_bar.left_px[0])
    )
    assert refined_slope == pytest.approx(true_bar_slope, abs=0.025)
    assert refined.left_base_px[0] == pytest.approx(true_left_base[0], abs=5.0)
    assert refined.right_base_px[0] == pytest.approx(true_right_base[0], abs=5.0)


def test_upright_centerline_uses_midpoint_of_paired_edges():
    left_edge = _line_from_points(np.array([180.0, 500.0]), np.array([178.0, 250.0]))
    right_edge = _line_from_points(np.array([206.0, 500.0]), np.array([202.0, 250.0]))

    centerline = _centerline_from_upright_edge_pair(
        [
            {"line": left_edge, "length": 250.0, "angle_from_vertical": 0.5},
            {"line": right_edge, "length": 250.0, "angle_from_vertical": 0.5},
        ],
        base_px=np.array([193.0, 500.0]),
        approximate_bar_y_px=250.0,
    )

    assert centerline is not None
    assert _x_at_y(centerline, 500.0) == pytest.approx(193.0, abs=0.5)
    assert _x_at_y(centerline, 250.0) == pytest.approx(190.0, abs=0.5)
