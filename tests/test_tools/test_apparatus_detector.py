"""Tests for the colour-agnostic, geometry-first apparatus detector.

These synthetic fixtures deliberately use *no* red pixels so they prove the
detector finds the apparatus from geometry alone, that it rejects tall
background poles lacking a crossbar/pad relationship, that it fits a pale
crossbar between standards, and that it picks up a landing-pad top edge
independently of the bar.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pose_estimation.apparatus_detector import (
    ApparatusConfig,
    detect_apparatus_geometry,
    detect_apparatus_geometry_stable,
)


def _blank(height: int = 540, width: int = 960, value: int = 120) -> np.ndarray:
    # Mid-grey background so both the pale standards and the dark crossbar carry
    # real edge contrast — mirroring daylight footage (dark bar against sky,
    # pale posts against trees).  Still entirely colourless (no red).
    return np.full((height, width, 3), value, dtype=np.uint8)


def _draw_post(frame: np.ndarray, x: int, y_base: int, y_top: int, colour, thickness: int = 6) -> None:
    cv2 = pytest.importorskip("cv2")
    cv2.line(frame, (x, y_base), (x, y_top), colour, thickness, cv2.LINE_AA)


def _draw_bar(frame: np.ndarray, x_left: int, x_right: int, y: int, colour, thickness: int = 5) -> None:
    cv2 = pytest.importorskip("cv2")
    cv2.line(frame, (x_left, y), (x_right, y), colour, thickness, cv2.LINE_AA)


def _make_pale_apparatus(
    *,
    left_x: int = 360,
    right_x: int = 600,
    y_base: int = 420,
    y_top: int = 250,
    bar_y: int = 252,
    with_bar: bool = True,
    with_pad: bool = False,
    pad_y: int = 360,
) -> np.ndarray:
    """Pale-grey standards on a dark background, with a darker crossbar.

    Mimics the daylight regime: no red anywhere, bright posts, dark bar.
    """
    frame = _blank()
    pale = (215, 215, 215)
    dark_bar = (40, 40, 40)
    _draw_post(frame, left_x, y_base, y_top, pale)
    _draw_post(frame, right_x, y_base, y_top, pale)
    if with_bar:
        _draw_bar(frame, left_x - 6, right_x + 6, bar_y, dark_bar, thickness=4)
    if with_pad:
        # A bright pad-top edge between the uprights, well below the bar.
        _draw_bar(frame, left_x + 4, right_x - 4, pad_y, (235, 235, 235), thickness=3)
    return frame


def test_detector_requires_no_red_pixels():
    pytest.importorskip("cv2")
    frame = _make_pale_apparatus(with_bar=True)
    assert frame[:, :, 2].max() == frame[:, :, 0].max()  # no red dominance anywhere

    det = detect_apparatus_geometry(frame)
    assert det is not None
    assert det.has_crossbar
    assert det.left_base_px[0] == pytest.approx(360, abs=20)
    assert det.right_base_px[0] == pytest.approx(600, abs=20)
    # Tops come from the crossbar/post intersection, so they sit at the bar y.
    assert det.left_top_px[1] == pytest.approx(252, abs=18)
    assert det.right_top_px[1] == pytest.approx(252, abs=18)


def test_detector_rejects_isolated_background_poles():
    pytest.importorskip("cv2")
    # Two tall poles, no crossbar joining them and no pad edge between them.
    frame = _blank()
    pale = (200, 200, 200)
    _draw_post(frame, 150, 430, 120, pale)  # tall, like a floodlight mast
    _draw_post(frame, 760, 430, 150, pale)  # second isolated pole, far apart
    det = detect_apparatus_geometry(frame)
    assert det is None


def test_detector_prefers_crossbar_pair_over_taller_pole_pair():
    pytest.importorskip("cv2")
    # A real apparatus pair (joined by a bar) plus a wider, *taller* pole pair
    # with no connecting bar.  The detector must choose the apparatus.
    frame = _make_pale_apparatus(left_x=420, right_x=620, y_base=420, y_top=260, bar_y=262)
    pale = (200, 200, 200)
    _draw_post(frame, 90, 430, 80, pale)   # tall background mast, left
    _draw_post(frame, 880, 430, 90, pale)  # tall background mast, right

    det = detect_apparatus_geometry(frame)
    assert det is not None
    assert det.has_crossbar
    assert det.left_base_px[0] == pytest.approx(420, abs=25)
    assert det.right_base_px[0] == pytest.approx(620, abs=25)


def test_detector_fits_pale_crossbar_slope_between_standards():
    cv2 = pytest.importorskip("cv2")
    frame = _blank()
    pale = (205, 205, 205)
    left_x, right_x = 380, 600
    y_base = 430
    # Leaning-perspective bar: left end higher than right.
    bar_left_y, bar_right_y = 250, 268
    _draw_post(frame, left_x, y_base, 240, pale)
    _draw_post(frame, right_x, y_base, 258, pale)
    cv2.line(frame, (left_x - 6, bar_left_y), (right_x + 6, bar_right_y), (55, 55, 55), 4, cv2.LINE_AA)

    det = detect_apparatus_geometry(frame)
    assert det is not None
    assert det.has_crossbar
    fitted_slope = float(
        (det.right_top_px[1] - det.left_top_px[1]) / (det.right_top_px[0] - det.left_top_px[0])
    )
    true_slope = (bar_right_y - bar_left_y) / (right_x - left_x)
    assert fitted_slope == pytest.approx(true_slope, abs=0.03)


def test_detector_picks_up_pad_top_edge_independently():
    pytest.importorskip("cv2")
    frame = _make_pale_apparatus(with_bar=True, with_pad=True, bar_y=252, pad_y=360)
    det = detect_apparatus_geometry(frame)
    assert det is not None
    assert det.has_pad
    assert det.pad_top is not None
    # The pad edge is a separate, lower line, not the crossbar.
    assert det.pad_top.y_center_px == pytest.approx(360, abs=18)
    assert det.pad_top.y_center_px > det.left_top_px[1] + 60


def test_detector_admits_pad_pair_without_a_crossbar():
    pytest.importorskip("cv2")
    # No bar drawn, but a clear pad-top edge between the standards: still admit,
    # flagged as no-crossbar (lower confidence path).
    frame = _make_pale_apparatus(with_bar=False, with_pad=True, pad_y=360)
    det = detect_apparatus_geometry(frame)
    assert det is not None
    assert not det.has_crossbar
    assert det.has_pad


def test_stable_aggregation_reports_consistent_geometry():
    pytest.importorskip("cv2")
    frames = [_make_pale_apparatus(with_bar=True) for _ in range(4)]
    stable = detect_apparatus_geometry_stable(frames)
    assert stable.is_stable
    assert stable.n_frames_used >= 3
    assert stable.crossbar_frac == pytest.approx(1.0)
    assert stable.left_base_px[0] == pytest.approx(360, abs=20)
    assert stable.right_base_px[0] == pytest.approx(600, abs=20)
