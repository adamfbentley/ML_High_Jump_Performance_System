"""Tests for background camera-motion estimation and stable-window detection."""

from __future__ import annotations

import numpy as np
import pytest

from src.pose_estimation.camera_motion import (
    MotionConfig,
    background_displacement,
    build_static_plate,
    estimate_camera_motion,
    estimate_homography,
    find_stable_window,
    remap_points_through_homography,
    stabilize_window,
)


def _textured_scene(height: int = 360, width: int = 640, seed: int = 0) -> np.ndarray:
    """A richly textured frame so ORB has plenty of stable background features."""
    rng = np.random.default_rng(seed)
    import cv2

    frame = np.full((height, width, 3), 30, dtype=np.uint8)
    for _ in range(140):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        r = int(rng.integers(4, 16))
        colour = tuple(int(c) for c in rng.integers(60, 255, size=3))
        if rng.random() < 0.5:
            cv2.circle(frame, (x, y), r, colour, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (x, y), (x + r, y + r), colour, -1)
    return frame


def _translate(frame: np.ndarray, tx: float, ty: float) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    m = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float64)
    return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def test_estimate_homography_recovers_known_translation():
    pytest.importorskip("cv2")
    base = _textured_scene(seed=1)
    shifted = _translate(base, 9.0, -4.0)
    h_mat, n_inliers = estimate_homography(base, shifted)
    assert h_mat is not None
    assert n_inliers >= 18
    disp = background_displacement(h_mat, width=base.shape[1], height=base.shape[0])
    assert disp == pytest.approx(np.hypot(9.0, 4.0), abs=1.5)


def test_estimate_homography_near_zero_for_static_pair():
    pytest.importorskip("cv2")
    base = _textured_scene(seed=2)
    h_mat, _ = estimate_homography(base, base.copy())
    assert h_mat is not None
    disp = background_displacement(h_mat, width=base.shape[1], height=base.shape[0])
    assert disp < 0.5


def test_foreground_mask_excludes_moving_athlete():
    cv2 = pytest.importorskip("cv2")
    # Static background; a bright "athlete" block translates a lot between frames.
    base = _textured_scene(seed=3)
    a = base.copy()
    b = base.copy()
    cv2.rectangle(a, (250, 150), (330, 320), (255, 255, 255), -1)
    cv2.rectangle(b, (250 + 40, 150), (330 + 40, 320), (255, 255, 255), -1)  # athlete moved
    fg_a = np.zeros(base.shape[:2], dtype=bool)
    fg_a[150:320, 250:330] = True
    fg_b = np.zeros(base.shape[:2], dtype=bool)
    fg_b[150:320, 250 + 40 : 330 + 40] = True

    h_mat, _ = estimate_homography(a, b, foreground_mask_a=fg_a, foreground_mask_b=fg_b)
    assert h_mat is not None
    disp = background_displacement(h_mat, width=base.shape[1], height=base.shape[0])
    # Background is static, so masked estimation should report ~no motion even
    # though a big foreground object moved 40 px.
    assert disp < 1.5


def test_find_stable_window_picks_still_segment():
    # disp[t] = motion into frame t. Frames 5..12 are still; ends are panning.
    disp = np.array(
        [0.0] + [8.0] * 4 + [0.4, 0.5, 0.3, 0.6, 0.4, 0.5, 0.2, 0.5] + [9.0] * 4,
        dtype=np.float64,
    )
    window = find_stable_window(disp, fps=30.0, config=MotionConfig(min_window_s=0.2))
    assert window is not None
    # Window seeds at the last panning frame (4) then includes the still run 5..12.
    assert window.start == 4
    assert window.end == 12
    assert window.max_disp_px <= 2.5
    assert window.duration_s == pytest.approx(window.n_frames / 30.0)


def test_find_stable_window_rejects_when_too_short():
    disp = np.array([0.0, 8.0, 0.3, 0.4, 9.0, 9.0], dtype=np.float64)
    window = find_stable_window(disp, fps=30.0, config=MotionConfig(min_window_s=0.5))
    assert window is None


def test_estimate_camera_motion_on_synthetic_pan_then_hold():
    pytest.importorskip("cv2")
    base = _textured_scene(seed=4)
    frames = []
    # 4 panning frames (6 px/frame), then 8 held frames (no motion).
    offset = 0.0
    for _ in range(4):
        offset += 6.0
        frames.append(_translate(base, offset, 0.0))
    for _ in range(8):
        frames.append(_translate(base, offset, 0.0))

    result = estimate_camera_motion(frames, fps=30.0, config=MotionConfig(min_window_s=0.2))
    assert result.stable_window is not None
    # The held segment is frames 4..11.
    assert result.stable_window.end == 11
    assert result.stable_window.start >= 3
    assert result.stable_window.n_frames >= 7
    # Panning frames register clearly more motion than the held ones.
    assert np.mean(result.disp_px[1:4]) > 3.0
    assert np.max(result.disp_px[5:12]) < 2.5


def test_stabilize_window_registers_panning_frames_to_reference():
    pytest.importorskip("cv2")
    base = _textured_scene(seed=5)
    # A window that is actually *panning* (6 px/frame): stabilization should
    # still register every frame to the reference with a small residual.
    indices = list(range(7))
    frames = [_translate(base, 6.0 * i, 0.0) for i in indices]
    ref = 3
    stab = stabilize_window(frames, indices, ref_index=ref)
    assert stab.n_registered == len(indices)
    assert stab.max_residual_px < 1.5  # leftover motion after warping is tiny
    assert np.allclose(stab.homographies[ref], np.eye(3))


def test_stabilize_window_remaps_point_into_reference_frame():
    pytest.importorskip("cv2")
    base = _textured_scene(seed=6)
    indices = list(range(5))
    # Frame k is the base shifted right by 10*k px. A world feature that sits at
    # x in the base appears at x + 10*k in frame k; remapping frame k -> ref(0)
    # must bring it back to its base location.
    frames = [_translate(base, 10.0 * i, 0.0) for i in indices]
    stab = stabilize_window(frames, indices, ref_index=0)
    k = 4
    pt_in_frame_k = np.array([[300.0 + 10.0 * k, 180.0]])
    remapped = remap_points_through_homography(pt_in_frame_k, stab.homographies[k])
    assert remapped[0, 0] == pytest.approx(300.0, abs=2.0)
    assert remapped[0, 1] == pytest.approx(180.0, abs=2.0)


def test_remap_preserves_nan_points():
    h = np.eye(3, dtype=np.float64)
    pts = np.array([[10.0, 20.0], [np.nan, np.nan]])
    out = remap_points_through_homography(pts, h)
    assert out[0] == pytest.approx([10.0, 20.0])
    assert np.isnan(out[1]).all()


def test_build_static_plate_removes_moving_athlete_stationary_clip():
    cv2 = pytest.importorskip("cv2")
    # Stationary camera: a bright "athlete" block sweeps across a fixed scene.
    # The temporal median must recover the background wherever the block is a
    # minority occupant (it is, at every column, since it only dwells briefly).
    base = _textured_scene(seed=11)
    h, w = base.shape[:2]
    indices = list(range(11))
    frames = []
    for i in indices:
        f = base.copy()
        x = 40 + i * 40  # block marches left->right across the frame
        cv2.rectangle(f, (x, 120), (x + 60, 300), (255, 255, 255), -1)
        frames.append(f)

    plate = build_static_plate(frames, indices, ref_index=5)
    assert plate.n_frames_used == len(indices)
    assert plate.plate.shape == base.shape
    # The plate must look like the clean background, not any single athlete frame.
    plate_err = np.abs(plate.plate.astype(float) - base.astype(float)).mean()
    frame_err = np.abs(frames[5].astype(float) - base.astype(float)).mean()
    assert plate_err < frame_err
    assert plate_err < 6.0  # background recovered to within a few grey levels
    # Interior coverage is full (every frame contributes; no warping on a
    # stationary clip).
    assert plate.coverage[h // 2, w // 2] == pytest.approx(1.0)


def test_build_static_plate_foreground_mask_excludes_athlete():
    cv2 = pytest.importorskip("cv2")
    # A block dwells at x=300 for the MAJORITY of frames (a plain median would
    # keep it) and steps aside for two frames that expose the background there.
    # The foreground mask must drop the dwelling block so the median recovers the
    # background from the minority clear views.
    base = _textured_scene(seed=12)
    indices = list(range(7))
    frames = []
    masks = []
    for i in indices:
        f = base.copy()
        x = 600 if i >= 5 else 300  # clears the x=300 region only in frames 5,6
        cv2.rectangle(f, (x, 150), (x + 50, 280), (255, 255, 255), -1)
        m = np.zeros(base.shape[:2], dtype=bool)
        m[150:280, x : x + 50] = True
        frames.append(f)
        masks.append(m)

    region = (slice(150, 280), slice(300, 350))
    # Without a mask the dwelling block survives the median.
    no_mask = build_static_plate(frames, indices, ref_index=3)
    no_mask_err = np.abs(no_mask.plate[region].astype(float) - base[region].astype(float)).mean()
    assert no_mask_err > 100.0  # block still present
    # With the mask it is excluded and the background is recovered.
    masked = build_static_plate(frames, indices, ref_index=3, foreground_masks=masks)
    masked_err = np.abs(masked.plate[region].astype(float) - base[region].astype(float)).mean()
    assert masked_err < 12.0
