"""Tests for decoded timing used by pose-overlay rendering."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest
from src.visualization.pose_overlay import _decoded_video_timing


class _FakeVideoCapture:
    def __init__(self, timestamps_ms: list[float], reported_fps: float) -> None:
        self._timestamps_ms = timestamps_ms
        self._reported_fps = reported_fps
        self._frame_index = -1

    def isOpened(self) -> bool:  # noqa: N802 - match OpenCV's API
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._reported_fps
        if prop == cv2.CAP_PROP_POS_MSEC and self._frame_index >= 0:
            return self._timestamps_ms[self._frame_index]
        return 0.0

    def grab(self) -> bool:
        self._frame_index += 1
        return self._frame_index < len(self._timestamps_ms)

    def release(self) -> None:
        pass


def test_decoded_video_timing_uses_frame_cadence_and_actual_count(monkeypatch) -> None:
    timestamps_ms = [0.0, 1000.0 / 30.0, 2000.0 / 30.0, 3000.0 / 30.0]
    fake_capture = _FakeVideoCapture(timestamps_ms, reported_fps=27.2)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: fake_capture)

    fps, frame_count = _decoded_video_timing(Path("clip.mov"))

    assert fps == pytest.approx(30.0)
    assert frame_count == 4
