"""Tests for the BVH generator."""

import numpy as np
import pytest
from pathlib import Path

from src.pose_estimation.skeleton.bvh_generator import (
    AnthropometricProfile,
    build_skeleton_rig,
    write_bvh,
)


@pytest.fixture
def profile():
    return AnthropometricProfile(height_cm=193.0, weight_kg=82.0)


def test_estimate_missing_fills_all_segments(profile):
    profile.estimate_missing()
    assert profile.leg_length_cm is not None
    assert profile.thigh_length_cm is not None
    assert profile.shank_length_cm is not None
    assert profile.trunk_length_cm is not None
    assert profile.shoulder_width_cm is not None


def test_build_skeleton_returns_hips_root(profile):
    root = build_skeleton_rig(profile)
    assert root.name == "Hips"
    assert len(root.children) == 3  # Spine, LeftUpLeg, RightUpLeg


def test_skeleton_has_correct_depth(profile):
    """Check the chain Hips → Spine → Spine1 → Neck → Head exists."""
    root = build_skeleton_rig(profile)
    spine = [c for c in root.children if c.name == "Spine"][0]
    spine1 = spine.children[0]
    assert spine1.name == "Spine1"
    neck = [c for c in spine1.children if c.name == "Neck"][0]
    head = neck.children[0]
    assert head.name == "Head"
    assert head.is_end_site


def test_write_bvh_creates_file(profile, tmp_path):
    root = build_skeleton_rig(profile)
    # Fake motion data: just zeros
    n_channels = 6 + 20 * 3  # root has 6, 20 other joints have 3 each
    motion = np.zeros((10, n_channels))
    out_path = tmp_path / "test.bvh"
    result = write_bvh(root, motion, fps=30.0, output_path=out_path)
    assert result.exists()
    content = result.read_text()
    assert "HIERARCHY" in content
    assert "MOTION" in content
    assert "Frames: 10" in content
