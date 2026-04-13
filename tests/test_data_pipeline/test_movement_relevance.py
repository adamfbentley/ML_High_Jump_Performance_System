"""Tests for movement relevance rankings in BiomechanicalSample.

Covers the updated MOVEMENT_RELEVANCE ordering reflecting Imogen's
athlete brief: single-leg drop jump > drop jump > CMJ for high jump
pre-training relevance.
"""

from __future__ import annotations

import pytest

from src.data_pipeline.sample import (
    MovementType,
    MOVEMENT_RELEVANCE,
    BiomechanicalSample,
    SubjectInfo,
    SessionContext,
)


# ── Ranking order (Imogen priority) ──────────────────────────────────────

def test_high_jump_is_highest():
    assert MOVEMENT_RELEVANCE[MovementType.HIGH_JUMP] == 1.0


def test_single_leg_drop_jump_exists():
    """SINGLE_LEG_DROP_JUMP must be in MovementType and MOVEMENT_RELEVANCE."""
    assert hasattr(MovementType, "SINGLE_LEG_DROP_JUMP")
    assert MovementType.SINGLE_LEG_DROP_JUMP in MOVEMENT_RELEVANCE


def test_single_leg_drop_jump_above_drop_jump():
    """Single-leg drop jump is more HJ-specific than regular box drop jump."""
    assert (
        MOVEMENT_RELEVANCE[MovementType.SINGLE_LEG_DROP_JUMP]
        > MOVEMENT_RELEVANCE[MovementType.DROP_JUMP]
    )


def test_drop_jump_above_cmj():
    """Box drop jump should rank higher than CMJ per Imogen."""
    assert (
        MOVEMENT_RELEVANCE[MovementType.DROP_JUMP]
        > MOVEMENT_RELEVANCE[MovementType.COUNTERMOVEMENT_JUMP]
    )


def test_cmj_below_drop_jump_but_above_vertical_jump():
    """CMJ relevance hierarchy sanity check."""
    cmj = MOVEMENT_RELEVANCE[MovementType.COUNTERMOVEMENT_JUMP]
    dj  = MOVEMENT_RELEVANCE[MovementType.DROP_JUMP]
    vj  = MOVEMENT_RELEVANCE[MovementType.VERTICAL_JUMP]
    assert dj > cmj
    assert cmj > vj or cmj >= vj  # CMJ and VJ should be close


def test_full_ranking_top_four():
    """Verify the top-4 ranking: HJ > SLDJ > DJ > CMJ."""
    hj   = MOVEMENT_RELEVANCE[MovementType.HIGH_JUMP]
    sldj = MOVEMENT_RELEVANCE[MovementType.SINGLE_LEG_DROP_JUMP]
    dj   = MOVEMENT_RELEVANCE[MovementType.DROP_JUMP]
    cmj  = MOVEMENT_RELEVANCE[MovementType.COUNTERMOVEMENT_JUMP]
    assert hj > sldj > dj > cmj


def test_all_movement_types_in_relevance_table():
    """Every MovementType member must have an entry in MOVEMENT_RELEVANCE."""
    for mt in MovementType:
        assert mt in MOVEMENT_RELEVANCE, (
            f"MovementType.{mt.name} missing from MOVEMENT_RELEVANCE"
        )


def test_all_relevance_scores_in_range():
    """All relevance scores must be in [0, 1]."""
    for mt, score in MOVEMENT_RELEVANCE.items():
        assert 0.0 <= score <= 1.0, (
            f"{mt.name} has out-of-range relevance {score}"
        )


# ── relevance_score property ──────────────────────────────────────────────

def test_sample_relevance_single_leg_drop_jump():
    sample = BiomechanicalSample(
        dataset_name="test",
        trial_id="t1",
        subject=SubjectInfo(subject_id="s1"),
        movement_type=MovementType.SINGLE_LEG_DROP_JUMP,
        fps=100.0,
    )
    assert sample.relevance_score == MOVEMENT_RELEVANCE[MovementType.SINGLE_LEG_DROP_JUMP]


def test_sample_relevance_drop_jump_above_cmj():
    dj_sample = BiomechanicalSample(
        dataset_name="test", trial_id="t1",
        subject=SubjectInfo(subject_id="s1"),
        movement_type=MovementType.DROP_JUMP, fps=100.0,
    )
    cmj_sample = BiomechanicalSample(
        dataset_name="test", trial_id="t2",
        subject=SubjectInfo(subject_id="s1"),
        movement_type=MovementType.COUNTERMOVEMENT_JUMP, fps=100.0,
    )
    assert dj_sample.relevance_score > cmj_sample.relevance_score


# ── SessionContext ────────────────────────────────────────────────────────

def test_session_context_can_be_added_to_sample():
    ctx = SessionContext(
        jump_number_in_session=3,
        rest_time_before_jump_s=120.0,
        heart_rate_bpm=145.0,
        days_since_last_training=1,
    )
    sample = BiomechanicalSample(
        dataset_name="test", trial_id="t1",
        subject=SubjectInfo(subject_id="s1"),
        fps=100.0,
        session_context=ctx,
    )
    assert sample.session_context is not None
    assert sample.session_context.jump_number_in_session == 3
    assert sample.session_context.rest_time_before_jump_s == 120.0


def test_session_context_defaults_to_none():
    """Samples without session context should have None."""
    sample = BiomechanicalSample(
        dataset_name="test", trial_id="t1",
        subject=SubjectInfo(subject_id="s1"),
        fps=100.0,
    )
    assert sample.session_context is None


def test_session_context_fields_all_optional():
    """SessionContext can be created with no fields populated."""
    ctx = SessionContext()
    assert ctx.jump_number_in_session is None
    assert ctx.heart_rate_bpm is None
    assert ctx.days_since_last_training is None


def test_get_window_preserves_session_context():
    """get_window() should propagate session_context to the sliced sample."""
    import numpy as np
    ctx = SessionContext(jump_number_in_session=5)
    sample = BiomechanicalSample(
        dataset_name="test", trial_id="t1",
        subject=SubjectInfo(subject_id="s1"),
        fps=100.0,
        session_context=ctx,
        com_position=np.zeros((100, 3)),
    )
    window = sample.get_window(10, 30)
    assert window.session_context is ctx
