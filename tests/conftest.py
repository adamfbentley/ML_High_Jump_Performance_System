"""Shared test fixtures for the high jump analysis research project."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_data_dir(project_root):
    """Return path to sample test data."""
    return project_root / "tests" / "fixtures"


@pytest.fixture
def sample_anthropometrics():
    """Typical anthropometric profile for a high jumper."""
    return {
        "height_cm": 193.0,
        "weight_kg": 82.0,
        "leg_length_cm": 96.0,
        "arm_span_cm": 198.0,
        "standing_reach_cm": 252.0,
        "gender": "male",
        "age_years": 24,
    }
