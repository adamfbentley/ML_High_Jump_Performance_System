"""Shared constants and configuration for the high jump analysis system."""

from __future__ import annotations

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Physics constants
GRAVITY_MPS2 = 9.81

# MediaPipe BlazePose landmark count
N_BLAZEPOSE_LANDMARKS = 33

# High jump specific constants
TYPICAL_BAR_HEIGHTS_CM = list(range(150, 250, 5))  # competition heights
MIN_FLIGHT_FRAMES = 5  # minimum frames to consider as flight phase

# Segment model (de Leva 1996) mass fractions - male
SEGMENT_MASS_FRACTIONS_MALE = {
    "head": 0.0694,
    "trunk": 0.4346,
    "upper_arm": 0.0271,
    "forearm_hand": 0.0228,
    "thigh": 0.1416,
    "shank": 0.0433,
    "foot": 0.0137,
}
