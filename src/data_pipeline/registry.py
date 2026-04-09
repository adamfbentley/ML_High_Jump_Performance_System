"""Public dataset registry for pre-training.

Central catalog of all supported datasets with download info, expected
file formats, and loader references. Each entry describes what's available
and how to obtain it — the actual loading logic is in loaders/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.utils.constants import DATA_DIR


# Default root for downloaded public datasets
PUBLIC_DATA_DIR = DATA_DIR / "public"


@dataclass
class DatasetInfo:
    """Metadata for a public biomechanics dataset."""

    name: str
    description: str
    url: str
    license: str
    citation: str

    # What modalities are available
    has_kinematics: bool = False
    has_dynamics: bool = False
    has_video: bool = False
    has_3d_poses: bool = False
    has_2d_poses: bool = False
    has_grf: bool = False
    has_joint_torques: bool = False
    has_com: bool = False

    # Expected file formats
    file_formats: list[str] = field(default_factory=list)

    # Number of subjects/trials (approximate)
    n_subjects_approx: int = 0
    n_hours_approx: float = 0.0

    # Which movement types are included
    movement_types: list[str] = field(default_factory=list)

    # Pre-training priority (higher = more useful for high jump)
    priority: int = 0

    @property
    def local_dir(self) -> Path:
        """Expected local directory for this dataset."""
        return PUBLIC_DATA_DIR / self.name


# ── Dataset Registry ────────────────────────────────────────────────────

DATASET_REGISTRY: dict[str, DatasetInfo] = {
    "addbiomechanics": DatasetInfo(
        name="addbiomechanics",
        description=(
            "Largest publicly available dataset with physically accurate human "
            "dynamics. Includes inverse dynamics, joint torques, GRF, and scaled "
            "musculoskeletal models across 273 subjects and 70+ hours of data."
        ),
        url="https://addbiomechanics.org/download_data.html",
        license="CC BY 4.0",
        citation=(
            "Werling et al. (2023). AddBiomechanics: Automating model scaling, "
            "inverse kinematics, and inverse dynamics from human motion data "
            "through sequential optimization. PLOS ONE."
        ),
        has_kinematics=True,
        has_dynamics=True,
        has_grf=True,
        has_joint_torques=True,
        has_com=True,
        file_formats=[".b3d"],
        n_subjects_approx=273,
        n_hours_approx=70,
        movement_types=["drop_jump", "squat", "running", "walking", "other"],
        priority=10,
    ),

    "biocv": DatasetInfo(
        name="biocv",
        description=(
            "Synchronized multi-camera video + marker-based motion capture + "
            "force plate data for 15 participants performing walking, running, "
            "countermovement jumps, and hopping."
        ),
        url="https://doi.org/10.1038/s41597-024-03463-1",
        license="CC BY 4.0",
        citation=(
            "BioCV: Synchronized video, motion capture and force plate dataset "
            "for the evaluation of markerless motion capture. "
            "Nature Scientific Data (2024)."
        ),
        has_kinematics=True,
        has_dynamics=True,
        has_video=True,
        has_grf=True,
        has_3d_poses=True,
        file_formats=[".c3d", ".mp4", ".csv"],
        n_subjects_approx=15,
        n_hours_approx=2,
        movement_types=["walking", "running", "cmj", "hopping"],
        priority=9,
    ),

    "opencap": DatasetInfo(
        name="opencap",
        description=(
            "Open-source markerless system using smartphone videos to estimate "
            "3D kinematics and dynamics. Validation dataset includes synchronized "
            "video + lab mocap + force plates for jumping tasks."
        ),
        url="https://opencap.ai",
        license="Research use",
        citation=(
            "Uhlrich et al. (2023). OpenCap: Human movement dynamics from "
            "smartphone videos. PLOS Computational Biology."
        ),
        has_kinematics=True,
        has_dynamics=True,
        has_video=True,
        has_grf=True,
        has_3d_poses=True,
        file_formats=[".trc", ".mot", ".mp4"],
        n_subjects_approx=100,
        n_hours_approx=5,
        movement_types=["drop_jump", "squat", "walking", "running"],
        priority=8,
    ),

    "athletepose3d": DatasetInfo(
        name="athletepose3d",
        description=(
            "~1.3 million frames of high-speed athletic motions including "
            "track & field and jumps. Designed for monocular 3D pose estimation "
            "in sports contexts with fast, dynamic movements."
        ),
        url="https://github.com/calvinyeungck/AthletePose3D",
        license="Research use",
        citation=(
            "AthletePose3D: 3D Human Pose Estimation for Sports Broadcasting."
        ),
        has_video=True,
        has_3d_poses=True,
        has_2d_poses=True,
        file_formats=[".json", ".mp4"],
        n_subjects_approx=50,
        n_hours_approx=10,
        movement_types=["sprinting", "jumping", "figure_skating", "other"],
        priority=7,
    ),

    "vertical_jump_imu": DatasetInfo(
        name="vertical_jump_imu",
        description=(
            "Vertical jump data combining optical mocap and inertial sensors. "
            "Good for multi-modal pre-training on jump metrics."
        ),
        url="https://zenodo.org/records/vertical-jump-inertial-optical",
        license="CC BY 4.0",
        citation="Vertical Jump Inertial and Optical Motion Capture Dataset.",
        has_kinematics=True,
        has_3d_poses=True,
        file_formats=[".c3d", ".csv"],
        n_subjects_approx=20,
        n_hours_approx=1,
        movement_types=["vertical_jump", "cmj"],
        priority=6,
    ),
}


def get_dataset_info(name: str) -> DatasetInfo:
    """Look up a dataset by name."""
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name]


def list_datasets(sort_by_priority: bool = True) -> list[DatasetInfo]:
    """List all registered datasets, optionally sorted by priority."""
    datasets = list(DATASET_REGISTRY.values())
    if sort_by_priority:
        datasets.sort(key=lambda d: d.priority, reverse=True)
    return datasets


def list_dynamics_datasets() -> list[DatasetInfo]:
    """List datasets that include dynamics data (GRF, torques)."""
    return [d for d in list_datasets() if d.has_dynamics]


def list_pose_datasets() -> list[DatasetInfo]:
    """List datasets that include 2D or 3D pose data."""
    return [d for d in list_datasets() if d.has_3d_poses or d.has_2d_poses]
