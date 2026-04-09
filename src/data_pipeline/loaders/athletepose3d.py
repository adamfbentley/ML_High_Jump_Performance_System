"""AthletePose3D dataset loader.

Loads 3D pose annotations from the AthletePose3D dataset for pre-training
pose estimation and 2D→3D lifting models on athletic movements.

This dataset provides ~1.3M frames of high-speed sports motion with
monocular 3D pose annotations — ideal for training robust pose extractors
on fast, dynamic movements like run-ups and takeoffs.

Requirements:
    No special dependencies beyond numpy.

Download:
    https://github.com/calvinyeungck/AthletePose3D
    Place files in data/public/athletepose3d/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Generator

import numpy as np

from src.data_pipeline.sample import (
    BiomechanicalSample,
    MovementType,
    SubjectInfo,
)

logger = logging.getLogger(__name__)

# AthletePose3D uses a COCO-like 17-joint skeleton
ATHLETEPOSE_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Sports keywords → movement types
_SPORT_MAP = {
    "track": MovementType.SPRINTING,
    "sprint": MovementType.SPRINTING,
    "running": MovementType.RUNNING,
    "high_jump": MovementType.HIGH_JUMP,
    "long_jump": MovementType.OTHER,
    "jump": MovementType.VERTICAL_JUMP,
    "figure_skating": MovementType.OTHER,
    "skating": MovementType.OTHER,
    "gymnastics": MovementType.OTHER,
}


def _classify_sport(name: str) -> MovementType:
    name_lower = name.lower()
    for keyword, mtype in _SPORT_MAP.items():
        if keyword in name_lower:
            return mtype
    return MovementType.OTHER


def load_athletepose3d_annotations(
    annotation_file: Path,
) -> list[BiomechanicalSample]:
    """Load annotations from a COCO-format JSON file.

    AthletePose3D annotations follow this structure:
    {
        "images": [{id, file_name, width, height, ...}],
        "annotations": [{
            "id", "image_id",
            "keypoints": [x1,y1,v1, x2,y2,v2, ...],  // 2D
            "keypoints_3d": [x1,y1,z1, x2,y2,z2, ...],  // 3D (if available)
            "category_id",
            ...
        }],
        "categories": [{"id", "name", "keypoints", ...}]
    }

    Frames from the same video sequence are grouped into samples.

    Args:
        annotation_file: Path to the COCO-format JSON annotation file.

    Returns:
        List of BiomechanicalSample grouped by video sequence.
    """
    annotation_file = Path(annotation_file)
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    logger.info(f"Loading annotations from {annotation_file.name}...")

    with open(annotation_file, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data.get("images", [])}
    annotations = data.get("annotations", [])

    # Group annotations by video/sequence
    # Try to extract sequence from image filename (e.g., "video001_frame042.jpg")
    sequences: dict[str, list[tuple[int, dict, dict]]] = {}
    for ann in annotations:
        img = images.get(ann.get("image_id"))
        if img is None:
            continue

        filename = img.get("file_name", "")
        # Heuristic: group by everything before the last underscore or number sequence
        parts = Path(filename).stem.rsplit("_", 1)
        seq_id = parts[0] if len(parts) > 1 else Path(filename).stem

        # Extract frame number for ordering
        frame_num = 0
        if len(parts) > 1:
            try:
                frame_num = int("".join(c for c in parts[1] if c.isdigit()) or "0")
            except ValueError:
                pass

        if seq_id not in sequences:
            sequences[seq_id] = []
        sequences[seq_id].append((frame_num, ann, img))

    samples = []
    for seq_id, frame_data in sequences.items():
        # Sort by frame number
        frame_data.sort(key=lambda x: x[0])

        n_frames = len(frame_data)
        if n_frames < 2:
            continue

        n_joints = len(ATHLETEPOSE_JOINT_NAMES)
        pose_2d = np.zeros((n_frames, n_joints, 3))  # x, y, confidence
        pose_3d = None
        frame_paths = []

        has_3d = False
        for i, (_, ann, img) in enumerate(frame_data):
            # 2D keypoints
            kpts = ann.get("keypoints", [])
            if len(kpts) >= n_joints * 3:
                pose_2d[i] = np.array(kpts[:n_joints * 3]).reshape(n_joints, 3)

            # 3D keypoints (if available)
            kpts_3d = ann.get("keypoints_3d", [])
            if len(kpts_3d) >= n_joints * 3:
                if pose_3d is None:
                    pose_3d = np.zeros((n_frames, n_joints, 3))
                pose_3d[i] = np.array(kpts_3d[:n_joints * 3]).reshape(n_joints, 3)
                has_3d = True

            frame_paths.append(img.get("file_name", ""))

        # Estimate fps from frame numbers if available
        frame_nums = [fd[0] for fd in frame_data]
        if len(set(frame_nums)) > 1:
            # Assume constant frame rate, estimate from frame number spacing
            diffs = np.diff(frame_nums)
            median_diff = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 1
            fps = 30.0 / median_diff  # assume 30fps video
        else:
            fps = 30.0

        # If 3D poses are in mm, convert to meters
        if pose_3d is not None and np.nanmax(np.abs(pose_3d)) > 10:
            pose_3d = pose_3d / 1000.0

        sample = BiomechanicalSample(
            dataset_name="athletepose3d",
            trial_id=seq_id,
            subject=SubjectInfo(subject_id=seq_id.split("_")[0]),
            movement_type=_classify_sport(seq_id),
            fps=fps,
            pose_2d=pose_2d,
            pose_3d=pose_3d if has_3d else None,
            pose_landmark_names=ATHLETEPOSE_JOINT_NAMES,
            frame_paths=frame_paths,
        )

        samples.append(sample)

    logger.info(f"Loaded {len(samples)} sequences ({sum(s.n_frames for s in samples)} frames)")
    return samples


def load_athletepose3d(
    data_dir: Path | str | None = None,
    movement_filter: list[MovementType] | None = None,
) -> Generator[BiomechanicalSample, None, None]:
    """Iterate over all AthletePose3D samples.

    Args:
        data_dir: Root directory. Defaults to data/public/athletepose3d/.
        movement_filter: Only yield matching movement types.

    Yields:
        BiomechanicalSample for each video sequence.
    """
    if data_dir is None:
        from src.data_pipeline.registry import DATASET_REGISTRY
        data_dir = DATASET_REGISTRY["athletepose3d"].local_dir

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"AthletePose3D data directory not found: {data_dir}\n"
            f"Download from: https://github.com/calvinyeungck/AthletePose3D"
        )

    # Find annotation JSON files
    json_files = sorted(data_dir.glob("**/*.json"))
    annotation_files = [
        f for f in json_files
        if "annotation" in f.stem.lower() or "keypoint" in f.stem.lower()
        or f.parent.name == "annotations"
    ]

    # If none match the name pattern, try all JSON files
    if not annotation_files:
        annotation_files = json_files

    if not annotation_files:
        raise FileNotFoundError(f"No annotation JSON files found in {data_dir}")

    for ann_file in annotation_files:
        try:
            samples = load_athletepose3d_annotations(ann_file)
        except Exception as e:
            logger.warning(f"Failed to load {ann_file}: {e}")
            continue

        for sample in samples:
            if movement_filter and sample.movement_type not in movement_filter:
                continue
            yield sample
