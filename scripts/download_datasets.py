"""Dataset download helper.

Prints instructions for downloading each public dataset and verifies
which datasets are already present locally.

Usage:
    python scripts/download_datasets.py              # check status of all
    python scripts/download_datasets.py --verify     # verify file counts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.registry import list_datasets, PUBLIC_DATA_DIR


# ── Download Instructions ──────────────────────────────────────────────

DOWNLOAD_INSTRUCTIONS = {
    "addbiomechanics": """
  ┌─────────────────────────────────────────────────────────────────┐
  │  AddBiomechanics (BEST FOR PINN PRE-TRAINING)                  │
  │                                                                 │
  │  1. Go to: https://addbiomechanics.org/download_data.html       │
  │  2. Create a free account and log in                            │
  │                                                                 │
  │  WINDOWS (recommended — no extra packages needed):              │
  │    • Download "OpenSim Results" export for each subject         │
  │    • Extracts to: subject_001/ with IK/, ID/, GRF/ subfolders   │
  │    • Place subject folders in data/public/addbiomechanics/      │
  │    • Layout: addbiomechanics/subject_001/IK/trial.mot           │
  │                                                                 │
  │  LINUX / macOS:                                                 │
  │    • Download .b3d files (richer data, binary format)           │
  │    • Place in data/public/addbiomechanics/*.b3d                 │
  │    • Requires: pip install nimblephysics                        │
  │                                                                 │
  │  Start with subjects tagged "drop_jump", "cmj" or "squat_jump"  │
  └─────────────────────────────────────────────────────────────────┘""",

    "biocv": """
  ┌─────────────────────────────────────────────────────────────────┐
  │  BioCV (VIDEO + MARKERS + FORCES)                              │
  │                                                                 │
  │  1. Paper: https://doi.org/10.1038/s41597-024-03463-1           │
  │  2. Data repository linked in the paper's Data Availability     │
  │  3. Download C3D files + video files                            │
  │  4. Place them in:                                              │
  │     data/public/biocv/                                          │
  │     Organize by subject: biocv/Subject01/CMJ/trial.c3d          │
  │                                                                 │
  │  Required Python package: pip install ezc3d                     │
  └─────────────────────────────────────────────────────────────────┘""",

    "opencap": """
  ┌─────────────────────────────────────────────────────────────────┐
  │  OpenCap (MARKERLESS VIDEO → KINEMATICS)                       │
  │                                                                 │
  │  1. Validation data: https://simtk.org/projects/opencap         │
  │  2. Download .trc and .mot files                                │
  │  3. Place them in:                                              │
  │     data/public/opencap/                                        │
  │                                                                 │
  │  Also see: https://opencap.ai for the platform itself           │
  │  No extra packages needed (text file parsing only).             │
  └─────────────────────────────────────────────────────────────────┘""",

    "athletepose3d": """
  ┌─────────────────────────────────────────────────────────────────┐
  │  AthletePose3D (SPORTS POSE ESTIMATION)                        │
  │                                                                 │
  │  1. GitHub: https://github.com/calvinyeungck/AthletePose3D      │
  │  2. Follow the download instructions in the repo README         │
  │  3. Place annotation JSONs in:                                  │
  │     data/public/athletepose3d/                                  │
  │                                                                 │
  │  No extra packages needed.                                      │
  └─────────────────────────────────────────────────────────────────┘""",

    "vertical_jump_imu": """
  ┌─────────────────────────────────────────────────────────────────┐
  │  Vertical Jump IMU (SUPPLEMENTARY)                             │
  │                                                                 │
  │  1. Zenodo: search "vertical jump inertial optical motion"      │
  │  2. Download C3D and CSV files                                  │
  │  3. Place them in:                                              │
  │     data/public/vertical_jump_imu/                              │
  │                                                                 │
  │  Uses same C3D loader as BioCV: pip install ezc3d               │
  └─────────────────────────────────────────────────────────────────┘""",
}


def check_dataset_status() -> dict[str, dict]:
    """Check which datasets are downloaded and count files."""
    status = {}

    for info in list_datasets():
        local_dir = info.local_dir
        exists = local_dir.exists()

        file_counts = {}
        if exists:
            for fmt in info.file_formats:
                files = list(local_dir.glob(f"**/*{fmt}"))
                file_counts[fmt] = len(files)

        total_files = sum(file_counts.values())

        status[info.name] = {
            "exists": exists,
            "path": str(local_dir),
            "file_counts": file_counts,
            "total_files": total_files,
            "priority": info.priority,
            "description": info.description[:80] + "...",
        }

    return status


def main():
    parser = argparse.ArgumentParser(description="Dataset download helper")
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify file counts for downloaded datasets",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Show instructions for a specific dataset only",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  PUBLIC DATASET DOWNLOAD HELPER")
    print("  High Jump Biomechanics Pre-Training Data")
    print("=" * 70)

    status = check_dataset_status()

    # Ensure the public data directory exists
    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        datasets_to_show = [args.dataset]
    else:
        datasets_to_show = list(status.keys())

    for name in datasets_to_show:
        info = status.get(name)
        if info is None:
            print(f"\n  Unknown dataset: {name}")
            continue

        # Status indicator
        if info["total_files"] > 0:
            indicator = f"✓ READY ({info['total_files']} files)"
        elif info["exists"]:
            indicator = "⚠ Directory exists but no data files found"
        else:
            indicator = "✗ NOT DOWNLOADED"

        print(f"\n  [{info['priority']:2d}] {name}: {indicator}")

        if args.verify and info["file_counts"]:
            for fmt, count in info["file_counts"].items():
                print(f"       {fmt}: {count} files")

        if info["total_files"] == 0:
            instructions = DOWNLOAD_INSTRUCTIONS.get(name, "  No instructions available.")
            print(instructions)

    # Summary
    ready = sum(1 for s in status.values() if s["total_files"] > 0)
    total = len(status)
    print(f"\n{'=' * 70}")
    print(f"  Status: {ready}/{total} datasets ready")
    print(f"  Data directory: {PUBLIC_DATA_DIR}")

    if ready == 0:
        print("\n  RECOMMENDED FIRST STEP:")
        print("  Download AddBiomechanics — it has the richest dynamics data")
        print("  for pre-training PINNs (joint torques, GRF, CoM trajectories).")
        print("  Then run: python scripts/pretrain_dynamics_pinn.py")

    print()


if __name__ == "__main__":
    main()
