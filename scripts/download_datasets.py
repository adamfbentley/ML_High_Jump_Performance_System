"""Dataset status checker and download instructions.

Run this to see which datasets are present and get download links.

Usage:
    python scripts/download_datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.registry import list_datasets, PUBLIC_DATA_DIR

# ── Where to get each dataset manually ───────────────────────────────

DOWNLOAD_INSTRUCTIONS: dict[str, str] = {
    "addbiomechanics": """
  WHERE:  https://addbiomechanics.org/download_data.html  (free account required)
  FORMAT: Download "OpenSim Results" export.
  LAYOUT: data/public/addbiomechanics/subject_001/IK/trial.mot
                                      subject_001/ID/trial.sto
                                      subject_001/GRF/trial.mot
                                      subject_001/bodyKinematics/trial.sto
  NOTE:   273 subjects — highest priority pre-training source.""",

    "biocv": """
  WHERE:  https://doi.org/10.1038/s41597-024-03463-1  (see Data Availability)
  FORMAT: C3D files (requires: pip install ezc3d)
  LAYOUT: data/public/biocv/Subject01/CMJ/trial.c3d""",

    "opencap": """
  WHERE:  https://simtk.org/projects/opencap  (free account required)
  FORMAT: .trc + .mot text files
  LAYOUT: data/public/opencap/<subject>/<trial>.trc""",

    "athletepose3d": """
  WHERE:  https://github.com/calvinyeungck/AthletePose3D  (license agreement)
  FORMAT: COCO JSON annotations
  LAYOUT: data/public/athletepose3d/annotations.json""",

    "cmj_grf_zenodo": """
  WHERE:  https://zenodo.org/record/19136480  (no account — direct download)
  FILE:   cmj_dataset_both.npz  (~17 MB)
  LAYOUT: data/public/cmj_grf_zenodo/cmj_dataset_both.npz""",

    "dvj_opensim_zenodo": """
  WHERE:  https://zenodo.org/record/18503500  (no account — direct download)
  FILE:   Data.zip  (~1.5 GB, extract in place)
  LAYOUT: data/public/dvj_opensim_zenodo/""",

    "cod_ik_id_zenodo": """
  WHERE:  https://zenodo.org/record/6949012  (no account — direct download)
  FILE:   Nitschke_et_al_Change_of_Direction_Marker_Simulation.zip  (~70 MB)
  LAYOUT: data/public/cod_ik_id_zenodo/""",
}


def main() -> None:
    print("\n" + "=" * 70)
    print("  DATASET STATUS — High Jump Biomechanics Pre-Training Data")
    print("=" * 70)

    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    missing = []

    for info in list_datasets():
        local_dir = info.local_dir
        file_counts = {
            fmt: len(list(local_dir.glob(f"**/*{fmt}")))
            for fmt in info.file_formats
        } if local_dir.exists() else {}
        total = sum(file_counts.values())

        if total > 0:
            print(f"  [{info.priority:2d}] {info.name}: ✓  {total} files")
        else:
            print(f"  [{info.priority:2d}] {info.name}: ✗  NOT PRESENT")
            missing.append(info.name)

    if missing:
        print(f"\n  {len(missing)} dataset(s) missing. Download instructions:\n")
        for name in missing:
            print(f"  ── {name} " + "─" * max(0, 50 - len(name)))
            instructions = DOWNLOAD_INSTRUCTIONS.get(name, "  No instructions recorded.")
            print(instructions)
            print()

    print(f"  Data directory: {PUBLIC_DATA_DIR}\n")


if __name__ == "__main__":
    main()
