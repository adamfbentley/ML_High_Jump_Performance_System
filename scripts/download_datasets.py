"""Dataset download helper.

Automatically downloads open-access datasets from Zenodo.
Prints step-by-step instructions for datasets that require registration.

Usage:
    python scripts/download_datasets.py              # check status + auto-download
    python scripts/download_datasets.py --auto       # download all Zenodo datasets
    python scripts/download_datasets.py --dataset cmj_grf_zenodo --auto
    python scripts/download_datasets.py --verify     # verify file counts
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.registry import list_datasets, PUBLIC_DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── Zenodo record definitions (auto-downloadable, CC-BY-4.0) ──────────

# Each entry: (zenodo_record_id, filename_on_zenodo, expected_md5, size_bytes)
# Verified from the Zenodo API (queried 2025).
ZENODO_DOWNLOADS: dict[str, list[tuple[str, str, str | None, int]]] = {
    # Preprocessed CMJ accelerometer + vertical GRF (Python NumPy .npz)
    # White (2026). Zenodo 19136480. CC-BY-4.0. 67 subj, 663 trials.
    "cmj_grf_zenodo": [
        (
            "19136480",
            "cmj_dataset_both.npz",
            "6435156f28f083387a6fff415cbd2f21",
            18_070_947,
        ),
    ],

    # Drop vertical jump — whole-body kinematics + GRF + EMG (OpenSim + C3D)
    # Zang & Wu (2026). Zenodo 18503500. CC-BY-4.0. 28 subjects.
    "dvj_opensim_zenodo": [
        (
            "18503500",
            "Data.zip",
            "a3b7e2c0ed48d727f434dfa8bfae380d",
            1_584_061_332,
        ),
    ],

    # Change-of-direction IK/ID + optimal control results (OpenSim + GRF)
    # Nitschke et al. (2022). Zenodo 6949012. CC-BY-4.0. 30 trials.
    "cod_ik_id_zenodo": [
        (
            "6949012",
            "Nitschke_et_al_Change_of_Direction_Marker_Simulation.zip",
            "a9a6e82269beaf8b3d444bc6b4b9ea61",
            73_192_918,
        ),
    ],
}


# ── Manual-download instructions ──────────────────────────────────────

MANUAL_INSTRUCTIONS: dict[str, str] = {
    "addbiomechanics": """
  ┌─────────────────────────────────────────────────────────────────┐
  │  AddBiomechanics  [HIGHEST PRIORITY — requires free account]   │
  │                                                                 │
  │  1. Register (free): https://addbiomechanics.org               │
  │  2. Go to: https://addbiomechanics.org/download_data.html       │
  │  3. Filter subjects by task: "drop_jump", "CMJ", "squat_jump"   │
  │  4. Download "OpenSim Results" export (zip, no extra installs)  │
  │     Each subject unzips to:  subject_001/IK/*.mot               │
  │                              subject_001/ID/*.sto               │
  │                              subject_001/GRF/*.mot              │
  │                              subject_001/bodyKinematics/*.sto   │
  │  5. Place subject folders in: data/public/addbiomechanics/      │
  │     i.e., data/public/addbiomechanics/subject_001/IK/trial.mot  │
  │                                                                 │
  │  273 subjects, 70+ hours — best PINN pre-training source.       │
  └─────────────────────────────────────────────────────────────────┘""",

    "biocv": """
  ┌─────────────────────────────────────────────────────────────────┐
  │  BioCV  [video + markers + force plates]                       │
  │                                                                 │
  │  1. Paper DOI: https://doi.org/10.1038/s41597-024-03463-1       │
  │  2. Follow "Data Availability" link in the paper for C3D files  │
  │  3. Place C3D files in: data/public/biocv/                      │
  │     Suggested layout: biocv/Subject01/CMJ/trial.c3d             │
  │                                                                 │
  │  Requires: pip install ezc3d                                    │
  └─────────────────────────────────────────────────────────────────┘""",

    "opencap": """
  ┌─────────────────────────────────────────────────────────────────┐
  │  OpenCap  [markerless → kinematics, requires free account]     │
  │                                                                 │
  │  1. Register: https://opencap.ai                               │
  │  2. SimTK validation data: https://simtk.org/projects/opencap   │
  │  3. Download .trc + .mot files                                  │
  │  4. Place in: data/public/opencap/                              │
  │                                                                 │
  │  No extra packages needed (text-file parsing only).             │
  └─────────────────────────────────────────────────────────────────┘""",

    "athletepose3d": """
  ┌─────────────────────────────────────────────────────────────────┐
  │  AthletePose3D  [sports pose, requires license agreement]      │
  │                                                                 │
  │  1. GitHub: https://github.com/calvinyeungck/AthletePose3D      │
  │  2. Sign license agreement, then download annotation JSONs      │
  │  3. Place in: data/public/athletepose3d/                        │
  └─────────────────────────────────────────────────────────────────┘""",
}


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    """Return hex MD5 of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download_zenodo_file(
    record_id: str,
    filename: str,
    dest_path: Path,
    expected_md5: str | None = None,
) -> bool:
    """Download a single file from Zenodo using requests with a progress bar.

    Uses the public Zenodo API — no authentication required for open-access
    records. Skips the download if the file already exists and the MD5 matches.

    Returns True on success, False on failure.
    """
    try:
        import requests
    except ImportError:
        logger.error("requests is not installed. Run: pip install requests")
        return False

    url = f"https://zenodo.org/api/records/{record_id}/files/{filename}/content"
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded and intact
    if dest_path.exists() and expected_md5:
        local_md5 = _md5(dest_path)
        if local_md5 == expected_md5:
            logger.info(f"  ✓ {filename} already downloaded and verified")
            return True
        logger.warning(f"  MD5 mismatch for {filename} — re-downloading")

    logger.info(f"  Downloading {filename} from Zenodo record {record_id} …")
    logger.info(f"  URL: {url}")

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))

            # Write to a temporary file, then move on success to avoid
            # partially-written files being mistaken for complete downloads.
            tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
            downloaded = 0
            last_report = time.time()

            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if now - last_report > 5:
                            if total:
                                pct = 100 * downloaded / total
                                mib = downloaded / (1 << 20)
                                total_mib = total / (1 << 20)
                                logger.info(
                                    f"    {mib:.0f} / {total_mib:.0f} MiB  ({pct:.0f}%)"
                                )
                            last_report = now

            # Verify MD5 before committing
            if expected_md5:
                actual_md5 = _md5(tmp_path)
                if actual_md5 != expected_md5:
                    logger.error(
                        f"  MD5 mismatch after download: expected {expected_md5}, "
                        f"got {actual_md5}. Deleting partial file."
                    )
                    tmp_path.unlink(missing_ok=True)
                    return False

            tmp_path.rename(dest_path)
            mib = dest_path.stat().st_size / (1 << 20)
            logger.info(f"  ✓ Saved {filename} ({mib:.1f} MiB)")
            return True

    except Exception as exc:
        logger.error(f"  ✗ Download failed: {exc}")
        Path(str(dest_path) + ".tmp").unlink(missing_ok=True)
        return False


def _extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """Extract a zip archive into dest_dir. Returns True on success."""
    if not zip_path.exists():
        return False
    logger.info(f"  Extracting {zip_path.name} → {dest_dir.name}/")
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest_dir)
        logger.info(f"  ✓ Extraction complete")
        return True
    except Exception as exc:
        logger.error(f"  ✗ Extraction failed: {exc}")
        return False


def download_zenodo_dataset(name: str, dest_dir: Path) -> bool:
    """Download all files for a Zenodo dataset and extract any zips.

    Returns True if all files were successfully obtained.
    """
    if name not in ZENODO_DOWNLOADS:
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for record_id, filename, expected_md5, _ in ZENODO_DOWNLOADS[name]:
        dest_path = dest_dir / filename
        ok = _download_zenodo_file(record_id, filename, dest_path, expected_md5)
        if not ok:
            all_ok = False
            continue

        # Extract zip archives after download
        if dest_path.suffix.lower() == ".zip" and dest_path.exists():
            _extract_zip(dest_path, dest_dir)

    return all_ok


def check_dataset_status() -> dict[str, dict]:
    """Check which datasets are downloaded and count data files."""
    status = {}

    for info in list_datasets():
        local_dir = info.local_dir
        exists = local_dir.exists()

        file_counts: dict[str, int] = {}
        if exists:
            for fmt in info.file_formats:
                files = list(local_dir.glob(f"**/*{fmt}"))
                file_counts[fmt] = len(files)

        total_files = sum(file_counts.values())
        auto_available = info.name in ZENODO_DOWNLOADS

        status[info.name] = {
            "exists": exists,
            "path": str(local_dir),
            "file_counts": file_counts,
            "total_files": total_files,
            "priority": info.priority,
            "description": info.description[:80] + "...",
            "auto_available": auto_available,
        }

    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset download helper")
    parser.add_argument(
        "--auto", action="store_true",
        help="Automatically download all Zenodo (no-auth) datasets",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify file counts for downloaded datasets",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Target a specific dataset name only",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  PUBLIC DATASET DOWNLOAD HELPER")
    print("  High Jump Biomechanics Pre-Training Data")
    print("=" * 70)

    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Auto-download tier (Zenodo, no auth required) ────────────────
    if args.auto:
        targets = [args.dataset] if args.dataset else list(ZENODO_DOWNLOADS.keys())
        print(f"\n  Auto-downloading {len(targets)} Zenodo dataset(s)...\n")
        for name in targets:
            dest = PUBLIC_DATA_DIR / name
            print(f"  [{name}]")
            ok = download_zenodo_dataset(name, dest)
            if not ok:
                print(f"  ✗ {name} download incomplete — check logs above.\n")
            else:
                print(f"  ✓ {name} ready in {dest}\n")

    # ── Status report ────────────────────────────────────────────────
    status = check_dataset_status()

    datasets_to_show = (
        [args.dataset] if (args.dataset and not args.auto) else list(status.keys())
    )

    print()
    for name in datasets_to_show:
        info = status.get(name)
        if info is None:
            print(f"  Unknown dataset: {name}")
            continue

        if info["total_files"] > 0:
            indicator = f"✓ READY ({info['total_files']} files)"
        elif info["exists"]:
            indicator = "⚠  Directory exists but no data files found"
        else:
            auto_tag = " [auto-downloadable]" if info["auto_available"] else ""
            indicator = f"✗ NOT DOWNLOADED{auto_tag}"

        print(f"  [{info['priority']:2d}] {name}: {indicator}")

        if args.verify and info["file_counts"]:
            for fmt, count in info["file_counts"].items():
                print(f"       {fmt}: {count} files")

        # Only print manual instructions for datasets not yet downloaded
        # and not on the Zenodo auto-download list.
        if info["total_files"] == 0 and not info["auto_available"]:
            instructions = MANUAL_INSTRUCTIONS.get(name)
            if instructions:
                print(instructions)

    # ── Summary ──────────────────────────────────────────────────────
    ready = sum(1 for s in status.values() if s["total_files"] > 0)
    auto_avail = sum(1 for s in status.values() if s["auto_available"])
    total = len(status)

    print(f"\n{'=' * 70}")
    print(f"  Status: {ready}/{total} datasets ready")
    print(f"  Data directory: {PUBLIC_DATA_DIR}")

    if ready == 0 and not args.auto:
        print(f"\n  {auto_avail} dataset(s) can be downloaded automatically:")
        for name in ZENODO_DOWNLOADS:
            print(f"    • {name}")
        print("\n  Run:  python scripts/download_datasets.py --auto")
        print("\n  After downloading, pre-train the PINN:")
        print("    python scripts/pretrain_dynamics_pinn.py")

    print()


if __name__ == "__main__":
    main()
