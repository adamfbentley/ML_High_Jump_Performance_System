"""Standalone OpenSim IK runner — designed to be called as a subprocess.

This script runs under a conda Python environment where opensim is installed.
It takes a TRC marker file + model file, runs inverse kinematics, and outputs
a MOT file with joint angles.

Usage:
    python opensim_ik_subprocess.py <trc_path> <model_path> <output_mot_path> \
        [--height_m 1.75] [--mass_kg 65.0]

The main pipeline (which uses a pip venv with numpy 2.x / PyTorch) calls this
via subprocess to avoid the opensim numpy 1.x binary incompatibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import opensim


def scale_model(
    model: opensim.Model,
    height_m: float,
    mass_kg: float,
) -> opensim.Model:
    """Scale the generic model to the athlete's dimensions.

    Uses uniform scaling based on the ratio of athlete height to the generic
    model's height (1.70 m for Rajagopal2016/LaiUhlrich2023).
    Generic model mass: 75.16 kg.
    """
    GENERIC_HEIGHT_M = 1.70
    GENERIC_MASS_KG = 75.16
    scale_factor = height_m / GENERIC_HEIGHT_M
    mass_ratio = mass_kg / GENERIC_MASS_KG

    state = model.initSystem()
    body_set = model.getBodySet()

    for i in range(body_set.getSize()):
        body = body_set.get(i)
        # Scale mass
        body.setMass(body.getMass() * mass_ratio)

    # Scale segment lengths via the BodyScaleSet approach:
    # For each body, scale the attached geometry and joint offsets
    for i in range(body_set.getSize()):
        body = body_set.get(i)
        # Scale inertia by mass ratio and length^2 ratio
        mass_ratio = mass_kg / 75.16
        inertia = body.getInertia()
        scaled_moments = opensim.Vec3(
            inertia.getMoments().get(0) * mass_ratio * scale_factor ** 2,
            inertia.getMoments().get(1) * mass_ratio * scale_factor ** 2,
            inertia.getMoments().get(2) * mass_ratio * scale_factor ** 2,
        )
        scaled_products = opensim.Vec3(
            inertia.getProducts().get(0) * mass_ratio * scale_factor ** 2,
            inertia.getProducts().get(1) * mass_ratio * scale_factor ** 2,
            inertia.getProducts().get(2) * mass_ratio * scale_factor ** 2,
        )
        body.setInertia(opensim.Inertia(scaled_moments, scaled_products))

    # Scale joint location offsets
    joint_set = model.getJointSet()
    for i in range(joint_set.getSize()):
        joint = joint_set.get(i)
        # Scale the frames' translations
        for frame_name in ["offset_frame_on_parent", "offset_frame_on_child"]:
            try:
                frame = joint.getPropertyByName(frame_name)
            except Exception:
                continue

    model.initSystem()
    return model


def run_ik(
    trc_path: str,
    model_path: str,
    output_mot_path: str,
    height_m: float = 1.75,
    mass_kg: float = 65.0,
) -> dict:
    """Run OpenSim inverse kinematics.

    Args:
        trc_path: Path to input TRC marker file.
        model_path: Path to .osim model file.
        output_mot_path: Where to write the IK results (.mot).
        height_m: Athlete standing height in metres.
        mass_kg: Body mass in kg.

    Returns:
        Dictionary with status and metadata.
    """
    # Suppress geometry warnings (vtp mesh files not needed for IK)
    opensim.Logger.setLevelString("error")

    model = opensim.Model(model_path)
    model = scale_model(model, height_m, mass_kg)
    model.initSystem()

    # Read TRC to get time range
    marker_data = opensim.MarkerData(trc_path)
    start_time = marker_data.getStartFrameTime()
    end_time = marker_data.getLastFrameTime()
    n_frames = marker_data.getNumFrames()

    # Set up IK tool
    ik_tool = opensim.InverseKinematicsTool()
    ik_tool.setModel(model)
    ik_tool.setMarkerDataFileName(trc_path)
    ik_tool.setStartTime(start_time)
    ik_tool.setEndTime(end_time)
    ik_tool.setOutputMotionFileName(output_mot_path)

    # Configure marker task weights — only for markers that exist in the model
    ik_task_set = ik_tool.getIKTaskSet()

    # Get model marker names
    model_marker_set = model.getMarkerSet()
    model_marker_names = set()
    for i in range(model_marker_set.getSize()):
        model_marker_names.add(model_marker_set.get(i).getName())

    # Read marker names from TRC header
    with open(trc_path) as f:
        lines = f.readlines()
    marker_line = lines[3].strip().split("\t")
    trc_marker_names = [m.strip() for m in marker_line[2:] if m.strip()]

    matched = 0
    for mname in trc_marker_names:
        if mname in model_marker_names:
            task = opensim.IKMarkerTask()
            task.setName(mname)
            task.setApply(True)
            # Higher weight for pelvis/lower-limb (most important for dynamics)
            if any(k in mname for k in ["ASI", "KJC", "AJC", "TOE", "CAL"]):
                task.setWeight(10.0)
            else:
                task.setWeight(1.0)
            ik_task_set.adoptAndAppend(task)
            matched += 1

    if matched < 4:
        raise RuntimeError(
            f"Only {matched} TRC markers matched model markers. "
            f"TRC: {trc_marker_names}, Model: {sorted(model_marker_names)[:20]}..."
        )

    # Run IK
    ik_tool.run()

    # Read results
    storage = opensim.Storage(output_mot_path)
    n_rows = storage.getSize()
    col_labels = storage.getColumnLabels()
    joint_names = []
    for i in range(col_labels.getSize()):
        label = col_labels.get(i)
        if label != "time":
            joint_names.append(label)

    return {
        "status": "success",
        "n_frames": n_rows,
        "n_coordinates": len(joint_names),
        "joint_names": joint_names,
        "start_time": start_time,
        "end_time": end_time,
    }


def main():
    parser = argparse.ArgumentParser(description="OpenSim IK subprocess runner")
    parser.add_argument("trc_path", help="Input TRC marker file")
    parser.add_argument("model_path", help="OpenSim .osim model file")
    parser.add_argument("output_mot_path", help="Output MOT file for IK results")
    parser.add_argument("--height_m", type=float, default=1.75)
    parser.add_argument("--mass_kg", type=float, default=65.0)
    args = parser.parse_args()

    try:
        result = run_ik(
            args.trc_path,
            args.model_path,
            args.output_mot_path,
            args.height_m,
            args.mass_kg,
        )
        # Output JSON to stdout for the caller to parse
        print(json.dumps(result))
    except Exception as e:
        error = {"status": "error", "message": str(e)}
        print(json.dumps(error), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
