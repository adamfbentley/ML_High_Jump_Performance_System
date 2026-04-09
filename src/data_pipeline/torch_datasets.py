"""PyTorch Dataset wrappers for pre-training.

Converts BiomechanicalSample streams into proper torch Datasets
that can be used with DataLoader for training PINNs and GNNs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_pipeline.sample import BiomechanicalSample


class DynamicsDataset(Dataset):
    """Dataset for pre-training dynamics PINNs (inverse dynamics, GRF prediction).

    Each item is a single time window containing:
        - input: [time, joint_angles, joint_angular_velocities, body_mass, height]
        - target_grf: ground reaction force
        - target_torques: joint torques
        - target_com_acc: center of mass acceleration

    Used with AddBiomechanics and BioCV data.
    """

    def __init__(
        self,
        samples: Sequence[BiomechanicalSample],
        window_size: int = 64,
        stride: int = 32,
        normalize_forces: bool = True,
    ):
        self.window_size = window_size
        self.normalize_forces = normalize_forces

        # Pre-process all samples into windows
        self.windows: list[dict[str, np.ndarray]] = []

        for sample in samples:
            if not sample.has_dynamics or not sample.has_kinematics:
                # Use whatever data is available for CoM-based dynamics
                if sample.com_acceleration is not None and sample.grf is not None:
                    self._add_com_windows(sample, window_size, stride)
                continue

            self._add_full_dynamics_windows(sample, window_size, stride)

    def _add_full_dynamics_windows(
        self, sample: BiomechanicalSample, window_size: int, stride: int
    ) -> None:
        """Extract windows from a sample with full dynamics data."""
        n_frames = sample.n_frames
        mass = sample.subject.body_mass_kg or 75.0
        height = sample.subject.height_m or 1.75

        for start in range(0, n_frames - window_size + 1, stride):
            end = start + window_size
            t = np.linspace(0, 1, window_size, dtype=np.float32)

            q = sample.joint_angles[start:end].astype(np.float32)
            qd = sample.joint_angular_velocities[start:end].astype(np.float32)

            # Anthropometric features (broadcast to each timestep)
            anthro = np.full((window_size, 2), [mass, height], dtype=np.float32)

            # Build input: time + joint angles + velocities + anthropometrics
            input_data = np.concatenate([
                t[:, None], q, qd, anthro
            ], axis=1)

            window = {"input": input_data}

            if sample.grf is not None:
                grf = sample.grf[start:end].astype(np.float32)
                if self.normalize_forces and mass > 0:
                    grf = grf / mass
                window["target_grf"] = grf

            if sample.joint_torques is not None:
                torques = sample.joint_torques[start:end].astype(np.float32)
                if self.normalize_forces and mass > 0:
                    torques = torques / mass
                window["target_torques"] = torques

            if sample.com_acceleration is not None:
                window["target_com_acc"] = sample.com_acceleration[start:end].astype(np.float32)

            if sample.com_position is not None:
                window["com_position"] = sample.com_position[start:end].astype(np.float32)

            # Metadata
            window["body_mass"] = np.float32(mass)
            window["relevance"] = np.float32(sample.relevance_score)

            self.windows.append(window)

    def _add_com_windows(
        self, sample: BiomechanicalSample, window_size: int, stride: int
    ) -> None:
        """Extract windows from a sample with only CoM + GRF data."""
        n_frames = sample.n_frames
        mass = sample.subject.body_mass_kg or 75.0

        for start in range(0, n_frames - window_size + 1, stride):
            end = start + window_size
            t = np.linspace(0, 1, window_size, dtype=np.float32)

            input_data = np.concatenate([
                t[:, None],
                sample.com_position[start:end].astype(np.float32),
                sample.com_velocity[start:end].astype(np.float32) if sample.com_velocity is not None else np.zeros((window_size, 3), dtype=np.float32),
            ], axis=1)

            window = {
                "input": input_data,
                "target_com_acc": sample.com_acceleration[start:end].astype(np.float32),
                "body_mass": np.float32(mass),
                "relevance": np.float32(sample.relevance_score),
            }

            if sample.grf is not None:
                grf = sample.grf[start:end].astype(np.float32)
                if self.normalize_forces and mass > 0:
                    grf = grf / mass
                window["target_grf"] = grf

            self.windows.append(window)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        window = self.windows[idx]
        return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v)
                for k, v in window.items()}


class FlightPhaseDataset(Dataset):
    """Dataset for pre-training the projectile PINN on real CoM flight trajectories.

    Extracts segments where both feet are off the ground (vertical GRF ≈ 0),
    perfect for enforcing d²y/dt² = -g.

    Used with AddBiomechanics and BioCV jump trials.
    """

    def __init__(
        self,
        samples: Sequence[BiomechanicalSample],
        min_flight_frames: int = 5,
        grf_threshold_n: float = 20.0,
    ):
        self.trajectories: list[dict[str, np.ndarray]] = []

        for sample in samples:
            if sample.com_position is None:
                continue

            # Detect flight phases from GRF
            if sample.grf is not None:
                grf_magnitude = np.linalg.norm(sample.grf, axis=1)
                in_air = grf_magnitude < grf_threshold_n
            elif sample.com_acceleration is not None:
                # No GRF — estimate flight as when vertical acc ≈ -g
                vertical_acc = sample.com_acceleration[:, 1]
                in_air = np.abs(vertical_acc + 9.81) < 2.0
            else:
                continue

            # Find contiguous flight segments
            changes = np.diff(in_air.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1

            if in_air[0]:
                starts = np.concatenate([[0], starts])
            if in_air[-1]:
                ends = np.concatenate([ends, [len(in_air)]])

            for s, e in zip(starts, ends):
                if e - s < min_flight_frames:
                    continue

                n = e - s
                t = np.arange(n, dtype=np.float32) / sample.fps
                pos = sample.com_position[s:e].astype(np.float32)

                traj = {
                    "time": t,
                    "position": pos,
                }

                if sample.com_velocity is not None:
                    traj["initial_velocity"] = sample.com_velocity[s].astype(np.float32)

                self.trajectories.append(traj)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj = self.trajectories[idx]
        return {k: torch.from_numpy(v) for k, v in traj.items()}


class PoseLiftingDataset(Dataset):
    """Dataset for pre-training 2D→3D pose lifting models.

    Each item is a single frame with 2D keypoints (input) and
    3D keypoints (target). Used with AthletePose3D data.
    """

    def __init__(
        self,
        samples: Sequence[BiomechanicalSample],
        augment: bool = True,
    ):
        self.augment = augment
        self.frames_2d: list[np.ndarray] = []
        self.frames_3d: list[np.ndarray] = []

        for sample in samples:
            if sample.pose_2d is None or sample.pose_3d is None:
                continue

            for t in range(sample.n_frames):
                kpts_2d = sample.pose_2d[t]  # (n_joints, 3): x, y, conf
                kpts_3d = sample.pose_3d[t]  # (n_joints, 3): x, y, z

                # Skip low-confidence frames
                if kpts_2d[:, 2].mean() < 0.3:
                    continue

                self.frames_2d.append(kpts_2d[:, :2].astype(np.float32))
                self.frames_3d.append(kpts_3d.astype(np.float32))

    def __len__(self) -> int:
        return len(self.frames_2d)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pose_2d = self.frames_2d[idx].copy()
        pose_3d = self.frames_3d[idx].copy()

        if self.augment:
            # Random horizontal flip
            if np.random.random() < 0.5:
                pose_2d[:, 0] = -pose_2d[:, 0]
                pose_3d[:, 0] = -pose_3d[:, 0]

            # Small random noise on 2D inputs (simulates detection jitter)
            pose_2d += np.random.randn(*pose_2d.shape).astype(np.float32) * 0.002

        return {
            "pose_2d": torch.from_numpy(pose_2d),
            "pose_3d": torch.from_numpy(pose_3d),
        }
