"""Skeleton Graph Neural Network for joint coupling.

Models the kinetic chain as a graph where joints are nodes and
bones are edges. Message passing propagates forces and moments
through the skeleton, coupling the individual joint PINNs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import torch_geometric
    from torch_geometric.nn import MessagePassing
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    MessagePassing = nn.Module  # fallback for type hints

# Skeleton graph topology: bone connections as (parent, child) pairs
# Indices correspond to joint order in the JointPINN ensemble
SKELETON_EDGES = [
    (0, 1),   # ankle → knee
    (1, 2),   # knee → hip
    (2, 3),   # hip → lumbar
    (3, 4),   # lumbar → shoulder_girdle
    (2, 5),   # hip → free_leg_hip (contralateral)
]

JOINT_NAMES = ["ankle", "knee", "hip", "lumbar", "shoulder_girdle", "free_leg"]


def build_skeleton_graph(
    joint_features: torch.Tensor,
    edges: list[tuple[int, int]] | None = None,
) -> "Data":
    """Build a PyTorch Geometric Data object for the skeleton graph.

    Args:
        joint_features: (N_joints, feature_dim) node features from joint PINNs.
        edges: List of (src, dst) pairs. Defaults to SKELETON_EDGES.

    Returns:
        PyG Data object ready for message passing.
    """
    if not HAS_PYG:
        raise ImportError(
            "torch_geometric is required. Install via: "
            "pip install torch-geometric"
        )
    if edges is None:
        edges = SKELETON_EDGES

    # Make bidirectional
    edge_list = edges + [(dst, src) for src, dst in edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return Data(x=joint_features, edge_index=edge_index)


class ForceMessageLayer(MessagePassing if HAS_PYG else nn.Module):
    """Message passing layer that propagates forces through the kinetic chain.

    Each message represents the force/moment transmitted along a bone
    from one joint to the next.
    """

    def __init__(self, node_dim: int, edge_dim: int = 0, hidden_dim: int = 64):
        if HAS_PYG:
            super().__init__(aggr="add")
        else:
            super().__init__()

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if HAS_PYG:
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        raise ImportError("torch_geometric required for forward pass")

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        if edge_attr is not None:
            inp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            inp = torch.cat([x_i, x_j], dim=-1)
        return self.message_mlp(inp)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class SkeletonGNN(nn.Module):
    """Full GNN that couples joint PINNs through the skeleton graph.

    Architecture:
        1. Each joint PINN produces per-joint features
        2. The GNN propagates forces/moments through the skeleton
        3. Updated joint features are used for final predictions
    """

    def __init__(
        self,
        node_dim: int = 5,      # output dim of each JointPINN
        hidden_dim: int = 64,
        n_message_passes: int = 3,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ForceMessageLayer(node_dim=node_dim, hidden_dim=hidden_dim)
            for _ in range(n_message_passes)
        ])
        self.readout = nn.Linear(node_dim, node_dim)

    def forward(self, data: "Data") -> torch.Tensor:
        """Run message passing over the skeleton graph.

        Args:
            data: PyG Data with node features from joint PINNs.

        Returns:
            (N_joints, node_dim) updated joint predictions after coupling.
        """
        x = data.x
        for layer in self.layers:
            x = layer(x, data.edge_index) + x  # residual connection
        return self.readout(x)
