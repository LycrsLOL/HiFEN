from __future__ import annotations

import torch
import torch.nn as nn


class RBFEdgeEncoder(nn.Module):
    def __init__(self, num_kernels: int = 16, cutoff: float = 10.0, extra_dim: int = 1):
        super().__init__()
        if num_kernels < 1:
            raise ValueError("num_kernels must be >= 1")
        centers = torch.linspace(0.0, float(cutoff), num_kernels)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / max((float(cutoff) / num_kernels) ** 2, 1e-6)
        self.extra_dim = int(extra_dim)
        self.out_dim = num_kernels + self.extra_dim

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr is None or edge_attr.dim() != 2:
            raise ValueError("edge_attr must be a 2D tensor")
        if edge_attr.size(1) != 1 + self.extra_dim:
            raise ValueError(f"edge_attr expected {1 + self.extra_dim} columns, got {edge_attr.size(1)}")
        distances = edge_attr[:, :1].clamp_min(0.0)
        rbf = torch.exp(-self.gamma * (distances - self.centers.view(1, -1)) ** 2)
        return torch.cat([rbf, edge_attr[:, 1:]], dim=1) if self.extra_dim else rbf
