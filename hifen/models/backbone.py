from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, GraphNorm
except Exception:  # pragma: no cover
    GATv2Conv = None
    GraphNorm = None


class ResidualGATBackbone(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, heads: int, edge_dim: int, dropout: float):
        super().__init__()
        if GATv2Conv is None or GraphNorm is None:
            raise RuntimeError("torch_geometric is required for ResidualGATBackbone")
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.dropout = float(dropout)
        self.convs = nn.ModuleList(
            [
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(num_layers)])

    def forward_layers(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> list[torch.Tensor]:
        end = len(self.convs) if end is None else int(end)
        start = int(start)
        if not 0 <= start <= end <= len(self.convs):
            raise ValueError(f"Invalid layer range [{start}, {end}) for {len(self.convs)} layers")
        outputs = []
        for conv, norm in zip(self.convs[start:end], self.norms[start:end]):
            residual = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x, batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
            outputs.append(x)
        return outputs

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> list[torch.Tensor]:
        return self.forward_layers(x, edge_index, edge_attr, batch)
