from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import global_add_pool
    from torch_geometric.utils import softmax
except Exception:  # pragma: no cover
    global_add_pool = None
    softmax = None


class LevelAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, 1), 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if global_add_pool is None or softmax is None:
            raise RuntimeError("torch_geometric is required for LevelAttentionPool")
        scores = self.gate(x).view(-1)
        weights = softmax(scores, batch).view(-1, 1)
        pooled = global_add_pool(x * weights, batch)
        return pooled, weights
