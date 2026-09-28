from __future__ import annotations

import torch


def sigmoid_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def apply_parent_child_consistency(
    probs: dict[str, torch.Tensor],
    parent_child_maps: dict[str, list[tuple[int, int]]],
) -> dict[str, torch.Tensor]:
    adjusted = {level: tensor.clone() for level, tensor in probs.items()}
    for parent_level, child_level, key in [
        ("ec1", "ec2", "ec1_to_ec2"),
        ("ec2", "ec3", "ec2_to_ec3"),
        ("ec3", "ec4", "ec3_to_ec4"),
    ]:
        pairs = parent_child_maps.get(key, [])
        if not pairs or parent_level not in adjusted or child_level not in adjusted:
            continue
        for parent_idx, child_idx in pairs:
            adjusted[child_level][..., child_idx] = torch.minimum(
                adjusted[child_level][..., child_idx],
                adjusted[parent_level][..., parent_idx],
            )
    return adjusted


def topk_labels(
    probs: torch.Tensor,
    idx_to_label: list[str],
    k: int,
    parent_lookup: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    if probs.dim() == 2:
        probs = probs[0]
    k = min(k, probs.numel())
    values, indices = torch.topk(probs, k=k)
    rows = []
    for rank, (value, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        label = idx_to_label[idx]
        rows.append(
            {
                "rank": rank,
                "label": label,
                "probability": float(value),
                "parent_label": parent_lookup.get(label) if parent_lookup else None,
            }
        )
    return rows
