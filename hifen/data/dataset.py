from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import torch

from .embedding import load_embedding
from .ec_labels import encode_ec_values

try:
    from torch_geometric.data import Dataset
except Exception:  # pragma: no cover
    Dataset = object


def _protected_site_nodes(data, *, protected_hops: int) -> torch.Tensor:
    num_nodes = int(data.x.size(0))
    protected = torch.zeros(num_nodes, dtype=torch.bool)
    y_site = getattr(data, "y_site", None)
    y_site_mask = getattr(data, "y_site_mask", None)
    if y_site is not None and y_site_mask is not None and y_site.size(0) == num_nodes:
        protected = ((y_site > 0) & (y_site_mask > 0)).view(num_nodes, -1).any(dim=1)
    edge_index = getattr(data, "edge_index", None)
    if protected_hops <= 0 or edge_index is None or edge_index.numel() == 0 or not bool(protected.any()):
        return protected
    src, dst = edge_index.long()
    for _ in range(int(protected_hops)):
        previous = protected.clone()
        protected[src[previous[dst]]] = True
        protected[dst[previous[src]]] = True
    return protected


def apply_tail_node_feature_mask(
    data,
    *,
    class_support: torch.Tensor,
    config: Mapping[str, object],
) -> tuple[bool, int, int]:
    level = str(config.get("level", "ec4"))
    target = getattr(data, f"y_{level}", None)
    if target is None or not hasattr(data, "x") or data.x is None:
        return False, 0, 0
    target = target.view(-1)
    support = torch.as_tensor(class_support, dtype=torch.float32).view(-1)
    if target.numel() != support.numel():
        raise ValueError(
            f"Tail node feature masking expected y_{level} size {support.numel()}, got {target.numel()}"
        )
    tail_max_support = max(int(config.get("tail_max_support", 10)), 1)
    tail_classes = (support > 0) & (support <= tail_max_support)
    eligible = bool(((target > 0) & tail_classes).any())
    if not eligible:
        return False, 0, 0
    probability = float(config.get("apply_probability", 1.0))
    if probability <= 0.0 or (probability < 1.0 and float(torch.rand(())) >= probability):
        return True, 0, 0

    num_nodes = int(data.x.size(0))
    min_nodes = max(int(config.get("min_nodes", 20)), 1)
    mask_ratio = float(config.get("mask_ratio", 0.1))
    if num_nodes < min_nodes or mask_ratio <= 0.0:
        return True, 0, 0
    if mask_ratio > 1.0:
        raise ValueError("train.data_augmentation.node_feature_mask.mask_ratio must be <= 1")

    protected = (
        _protected_site_nodes(data, protected_hops=int(config.get("protected_hops", 1)))
        if bool(config.get("protect_site_positive", True))
        else torch.zeros(num_nodes, dtype=torch.bool)
    )
    candidates = torch.where(~protected)[0]
    mask_count = min(max(int(round(int(candidates.numel()) * mask_ratio)), 1), int(candidates.numel()))
    if mask_count <= 0:
        return True, 0, int(protected.sum().item())
    selected = candidates[torch.randperm(candidates.numel())[:mask_count]]
    fill_value = str(config.get("fill_value", "zero")).strip().lower()
    if fill_value in {"zero", "zeros", "0"}:
        data.x[selected] = 0
    elif fill_value in {"protein_mean", "mean"}:
        data.x[selected] = data.x.mean(dim=0, keepdim=True)
    else:
        raise ValueError(f"Unsupported node feature mask fill_value: {fill_value!r}")
    return True, mask_count, int(protected.sum().item())


def clear_ignored_label_indices(
    data,
    ignored_label_indices: Mapping[str, list[int] | tuple[int, ...]],
) -> None:
    """Clear legacy placeholder classes before loss and metric computation."""
    for level, raw_indices in ignored_label_indices.items():
        if not raw_indices:
            continue
        target = getattr(data, f"y_{level}", None)
        if target is None:
            continue
        flattened = target.view(-1).clone()
        indices = torch.as_tensor(raw_indices, dtype=torch.long)
        if indices.numel() and int(indices.max()) >= flattened.numel():
            raise ValueError(
                f"Ignored y_{level} index {int(indices.max())} exceeds target size "
                f"{flattened.numel()}"
            )
        flattened[indices] = 0
        setattr(data, f"y_{level}", flattened.view(target.shape))


def replace_graph_ec_labels(
    data,
    ec_value: object,
    label_maps: Mapping[str, Mapping[str, int]],
    ignored_ec_value: object | None = None,
) -> None:
    """Encode labels and per-class supervision masks at graph load time.

    ``ec_value`` contains the labels supervised as positive for this sample.
    Labels in ``ignored_ec_value`` remain unknown: they are neither positives nor
    negatives. Positive labels take precedence when an ignored co-label shares an
    ancestor with the supervised target.
    """
    labels = encode_ec_values(ec_value, label_maps)  # type: ignore[arg-type]
    ignored = encode_ec_values(ignored_ec_value, label_maps)  # type: ignore[arg-type]
    for level, target in labels.items():
        setattr(data, f"y_{level}", target.float())
        target_mask = torch.ones_like(target, dtype=torch.float32)
        target_mask[ignored[level] > 0] = 0.0
        target_mask[target > 0] = 1.0
        setattr(data, f"y_{level}_mask", target_mask)


class ThinGraphDataset(Dataset):
    def __init__(
        self,
        graph_paths: list[str | Path],
        embedding_dtype: torch.dtype = torch.float32,
        *,
        node_feature_mask_config: Mapping[str, object] | None = None,
        node_feature_mask_support: torch.Tensor | None = None,
        ignored_label_indices: Mapping[str, list[int] | tuple[int, ...]] | None = None,
        ec_values_by_path: Mapping[str, object] | None = None,
        ignored_ec_values_by_path: Mapping[str, object] | None = None,
        label_maps: Mapping[str, Mapping[str, int]] | None = None,
    ):
        super().__init__()
        self.graph_paths = [Path(path) for path in graph_paths]
        self.embedding_dtype = embedding_dtype
        self.node_feature_mask_config = dict(node_feature_mask_config or {})
        self.node_feature_mask_support = (
            torch.as_tensor(node_feature_mask_support, dtype=torch.float32)
            if node_feature_mask_support is not None
            else None
        )
        self.ignored_label_indices = {
            str(level): tuple(int(index) for index in indices)
            for level, indices in (ignored_label_indices or {}).items()
        }
        self.ec_values_by_path = {
            str(path): value for path, value in (ec_values_by_path or {}).items()
        }
        self.ignored_ec_values_by_path = {
            str(path): value for path, value in (ignored_ec_values_by_path or {}).items()
        }
        self.label_maps = {
            str(level): dict(level_map)
            for level, level_map in (label_maps or {}).items()
        }
        if (self.ec_values_by_path or self.ignored_ec_values_by_path) and not self.label_maps:
            raise ValueError("label_maps are required when per-path EC supervision is provided")

    def len(self) -> int:
        return len(self.graph_paths)

    def __len__(self) -> int:
        return self.len()

    def get(self, idx: int):
        path = self.graph_paths[idx]
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not hasattr(data, "x") or data.x is None:
            embedding_path = getattr(data, "embedding_path", None)
            if not embedding_path:
                raise ValueError(f"{path} is missing x and embedding_path")
            embedding = load_embedding(embedding_path, dtype=self.embedding_dtype)
            embedding_indices = getattr(data, "embedding_indices", None)
            if embedding_indices is not None:
                embedding_indices = torch.as_tensor(embedding_indices, dtype=torch.long)
                data.x = embedding[embedding_indices]
            else:
                start = int(getattr(data, "embedding_start", 0) or 0)
                end = getattr(data, "embedding_end", None)
                end = int(end) if end is not None else embedding.size(0)
                data.x = embedding[start:end]
        data.edge_index = data.edge_index.long()
        data.edge_attr = data.edge_attr.float()
        if hasattr(data, "annotation_x") and data.annotation_x is not None:
            data.annotation_x = data.annotation_x.float()
        if hasattr(data, "annotation_mask") and data.annotation_mask is not None:
            data.annotation_mask = data.annotation_mask.float()
        if hasattr(data, "y_site") and data.y_site is not None:
            data.y_site = data.y_site.float()
        if hasattr(data, "y_site_mask") and data.y_site_mask is not None:
            data.y_site_mask = data.y_site_mask.float()
        if str(path) in self.ec_values_by_path:
            replace_graph_ec_labels(
                data,
                self.ec_values_by_path[str(path)],
                self.label_maps,
                self.ignored_ec_values_by_path.get(str(path)),
            )
        clear_ignored_label_indices(data, self.ignored_label_indices)
        augmentation_eligible = False
        augmentation_masked_nodes = 0
        augmentation_protected_nodes = 0
        if self.node_feature_mask_support is not None and bool(self.node_feature_mask_config.get("enabled", False)):
            augmentation_eligible, augmentation_masked_nodes, augmentation_protected_nodes = (
                apply_tail_node_feature_mask(
                    data,
                    class_support=self.node_feature_mask_support,
                    config=self.node_feature_mask_config,
                )
            )
        data.augmentation_eligible = torch.tensor([int(augmentation_eligible)], dtype=torch.long)
        data.augmentation_applied = torch.tensor([int(augmentation_masked_nodes > 0)], dtype=torch.long)
        data.augmentation_masked_nodes = torch.tensor([augmentation_masked_nodes], dtype=torch.long)
        data.augmentation_protected_nodes = torch.tensor([augmentation_protected_nodes], dtype=torch.long)
        for attr in (
            "embedding_indices",
            "sequence_positions",
            "alignment_mode",
            "aligned_residues",
            "structure_residues",
            "aligned_fraction",
            "node_confidence_schema_version",
        ):
            if hasattr(data, attr):
                delattr(data, attr)
        data.graph_path = str(path)
        return data

    def __getitem__(self, idx: int):
        return self.get(idx)
