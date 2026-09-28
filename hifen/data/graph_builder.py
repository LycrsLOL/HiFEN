from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from torch_geometric.data import Data
except Exception:  # pragma: no cover
    Data = None


ANNOTATION_DIM = 12
SITE_TASKS = (
    "active_or_catalytic_site",
    "binding_site",
    "domain_region",
    "motif_region",
)


def build_edges_from_coords(
    coords: np.ndarray,
    cutoff: float = 10.0,
    sequence_positions: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [num_nodes, 3]")
    n = coords.shape[0]
    if n == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 2), dtype=torch.float32)
    if sequence_positions is not None and len(sequence_positions) != n:
        raise ValueError("sequence_positions must match coords length")

    def is_sequence_neighbor(i: int, j: int) -> bool:
        if sequence_positions is None:
            return abs(i - j) == 1
        return abs(int(sequence_positions[i]) - int(sequence_positions[j])) == 1

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(coords)
        pairs = tree.query_pairs(float(cutoff), output_type="ndarray")
        src: list[int] = []
        dst: list[int] = []
        dist: list[float] = []
        seq_flag: list[float] = []
        seen: set[tuple[int, int]] = set()

        def add_edge(i: int, j: int, is_seq: bool) -> None:
            for a, b in ((i, j), (j, i)):
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                src.append(a)
                dst.append(b)
                dist.append(float(np.linalg.norm(coords[a] - coords[b])))
                seq_flag.append(1.0 if is_seq else 0.0)

        for i, j in pairs.tolist():
            add_edge(int(i), int(j), is_sequence_neighbor(int(i), int(j)))
        for i in range(n - 1):
            if is_sequence_neighbor(i, i + 1):
                add_edge(i, i + 1, True)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(list(zip(dist, seq_flag)), dtype=torch.float32)
        return edge_index, edge_attr
    except Exception:
        logging.debug("scipy cKDTree unavailable; falling back to quadratic edge construction.")

    src: list[int] = []
    dst: list[int] = []
    dist: list[float] = []
    seq_flag: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            is_seq = is_sequence_neighbor(i, j)
            if d <= cutoff or is_seq:
                for a, b in ((i, j), (j, i)):
                    src.append(a)
                    dst.append(b)
                    dist.append(d)
                    seq_flag.append(1.0 if is_seq else 0.0)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(list(zip(dist, seq_flag)), dtype=torch.float32)
    return edge_index, edge_attr


def empty_annotation(num_nodes: int, annotation_dim: int = ANNOTATION_DIM) -> tuple[torch.Tensor, torch.Tensor]:
    annotation_x = torch.zeros((num_nodes, annotation_dim), dtype=torch.float32)
    annotation_mask = torch.zeros((num_nodes, annotation_dim), dtype=torch.float32)
    return annotation_x, annotation_mask


_GRAPH_DEDUP_EXCLUDED_ATTRS = {"protein_idx", "embedding_source_path", "structure_path", "embedding_path"}


def _graph_content_identical(new_data: Any, shared_data: Any) -> bool:
    try:
        new_attrs = dict(new_data.to_dict())
        shared_attrs = dict(shared_data.to_dict())
    except Exception:
        return False
    new_keys = {key for key in new_attrs if key not in _GRAPH_DEDUP_EXCLUDED_ATTRS}
    shared_keys = {key for key in shared_attrs if key not in _GRAPH_DEDUP_EXCLUDED_ATTRS}
    if new_keys != shared_keys:
        return False
    for key in new_keys:
        left = new_attrs[key]
        right = shared_attrs[key]
        if torch.is_tensor(left) or torch.is_tensor(right):
            if not (torch.is_tensor(left) and torch.is_tensor(right)):
                return False
            if left.shape != right.shape or left.dtype != right.dtype:
                return False
            if not torch.equal(left.detach().cpu(), right.detach().cpu()):
                return False
        elif left != right:
            return False
    return True


def save_thin_graph(
    out_path: str | Path,
    embedding_path: str | Path | None,
    structure_path: str | Path | None,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    labels: dict[str, torch.Tensor],
    annotation_x: torch.Tensor | None = None,
    annotation_mask: torch.Tensor | None = None,
    site_targets: torch.Tensor | None = None,
    site_target_mask: torch.Tensor | None = None,
    node_confidence: torch.Tensor | None = None,
    node_confidence_mask: torch.Tensor | None = None,
    node_features: torch.Tensor | None = None,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
    dedup_reference: str | Path | None = None,
) -> Path:
    if Data is None:
        raise RuntimeError("torch_geometric is required to save thin graph Data objects")
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        logging.info("Thin graph exists, skipping: %s", out_path)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_nodes = annotation_x.size(0) if annotation_x is not None else int(edge_index.max().item() + 1 if edge_index.numel() else 0)
    if annotation_x is None or annotation_mask is None:
        annotation_x, annotation_mask = empty_annotation(num_nodes)
    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    if node_features is not None:
        data.x = node_features.detach().cpu()
        if embedding_path:
            data.embedding_source_path = str(embedding_path)
    elif embedding_path:
        data.embedding_path = str(embedding_path)
    data.structure_path = str(structure_path) if structure_path else ""
    data.annotation_x = annotation_x
    data.annotation_mask = annotation_mask
    if site_targets is not None:
        data.y_site = site_targets.float()
    if site_target_mask is not None:
        data.y_site_mask = site_target_mask.float()
    if node_confidence is not None:
        data.node_confidence = node_confidence.float().view(-1)
    if node_confidence_mask is not None:
        data.node_confidence_mask = node_confidence_mask.float().view(-1)
    for key, value in labels.items():
        setattr(data, f"y_{key}", value.float())
    for key, value in (metadata or {}).items():
        setattr(data, key, value)
    if dedup_reference is not None and not out_path.exists():
        reference_path = Path(dedup_reference) / out_path.name
        if reference_path.exists() and reference_path.stat().st_size > 0:
            try:
                shared_data = torch.load(reference_path, map_location="cpu", weights_only=False)
                if _graph_content_identical(data, shared_data):
                    out_path.symlink_to(reference_path.resolve())
                    logging.info("Reused identical graph via symlink: %s -> %s", out_path, reference_path)
                    return out_path
            except Exception as dedup_exc:
                logging.debug("Graph dedup check failed for %s: %s", out_path, dedup_exc)
    torch.save(data, out_path)
    return out_path


def toy_thin_graph(num_nodes: int = 8, embedding_dim: int = 16) -> Any:
    if Data is None:
        raise RuntimeError("torch_geometric is required for toy graphs")
    coords = np.stack([np.arange(num_nodes), np.zeros(num_nodes), np.zeros(num_nodes)], axis=1).astype(np.float32)
    edge_index, edge_attr = build_edges_from_coords(coords, cutoff=2.0)
    annotation_x, annotation_mask = empty_annotation(num_nodes)
    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    data.x = torch.randn(num_nodes, embedding_dim)
    data.annotation_x = annotation_x
    data.annotation_mask = annotation_mask
    data.y_site = torch.zeros((num_nodes, len(SITE_TASKS)), dtype=torch.float32)
    data.y_site_mask = torch.zeros_like(data.y_site)
    data.batch = torch.zeros(num_nodes, dtype=torch.long)
    return data
