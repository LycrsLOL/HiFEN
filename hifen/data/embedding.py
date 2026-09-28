from __future__ import annotations

from pathlib import Path

import torch


def load_embedding(path: str | Path, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(tensor, dict):
        for key in ("embedding", "x", "representations", "per_residue_embedding"):
            if key in tensor:
                tensor = tensor[key]
                break
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.dim() == 3 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D residue embedding from {path}, got shape {tuple(tensor.shape)}")
    return tensor.to(dtype=dtype)


def expected_embedding_paths(embedding_dir: str | Path, uniprot_id: str, pdb_id: str | None = None) -> list[Path]:
    root = Path(embedding_dir)
    paths = []
    if pdb_id:
        paths.append(root / f"{uniprot_id}_{pdb_id}.pt")
    paths.append(root / f"{uniprot_id}.pt")
    return paths
