from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable

from ..config import ensure_dirs
from .ec_labels import LEVELS, build_label_maps, build_parent_child_maps, encode_ec_values
from .graph_preparation import _norm
from ..utils import write_json


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def graph_path_for_row(config: dict, dataset: str, row: dict[str, str]) -> Path | None:
    uniprot_id = _norm(row.get("uniprot_id"))
    if not uniprot_id:
        return None
    pdb_id = _norm(row.get("pdb_id"))
    graph_name = f"{uniprot_id}_{pdb_id}.pt" if pdb_id else f"{uniprot_id}.pt"
    save_dir = Path(config["storage"]["save_dir"])
    output_subdir = config["datasets"][dataset].get("output_subdir", dataset)
    return save_dir / output_subdir / "graph" / graph_name


def build_current_label_maps(config: dict) -> tuple[dict[str, dict[str, int]], dict[str, list[tuple[int, int]]]]:
    train_csv = Path(config["datasets"]["train"]["csv"])
    train_rows = read_csv_rows(train_csv)
    label_maps = build_label_maps(row.get("ec_numbers") for row in train_rows)
    parent_child_maps = build_parent_child_maps(label_maps)
    return label_maps, parent_child_maps


def save_label_maps(config: dict, label_maps: dict[str, dict[str, int]], parent_child_maps: dict[str, list[tuple[int, int]]]) -> None:
    out_dir = Path(config["storage"]["label_map_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "ec_label_maps.json", label_maps)
    write_json(out_dir / "parent_child_maps.json", parent_child_maps)


def _labels_need_update(data, labels: dict[str, torch.Tensor]) -> bool:
    for level in LEVELS:
        expected = labels[level].float().view(-1)
        current = getattr(data, f"y_{level}", None)
        if current is None:
            return True
        current = current.detach().cpu().float().view(-1)
        if current.numel() != expected.numel():
            return True
        if not torch.equal(current, expected):
            return True
    return False


def repair_graph_labels(
    config: dict,
    dataset_names: Iterable[str] | None = None,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    overwrite_label_maps: bool = True,
) -> dict[str, int]:
    ensure_dirs(config)
    selected = list(dataset_names or config.get("datasets", {}).keys())
    label_maps, parent_child_maps = build_current_label_maps(config)
    if overwrite_label_maps and not dry_run:
        save_label_maps(config, label_maps, parent_child_maps)

    stats = {
        "checked": 0,
        "updated": 0,
        "already_current": 0,
        "missing_graph": 0,
        "failed": 0,
    }
    processed = 0
    for dataset in selected:
        rows = read_csv_rows(config["datasets"][dataset]["csv"])
        progress = tqdm(rows, desc=f"Repairing {dataset} graph labels", unit="graph", dynamic_ncols=True)
        for row in progress:
            if limit is not None and processed >= limit:
                return stats
            processed += 1
            graph_path = graph_path_for_row(config, dataset, row)
            if graph_path is None or not graph_path.exists():
                stats["missing_graph"] += 1
                continue
            stats["checked"] += 1
            labels = encode_ec_values(row.get("ec_numbers"), label_maps)
            try:
                data = torch.load(graph_path, map_location="cpu", weights_only=False)
                if not _labels_need_update(data, labels):
                    stats["already_current"] += 1
                    continue
                if not dry_run:
                    for level, value in labels.items():
                        setattr(data, f"y_{level}", value.float())
                    tmp_path = graph_path.with_suffix(graph_path.suffix + ".tmp")
                    torch.save(data, tmp_path)
                    tmp_path.replace(graph_path)
                stats["updated"] += 1
            except Exception as exc:  # pragma: no cover - depends on data files
                stats["failed"] += 1
                logging.warning("Failed to repair labels for %s: %s", graph_path, exc)
    return stats

