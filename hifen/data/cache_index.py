from __future__ import annotations

import logging
import os
import re
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from ..config import dataset_resource_dirs


UNIPROT_RE = re.compile(r"^[A-Z0-9]{6,10}$")
NCBI_PROTEIN_RE = re.compile(r"^[A-Z]{2}_[A-Z0-9]+(?:\.\d+)?$")
PDB_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


@dataclass
class CacheRecord:
    dataset_name: str
    resource_type: str
    uniprot_id: str | None
    pdb_id: str | None
    file_path: str
    file_size: int
    suffix: str
    basename: str
    modified_time: float
    parser_status: str


def _parse_resource_name(path: Path, resource_type: str) -> tuple[str | None, str | None, str]:
    stem = path.stem
    if NCBI_PROTEIN_RE.match(stem.upper()):
        return stem, None, "ok"
    tokens = re.split(r"[_\-.]", stem)
    uniprot_id: str | None = None
    pdb_id: str | None = None

    for token in tokens:
        t = token.upper()
        if uniprot_id is None and UNIPROT_RE.match(t):
            uniprot_id = token
        if pdb_id is None and PDB_RE.match(t):
            pdb_id = token.lower()

    if resource_type == "predicted" and uniprot_id is None and UNIPROT_RE.match(stem.upper()):
        uniprot_id = stem
    if resource_type == "embedding" and "_" in stem:
        parts = stem.split("_")
        if len(parts) >= 2:
            if UNIPROT_RE.match(parts[0].upper()):
                uniprot_id = parts[0]
            if PDB_RE.match(parts[-1].upper()):
                pdb_id = parts[-1].lower()
    if resource_type == "complete" and "_" in stem:
        parts = stem.split("_")
        if len(parts) >= 2:
            if UNIPROT_RE.match(parts[0].upper()):
                uniprot_id = parts[0]
            if PDB_RE.match(parts[-1].upper()):
                pdb_id = parts[-1].lower()
    if uniprot_id is None and "_" in stem:
        parts = stem.split("_")
        first = parts[0] if parts else ""
        last = parts[-1] if parts else ""
        if PDB_RE.match(first.upper()) and PDB_RE.match(last.upper()):
            uniprot_id = first
            pdb_id = last.lower()

    status = "ok" if uniprot_id else "unparsed_uniprot"
    return uniprot_id, pdb_id, status


def scan_resource_dir(dataset_name: str, resource_type: str, root: Path, recursive: bool = True) -> list[CacheRecord]:
    archive = root.with_suffix(".zip")
    if not root.exists():
        if resource_type == "crystal" and archive.exists() and archive.is_file():
            stat = archive.stat()
            return [
                CacheRecord(
                    dataset_name=dataset_name,
                    resource_type=resource_type,
                    uniprot_id=None,
                    pdb_id=None,
                    file_path=str(archive),
                    file_size=stat.st_size,
                    suffix=archive.suffix.lower(),
                    basename=archive.name,
                    modified_time=stat.st_mtime,
                    parser_status="archive_available",
                )
            ]
        logging.info("Cache directory does not exist; skipping optional resource scan: %s", root)
        return []
    pattern = "**/*" if recursive else "*"
    records: list[CacheRecord] = []
    for path in root.glob(pattern):
        if not path.is_file():
            continue
        if resource_type in {"crystal", "predicted", "complete"} and path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
            continue
        try:
            stat = path.stat()
        except OSError as exc:
            logging.warning("Could not stat %s: %s", path, exc)
            continue
        uniprot_id, pdb_id, status = _parse_resource_name(path, resource_type)
        records.append(
            CacheRecord(
                dataset_name=dataset_name,
                resource_type=resource_type,
                uniprot_id=uniprot_id,
                pdb_id=pdb_id,
                file_path=str(path),
                file_size=stat.st_size,
                suffix=path.suffix.lower(),
                basename=path.name,
                modified_time=stat.st_mtime,
                parser_status=status,
            )
        )
    if not records and resource_type == "crystal" and archive.exists() and archive.is_file():
        stat = archive.stat()
        records.append(
            CacheRecord(
                dataset_name=dataset_name,
                resource_type=resource_type,
                uniprot_id=None,
                pdb_id=None,
                file_path=str(archive),
                file_size=stat.st_size,
                suffix=archive.suffix.lower(),
                basename=archive.name,
                modified_time=stat.st_mtime,
                parser_status="archive_available",
            )
        )
    return records


def _record_dicts(records: list[CacheRecord]) -> list[dict]:
    return [asdict(record) for record in records]


def build_manifest(config: dict, dataset_names: Iterable[str] | None = None):
    datasets = config.get("datasets", {}) or {}
    selected = list(dataset_names or datasets.keys())
    recursive = bool((config.get("deployment", {}) or {}).get("recursive_index", True))
    records: list[CacheRecord] = []
    for dataset_name in selected:
        dirs = dataset_resource_dirs(config, dataset_name)
        for resource_type, root in dirs.items():
            records.extend(scan_resource_dir(dataset_name, resource_type, root, recursive=recursive))
    rows = _record_dicts(records)
    rows = sorted(
        rows,
        key=lambda row: (
            row.get("dataset_name") or "",
            row.get("resource_type") or "",
            row.get("uniprot_id") or "",
            row.get("pdb_id") or "",
            row.get("file_path") or "",
        ),
    )
    if pd is not None:
        return pd.DataFrame(rows)
    return rows


def save_manifest(frame, config: dict, name: str = "cache_manifest.csv") -> Path:
    out_dir = Path(config["storage"]["manifest_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    if pd is not None and hasattr(frame, "to_csv"):
        frame.to_csv(out_path, index=False)
        row_count = len(frame)
    else:
        fieldnames = [
            "dataset_name",
            "resource_type",
            "uniprot_id",
            "pdb_id",
            "file_path",
            "file_size",
            "suffix",
            "basename",
            "modified_time",
            "parser_status",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(frame)
        row_count = len(frame)
    logging.info("Saved manifest: %s (%d rows)", out_path, row_count)
    return out_path


def summarize_manifest(frame):
    if pd is not None and hasattr(frame, "empty"):
        if frame.empty:
            return pd.DataFrame(columns=["dataset_name", "resource_type", "files", "size_gib"])
        summary = (
            frame.groupby(["dataset_name", "resource_type"], dropna=False)
            .agg(files=("file_path", "count"), size_bytes=("file_size", "sum"))
            .reset_index()
        )
        summary["size_gib"] = summary["size_bytes"] / (1024 ** 3)
        return summary.drop(columns=["size_bytes"])

    buckets: dict[tuple[str, str], dict[str, object]] = {}
    for row in frame:
        key = (row.get("dataset_name") or "", row.get("resource_type") or "")
        bucket = buckets.setdefault(
            key,
            {"dataset_name": key[0], "resource_type": key[1], "files": 0, "size_gib": 0.0},
        )
        bucket["files"] = int(bucket["files"]) + 1
        bucket["size_gib"] = float(bucket["size_gib"]) + float(row.get("file_size") or 0) / (1024 ** 3)
    return list(buckets.values())


def format_summary(summary) -> str:
    if hasattr(summary, "to_string"):
        return summary.to_string(index=False)
    lines = ["dataset_name resource_type files size_gib"]
    for row in summary:
        lines.append(
            f"{row['dataset_name']} {row['resource_type']} {row['files']} {float(row['size_gib']):.3f}"
        )
    return "\n".join(lines)

