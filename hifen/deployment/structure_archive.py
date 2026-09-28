from __future__ import annotations

import io
import json
import logging
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


STRUCTURE_SUFFIXES = (".pdb", ".cif", ".mmcif")


@dataclass
class StructureArchiveResult:
    status: str
    archive_path: Path | None = None
    archived_files: int = 0
    deleted_files: int = 0
    bytes_reclaimed: int = 0
    message: str = ""


def _norm(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return None
    return text


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value)
    return safe or "structure"


def _graph_is_self_contained(graph_path: Path) -> bool:
    try:
        try:
            data = torch.load(graph_path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(graph_path, map_location="cpu")
    except Exception as exc:
        logging.warning("Could not inspect graph before structure archive: %s (%s)", graph_path, exc)
        return False
    x = getattr(data, "x", None)
    if x is None or not hasattr(x, "numel") or int(x.numel()) == 0:
        return False
    return not bool(getattr(data, "embedding_path", ""))


def _existing_files(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not path.exists() or not path.is_file():
            continue
        seen.add(resolved)
        files.append(path)
    return files


def _sidecars_for(path: Path) -> list[Path]:
    return [path.with_suffix(path.suffix + ".json"), path.with_suffix(".json")]


def _candidate_structure_files(root: Path, uniprot_id: str, pdb_id: str | None, structure_path: str | Path | None) -> list[Path]:
    candidates: list[Path] = []
    if structure_path:
        structure = Path(structure_path)
        candidates.append(structure)
        candidates.extend(_sidecars_for(structure))

    predicted_dir = root / "predicted"
    complete_dir = root / "complete"
    crystal_dir = root / "crystal"

    predicted_stems = [uniprot_id]
    complete_stems = [uniprot_id]
    crystal_stems: list[str] = []
    if pdb_id:
        pdb_lower = pdb_id.lower()
        predicted_stems.insert(0, f"{uniprot_id}_{pdb_id}")
        complete_stems.insert(0, f"{uniprot_id}_{pdb_id}")
        crystal_stems.extend(
            [
                f"{uniprot_id}_{pdb_lower}",
                f"{uniprot_id}_{pdb_id}",
                f"{pdb_id}_{uniprot_id}",
                f"{pdb_id.upper()}_{pdb_lower}",
                pdb_id,
                pdb_lower,
            ]
        )

    for directory, stems in (
        (predicted_dir, predicted_stems),
        (complete_dir, complete_stems),
        (crystal_dir, crystal_stems),
    ):
        for stem in stems:
            for suffix in STRUCTURE_SUFFIXES:
                path = directory / f"{stem}{suffix}"
                candidates.append(path)
                candidates.extend(_sidecars_for(path))
    return _existing_files(candidates)


def _paths_are_inside(root: Path, paths: Iterable[Path]) -> bool:
    root_resolved = root.resolve()
    for path in paths:
        try:
            path.resolve().relative_to(root_resolved)
        except ValueError:
            logging.warning("Refusing to archive/delete structure outside dataset root: %s", path)
            return False
    return True


def _archive_suffix(archive_format: str) -> str:
    value = archive_format.strip().lower()
    if value in {"tar.zst", "zst", "tzst"}:
        return ".tar.zst"
    if value in {"tar.gz", "tgz", "gz"}:
        return ".tar.gz"
    if value == "tar":
        return ".tar"
    raise ValueError(f"Unsupported deployment.structure_archive_format: {archive_format!r}")


def _add_manifest(tar: tarfile.TarFile, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    info = tarfile.TarInfo("manifest.json")
    info.size = len(data)
    info.mtime = time.time()
    tar.addfile(info, io.BytesIO(data))


def _write_archive_stream(
    archive_path: Path,
    *,
    root: Path,
    files: list[Path],
    manifest: dict[str, Any],
    archive_format: str,
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_name(archive_path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)
    fmt = archive_format.strip().lower()

    if fmt in {"tar.zst", "zst", "tzst"}:
        zstd = shutil.which("zstd")
        if not zstd:
            raise RuntimeError("zstd executable is required for structure_archive_format=tar.zst")
        proc = subprocess.Popen([zstd, "-T0", "-q", "-f", "-o", str(tmp_path)], stdin=subprocess.PIPE)
        assert proc.stdin is not None
        try:
            with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
                for path in files:
                    tar.add(path, arcname=str(path.resolve().relative_to(root.resolve())))
                _add_manifest(tar, manifest)
            proc.stdin.close()
            return_code = proc.wait()
        except Exception:
            proc.kill()
            proc.wait()
            tmp_path.unlink(missing_ok=True)
            raise
        if return_code != 0:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"zstd failed while writing {archive_path}")
    else:
        mode = "w:gz" if fmt in {"tar.gz", "tgz", "gz"} else "w"
        with tarfile.open(tmp_path, mode) as tar:
            for path in files:
                tar.add(path, arcname=str(path.resolve().relative_to(root.resolve())))
            _add_manifest(tar, manifest)

    if tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Structure archive is empty: {tmp_path}")
    tmp_path.replace(archive_path)


def _archive_is_readable(archive_path: Path, archive_format: str) -> bool:
    fmt = archive_format.strip().lower()
    try:
        if fmt in {"tar.zst", "zst", "tzst"}:
            zstd = shutil.which("zstd")
            if not zstd:
                return False
            completed = subprocess.run([zstd, "-q", "-t", str(archive_path)], check=False)
            return completed.returncode == 0
        mode = "r:gz" if fmt in {"tar.gz", "tgz", "gz"} else "r"
        with tarfile.open(archive_path, mode) as tar:
            return bool(tar.getmembers())
    except Exception as exc:
        logging.warning("Structure archive validation failed for %s: %s", archive_path, exc)
        return False


def _delete_files(files: Iterable[Path]) -> tuple[int, int]:
    deleted = 0
    bytes_reclaimed = 0
    for path in files:
        try:
            size = path.stat().st_size
            path.unlink()
            deleted += 1
            bytes_reclaimed += size
        except FileNotFoundError:
            continue
        except OSError as exc:
            logging.warning("Could not delete archived structure file %s: %s", path, exc)
    return deleted, bytes_reclaimed


def archive_graph_input_structures(
    config: dict,
    *,
    dataset: str,
    row: dict[str, str],
    graph_path: str | Path,
    structure_path: str | Path | None,
) -> StructureArchiveResult:
    deployment = config.get("deployment", {}) or {}
    if not bool(deployment.get("compress_structures_after_graph", False)):
        return StructureArchiveResult(status="disabled")

    storage = config.get("storage", {}) or {}
    datasets = config.get("datasets", {}) or {}
    spec = datasets.get(dataset, {}) or {}
    output_subdir = spec.get("output_subdir", dataset)
    root = Path(storage["save_dir"]) / output_subdir
    uniprot_id = _norm(row.get("uniprot_id"))
    if not uniprot_id:
        return StructureArchiveResult(status="skipped_no_uniprot")
    pdb_id = _norm(row.get("pdb_id"))

    graph_path = Path(graph_path)
    if bool(deployment.get("require_self_contained_graph_before_structure_delete", True)):
        if not _graph_is_self_contained(graph_path):
            return StructureArchiveResult(status="graph_not_self_contained")

    files = _candidate_structure_files(root, uniprot_id, pdb_id, structure_path)
    if not files:
        return StructureArchiveResult(status="no_raw_structures")
    if not _paths_are_inside(root, files):
        return StructureArchiveResult(status="unsafe_path")

    archive_format = str(deployment.get("structure_archive_format", "tar.zst"))
    suffix = _archive_suffix(archive_format)
    archive_root = Path(storage.get("structure_archive_dir") or (root / "structure_archives"))
    stem = _safe_name(f"{uniprot_id}_{pdb_id}" if pdb_id else uniprot_id)
    archive_path = archive_root / stem[:2] / f"{stem}{suffix}"
    manifest = {
        "dataset": dataset,
        "uniprot_id": uniprot_id,
        "pdb_id": pdb_id or "",
        "graph_path": str(graph_path),
        "structure_path": str(structure_path or ""),
        "archived_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [
            {
                "path": str(path.resolve().relative_to(root.resolve())),
                "size": path.stat().st_size,
                "mtime": path.stat().st_mtime,
            }
            for path in files
        ],
    }

    if not archive_path.exists():
        _write_archive_stream(
            archive_path,
            root=root,
            files=files,
            manifest=manifest,
            archive_format=archive_format,
        )
    if not _archive_is_readable(archive_path, archive_format):
        return StructureArchiveResult(status="archive_invalid", archive_path=archive_path, archived_files=len(files))

    delete_raw = bool(deployment.get("delete_raw_structures_after_compress", False))
    deleted = 0
    bytes_reclaimed = 0
    if delete_raw:
        deleted, bytes_reclaimed = _delete_files(files)
    return StructureArchiveResult(
        status="archived_deleted" if delete_raw else "archived",
        archive_path=archive_path,
        archived_files=len(files),
        deleted_files=deleted,
        bytes_reclaimed=bytes_reclaimed,
    )
