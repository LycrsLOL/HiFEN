from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


_SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LEVELS = ("ec1", "ec2", "ec3", "ec4")


def _proc_stat(pid: int) -> tuple[int, int, int] | None:
    """Return (parent pid, session id, start ticks) for a Linux process."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = text.rsplit(")", 1)[1].split()
        return int(fields[1]), int(fields[3]), int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _proc_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [item.decode("utf-8", errors="replace") for item in raw.split(b"\0") if item]


def _proc_children(pid: int) -> list[int]:
    try:
        text = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="utf-8")
    except OSError:
        return []
    result: list[int] = []
    for token in text.split():
        try:
            result.append(int(token))
        except ValueError:
            continue
    return result


def _ancestor_distances(pid: int, limit: int = 12) -> dict[int, int]:
    result: dict[int, int] = {}
    current = pid
    for distance in range(limit + 1):
        if current <= 1 or current in result:
            break
        result[current] = distance
        stat = _proc_stat(current)
        if stat is None:
            break
        current = stat[0]
    return result


def _tee_output_argument(argv: list[str]) -> str | None:
    if not argv or Path(argv[0]).name != "tee":
        return None
    outputs: list[str] = []
    options_done = False
    for argument in argv[1:]:
        if not options_done and argument == "--":
            options_done = True
            continue
        if not options_done and argument.startswith("-"):
            continue
        outputs.append(argument)
    log_outputs = [argument for argument in outputs if argument.lower().endswith(".log")]
    return log_outputs[0] if log_outputs else (outputs[0] if outputs else None)


def _resolve_process_path(pid: int, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    try:
        cwd = Path(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        cwd = Path.cwd()
    return (cwd / path).resolve()


def _candidate_tee_processes(pid: int) -> list[tuple[int, int]]:
    """Return (tee pid, ancestor distance), preferring the current process tree."""
    ancestors = _ancestor_distances(pid)
    candidates: dict[int, int] = {}
    for ancestor, distance in ancestors.items():
        for child in _proc_children(ancestor):
            if _tee_output_argument(_proc_cmdline(child)) is not None:
                candidates[child] = min(candidates.get(child, distance), distance)

    current_stat = _proc_stat(pid)
    current_session = current_stat[1] if current_stat is not None else None
    try:
        proc_entries = list(Path("/proc").iterdir())
    except OSError:
        proc_entries = []
    for entry in proc_entries:
        if not entry.name.isdigit():
            continue
        candidate_pid = int(entry.name)
        if candidate_pid in candidates:
            continue
        stat = _proc_stat(candidate_pid)
        if stat is None or (current_session is not None and stat[1] != current_session):
            continue
        if _tee_output_argument(_proc_cmdline(candidate_pid)) is not None:
            candidates[candidate_pid] = 99
    return sorted(candidates.items())


def detect_tee_log_path(
    configured_log_dir: str | os.PathLike[str] | None = None,
    *,
    pid: int | None = None,
    max_start_delta_seconds: float = 180.0,
) -> Path | None:
    """Detect the output file of a sibling ``tee`` process in the same pipeline."""
    current_pid = int(pid or os.getpid())
    current_stat = _proc_stat(current_pid)
    if current_stat is None:
        return None
    current_start = current_stat[2]
    ticks_per_second = float(os.sysconf("SC_CLK_TCK"))
    configured_dir = Path(configured_log_dir).expanduser() if configured_log_dir else None
    preferred_dir = configured_dir.resolve() if configured_dir is not None else None
    ranked: list[tuple[int, float, int, str, Path]] = []
    for candidate_pid, ancestor_distance in _candidate_tee_processes(current_pid):
        argv = _proc_cmdline(candidate_pid)
        output = _tee_output_argument(argv)
        stat = _proc_stat(candidate_pid)
        if output is None or stat is None:
            continue
        start_delta = abs(stat[2] - current_start) / ticks_per_second
        if start_delta > max_start_delta_seconds:
            continue
        path = _resolve_process_path(candidate_pid, output)
        preferred = 0 if preferred_dir is not None and path.parent == preferred_dir else 1
        reported_path = (configured_dir / path.name) if preferred == 0 and configured_dir is not None else path
        ranked.append((preferred, start_delta, ancestor_distance, str(reported_path), reported_path))
    if not ranked:
        return None
    return min(ranked)[4]


def tag_from_log_path(path: str | os.PathLike[str]) -> str:
    name = Path(path).name
    tag = name[:-4] if name.lower().endswith(".log") else Path(name).stem
    if not _SAFE_TAG.fullmatch(tag):
        raise ValueError(
            "tee log basename must contain only letters, numbers, '.', '_' or '-': "
            f"{name!r}"
        )
    return tag


def apply_training_artifact_routing(config: dict[str, Any], tee_log_path: Path) -> dict[str, Any]:
    """Route one training run to a tag-named directory on the data disk."""
    tag = tag_from_log_path(tee_log_path)
    storage = config.setdefault("storage", {})
    project_artifact_dir = Path(storage["project_artifact_dir"]).expanduser().resolve()
    run_dir = project_artifact_dir / "experiments" / tag
    storage["checkpoint_dir"] = str(run_dir)
    storage["metrics_dir"] = str(run_dir)
    storage["reproducibility_dir"] = str(run_dir)
    run = config.setdefault("run", {})
    run["artifact_tag"] = tag
    run["artifact_dir"] = str(run_dir)
    run["tee_log_path"] = str(tee_log_path)
    run["artifact_naming_source"] = "tee_log_basename"
    return config


def artifact_tag(config: dict[str, Any]) -> str | None:
    value = str((config.get("run", {}) or {}).get("artifact_tag") or "").strip()
    return value or None


def artifact_filename(config: dict[str, Any], plain_name: str) -> str:
    tag = artifact_tag(config)
    return f"{tag}_{plain_name}" if tag else plain_name


def synchronize_training_artifact_config(config: dict[str, Any], context: dict[str, Any] | None) -> None:
    if not context:
        return
    storage = config.setdefault("storage", {})
    run = config.setdefault("run", {})
    for key in ("checkpoint_dir", "metrics_dir", "reproducibility_dir"):
        if context.get(key):
            storage[key] = str(context[key])
    for key in ("artifact_tag", "artifact_dir", "tee_log_path", "artifact_naming_source"):
        if context.get(key):
            run[key] = str(context[key])
        else:
            run.pop(key, None)


def training_artifact_context(config: dict[str, Any]) -> dict[str, str]:
    tag = artifact_tag(config)
    storage = config.get("storage", {}) or {}
    run = config.get("run", {}) or {}
    return {
        "checkpoint_dir": str(storage.get("checkpoint_dir") or ""),
        "metrics_dir": str(storage.get("metrics_dir") or ""),
        "reproducibility_dir": str(storage.get("reproducibility_dir") or ""),
        "artifact_tag": tag or "",
        "artifact_dir": str(run.get("artifact_dir") or ""),
        "tee_log_path": str(run.get("tee_log_path") or ""),
        "artifact_naming_source": str(run.get("artifact_naming_source") or ""),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _data_file_paths(config: dict[str, Any]) -> dict[str, Path]:
    storage = config.get("storage", {}) or {}
    datasets = config.get("datasets", {}) or {}
    split_dir = Path(storage["split_dir"])
    label_dir = Path(storage["label_map_dir"])
    paths = {
        "train_csv": Path(datasets["train"]["csv"]),
        "train_split": split_dir / "train_graphs.txt",
        "val_split": split_dir / "val_graphs.txt",
        "test_split": split_dir / "test_graphs.txt",
        "split_summary": split_dir / "split_summary.json",
        "ec_label_maps": label_dir / "ec_label_maps.json",
        "parent_child_maps": label_dir / "parent_child_maps.json",
    }
    return {name: path for name, path in paths.items() if path.is_file()}


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def write_training_run_inputs(
    config: dict[str, Any],
    *,
    split_paths: dict[str, list[str]],
    label_maps: dict[str, dict[str, int]],
) -> dict[str, str]:
    tag = artifact_tag(config)
    if not tag:
        return {}
    run_dir = Path((config.get("run", {}) or {})["artifact_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = run_dir / artifact_filename(config, "train.yaml")
    fingerprint_path = run_dir / artifact_filename(config, "data_fingerprint.json")

    yaml_text = yaml.safe_dump(
        _json_ready(config),
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )
    _atomic_write_text(yaml_path, yaml_text)

    data_files = _data_file_paths(config)
    split_summary_path = data_files.get("split_summary")
    split_summary = (
        json.loads(split_summary_path.read_text(encoding="utf-8-sig"))
        if split_summary_path is not None
        else {}
    )
    fingerprint = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_tag": tag,
        "tee_log_path": str((config.get("run", {}) or {}).get("tee_log_path") or ""),
        "config_path": str(config.get("_config_path") or ""),
        "profile": str(config.get("_profile") or ""),
        "training_seed": int((config.get("train", {}) or {}).get("seed", 42)),
        "split_seed": split_summary.get("seed"),
        "split_counts": {name: len(paths) for name, paths in split_paths.items()},
        "split_paths_all_exist": all(Path(path).is_file() for paths in split_paths.values() for path in paths),
        "label_dimensions": {level: len(label_maps[level]) for level in _LEVELS},
        "data_files": {name: _file_record(path) for name, path in data_files.items()},
        "split_nonempty_line_counts": {
            name: _line_count(path)
            for name, path in data_files.items()
            if name in {"train_split", "val_split", "test_split"}
        },
        "split_summary": split_summary,
    }
    _atomic_write_text(
        fingerprint_path,
        json.dumps(fingerprint, ensure_ascii=False, indent=2) + "\n",
    )
    return {
        "train_yaml": str(yaml_path),
        "data_fingerprint": str(fingerprint_path),
    }


def _copy_log(source: str | os.PathLike[str] | None, destination: Path) -> str | None:
    if not source:
        return None
    source_path = Path(source)
    if not source_path.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return str(destination)


def _write_checksums(path: Path, files: Iterable[Path]) -> None:
    unique = sorted({item.resolve() for item in files if item.is_file() and item.resolve() != path.resolve()})
    lines = [f"{_sha256(item)}  {item}\n" for item in unique]
    _atomic_write_text(path, "".join(lines))


def finalize_training_run_artifacts(
    config: dict[str, Any],
    result: dict[str, Any] | None,
    *,
    status: str,
    error: str | None = None,
) -> dict[str, str]:
    tag = artifact_tag(config)
    if not tag:
        return {}
    run = config.get("run", {}) or {}
    run_dir = Path(run["artifact_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    tee_copy = _copy_log(run.get("tee_log_path"), run_dir / f"{tag}.log")
    internal_copy = _copy_log(run.get("internal_log_path"), run_dir / f"{tag}_internal.log")
    metadata_path = run_dir / artifact_filename(config, "archive_metadata.json")
    checksum_path = run_dir / artifact_filename(config, "artifacts.sha256")
    metadata = {
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "error": error,
        "artifact_tag": tag,
        "artifact_dir": str(run_dir),
        "tee_log_source": str(run.get("tee_log_path") or ""),
        "tee_log_copy": tee_copy,
        "internal_log_source": str(run.get("internal_log_path") or ""),
        "internal_log_copy": internal_copy,
        "config_path": str(config.get("_config_path") or ""),
        "profile": str(config.get("_profile") or ""),
        "training_seed": int((config.get("train", {}) or {}).get("seed", 42)),
        "result": _json_ready(result or {}),
    }
    _atomic_write_text(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    artifact_files = list(run_dir.glob(f"{tag}*"))
    _write_checksums(checksum_path, artifact_files)
    return {
        "artifact_dir": str(run_dir),
        "archive_metadata": str(metadata_path),
        "checksums": str(checksum_path),
    }
