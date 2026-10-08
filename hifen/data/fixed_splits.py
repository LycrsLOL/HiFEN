"""Portable, immutable graph assignments for the published S2 cohort."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPLIT_NAMES = ("train", "val", "test")


def fixed_split_directory(config: dict) -> Path | None:
    value = (config.get("train", {}) or {}).get("fixed_split_dir")
    if value is None or value is False or value == "":
        return None
    directory = Path(value)
    if not directory.is_absolute():
        config_dir = config.get("_config_dir")
        root = Path(config_dir).parent if config_dir else Path.cwd()
        directory = root / directory
    return directory.resolve()


def load_fixed_splits(config: dict, limit: int | None = None) -> tuple[dict[str, list[str]], dict]:
    """Resolve basenames in the fixed cohort against this profile's graph cache.

    Extra prepared graphs are ignored. Missing cohort graphs are fatal; there is
    deliberately no automatic repartitioning or silent dropping of assignments.
    ``limit`` selects the first N records within each fixed split for smoke runs.
    """
    directory = fixed_split_directory(config)
    if directory is None:
        raise ValueError("train.fixed_split_dir is not configured")
    if (config.get("train", {}) or {}).get("force_resplit", False):
        raise ValueError("force_resplit conflicts with fixed S2 assignments; disable fixed_split_dir explicitly for a new cohort")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    summary_path = directory / "split_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing fixed split summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if summary.get("strategy") != "fixed_s2" or summary.get("manifest_format") != "graph_basename":
        raise ValueError("Expected a fixed_s2 summary with graph_basename manifests")

    names_by_split = {}
    seen = set()
    for split in SPLIT_NAMES:
        path = directory / f"{split}_graphs.txt"
        contents = path.read_bytes()
        expected_hash = summary.get("manifest_sha256", {}).get(split)
        if not expected_hash or hashlib.sha256(contents).hexdigest() != expected_hash:
            raise ValueError(f"Fixed {split} manifest checksum mismatch: {path}")
        names = [line.strip() for line in contents.decode("utf-8-sig").splitlines() if line.strip()]
        if not names or len(names) != summary.get("counts", {}).get(split):
            raise ValueError(f"Fixed {split} manifest count mismatch")
        for name in names:
            if Path(name).name != name or "\\" in name or not name.endswith(".pt"):
                raise ValueError(f"Expected a graph basename, got {name!r}")
            if name in seen:
                raise ValueError(f"Duplicate or cross-split graph assignment: {name}")
            seen.add(name)
        names_by_split[split] = names
    if len(seen) != summary.get("total_records"):
        raise ValueError("Fixed split total-record count mismatch")

    graph_dir = Path(config["storage"]["graph_dir"])
    paths = {split: [str(graph_dir / name) for name in (names[:limit] if limit is not None else names)]
             for split, names in names_by_split.items()}
    missing = [path for values in paths.values() for path in values if not Path(path).is_file()]
    if missing:
        examples = ", ".join(missing[:5])
        raise FileNotFoundError(f"Fixed S2 cohort is missing {len(missing)} prepared graphs; assignments were not changed. Examples: {examples}")
    return paths, {**summary, "resolved_graph_dir": str(graph_dir)}
