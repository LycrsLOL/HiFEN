from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable


_GRAPHQL_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_entry_info {
      experimental_method
      resolution_combined
    }
    exptl {
      method
    }
    polymer_entities {
      rcsb_polymer_entity_container_identifiers {
        reference_sequence_identifiers {
          database_accession
          database_name
        }
      }
      entity_poly {
        pdbx_seq_one_letter_code_can
      }
    }
  }
}
"""


def _norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _clean_sequence(value: Any) -> str:
    return "".join(char for char in _norm(value).upper() if "A" <= char <= "Z")


def _method_rank(method: str) -> int:
    normalized = method.upper()
    if "X-RAY" in normalized or "X-RAY" in normalized.replace("_", "-") or "NEUTRON" in normalized:
        return 0
    if "ELECTRON" in normalized or "CRYO-EM" in normalized:
        return 1
    if "NMR" in normalized:
        return 2
    return 3


def _entry_method(entry: dict[str, Any]) -> str:
    methods = [
        _norm(item.get("method"))
        for item in (entry.get("exptl") or [])
        if isinstance(item, dict)
    ]
    methods = [method for method in methods if method]
    if methods:
        return methods[0]
    return _norm((entry.get("rcsb_entry_info") or {}).get("experimental_method")) or "unknown"


def _entry_resolution(entry: dict[str, Any]) -> float | None:
    values = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or []
    valid: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            valid.append(number)
    return min(valid) if valid else None


def _entity_matches_uniprot(entity: dict[str, Any], uniprot_id: str) -> bool:
    identifiers = (
        entity.get("rcsb_polymer_entity_container_identifiers") or {}
    ).get("reference_sequence_identifiers") or []
    target = uniprot_id.upper()
    return any(
        _norm(identifier.get("database_name")).lower() == "uniprot"
        and _norm(identifier.get("database_accession")).upper() == target
        for identifier in identifiers
        if isinstance(identifier, dict)
    )


def _sequence_similarity(target: str, candidate: str) -> tuple[float, float]:
    if not target or not candidate:
        return 0.0, 0.0
    matcher = SequenceMatcher(None, target, candidate, autojunk=False)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    coverage = min(matches / len(target), 1.0)
    identity = min(matches / len(candidate), 1.0)
    return coverage, identity


def score_entry(entry: dict[str, Any] | None, uniprot_id: str, sequence: str, pdb_id: str) -> dict[str, Any]:
    if not entry:
        return {
            "pdb_id": pdb_id.lower(),
            "metadata_available": False,
            "method": "unknown",
            "method_rank": 3,
            "resolution": None,
            "sequence_coverage": 0.0,
            "sequence_identity": 0.0,
            "uniprot_reference_match": False,
        }

    entities = [entity for entity in (entry.get("polymer_entities") or []) if isinstance(entity, dict)]
    matched_entities = [entity for entity in entities if _entity_matches_uniprot(entity, uniprot_id)]
    candidates = matched_entities or entities
    best_coverage = 0.0
    best_identity = 0.0
    target = _clean_sequence(sequence)
    for entity in candidates:
        candidate = _clean_sequence((entity.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can"))
        coverage, identity = _sequence_similarity(target, candidate)
        if (coverage, identity) > (best_coverage, best_identity):
            best_coverage, best_identity = coverage, identity

    method = _entry_method(entry)
    return {
        "pdb_id": pdb_id.lower(),
        "metadata_available": True,
        "method": method,
        "method_rank": _method_rank(method),
        "resolution": _entry_resolution(entry),
        "sequence_coverage": best_coverage,
        "sequence_identity": best_identity,
        "uniprot_reference_match": bool(matched_entities),
    }


def quality_sort_key(score: dict[str, Any]) -> tuple[Any, ...]:
    resolution = score.get("resolution")
    return (
        0 if score.get("metadata_available") else 1,
        int(score.get("method_rank", 3)),
        float(resolution) if resolution is not None else float("inf"),
        -float(score.get("sequence_coverage", 0.0)),
        -float(score.get("sequence_identity", 0.0)),
        str(score.get("pdb_id", "")).lower(),
    )


def _fetch_entry_metadata(
    pdb_ids: list[str],
    *,
    api_url: str,
    batch_size: int,
    timeout: float,
    retries: int,
    retry_backoff_seconds: float,
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for start in range(0, len(pdb_ids), batch_size):
        batch = pdb_ids[start : start + batch_size]
        payload = json.dumps(
            {"query": _GRAPHQL_QUERY, "variables": {"ids": [pdb_id.upper() for pdb_id in batch]}}
        ).encode("utf-8")
        request = urllib.request.Request(
            api_url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "HiFEN/1.0"},
        )
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    result = json.load(response)
                if result.get("errors"):
                    raise RuntimeError(str(result["errors"]))
                for entry in (result.get("data") or {}).get("entries") or []:
                    if isinstance(entry, dict) and _norm(entry.get("rcsb_id")):
                        entries[_norm(entry["rcsb_id"]).lower()] = entry
                break
            except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
                if attempt >= retries:
                    logging.warning(
                        "RCSB PDB quality metadata batch failed (%d entries): %s",
                        len(batch),
                        exc,
                    )
                    break
                time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
    return entries


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if payload.get("version") != 1 or not isinstance(payload.get("scores"), dict):
        return {}
    return payload["scores"]


def _write_cache(path: Path, scores: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps({"version": 1, "scores": scores}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def rank_pdb_candidate_groups(
    groups: dict[str, list[dict[str, str]]],
    *,
    cache_path: str | Path,
    api_url: str = "https://data.rcsb.org/graphql",
    batch_size: int = 100,
    timeout: float = 60.0,
    retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    has_existing_graph: Callable[[dict[str, str]], bool] | None = None,
) -> dict[str, list[dict[str, str]]]:
    cache_path = Path(cache_path)
    scores = _load_cache(cache_path)
    pairs: dict[str, dict[str, str]] = {}
    locked_groups = 0

    for uniprot_id, rows in groups.items():
        if has_existing_graph and any(has_existing_graph(row) for row in rows):
            locked_groups += 1
            continue
        for row in rows:
            pdb_id = _norm(row.get("pdb_id")).lower()
            if not pdb_id:
                continue
            key = f"{uniprot_id.upper()}|{pdb_id}"
            pairs.setdefault(key, row)

    missing_pairs = {key: row for key, row in pairs.items() if key not in scores}
    missing_pdb_ids = sorted({_norm(row.get("pdb_id")).lower() for row in missing_pairs.values() if _norm(row.get("pdb_id"))})
    metadata = _fetch_entry_metadata(
        missing_pdb_ids,
        api_url=api_url,
        batch_size=max(int(batch_size), 1),
        timeout=float(timeout),
        retries=max(int(retries), 1),
        retry_backoff_seconds=max(float(retry_backoff_seconds), 0.0),
    ) if missing_pdb_ids else {}

    new_scores = 0
    for key, row in missing_pairs.items():
        uniprot_id, pdb_id = key.split("|", 1)
        entry = metadata.get(pdb_id)
        if entry is not None:
            scores[key] = score_entry(entry, uniprot_id, row.get("sequence", ""), pdb_id)
            new_scores += 1
    if new_scores:
        _write_cache(cache_path, scores)

    ranked: dict[str, list[dict[str, str]]] = {}
    missing_metadata = 0
    for uniprot_id, rows in groups.items():
        def row_key(row: dict[str, str]) -> tuple[Any, ...]:
            pdb_id = _norm(row.get("pdb_id")).lower()
            existing_rank = 0 if has_existing_graph and has_existing_graph(row) else 1
            score = scores.get(f"{uniprot_id.upper()}|{pdb_id}") or score_entry(None, uniprot_id, "", pdb_id)
            return (existing_rank, *quality_sort_key(score))

        ranked_rows = sorted(rows, key=row_key)
        ranked[uniprot_id] = ranked_rows
        for row in ranked_rows:
            pdb_id = _norm(row.get("pdb_id")).lower()
            score = scores.get(f"{uniprot_id.upper()}|{pdb_id}")
            if score is None or not score.get("metadata_available"):
                missing_metadata += 1

    logging.info(
        "PDB candidate quality ranking | groups=%d locked_existing=%d candidates=%d "
        "new_scores=%d missing_metadata=%d cache=%s",
        len(groups),
        locked_groups,
        sum(len(rows) for rows in groups.values()),
        new_scores,
        missing_metadata,
        cache_path,
    )
    return ranked
