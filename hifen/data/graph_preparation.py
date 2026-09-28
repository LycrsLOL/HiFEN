from __future__ import annotations

import csv
import json
import logging
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .annotation_parser import parse_annotation_row
from .cache_index import build_manifest
from ..config import ensure_dirs
from .ec_labels import LEVELS, build_label_maps, build_parent_child_maps, encode_ec_values
from .embedding import load_embedding
from .graph_builder import ANNOTATION_DIM, SITE_TASKS, build_edges_from_coords, empty_annotation, save_thin_graph
from ..deployment.structure_archive import archive_graph_input_structures
from ..utils import write_json


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",
    "GLX": "Z",
    "SEC": "U",
    "PYL": "O",
    "MSE": "M",
}


@dataclass
class StructureResidue:
    coord: np.ndarray
    aa: str
    chain_id: str
    residue_id: str
    bfactor: float | None = None


@dataclass
class StructureAlignment:
    residues: list[StructureResidue]
    coords: np.ndarray
    sequence_positions: list[int]
    embedding_indices: list[int]
    alignment_mode: str
    aligned_residues: int
    structure_residues: int
    aligned_fraction: float


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def _manifest_rows(manifest: Any) -> list[dict[str, Any]]:
    if hasattr(manifest, "to_dict"):
        return manifest.to_dict("records")
    return list(manifest)


def _norm(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return None
    return text


def _clean_sequence(value: Any) -> str:
    text = _norm(value)
    if not text:
        return ""
    return "".join(text.split()).upper().replace("*", "")


def _aa3_to_1(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return "X"
    if len(text) == 1:
        return text
    return AA3_TO_1.get(text, "X")


def _build_resource_index(rows: Iterable[dict[str, Any]]) -> tuple[dict[tuple[str, str, str, str], list[str]], dict[tuple[str, str, str], list[str]]]:
    exact: dict[tuple[str, str, str, str], list[str]] = {}
    by_uniprot: dict[tuple[str, str, str], list[str]] = {}
    for row in rows:
        dataset = _norm(row.get("dataset_name"))
        resource = _norm(row.get("resource_type"))
        uniprot = _norm(row.get("uniprot_id"))
        pdb_id = _norm(row.get("pdb_id"))
        path = _norm(row.get("file_path"))
        if not dataset or not resource or not uniprot or not path:
            continue
        uniprot_key = (dataset, resource, uniprot)
        by_uniprot.setdefault(uniprot_key, []).append(path)
        if pdb_id:
            exact.setdefault((dataset, resource, uniprot, pdb_id.lower()), []).append(path)
    return exact, by_uniprot


def _first_existing(paths: Iterable[str]) -> str | None:
    for path in paths:
        if path and Path(path).exists():
            return path
    return None


def _select_embedding(
    dataset: str,
    uniprot_id: str,
    pdb_id: str | None,
    exact: dict[tuple[str, str, str, str], list[str]],
    by_uniprot: dict[tuple[str, str, str], list[str]],
) -> str | None:
    if pdb_id:
        found = _first_existing(exact.get((dataset, "embedding", uniprot_id, pdb_id.lower()), []))
        if found:
            return found
    return _first_existing(by_uniprot.get((dataset, "embedding", uniprot_id), []))


def _select_structure(
    dataset: str,
    uniprot_id: str,
    pdb_id: str | None,
    exact: dict[tuple[str, str, str, str], list[str]],
    by_uniprot: dict[tuple[str, str, str], list[str]],
    predicted_only: bool = False,
) -> str | None:
    resources = ("predicted",) if predicted_only else ("complete", "crystal", "predicted")
    for resource in resources:
        if pdb_id:
            found = _first_existing(exact.get((dataset, resource, uniprot_id, pdb_id.lower()), []))
            if found:
                return found
        found = _first_existing(by_uniprot.get((dataset, resource, uniprot_id), []))
        if found:
            return found
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _mmcif_field(block: dict[str, Any], names: Iterable[str]) -> list[Any]:
    for name in names:
        values = _as_list(block.get(name))
        if values:
            return values
    return []


def _first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if text:
                return text
    return ""


def _list_value(values: list[Any], index: int, default: Any = None) -> Any:
    return values[index] if index < len(values) else default


def _select_coord_group(groups: list[list[StructureResidue]], target_lengths: Iterable[int] | None = None) -> list[StructureResidue]:
    groups = [group for group in groups if group]
    if not groups:
        raise ValueError("No residue coordinates parsed")
    targets = {int(length) for length in (target_lengths or []) if int(length) > 0}
    if targets:
        groups = sorted(groups, key=lambda group: (min(abs(len(group) - target) for target in targets), -len(group)))
    return groups[0]


def _residue_records(residues: dict[tuple[str, str, str], dict[str, Any]]) -> list[StructureResidue]:
    records: list[StructureResidue] = []
    for residue in residues.values():
        if residue["ca"] is not None:
            coord = residue["ca"]
        elif residue["atoms"]:
            coord = np.asarray(residue["atoms"], dtype=np.float32).mean(axis=0)
        else:
            continue
        records.append(
            StructureResidue(
                coord=np.asarray(coord, dtype=np.float32),
                aa=residue["aa"],
                chain_id=residue["chain_id"],
                residue_id=residue["residue_id"],
                bfactor=(
                    float(residue["ca_bfactor"])
                    if residue.get("ca_bfactor") is not None
                    else (
                        float(np.mean(residue["bfactors"]))
                        if residue.get("bfactors")
                        else None
                    )
                ),
            )
        )
    return records


def _load_residue_groups_from_mmcif_atom_site(path: Path) -> list[list[StructureResidue]]:
    first_line = _first_nonempty_line(path)
    if not first_line.startswith("data_"):
        raise ValueError("Invalid mmCIF payload; first nonempty line is not data_")
    try:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    except Exception as exc:
        raise RuntimeError("Biopython is required to parse mmCIF atom_site tables.") from exc

    block = MMCIF2Dict(str(path))
    atom_names = _mmcif_field(block, ("_atom_site.label_atom_id", "_atom_site.auth_atom_id"))
    xs = _mmcif_field(block, ("_atom_site.Cartn_x",))
    ys = _mmcif_field(block, ("_atom_site.Cartn_y",))
    zs = _mmcif_field(block, ("_atom_site.Cartn_z",))
    if not atom_names or not xs or not ys or not zs:
        raise ValueError(f"No atom_site coordinates parsed from {path}")

    groups = _mmcif_field(block, ("_atom_site.group_PDB",))
    comp_ids = _mmcif_field(block, ("_atom_site.label_comp_id", "_atom_site.auth_comp_id"))
    chains = _mmcif_field(block, ("_atom_site.auth_asym_id", "_atom_site.label_asym_id"))
    seq_ids = _mmcif_field(block, ("_atom_site.auth_seq_id", "_atom_site.label_seq_id"))
    ins_codes = _mmcif_field(block, ("_atom_site.pdbx_PDB_ins_code",))
    models = _mmcif_field(block, ("_atom_site.pdbx_PDB_model_num",))
    bfactors = _mmcif_field(block, ("_atom_site.B_iso_or_equiv",))
    first_model = str(models[0]) if models else None

    residues_by_chain: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for idx in range(min(len(atom_names), len(xs), len(ys), len(zs))):
        model = str(_list_value(models, idx, first_model)) if first_model is not None else None
        if first_model is not None and model != first_model:
            continue
        group = str(_list_value(groups, idx, "ATOM")).upper()
        if group != "ATOM":
            continue
        seq_id = str(_list_value(seq_ids, idx, "")).strip()
        if seq_id in {"", ".", "?"}:
            continue
        try:
            coord = np.asarray([float(xs[idx]), float(ys[idx]), float(zs[idx])], dtype=np.float32)
        except (TypeError, ValueError):
            continue
        try:
            bfactor = float(_list_value(bfactors, idx))
        except (TypeError, ValueError):
            bfactor = None
        chain = str(_list_value(chains, idx, "")).strip()
        ins_code = str(_list_value(ins_codes, idx, "")).strip()
        atom_name = str(atom_names[idx]).strip().upper()
        residue_id = seq_id if ins_code in {"", ".", "?"} else f"{seq_id}{ins_code}"
        key = (chain, seq_id, "" if ins_code in {".", "?"} else ins_code)
        residues = residues_by_chain.setdefault(chain, {})
        residue = residues.setdefault(
            key,
            {
                "ca": None,
                "ca_bfactor": None,
                "atoms": [],
                "bfactors": [],
                "aa": _aa3_to_1(_list_value(comp_ids, idx, "")),
                "chain_id": chain,
                "residue_id": residue_id,
            },
        )
        if atom_name == "CA":
            residue["ca"] = coord
            residue["ca_bfactor"] = bfactor
        residue["atoms"].append(coord)
        if bfactor is not None and np.isfinite(bfactor):
            residue["bfactors"].append(bfactor)

    chain_groups = [_residue_records(residues) for residues in residues_by_chain.values()]
    all_residues = [residue for group in chain_groups for residue in group]
    groups = [all_residues] + chain_groups if len(chain_groups) > 1 else chain_groups
    if not any(groups):
        raise ValueError(f"No residue coordinates parsed from {path}")
    return groups


def _load_residue_groups_from_pdb(path: Path) -> list[list[StructureResidue]]:
    try:
        from Bio.PDB import PDBParser
    except Exception as exc:
        raise RuntimeError("Biopython is required to parse structures for graph building.") from exc

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    chain_groups: list[list[StructureResidue]] = []
    for model in structure:
        for chain in model:
            chain_residues: list[StructureResidue] = []
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    coord = residue["CA"].get_coord().astype(np.float32)
                    bfactor = float(residue["CA"].get_bfactor())
                elif len(residue):
                    coord = np.asarray([atom.get_coord() for atom in residue], dtype=np.float32).mean(axis=0)
                    bfactor = float(np.mean([atom.get_bfactor() for atom in residue]))
                else:
                    continue
                ins_code = str(residue.id[2]).strip()
                residue_id = str(residue.id[1]) if not ins_code else f"{residue.id[1]}{ins_code}"
                chain_residues.append(
                    StructureResidue(
                        coord=np.asarray(coord, dtype=np.float32),
                        aa=_aa3_to_1(residue.get_resname()),
                        chain_id=str(chain.id),
                        residue_id=residue_id,
                        bfactor=bfactor if np.isfinite(bfactor) else None,
                    )
                )
            if chain_residues:
                chain_groups.append(chain_residues)
        break
    all_residues = [residue for group in chain_groups for residue in group]
    groups = [all_residues] + chain_groups if len(chain_groups) > 1 else chain_groups
    if not any(groups):
        raise ValueError(f"No residue coordinates parsed from {path}")
    return groups


def _load_residue_groups_from_structure(path: str | Path) -> list[list[StructureResidue]]:
    path = Path(path)
    if path.suffix.lower() in {".cif", ".mmcif"}:
        return _load_residue_groups_from_mmcif_atom_site(path)
    return _load_residue_groups_from_pdb(path)


def _load_residues_from_structure(path: str | Path, target_lengths: Iterable[int] | None = None) -> list[StructureResidue]:
    return _select_coord_group(_load_residue_groups_from_structure(path), target_lengths=target_lengths)


def _coords_from_residues(residues: list[StructureResidue]) -> np.ndarray:
    coords = np.asarray([residue.coord for residue in residues], dtype=np.float32)
    if coords.size == 0:
        raise ValueError("No residue coordinates parsed")
    return coords


def _load_coords_from_structure(path: str | Path, target_lengths: Iterable[int] | None = None) -> np.ndarray:
    return _coords_from_residues(_load_residues_from_structure(path, target_lengths=target_lengths))


def _structure_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    candidates = (path.with_suffix(path.suffix + ".json"), path.with_suffix(".json"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _residue_confidence(
    residues: list[StructureResidue],
    structure_path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return normalized pLDDT, availability mask, and provenance.

    AlphaFold stores pLDDT in the B-factor column. Experimental B-factors have
    a different meaning and are deliberately masked instead of being silently
    reinterpreted as confidence.
    """

    path = Path(structure_path)
    metadata = _structure_metadata(path)
    mode = str(metadata.get("mode") or "").strip().lower()
    parent = path.parent.name.lower()
    predicted_residue_ids: set[str] | None
    if parent == "predicted" or mode.startswith("alphafold_only"):
        predicted_residue_ids = None
        source = "alphafold_plddt"
    elif mode == "crystal_with_alphafold_grafts":
        predicted_residue_ids = {
            str(int(value))
            for value in metadata.get("predicted_output_indices", [])
            if str(value).strip()
        }
        source = "hybrid_alphafold_grafts" if predicted_residue_ids else "unavailable_hybrid_provenance"
    else:
        predicted_residue_ids = set()
        source = "unavailable_experimental_bfactor_not_plddt"

    values = torch.zeros(len(residues), dtype=torch.float32)
    mask = torch.zeros(len(residues), dtype=torch.float32)
    for index, residue in enumerate(residues):
        eligible = predicted_residue_ids is None or residue.residue_id in predicted_residue_ids
        if not eligible or residue.bfactor is None or not np.isfinite(residue.bfactor):
            continue
        confidence = float(residue.bfactor)
        if confidence > 1.5:
            confidence /= 100.0
        if 0.0 <= confidence <= 1.0:
            values[index] = confidence
            mask[index] = 1.0
    return values, mask, source


def _embedding_target_lengths(embedding_path: str) -> set[int]:
    rows = int(load_embedding(embedding_path, dtype=torch.float32).size(0))
    return {rows, rows - 1, rows - 2}


def _embedding_slice_from_rows(rows: int, num_nodes: int, max_delta: int = 0) -> tuple[int, int] | None:
    if rows == num_nodes:
        return 0, num_nodes
    if rows == num_nodes + 2:
        return 1, num_nodes + 1
    if rows == num_nodes + 1:
        return 1, num_nodes + 1
    if max_delta > 0 and rows > num_nodes and rows - num_nodes <= max_delta:
        return 0, num_nodes
    return None


def _embedding_slice(embedding_path: str, num_nodes: int, max_delta: int = 0) -> tuple[int, int] | None:
    emb = load_embedding(embedding_path, dtype=torch.float32)
    return _embedding_slice_from_rows(int(emb.size(0)), num_nodes, max_delta=max_delta)


def _graph_feature_dtype(config: dict) -> torch.dtype:
    storage = config.get("storage", {}) or {}
    value = str(storage.get("graph_embedding_dtype", "float16")).strip().lower()
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported storage.graph_embedding_dtype: {value!r}")


def _embedding_indices_for_sequence_positions(
    sequence_positions: list[int],
    sequence_length: int,
    embedding_rows: int,
) -> list[int] | None:
    if sequence_length <= 0:
        return None
    if embedding_rows == sequence_length:
        offset = 0
    elif embedding_rows in {sequence_length + 1, sequence_length + 2}:
        offset = 1
    else:
        return None
    indices = [offset + int(position) - 1 for position in sequence_positions]
    if any(index < 0 or index >= embedding_rows for index in indices):
        return None
    return indices


def _exact_sequence_mapping(full_sequence: str, structure_sequence: str) -> tuple[list[int | None], str] | None:
    if not full_sequence or not structure_sequence:
        return None
    if len(full_sequence) == len(structure_sequence):
        comparable = [(a, b) for a, b in zip(full_sequence, structure_sequence) if b != "X"]
        matches = sum(1 for a, b in comparable if a == b)
        if not comparable or matches / len(comparable) >= 0.8:
            return list(range(1, len(structure_sequence) + 1)), "same_length"
    start = full_sequence.find(structure_sequence)
    if start >= 0:
        return list(range(start + 1, start + len(structure_sequence) + 1)), "substring"
    return None


def _pairwise_sequence_mapping(full_sequence: str, structure_sequence: str) -> tuple[list[int | None], str] | None:
    try:
        from Bio.Align import PairwiseAligner
    except Exception:
        return None
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -1.0
    alignments = aligner.align(full_sequence, structure_sequence)
    if not alignments:
        return None
    alignment = alignments[0]
    mapping: list[int | None] = [None] * len(structure_sequence)
    target_blocks, query_blocks = alignment.aligned
    for (target_start, target_end), (query_start, query_end) in zip(target_blocks, query_blocks):
        block_len = min(int(target_end - target_start), int(query_end - query_start))
        for offset in range(block_len):
            mapping[int(query_start) + offset] = int(target_start) + offset + 1
    return mapping, "pairwise_local"


def _align_residue_group(
    residues: list[StructureResidue],
    full_sequence: str,
    embedding_rows: int,
    min_aligned_fraction: float,
) -> StructureAlignment | None:
    if not residues or not full_sequence:
        return None
    structure_sequence = "".join(residue.aa for residue in residues)
    result = _exact_sequence_mapping(full_sequence, structure_sequence)
    if result is None:
        result = _pairwise_sequence_mapping(full_sequence, structure_sequence)
    if result is None:
        return None
    mapping, mode = result
    aligned_items = [(idx, int(position)) for idx, position in enumerate(mapping) if position is not None]
    aligned_fraction = len(aligned_items) / max(len(residues), 1)
    if not aligned_items or aligned_fraction < min_aligned_fraction:
        return None
    sequence_positions = [position for _, position in aligned_items]
    embedding_indices = _embedding_indices_for_sequence_positions(
        sequence_positions,
        sequence_length=len(full_sequence),
        embedding_rows=embedding_rows,
    )
    if embedding_indices is None:
        return None
    aligned_residues = [residues[idx] for idx, _ in aligned_items]
    return StructureAlignment(
        residues=aligned_residues,
        coords=_coords_from_residues(aligned_residues),
        sequence_positions=sequence_positions,
        embedding_indices=embedding_indices,
        alignment_mode=mode,
        aligned_residues=len(aligned_items),
        structure_residues=len(residues),
        aligned_fraction=aligned_fraction,
    )


def _best_structure_alignment(
    groups: list[list[StructureResidue]],
    full_sequence: str,
    embedding_rows: int,
    min_aligned_fraction: float,
) -> StructureAlignment | None:
    alignments = [
        alignment
        for group in groups
        if (alignment := _align_residue_group(group, full_sequence, embedding_rows, min_aligned_fraction)) is not None
    ]
    if not alignments:
        return None
    return sorted(
        alignments,
        key=lambda item: (
            item.aligned_residues,
            item.aligned_fraction,
            -abs(len(full_sequence) - item.aligned_residues),
        ),
        reverse=True,
    )[0]


def _annotation_input_available(row: dict[str, str], parsed=None) -> bool:
    """Return whether domain/motif tools ran, independent of site-label availability."""
    explicit = _norm(row.get("tool_annotation_available"))
    if explicit is not None:
        normalized = explicit.lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
        raise ValueError(f"Unsupported tool_annotation_available value: {explicit!r}")
    parsed = parsed or parse_annotation_row(row)
    return bool(
        parsed.domain_regions
        or parsed.motif_regions
        or _norm(row.get("domain_source"))
        or _norm(row.get("motif_source"))
        or _norm(row.get("domain_tool_version"))
        or _norm(row.get("motif_tool_version"))
    )


def _annotation_features(
    row: dict[str, str],
    num_nodes: int,
    sequence_positions: list[int] | None = None,
    *,
    annotation_feature_mode: str = "domain_motif",
    use_site_truth_as_input: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    parsed = parse_annotation_row(row)
    annotation_x, annotation_mask = empty_annotation(num_nodes, ANNOTATION_DIM)
    mode = str(annotation_feature_mode or "none").strip().lower()
    if mode not in {"none", "domain_motif", "all"}:
        raise ValueError(f"Unsupported annotation_feature_mode: {annotation_feature_mode}")
    if mode == "none":
        return annotation_x, annotation_mask

    include_site_truth = mode == "all" and bool(use_site_truth_as_input)
    missing_active = not parsed.active_sites and not parsed.catalytic_sites
    missing_binding = not parsed.binding_sites
    missing_domain = not parsed.domain_regions
    missing_motif = not parsed.motif_regions
    annotation_input_available = _annotation_input_available(row, parsed)

    def mark_points(points: Iterable[int], channel: int) -> None:
        point_set = {int(point) for point in points}
        if sequence_positions is not None:
            for idx, position in enumerate(sequence_positions):
                if int(position) in point_set:
                    annotation_x[idx, channel] = 1.0
                    annotation_mask[idx, channel] = 1.0
        else:
            for point in point_set:
                idx = int(point) - 1
                if 0 <= idx < num_nodes:
                    annotation_x[idx, channel] = 1.0
                    annotation_mask[idx, channel] = 1.0

    def mark_regions(regions: Iterable[tuple[int, int, str | None]], channel: int) -> None:
        for start, end, _ in regions:
            if sequence_positions is not None:
                lo = int(start)
                hi = int(end)
                for idx, position in enumerate(sequence_positions):
                    if lo <= int(position) <= hi:
                        annotation_x[idx, channel] = 1.0
                        annotation_mask[idx, channel] = 1.0
            else:
                lo = max(int(start) - 1, 0)
                hi = min(int(end), num_nodes)
                if lo < hi:
                    annotation_x[lo:hi, channel] = 1.0
                    annotation_mask[lo:hi, channel] = 1.0

    if include_site_truth:
        mark_points(parsed.active_sites, 0)
        mark_points(parsed.catalytic_sites, 1)
        mark_points(parsed.binding_sites, 2)
    if annotation_input_available:
        mark_regions(parsed.domain_regions, 3)
        mark_regions(parsed.motif_regions, 4)

    if include_site_truth:
        annotation_x[:, 8] = 1.0 if missing_active else 0.0
        annotation_x[:, 9] = 1.0 if missing_binding else 0.0
        annotation_mask[:, 8:10] = 1.0
    if annotation_input_available:
        annotation_x[:, 10] = 1.0 if missing_domain else 0.0
        annotation_x[:, 11] = 1.0 if missing_motif else 0.0
        annotation_mask[:, 10:12] = 1.0
    return annotation_x, annotation_mask


def _annotation_targets(
    row: dict[str, str],
    num_nodes: int,
    sequence_positions: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    parsed = parse_annotation_row(row)
    targets = torch.zeros((num_nodes, len(SITE_TASKS)), dtype=torch.float32)
    target_mask = torch.zeros_like(targets)

    def mark_points(points: Iterable[int], channel: int) -> None:
        point_set = {int(point) for point in points}
        if not point_set:
            return
        if sequence_positions is not None:
            for idx, position in enumerate(sequence_positions):
                if int(position) in point_set:
                    targets[idx, channel] = 1.0
        else:
            for point in point_set:
                idx = int(point) - 1
                if 0 <= idx < num_nodes:
                    targets[idx, channel] = 1.0
        if targets[:, channel].any():
            target_mask[:, channel] = 1.0

    def mark_regions(regions: Iterable[tuple[int, int, str | None]], channel: int) -> None:
        regions = list(regions)
        if not regions:
            return
        for start, end, _ in regions:
            if sequence_positions is not None:
                for idx, position in enumerate(sequence_positions):
                    if int(start) <= int(position) <= int(end):
                        targets[idx, channel] = 1.0
            else:
                lo = max(int(start) - 1, 0)
                hi = min(int(end), num_nodes)
                if lo < hi:
                    targets[lo:hi, channel] = 1.0
        if targets[:, channel].any():
            target_mask[:, channel] = 1.0

    mark_points(parsed.active_sites | parsed.catalytic_sites, 0)
    mark_points(parsed.binding_sites, 1)
    mark_regions(parsed.domain_regions, 2)
    mark_regions(parsed.motif_regions, 3)
    return targets, target_mask


def _save_label_maps(config: dict, label_maps: dict[str, dict[str, int]], parent_child_maps: dict[str, list[tuple[int, int]]]) -> None:
    out_dir = Path(config["storage"]["label_map_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "ec_label_maps.json", label_maps)
    write_json(out_dir / "parent_child_maps.json", parent_child_maps)


def _graph_stats() -> dict[str, int]:
    return {
        "written": 0,
        "skipped_existing": 0,
        "skipped_not_in_split": 0,
        "crystal_fallback_predicted": 0,
        "reused_identical": 0,
        "missing_embedding": 0,
        "transient_embedding_generated": 0,
        "missing_structure": 0,
        "mismatch": 0,
        "aligned_fragment": 0,
        "failed": 0,
        "structure_archived": 0,
        "structure_deleted": 0,
        "structure_archive_bytes_reclaimed": 0,
        "structure_archive_failed": 0,
        "structure_archive_skipped": 0,
    }


def _direct_embedding_path(root: Path, uniprot_id: str, pdb_id: str | None) -> str | None:
    embedding_dir = root / "embedding"
    candidates: list[Path] = []
    if pdb_id:
        candidates.append(embedding_dir / f"{uniprot_id}_{pdb_id}.pt")
    candidates.append(embedding_dir / f"{uniprot_id}.pt")
    found = _first_existing(str(path) for path in candidates)
    return found


def _direct_structure_path(root: Path, uniprot_id: str, pdb_id: str | None, predicted_only: bool = False) -> str | None:
    candidates: list[Path] = []
    complete_dir = root / "complete"
    crystal_dir = root / "crystal"
    predicted_dir = root / "predicted"
    if not predicted_only:
        if pdb_id:
            pdb_lower = pdb_id.lower()
            for suffix in (".pdb", ".cif", ".mmcif"):
                candidates.append(complete_dir / f"{uniprot_id}_{pdb_id}{suffix}")
                candidates.append(complete_dir / f"{uniprot_id}_{pdb_lower}{suffix}")
                candidates.append(crystal_dir / f"{uniprot_id}_{pdb_lower}{suffix}")
                candidates.append(crystal_dir / f"{uniprot_id}_{pdb_id}{suffix}")
        for suffix in (".pdb", ".cif", ".mmcif"):
            candidates.append(complete_dir / f"{uniprot_id}{suffix}")
    for suffix in (".pdb", ".cif", ".mmcif"):
        candidates.append(predicted_dir / f"{uniprot_id}{suffix}")
    return _first_existing(str(path) for path in candidates)


class ThinGraphBuildSession:
    """Stateful graph builder used by deploy-data pipeline mode."""

    def __init__(
        self,
        config: dict,
        manifest: Any | None = None,
        dataset_names: Iterable[str] | None = None,
        *,
        dry_run: bool = False,
        overwrite: bool = False,
    ) -> None:
        ensure_dirs(config)
        self.config = config
        self.datasets = config.get("datasets", {}) or {}
        self.selected = list(dataset_names or self.datasets.keys())
        self.save_dir = Path(config["storage"]["save_dir"])
        graph_config = config.get("graph", {}) or {}
        self.cutoff = float(graph_config.get("cutoff", 10.0))
        self.max_embedding_structure_delta = int(graph_config.get("max_embedding_structure_delta", 32))
        self.use_sequence_alignment = bool(graph_config.get("use_sequence_structure_alignment", True))
        self.min_aligned_fraction = float(graph_config.get("min_structure_alignment_fraction", 0.8))
        feature_config = config.get("features", {}) or {}
        self.annotation_feature_mode = str(feature_config.get("annotation_feature_mode", "domain_motif"))
        self.use_site_truth_as_input = bool(feature_config.get("use_site_truth_as_input", False))
        storage_config = config.get("storage", {}) or {}
        self.deployment_config = config.get("deployment", {}) or {}
        self.predicted_structure_only = bool(self.deployment_config.get("predicted_structure_only", False))
        self.save_embedding_in_graph = bool(storage_config.get("save_embedding_in_graph", False))
        self.graph_feature_dtype = _graph_feature_dtype(config)
        self.allow_transient_embedding = self.save_embedding_in_graph and bool(self.deployment_config.get("allow_generate_embedding", False))
        self.dry_run = dry_run
        self.overwrite = overwrite
        self.split_keep_set = _load_split_graph_keep_set(config)
        dedup_reference_config = graph_config.get("dedup_identical_against")
        self.dedup_reference = Path(dedup_reference_config) if dedup_reference_config else None
        self.transient_state: dict[str, Any] = {}
        self.stats = _graph_stats()
        self.processed = 0
        self.lock = threading.Lock()

        train_rows = _read_csv_rows(self.datasets["train"]["csv"])
        self.label_maps = build_label_maps(row.get("ec_numbers") for row in train_rows)
        self.parent_child_maps = build_parent_child_maps(self.label_maps)
        if not dry_run:
            _save_label_maps(config, self.label_maps, self.parent_child_maps)
        if manifest is None:
            manifest = build_manifest(config, dataset_names=self.selected)
        self.exact, self.by_uniprot = _build_resource_index(_manifest_rows(manifest))

    def load_or_generate_embedding(self, row: dict[str, str], path: str | None) -> tuple[torch.Tensor | None, str | None, bool]:
        if path:
            return load_embedding(path, dtype=torch.float32), path, False
        if not self.allow_transient_embedding:
            return None, None, False
        sequence = _clean_sequence(row.get("sequence"))
        if not sequence:
            return None, None, False
        if "model" not in self.transient_state:
            from ..deployment.resource_manager import _embedding_validation_issue, _extract_residue_embedding, _load_esm3_model

            esm_config = self.config.get("esm", {}) or {}
            device = str(esm_config.get("device") or self.deployment_config.get("embedding_device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
            self.transient_state["device"] = device
            self.transient_state["fail_abs_threshold"] = float(esm_config.get("embedding_fail_abs_threshold", 1e20))
            self.transient_state["_extract"] = _extract_residue_embedding
            self.transient_state["_validate"] = _embedding_validation_issue
            self.transient_state["model"] = _load_esm3_model(self.config, device)
        try:
            embedding, _source = self.transient_state["_extract"](self.transient_state["model"], sequence, self.transient_state["device"])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logging.warning("CUDA OOM during transient embedding, retrying after cache clear")
            embedding, _source = self.transient_state["_extract"](self.transient_state["model"], sequence, self.transient_state["device"])
        embedding = embedding.detach().cpu().to(dtype=torch.float32)
        issue = self.transient_state["_validate"](embedding, fail_abs_threshold=self.transient_state["fail_abs_threshold"])
        if issue:
            raise ValueError(f"transient embedding invalid: {issue}")
        return embedding, None, True

    def build_row(self, dataset: str, row: dict[str, str]) -> bool:
        with self.lock:
            self.processed += 1
            uniprot_id = _norm(row.get("uniprot_id"))
            if not uniprot_id:
                self.stats["failed"] += 1
                return False
            pdb_id = _norm(row.get("pdb_id"))
            if self.split_keep_set is not None and (f"{uniprot_id}_{pdb_id}" if pdb_id else uniprot_id) not in self.split_keep_set:
                self.stats["skipped_not_in_split"] += 1
                return True
            output_subdir = self.datasets[dataset].get("output_subdir", dataset)
            root = self.save_dir / output_subdir
            graph_name = f"{uniprot_id}_{pdb_id}.pt" if pdb_id else f"{uniprot_id}.pt"
            out_path = root / "graph" / graph_name
            if out_path.exists() and not self.overwrite:
                self.stats["skipped_existing"] += 1
                return True
            embedding_path = _select_embedding(dataset, uniprot_id, pdb_id, self.exact, self.by_uniprot)
            if not embedding_path:
                embedding_path = _direct_embedding_path(root, uniprot_id, pdb_id)
            structure_path = _direct_structure_path(root, uniprot_id, pdb_id, predicted_only=self.predicted_structure_only)
            if not structure_path:
                structure_path = _select_structure(dataset, uniprot_id, pdb_id, self.exact, self.by_uniprot, predicted_only=self.predicted_structure_only)
            if not structure_path:
                self.stats["missing_structure"] += 1
                return False
            if self.dry_run:
                self.stats["written"] += 1
                return True
            try:
                embedding, embedding_source_path, generated_transient = self.load_or_generate_embedding(row, embedding_path)
                if embedding is None:
                    self.stats["missing_embedding"] += 1
                    return False
                if generated_transient:
                    self.stats["transient_embedding_generated"] += 1
                embedding_rows = int(embedding.size(0))
                row_sequence = _clean_sequence(row.get("sequence"))
                structure_path = _maybe_predicted_fallback(
                    structure_path,
                    root,
                    uniprot_id,
                    row_sequence,
                    embedding_rows,
                    self.use_sequence_alignment,
                    self.min_aligned_fraction,
                )
                residue_groups = _load_residue_groups_from_structure(structure_path)
                alignment = None
                if self.use_sequence_alignment and row_sequence:
                    alignment = _best_structure_alignment(
                        residue_groups,
                        full_sequence=row_sequence,
                        embedding_rows=embedding_rows,
                        min_aligned_fraction=self.min_aligned_fraction,
                    )
                if alignment is not None:
                    graph_residues = alignment.residues
                    coords = alignment.coords
                    sequence_positions = alignment.sequence_positions
                    embedding_indices = alignment.embedding_indices
                    slice_pair = (min(embedding_indices), max(embedding_indices) + 1)
                    if alignment.aligned_residues != len(row_sequence):
                        self.stats["aligned_fragment"] += 1
                    edge_index, edge_attr = build_edges_from_coords(coords, cutoff=self.cutoff, sequence_positions=sequence_positions)
                    annotation_x, annotation_mask = _annotation_features(
                        row,
                        coords.shape[0],
                        sequence_positions=sequence_positions,
                        annotation_feature_mode=self.annotation_feature_mode,
                        use_site_truth_as_input=self.use_site_truth_as_input,
                    )
                    site_targets, site_target_mask = _annotation_targets(row, coords.shape[0], sequence_positions=sequence_positions)
                    alignment_metadata: dict[str, Any] = {
                        "embedding_indices": torch.tensor(embedding_indices, dtype=torch.long),
                        "sequence_positions": torch.tensor(sequence_positions, dtype=torch.long),
                        "alignment_mode": alignment.alignment_mode,
                        "aligned_residues": alignment.aligned_residues,
                        "structure_residues": alignment.structure_residues,
                        "aligned_fraction": float(alignment.aligned_fraction),
                    }
                else:
                    graph_residues = _select_coord_group(
                        residue_groups,
                        target_lengths={embedding_rows, embedding_rows - 1, embedding_rows - 2},
                    )
                    coords = _coords_from_residues(graph_residues)
                    slice_pair = _embedding_slice_from_rows(
                        embedding_rows,
                        coords.shape[0],
                        max_delta=self.max_embedding_structure_delta,
                    )
                    if slice_pair is None:
                        self.stats["mismatch"] += 1
                        return False
                    edge_index, edge_attr = build_edges_from_coords(coords, cutoff=self.cutoff)
                    annotation_x, annotation_mask = _annotation_features(
                        row,
                        coords.shape[0],
                        annotation_feature_mode=self.annotation_feature_mode,
                        use_site_truth_as_input=self.use_site_truth_as_input,
                    )
                    site_targets, site_target_mask = _annotation_targets(row, coords.shape[0])
                    alignment_metadata = {}
                if alignment is not None:
                    node_features = embedding[torch.as_tensor(embedding_indices, dtype=torch.long)]
                else:
                    node_features = embedding[slice_pair[0] : slice_pair[1]]
                if node_features.size(0) != coords.shape[0]:
                    self.stats["mismatch"] += 1
                    return False
                node_confidence, node_confidence_mask, node_confidence_source = _residue_confidence(
                    graph_residues,
                    structure_path,
                )
                node_features = node_features.to(dtype=self.graph_feature_dtype) if self.save_embedding_in_graph else None
                labels = encode_ec_values(row.get("ec_numbers"), self.label_maps)
                metadata = {
                    "uniprot_id": uniprot_id,
                    "pdb_id": pdb_id or "",
                    "protein_idx": self.processed - 1,
                    "embedding_start": slice_pair[0],
                    "embedding_end": slice_pair[1],
                    "embedding_storage": "graph_x" if self.save_embedding_in_graph else "external_path",
                    "embedding_source_path": str(embedding_source_path or ""),
                    "node_count": coords.shape[0],
                    "node_confidence_source": node_confidence_source,
                    "annotation_schema_version": 4,
                    "annotation_feature_mode": self.annotation_feature_mode,
                    "site_truth_as_input": self.use_site_truth_as_input,
                    "annotation_available": _annotation_input_available(row),
                    "tool_annotation_available": _annotation_input_available(row),
                    "annotation_provenance_json": json.dumps(
                        {
                            "active_site_source": row.get("active_site_source") or row.get("annotation_source") or "",
                            "binding_site_source": row.get("binding_site_source") or row.get("annotation_source") or "",
                            "domain_source": row.get("domain_source") or "",
                            "motif_source": row.get("motif_source") or "",
                            "domain_tool_version": row.get("domain_tool_version") or "",
                            "motif_tool_version": row.get("motif_tool_version") or "",
                        },
                        sort_keys=True,
                    ),
                }
                metadata.update(alignment_metadata)
                save_thin_graph(
                    out_path,
                    embedding_path=embedding_source_path,
                    structure_path=structure_path,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    labels=labels,
                    annotation_x=annotation_x,
                    annotation_mask=annotation_mask,
                    site_targets=site_targets,
                    site_target_mask=site_target_mask,
                    node_confidence=node_confidence,
                    node_confidence_mask=node_confidence_mask,
                    node_features=node_features,
                    metadata=metadata,
                    overwrite=self.overwrite,
                    dedup_reference=self.dedup_reference,
                )
                self.stats["written"] += 1
                if out_path.is_symlink():
                    self.stats["reused_identical"] += 1
                if bool(self.deployment_config.get("compress_structures_after_graph", False)):
                    try:
                        archive_result = archive_graph_input_structures(
                            self.config,
                            dataset=dataset,
                            row=row,
                            graph_path=out_path,
                            structure_path=structure_path,
                        )
                        if archive_result.status in {"archived", "archived_deleted"}:
                            self.stats["structure_archived"] += 1
                            self.stats["structure_deleted"] += archive_result.deleted_files
                            self.stats["structure_archive_bytes_reclaimed"] += archive_result.bytes_reclaimed
                        elif archive_result.status != "disabled":
                            self.stats["structure_archive_skipped"] += 1
                    except Exception as archive_exc:
                        self.stats["structure_archive_failed"] += 1
                        logging.warning("Structure archive failed after graph write for %s/%s: %s", uniprot_id, pdb_id or "", archive_exc)
                return True
            except Exception as exc:
                self.stats["failed"] += 1
                logging.warning("Pipeline graph build failed for %s/%s: %s", uniprot_id, pdb_id or "", exc)
                return False


def _load_split_graph_keep_set(config: dict) -> set[str] | None:
    graph_config = config.get("graph", {}) or {}
    if not bool(graph_config.get("write_split_graphs_only", False)):
        return None
    split_dir = Path(str((config.get("storage", {}) or {}).get("split_dir", "")))
    if not str(split_dir) or not split_dir.exists():
        logging.warning("write_split_graphs_only enabled but storage.split_dir missing: %s", split_dir)
        return None
    keep: set[str] = set()
    for path in sorted(split_dir.glob("*.txt")):
        for line in path.read_text(errors="ignore").splitlines():
            name = line.strip()
            if not name:
                continue
            if "," in name:
                name = name.split(",")[0].strip()
            name = Path(name).name
            if name.endswith(".pt"):
                name = name[:-3]
            if name and name.lower() not in {"name", "protein_id", "id", "uniprot_id", "graph"}:
                keep.add(name)
    logging.info("Split graph filter enabled: %d keep names from %s", len(keep), split_dir)
    return keep


def _maybe_predicted_fallback(structure_path, root, uniprot_id, row_sequence, embedding_rows, use_alignment, min_fraction):
    """Without the completion pipeline, crystals whose coverage is too low for
    alignment cannot produce a graph after structure fusion.
    Fall back to the full-coverage predicted structure, mirroring what a
    real deployment without completion would do."""
    if not structure_path or not use_alignment or not row_sequence:
        return structure_path
    sp = str(structure_path)
    if "/crystal/" not in sp:
        return structure_path
    try:
        groups = _load_residue_groups_from_structure(sp)
        alignment = _best_structure_alignment(
            groups,
            full_sequence=row_sequence,
            embedding_rows=embedding_rows,
            min_aligned_fraction=min_fraction,
        )
    except Exception:
        return structure_path
    if alignment is not None:
        return structure_path
    predicted_path = Path(root) / "predicted" / f"{uniprot_id}.cif"
    if predicted_path.exists() and str(predicted_path) != sp:
        logging.info("Crystal coverage too low for alignment; using predicted structure for %s", uniprot_id)
        return str(predicted_path)
    return structure_path


def prepare_thin_graphs(
    config: dict,
    manifest: Any | None = None,
    dataset_names: Iterable[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    ensure_dirs(config)
    datasets = config.get("datasets", {}) or {}
    selected = list(dataset_names or datasets.keys())
    save_dir = Path(config["storage"]["save_dir"])
    graph_config = config.get("graph", {}) or {}
    cutoff = float(graph_config.get("cutoff", 10.0))
    max_embedding_structure_delta = int(graph_config.get("max_embedding_structure_delta", 32))
    use_sequence_alignment = bool(graph_config.get("use_sequence_structure_alignment", True))
    min_aligned_fraction = float(graph_config.get("min_structure_alignment_fraction", 0.8))
    feature_config = config.get("features", {}) or {}
    annotation_feature_mode = str(feature_config.get("annotation_feature_mode", "domain_motif"))
    use_site_truth_as_input = bool(feature_config.get("use_site_truth_as_input", False))
    storage_config = config.get("storage", {}) or {}
    deployment_config = config.get("deployment", {}) or {}
    predicted_structure_only = bool(deployment_config.get("predicted_structure_only", False))
    save_embedding_in_graph = bool(storage_config.get("save_embedding_in_graph", False))
    graph_feature_dtype = _graph_feature_dtype(config)
    allow_transient_embedding = save_embedding_in_graph and bool(deployment_config.get("allow_generate_embedding", False))
    transient_state: dict[str, Any] = {}
    split_keep_set = _load_split_graph_keep_set(config)
    dedup_reference_config = graph_config.get("dedup_identical_against")
    dedup_reference = Path(dedup_reference_config) if dedup_reference_config else None

    def load_or_generate_embedding(row: dict[str, str], path: str | None) -> tuple[torch.Tensor | None, str | None, bool]:
        if path:
            return load_embedding(path, dtype=torch.float32), path, False
        if not allow_transient_embedding:
            return None, None, False
        sequence = _clean_sequence(row.get("sequence"))
        if not sequence:
            return None, None, False
        if "model" not in transient_state:
            from ..deployment.resource_manager import _embedding_validation_issue, _extract_residue_embedding, _load_esm3_model

            esm_config = config.get("esm", {}) or {}
            device = str(esm_config.get("device") or deployment_config.get("embedding_device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
            transient_state["device"] = device
            transient_state["fail_abs_threshold"] = float(esm_config.get("embedding_fail_abs_threshold", 1e20))
            transient_state["_extract"] = _extract_residue_embedding
            transient_state["_validate"] = _embedding_validation_issue
            transient_state["model"] = _load_esm3_model(config, device)
        try:
            embedding, _source = transient_state["_extract"](transient_state["model"], sequence, transient_state["device"])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logging.warning("CUDA OOM during transient embedding, retrying after cache clear")
            embedding, _source = transient_state["_extract"](transient_state["model"], sequence, transient_state["device"])
        embedding = embedding.detach().cpu().to(dtype=torch.float32)
        issue = transient_state["_validate"](embedding, fail_abs_threshold=transient_state["fail_abs_threshold"])
        if issue:
            raise ValueError(f"transient embedding invalid: {issue}")
        return embedding, None, True

    train_csv = datasets["train"]["csv"]
    train_rows = _read_csv_rows(train_csv)
    label_maps = build_label_maps(row.get("ec_numbers") for row in train_rows)
    parent_child_maps = build_parent_child_maps(label_maps)
    if not dry_run:
        _save_label_maps(config, label_maps, parent_child_maps)

    if manifest is None:
        manifest = build_manifest(config, dataset_names=selected)
    exact, by_uniprot = _build_resource_index(_manifest_rows(manifest))

    stats = {
        "written": 0,
        "skipped_existing": 0,
        "skipped_not_in_split": 0,
        "crystal_fallback_predicted": 0,
        "reused_identical": 0,
        "missing_embedding": 0,
        "transient_embedding_generated": 0,
        "missing_structure": 0,
        "mismatch": 0,
        "aligned_fragment": 0,
        "failed": 0,
        "structure_archived": 0,
        "structure_deleted": 0,
        "structure_archive_bytes_reclaimed": 0,
        "structure_archive_failed": 0,
        "structure_archive_skipped": 0,
    }
    failure_reasons: Counter[str] = Counter()
    failure_examples: defaultdict[str, list[str]] = defaultdict(list)
    processed = 0
    for dataset in selected:
        csv_path = datasets[dataset]["csv"]
        rows = _read_csv_rows(csv_path)
        if limit is not None and processed >= limit:
            logging.info("Reached graph limit=%s", limit)
            return stats
        if limit is not None:
            rows = rows[: max(limit - processed, 0)]
        progress = tqdm(rows, desc=f"Building {dataset} graphs", unit="protein", dynamic_ncols=True)
        for row in progress:
            processed += 1
            uniprot_id = _norm(row.get("uniprot_id"))
            if not uniprot_id:
                stats["failed"] += 1
                continue
            pdb_id = _norm(row.get("pdb_id"))
            if split_keep_set is not None and (f"{uniprot_id}_{pdb_id}" if pdb_id else uniprot_id) not in split_keep_set:
                stats["skipped_not_in_split"] += 1
                continue
            graph_name = f"{uniprot_id}_{pdb_id}.pt" if pdb_id else f"{uniprot_id}.pt"
            output_subdir = datasets[dataset].get("output_subdir", dataset)
            out_path = save_dir / output_subdir / "graph" / graph_name
            if out_path.exists() and not overwrite:
                stats["skipped_existing"] += 1
                continue
            embedding_path = _select_embedding(dataset, uniprot_id, pdb_id, exact, by_uniprot)
            structure_path = _select_structure(dataset, uniprot_id, pdb_id, exact, by_uniprot, predicted_only=predicted_structure_only)
            if not structure_path:
                structure_path = _direct_structure_path(save_dir / output_subdir, uniprot_id, pdb_id, predicted_only=predicted_structure_only)
            if not structure_path:
                stats["missing_structure"] += 1
                continue
            if dry_run:
                stats["written"] += 1
                continue
            try:
                embedding, embedding_source_path, generated_transient = load_or_generate_embedding(row, embedding_path)
                if embedding is None:
                    stats["missing_embedding"] += 1
                    continue
                if generated_transient:
                    stats["transient_embedding_generated"] += 1
                embedding_rows = int(embedding.size(0))
                row_sequence = _clean_sequence(row.get("sequence"))
                structure_path = _maybe_predicted_fallback(
                    structure_path,
                    save_dir / output_subdir,
                    uniprot_id,
                    row_sequence,
                    embedding_rows,
                    use_sequence_alignment,
                    min_aligned_fraction,
                )
                residue_groups = _load_residue_groups_from_structure(structure_path)
                alignment = None
                if use_sequence_alignment and row_sequence:
                    alignment = _best_structure_alignment(
                        residue_groups,
                        full_sequence=row_sequence,
                        embedding_rows=embedding_rows,
                        min_aligned_fraction=min_aligned_fraction,
                    )
                if alignment is not None:
                    graph_residues = alignment.residues
                    coords = alignment.coords
                    sequence_positions = alignment.sequence_positions
                    embedding_indices = alignment.embedding_indices
                    slice_pair = (min(embedding_indices), max(embedding_indices) + 1)
                    if alignment.aligned_residues != len(row_sequence):
                        stats["aligned_fragment"] += 1
                    edge_index, edge_attr = build_edges_from_coords(
                        coords,
                        cutoff=cutoff,
                        sequence_positions=sequence_positions,
                    )
                    annotation_x, annotation_mask = _annotation_features(
                        row,
                        coords.shape[0],
                        sequence_positions=sequence_positions,
                        annotation_feature_mode=annotation_feature_mode,
                        use_site_truth_as_input=use_site_truth_as_input,
                    )
                    site_targets, site_target_mask = _annotation_targets(
                        row,
                        coords.shape[0],
                        sequence_positions=sequence_positions,
                    )
                    alignment_metadata: dict[str, Any] = {
                        "embedding_indices": torch.tensor(embedding_indices, dtype=torch.long),
                        "sequence_positions": torch.tensor(sequence_positions, dtype=torch.long),
                        "alignment_mode": alignment.alignment_mode,
                        "aligned_residues": alignment.aligned_residues,
                        "structure_residues": alignment.structure_residues,
                        "aligned_fraction": float(alignment.aligned_fraction),
                    }
                else:
                    graph_residues = _select_coord_group(
                        residue_groups,
                        target_lengths={embedding_rows, embedding_rows - 1, embedding_rows - 2},
                    )
                    coords = _coords_from_residues(graph_residues)
                    slice_pair = _embedding_slice_from_rows(
                        embedding_rows,
                        coords.shape[0],
                        max_delta=max_embedding_structure_delta,
                    )
                    if slice_pair is None:
                        stats["mismatch"] += 1
                        continue
                    edge_index, edge_attr = build_edges_from_coords(coords, cutoff=cutoff)
                    annotation_x, annotation_mask = _annotation_features(
                        row,
                        coords.shape[0],
                        annotation_feature_mode=annotation_feature_mode,
                        use_site_truth_as_input=use_site_truth_as_input,
                    )
                    site_targets, site_target_mask = _annotation_targets(row, coords.shape[0])
                    alignment_metadata = {}
                if alignment is not None:
                    node_features = embedding[torch.as_tensor(embedding_indices, dtype=torch.long)]
                else:
                    node_features = embedding[slice_pair[0] : slice_pair[1]]
                if node_features.size(0) != coords.shape[0]:
                    stats["mismatch"] += 1
                    continue
                node_confidence, node_confidence_mask, node_confidence_source = _residue_confidence(
                    graph_residues,
                    structure_path,
                )
                node_features = node_features.to(dtype=graph_feature_dtype) if save_embedding_in_graph else None
                labels = encode_ec_values(row.get("ec_numbers"), label_maps)
                metadata = {
                    "uniprot_id": uniprot_id,
                    "pdb_id": pdb_id or "",
                    "protein_idx": processed - 1,
                    "embedding_start": slice_pair[0],
                    "embedding_end": slice_pair[1],
                    "embedding_storage": "graph_x" if save_embedding_in_graph else "external_path",
                    "embedding_source_path": str(embedding_source_path or ""),
                    "node_count": coords.shape[0],
                    "node_confidence_source": node_confidence_source,
                    "annotation_schema_version": 4,
                    "annotation_feature_mode": annotation_feature_mode,
                    "site_truth_as_input": use_site_truth_as_input,
                    "annotation_available": _annotation_input_available(row),
                    "tool_annotation_available": _annotation_input_available(row),
                    "annotation_provenance_json": json.dumps(
                        {
                            "active_site_source": row.get("active_site_source") or row.get("annotation_source") or "",
                            "binding_site_source": row.get("binding_site_source") or row.get("annotation_source") or "",
                            "domain_source": row.get("domain_source") or "",
                            "motif_source": row.get("motif_source") or "",
                            "domain_tool_version": row.get("domain_tool_version") or "",
                            "motif_tool_version": row.get("motif_tool_version") or "",
                        },
                        sort_keys=True,
                    ),
                }
                metadata.update(alignment_metadata)
                save_thin_graph(
                    out_path,
                    embedding_path=embedding_source_path,
                    structure_path=structure_path,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    labels=labels,
                    annotation_x=annotation_x,
                    annotation_mask=annotation_mask,
                    site_targets=site_targets,
                    site_target_mask=site_target_mask,
                    node_confidence=node_confidence,
                    node_confidence_mask=node_confidence_mask,
                    node_features=node_features,
                    metadata=metadata,
                    overwrite=overwrite,
                    dedup_reference=dedup_reference,
                )
                stats["written"] += 1
                if out_path.is_symlink():
                    stats["reused_identical"] += 1
                if bool(deployment_config.get("compress_structures_after_graph", False)):
                    try:
                        archive_result = archive_graph_input_structures(
                            config,
                            dataset=dataset,
                            row=row,
                            graph_path=out_path,
                            structure_path=structure_path,
                        )
                        if archive_result.status in {"archived", "archived_deleted"}:
                            stats["structure_archived"] += 1
                            stats["structure_deleted"] += archive_result.deleted_files
                            stats["structure_archive_bytes_reclaimed"] += archive_result.bytes_reclaimed
                        elif archive_result.status != "disabled":
                            stats["structure_archive_skipped"] += 1
                            logging.debug(
                                "Structure archive skipped for %s/%s: %s",
                                uniprot_id,
                                pdb_id or "",
                                archive_result.status,
                            )
                    except Exception as archive_exc:
                        stats["structure_archive_failed"] += 1
                        logging.warning(
                            "Structure archive failed after graph write for %s/%s: %s",
                            uniprot_id,
                            pdb_id or "",
                            archive_exc,
                        )
            except Exception as exc:
                reason = str(exc)
                failure_reasons[reason] += 1
                if len(failure_examples[reason]) < 5:
                    failure_examples[reason].append(f"{uniprot_id}/{pdb_id or ''}")
                stats["failed"] += 1
    if failure_reasons:
        logging.warning(
            "Graph build failures summarized by reason: %s; examples: %s",
            dict(failure_reasons),
            dict(failure_examples),
        )
    return stats
