from __future__ import annotations

import copy
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

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
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
}


@dataclass
class ResidueRecord:
    index: int
    aa: str
    residue: Any
    plddt: float | None


@dataclass
class ChainRecord:
    chain_id: str
    sequence: str
    residues: list[ResidueRecord]


@dataclass
class CompletionResult:
    output_path: Path
    mode: str
    crystal_residues: int
    predicted_residues: int
    skipped_predicted_low_plddt: int
    anchors: int
    rmsd: float | None


def _clean_sequence(value: str | None) -> str:
    return "".join(str(value or "").split()).upper().replace("*", "")


def _aa3_to_1(value: Any) -> str:
    return AA3_TO_1.get(str(value or "").strip().upper(), "X")


def _valid_structure(path: Path | None) -> bool:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle):
                if line_number >= 200:
                    break
                text = line.strip()
                if not text:
                    continue
                if path.suffix.lower() in {".cif", ".mmcif"}:
                    return text.startswith("data_")
                if text.lower().startswith(("<!doctype html", "<html")):
                    return False
                if text.startswith(("ATOM", "HETATM", "HEADER", "TITLE", "MODEL")):
                    return True
    except OSError:
        return False
    return False


def _parser_for(path: Path):
    try:
        from Bio.PDB import MMCIFParser, PDBParser
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Biopython is required for structure completion.") from exc
    if path.suffix.lower() in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def _load_chains(path: Path) -> list[ChainRecord]:
    try:
        from Bio.PDB.Polypeptide import is_aa
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Biopython is required for structure completion.") from exc
    structure = _parser_for(path).get_structure(path.stem, str(path))
    chains: list[ChainRecord] = []
    for model in structure:
        for chain in model:
            residues: list[ResidueRecord] = []
            for residue in chain:
                if not is_aa(residue, standard=False):
                    continue
                aa = _aa3_to_1(residue.get_resname())
                atoms = list(residue.get_atoms())
                plddt = float(np.mean([atom.get_bfactor() for atom in atoms])) if atoms else None
                residues.append(ResidueRecord(len(residues), aa, residue, plddt))
            if residues:
                chains.append(ChainRecord(str(chain.id), "".join(r.aa for r in residues), residues))
        break
    if not chains:
        raise ValueError(f"No protein residues parsed from {path}")
    return chains


def _alignment_map(query: str, target: str) -> tuple[dict[int, int], int]:
    try:
        from Bio.Align import PairwiseAligner
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Biopython PairwiseAligner is required for structure completion.") from exc
    query = _clean_sequence(query)
    target = _clean_sequence(target)
    if not query or not target:
        return {}, 0
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(target, query)[0]
    mapping: dict[int, int] = {}
    matches = 0
    for (target_start, target_end), (query_start, query_end) in zip(*alignment.aligned):
        length = min(int(target_end - target_start), int(query_end - query_start))
        for offset in range(length):
            q_idx = int(query_start) + offset
            t_pos = int(target_start) + offset + 1
            mapping[q_idx] = t_pos
            if target[t_pos - 1] == query[q_idx]:
                matches += 1
    return mapping, matches


def _select_chain(path: Path, full_sequence: str) -> tuple[ChainRecord, dict[int, int], int]:
    chains = _load_chains(path)
    if not full_sequence:
        chain = max(chains, key=lambda c: len(c.residues))
        return chain, {record.index: record.index + 1 for record in chain.residues}, len(chain.residues)
    scored: list[tuple[int, int, ChainRecord, dict[int, int]]] = []
    for chain in chains:
        mapping, matches = _alignment_map(chain.sequence, full_sequence)
        scored.append((matches, len(mapping), chain, mapping))
    matches, _, chain, mapping = max(scored, key=lambda item: (item[0], item[1], len(item[2].residues)))
    return chain, mapping, matches


def _ca_atom(record: ResidueRecord):
    return record.residue["CA"] if "CA" in record.residue else None


def _apply_superposition(
    crystal_by_pos: dict[int, ResidueRecord],
    predicted_by_pos: dict[int, ResidueRecord],
    min_anchor_residues: int,
) -> tuple[int, float | None]:
    try:
        from Bio.PDB import Superimposer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Biopython is required for structure completion.") from exc
    fixed = []
    moving = []
    for pos in sorted(set(crystal_by_pos) & set(predicted_by_pos)):
        ca_fixed = _ca_atom(crystal_by_pos[pos])
        ca_moving = _ca_atom(predicted_by_pos[pos])
        if ca_fixed is not None and ca_moving is not None:
            fixed.append(ca_fixed)
            moving.append(ca_moving)
    if len(fixed) < min_anchor_residues:
        return len(fixed), None
    sup = Superimposer()
    sup.set_atoms(fixed, moving)
    all_atoms = []
    for record in predicted_by_pos.values():
        all_atoms.extend(record.residue.get_atoms())
    sup.apply(all_atoms)
    return len(fixed), float(sup.rms)


def _write_pdb(output_path: Path, residues: Iterable[tuple[int, ResidueRecord]]) -> int:
    try:
        from Bio.PDB import Chain, Model, PDBIO, Structure
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Biopython is required for structure completion.") from exc
    structure = Structure.Structure(output_path.stem)
    model = Model.Model(0)
    chain = Chain.Chain("A")
    written = 0
    for new_index, (_, record) in enumerate(residues, start=1):
        residue = copy.deepcopy(record.residue)
        if hasattr(residue, "detach_parent"):
            residue.detach_parent()
        residue.id = (" ", new_index, " ")
        try:
            chain.add(residue)
        except Exception:
            logging.debug("Skipping duplicate residue while writing %s at output index %d", output_path, new_index)
            continue
        written += 1
    model.add(chain)
    structure.add(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path))
    return written


def _write_metadata(output_path: Path, payload: dict[str, Any]) -> None:
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_predicted_only(predicted_path: Path, output_path: Path, mode: str, overwrite: bool) -> CompletionResult | None:
    if output_path.exists() and not overwrite:
        return CompletionResult(output_path, mode, 0, 0, 0, 0, None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_suffix = predicted_path.suffix.lower()
    output_suffix = output_path.suffix.lower()
    if source_suffix in {".cif", ".mmcif"} and output_suffix == ".pdb":
        try:
            from Bio.PDB import PDBIO
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Biopython is required for mmCIF-to-PDB conversion.") from exc
        structure = _parser_for(predicted_path).get_structure(predicted_path.stem, str(predicted_path))
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(output_path))
    else:
        shutil.copy2(predicted_path, output_path)
    _write_metadata(
        output_path,
        {
            "mode": mode,
            "crystal_path": "",
            "predicted_path": str(predicted_path),
            "crystal_residues": 0,
            "predicted_residues": None,
            "note": "No usable experimental crystal structure was available; complete structure is the predicted model.",
        },
    )
    return CompletionResult(output_path, mode, 0, 0, 0, 0, None)


def complete_structure_if_needed(
    crystal_path: str | Path | None,
    predicted_path: str | Path | None,
    output_path: str | Path,
    allow_completion: bool = False,
    overwrite: bool = False,
    full_sequence: str | None = None,
    min_predicted_plddt: float = 50.0,
    min_anchor_residues: int = 8,
    max_anchor_rmsd: float = 8.0,
) -> CompletionResult | None:
    output_path = Path(output_path)
    if not allow_completion:
        logging.info("Completion disabled; not generating %s", output_path)
        return None
    if output_path.exists() and not overwrite:
        logging.info("Completed structure exists, skipping: %s", output_path)
        return CompletionResult(output_path, "existing", 0, 0, 0, 0, None)

    crystal = Path(crystal_path) if crystal_path else None
    predicted = Path(predicted_path) if predicted_path else None
    if not _valid_structure(predicted):
        logging.info("No valid predicted structure available for completion: %s", predicted or "")
        return None
    if not _valid_structure(crystal):
        return _copy_predicted_only(predicted, output_path, "alphafold_only", overwrite)

    sequence = _clean_sequence(full_sequence)
    crystal_chain, crystal_map, crystal_matches = _select_chain(crystal, sequence)
    predicted_chain, predicted_map, predicted_matches = _select_chain(predicted, sequence or crystal_chain.sequence)
    crystal_by_pos = {crystal_map[r.index]: r for r in crystal_chain.residues if r.index in crystal_map}
    predicted_by_pos = {predicted_map[r.index]: r for r in predicted_chain.residues if r.index in predicted_map}
    if not crystal_by_pos:
        logging.warning("No crystal residues mapped to target sequence for %s", crystal)
        return _copy_predicted_only(predicted, output_path, "alphafold_only_unmapped_crystal", overwrite)
    if not predicted_by_pos:
        logging.warning("No predicted residues mapped to target sequence for %s", predicted)
        return None

    anchors, rmsd = _apply_superposition(crystal_by_pos, predicted_by_pos, min_anchor_residues=min_anchor_residues)
    if anchors < min_anchor_residues:
        logging.warning(
            "Crystal graft skipped for %s: only %d alignment anchors; using predicted-only structure",
            output_path,
            anchors,
        )
        return _copy_predicted_only(
            predicted,
            output_path,
            "alphafold_only_insufficient_anchors",
            overwrite,
        )
    if rmsd is not None and rmsd > max_anchor_rmsd:
        logging.warning(
            "Crystal graft skipped for %s: anchor RMSD %.3f > %.3f; using predicted-only structure",
            output_path,
            rmsd,
            max_anchor_rmsd,
        )
        return _copy_predicted_only(
            predicted,
            output_path,
            "alphafold_only_high_anchor_rmsd",
            overwrite,
        )

    merged: list[tuple[int, ResidueRecord]] = []
    predicted_added = 0
    low_plddt = 0
    max_pos = max(max(crystal_by_pos), max(predicted_by_pos))
    for pos in range(1, max_pos + 1):
        if pos in crystal_by_pos:
            merged.append((pos, crystal_by_pos[pos]))
            continue
        record = predicted_by_pos.get(pos)
        if record is None:
            continue
        if record.plddt is not None and record.plddt < min_predicted_plddt:
            low_plddt += 1
            continue
        merged.append((pos, record))
        predicted_added += 1

    if not merged:
        return None
    written = _write_pdb(output_path, merged)
    if written == 0:
        return None
    predicted_output_indices = [
        output_index
        for output_index, (sequence_position, _) in enumerate(merged, start=1)
        if sequence_position not in crystal_by_pos
    ]
    predicted_sequence_positions = [
        sequence_position
        for sequence_position, _ in merged
        if sequence_position not in crystal_by_pos
    ]
    result = CompletionResult(
        output_path=output_path,
        mode="crystal_with_alphafold_grafts" if predicted_added else "crystal_only_copy",
        crystal_residues=sum(1 for pos, _ in merged if pos in crystal_by_pos),
        predicted_residues=predicted_added,
        skipped_predicted_low_plddt=low_plddt,
        anchors=anchors,
        rmsd=rmsd,
    )
    _write_metadata(
        output_path,
        {
            "mode": result.mode,
            "crystal_path": str(crystal),
            "predicted_path": str(predicted),
            "full_sequence_length": len(sequence) if sequence else None,
            "crystal_chain": crystal_chain.chain_id,
            "predicted_chain": predicted_chain.chain_id,
            "crystal_sequence_matches": crystal_matches,
            "predicted_sequence_matches": predicted_matches,
            "crystal_residues": result.crystal_residues,
            "predicted_residues": result.predicted_residues,
            "predicted_output_indices": predicted_output_indices,
            "predicted_sequence_positions": predicted_sequence_positions,
            "skipped_predicted_low_plddt": result.skipped_predicted_low_plddt,
            "min_predicted_plddt": min_predicted_plddt,
            "anchors": anchors,
            "anchor_rmsd": rmsd,
            "note": "Experimental coordinates are kept where available; predicted residues fill only unmapped positions.",
        },
    )
    logging.debug(
        "Completed structure written: %s mode=%s crystal=%d predicted=%d anchors=%d rmsd=%s",
        output_path,
        result.mode,
        result.crystal_residues,
        result.predicted_residues,
        anchors,
        f"{rmsd:.3f}" if rmsd is not None else "NA",
    )
    return result
