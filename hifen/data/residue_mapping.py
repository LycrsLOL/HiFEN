from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass(frozen=True)
class ResidueMap:
    node_to_uniprot: dict[int, int]
    mapped_fraction: float
    method: str
    warning: str | None = None


def direct_uniprot_map(num_nodes: int, start: int = 1) -> ResidueMap:
    return ResidueMap({idx: start + idx for idx in range(num_nodes)}, 1.0, "direct")


def align_structure_to_uniprot(structure_sequence: str, uniprot_sequence: str) -> ResidueMap:
    matcher = SequenceMatcher(None, structure_sequence, uniprot_sequence, autojunk=False)
    mapping: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapping[block.a + offset] = block.b + offset + 1
    fraction = len(mapping) / max(len(structure_sequence), 1)
    warning = None
    if fraction < 0.8:
        warning = f"Low structure-to-UniProt mapping fraction: {fraction:.3f}"
    return ResidueMap(mapping, fraction, "sequence_alignment", warning)
