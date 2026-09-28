from __future__ import annotations


def infer_domain_motif(sequence: str, backend: str = "auto") -> dict[str, object]:
    return {
        "domain_regions": [],
        "motif_regions": [],
        "backend": backend,
        "available": False,
        "warnings": ["No domain/motif backend configured; returning missing annotations."],
    }


def infer_structure_pockets(structure_path: str | None) -> dict[str, object]:
    return {
        "binding_like_regions": [],
        "available": False,
        "warnings": ["No pocket inference backend configured; returning missing annotations."],
    }
