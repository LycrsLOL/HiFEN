from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..utils import valid_text


RANGE_RE = re.compile(r"^(?:(?P<label>[^:;]+):)?(?P<start>-?\d+)(?:-(?P<end>-?\d+))?$")


@dataclass
class ParsedAnnotations:
    active_sites: set[int] = field(default_factory=set)
    catalytic_sites: set[int] = field(default_factory=set)
    binding_sites: set[int] = field(default_factory=set)
    domain_regions: list[tuple[int, int, str | None]] = field(default_factory=list)
    motif_regions: list[tuple[int, int, str | None]] = field(default_factory=list)
    available: bool = False
    source: str | None = None
    warnings: list[str] = field(default_factory=list)


def _tokens(value: Any) -> list[str]:
    if not valid_text(value):
        return []
    return [token.strip() for token in str(value).replace(",", ";").split(";") if token.strip()]


def parse_points(value: Any, field_name: str = "sites") -> tuple[set[int], list[str]]:
    points: set[int] = set()
    warnings: list[str] = []
    for token in _tokens(value):
        match = RANGE_RE.match(token)
        if not match:
            warnings.append(f"{field_name}: could not parse token '{token}'")
            continue
        start = int(match.group("start"))
        end_text = match.group("end")
        if end_text is None:
            points.add(start)
        else:
            end = int(end_text)
            lo, hi = sorted((start, end))
            points.update(range(lo, hi + 1))
    return points, warnings


def parse_regions(value: Any, field_name: str = "regions") -> tuple[list[tuple[int, int, str | None]], list[str]]:
    regions: list[tuple[int, int, str | None]] = []
    warnings: list[str] = []
    for token in _tokens(value):
        match = RANGE_RE.match(token)
        if not match:
            warnings.append(f"{field_name}: could not parse token '{token}'")
            continue
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        lo, hi = sorted((start, end))
        regions.append((lo, hi, match.group("label")))
    return regions, warnings


def parse_annotation_row(row: dict[str, Any] | Any) -> ParsedAnnotations:
    get = row.get if hasattr(row, "get") else lambda key, default=None: getattr(row, key, default)
    parsed = ParsedAnnotations(source=get("annotation_source"))
    parsed.active_sites, w = parse_points(get("active_sites"), "active_sites")
    parsed.warnings.extend(w)
    parsed.catalytic_sites, w = parse_points(get("catalytic_sites"), "catalytic_sites")
    parsed.warnings.extend(w)
    parsed.binding_sites, w = parse_points(get("binding_sites"), "binding_sites")
    parsed.warnings.extend(w)
    # InterPro/Pfam accessions are categorical cross-check metadata, not
    # residue-coordinate ranges. Only explicit range columns may contribute
    # spatial annotation features.
    parsed.domain_regions, w = parse_regions(get("domain_ranges"), "domain_ranges")
    parsed.warnings.extend(w)
    parsed.motif_regions, w = parse_regions(get("motif_ranges"), "motif_ranges")
    parsed.warnings.extend(w)
    parsed.available = any(
        [
            parsed.active_sites,
            parsed.catalytic_sites,
            parsed.binding_sites,
            parsed.domain_regions,
            parsed.motif_regions,
        ]
    )
    for warning in parsed.warnings:
        logging.warning(warning)
    return parsed

