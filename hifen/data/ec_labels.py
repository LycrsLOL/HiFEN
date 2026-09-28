from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import torch

from ..utils import valid_text


LEVELS = ("ec1", "ec2", "ec3", "ec4")


@dataclass(frozen=True)
class ParsedEC:
    ec1: str | None
    ec2: str | None
    ec3: str | None
    ec4: str | None


def split_ec_tokens(value: object) -> list[str]:
    if not valid_text(value):
        return []
    tokens = []
    for token in str(value).replace("|", ";").replace(",", ";").split(";"):
        token = token.strip()
        if token and token.lower() not in {"nan", "none", "null"}:
            tokens.append(token)
    return sorted(set(tokens))


def parse_ec(ec: str) -> ParsedEC | None:
    parts = [part.strip() for part in str(ec).split(".")]
    if not parts or parts[0] in {"", "-"}:
        return None
    levels: list[str | None] = [None, None, None, None]
    prefix: list[str] = []
    for idx, part in enumerate(parts[:4]):
        if part in {"", "-"}:
            break
        prefix.append(part)
        levels[idx] = ".".join(prefix)
    if levels[0] is None:
        return None
    return ParsedEC(*levels)


def parse_ec_list(value: object) -> list[ParsedEC]:
    parsed = []
    for token in split_ec_tokens(value):
        item = parse_ec(token)
        if item is None:
            logging.warning("Skipping invalid EC token: %s", token)
            continue
        parsed.append(item)
    return parsed


def build_label_maps(ec_values: Iterable[object]) -> dict[str, dict[str, int]]:
    labels = {level: set() for level in LEVELS}
    for value in ec_values:
        for parsed in parse_ec_list(value):
            for level in LEVELS:
                label = getattr(parsed, level)
                if label is not None:
                    labels[level].add(label)
    return {level: {label: idx for idx, label in enumerate(sorted(values))} for level, values in labels.items()}


def encode_ec_values(value: object, label_maps: dict[str, dict[str, int]]) -> dict[str, torch.Tensor]:
    encoded = {
        level: torch.zeros(len(label_maps[level]), dtype=torch.float32)
        for level in LEVELS
    }
    for parsed in parse_ec_list(value):
        for level in LEVELS:
            label = getattr(parsed, level)
            if label is not None and label in label_maps[level]:
                encoded[level][label_maps[level][label]] = 1.0
    return encoded


def build_parent_child_maps(label_maps: dict[str, dict[str, int]]) -> dict[str, list[tuple[int, int]]]:
    maps: dict[str, list[tuple[int, int]]] = {}
    for parent_level, child_level in [("ec1", "ec2"), ("ec2", "ec3"), ("ec3", "ec4")]:
        pairs = []
        parent_map = label_maps[parent_level]
        child_map = label_maps[child_level]
        for child_label, child_idx in child_map.items():
            parent_label = ".".join(child_label.split(".")[:-1])
            parent_idx = parent_map.get(parent_label)
            if parent_idx is not None:
                pairs.append((parent_idx, child_idx))
        maps[f"{parent_level}_to_{child_level}"] = pairs
    return maps

