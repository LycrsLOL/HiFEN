"""Resolve S2 rows against existing graph manifests without guessing truncated IDs.

Uses only the Python standard library. The previous manifests and source CSV
are required because S2 contains repeated and truncated display identifiers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile

SPLITS = ("train", "val", "test")
LEVELS = ("ec1", "ec2", "ec3", "ec4")
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_s2(path: Path) -> dict[str, list[tuple[int, str]]]:
    result = {split: [] for split in SPLITS}
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(item.itertext()) for item in ET.fromstring(
                archive.read("xl/sharedStrings.xml")
            ).findall("m:si", NS)]
        for _, row in ET.iterparse(archive.open("xl/worksheets/sheet1.xml"), events=("end",)):
            if row.tag != f"{{{NS['m']}}}row":
                continue
            values = {}
            for cell in row.findall("m:c", NS):
                value = cell.find("m:v", NS)
                text = value.text if value is not None else ""
                if cell.get("t") == "s":
                    text = shared[int(text)]
                elif cell.get("t") == "inlineStr":
                    text = "".join(cell.find("m:is", NS).itertext())
                values[re.sub(r"\d+", "", cell.get("r"))] = text
            number = int(row.get("r"))
            if number == 1:
                if values.get("A") != "split" or values.get("B") != "uniprot_id":
                    raise ValueError("S2 must have split and uniprot_id columns in A and B")
            elif values:
                split, uid = values.get("A"), values.get("B")
                if split not in SPLITS or not uid:
                    raise ValueError(f"Invalid S2 assignment at row {number}")
                result[split].append((number, uid))
            row.clear()
    if not all(result.values()):
        raise ValueError("S2 must contain all three nonempty splits")
    return result


def ec_labels(value: str) -> dict[str, set[str]]:
    result = {level: set() for level in LEVELS}
    for token in re.split(r"[;,|]", value or ""):
        prefix = []
        for index, part in enumerate(token.strip().split(".")[:4]):
            part = part.strip()
            if part in {"", "-"}:
                break
            prefix.append(part)
            result[LEVELS[index]].add(".".join(prefix))
    return result


def resolve(s2: dict, previous_dir: Path) -> tuple[dict, dict]:
    retained, removed = {}, {}
    for split in SPLITS:
        old = [Path(line).name for line in (previous_dir / f"{split}_graphs.txt").read_text().split()]
        display = [Path(name).stem.split("_")[0] for name in old]
        old_counts = Counter(display)
        requested = [uid for _, uid in s2[split]]
        requested_counts = Counter(requested)
        if requested_counts - old_counts:
            raise ValueError(f"S2 contains unmatched identifiers in {split}")
        # A repeated display ID may only be retained in full or removed in full.
        # Otherwise its source graph would be ambiguous and must not be guessed.
        ambiguous = [uid for uid, count in requested_counts.items() if count != old_counts[uid]]
        if ambiguous:
            raise ValueError(f"Ambiguous partial retention in {split}: {ambiguous[:10]}")
        retained[split] = [name for name, uid in zip(old, display) if uid in requested_counts]
        removed[split] = [name for name, uid in zip(old, display) if uid not in requested_counts]
        if [Path(name).stem.split("_")[0] for name in retained[split]] != requested:
            raise ValueError(f"S2 row order does not match the retained {split} manifest")
    flat = [name for names in retained.values() for name in names]
    if len(flat) != len(set(flat)):
        raise ValueError("Retained graph IDs are duplicated or cross split boundaries")
    return retained, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s2", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--previous-split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    s2 = read_s2(args.s2)
    retained, removed = resolve(s2, args.previous_split_dir)
    required = {name for groups in (retained, removed) for names in groups.values() for name in names}
    source = {}
    catalogue = {level: set() for level in LEVELS}
    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            uid = row["uniprot_id"].strip()
            pdb = row.get("pdb_id", "").strip()
            if pdb.lower() in {"nan", "none", "null"}:
                pdb = ""
            name = f"{uid}_{pdb}.pt" if pdb else f"{uid}.pt"
            for level, labels in ec_labels(row.get("ec_numbers", "")).items():
                catalogue[level].update(labels)
            if name in required:
                if name in source:
                    raise ValueError(f"Multiple CSV records resolve to {name}")
                source[name] = row
    if required - source.keys():
        raise ValueError(f"Graph IDs missing from source CSV: {sorted(required-source.keys())[:10]}")

    groups = ("homology_cluster", "sequence_cluster", "uniprot_id", "source_record_id")
    group_ids = {split: {key: set() for key in groups} for split in SPLITS}
    supports = {split: {level: Counter() for level in LEVELS} for split in SPLITS}
    eligible = {split: Counter() for split in SPLITS}
    for split, names in retained.items():
        for name in names:
            row = source[name]
            for key in groups:
                value = row.get(key, "").strip()
                if value and value.lower() not in {"nan", "none", "null"}:
                    group_ids[split][key].add(value)
            positive = row.get("ec_numbers_supervised", row.get("ec_numbers", ""))
            for level, labels in ec_labels(positive).items():
                supports[split][level].update(labels)
                eligible[split][level] += bool(labels)
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    overlaps = {key: {f"{a}/{b}": len(group_ids[a][key] & group_ids[b][key]) for a, b in pairs} for key in groups}
    coverage = {level: {
        "catalogue_classes": len(catalogue[level]),
        "observed_classes": {split: len(supports[split][level]) for split in SPLITS},
        "eligible_graphs": {split: eligible[split][level] for split in SPLITS},
        "validation_labels_absent_from_training": len(supports["val"][level].keys() - supports["train"][level].keys()),
        "test_labels_absent_from_training": len(supports["test"][level].keys() - supports["train"][level].keys()),
    } for level in LEVELS}
    if any(count for values in overlaps.values() for count in values.values()):
        raise ValueError("Resolved S2 membership has cross-split grouping identifiers")
    if any(v["validation_labels_absent_from_training"] or v["test_labels_absent_from_training"] for v in coverage.values()):
        raise ValueError("Resolved S2 membership is not closed-set")

    output = args.output_dir
    (output / "splits").mkdir(parents=True, exist_ok=True)
    (output / "supplementary data").mkdir(parents=True, exist_ok=True)
    destination = output / "supplementary data" / "Supplementary Table S2.xlsx"
    if args.s2.resolve() != destination.resolve():
        shutil.copy2(args.s2, destination)
    for split, names in retained.items():
        (output / f"{split}_graphs.txt").write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    with (output / "splits/s2_membership.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("s2_row", "split", "s2_display_id", "uniprot_id", "pdb_id", "graph_file"))
        for split in SPLITS:
            for (number, display), name in zip(s2[split], retained[split]):
                row = source[name]
                writer.writerow((number, split, display, row["uniprot_id"], row.get("pdb_id", ""), name))
    with (output / "splits/s2_excluded_graphs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("previous_split", "uniprot_id", "pdb_id", "graph_file", "reason"))
        for split, names in removed.items():
            for name in names:
                row = source[name]
                writer.writerow((split, row["uniprot_id"], row.get("pdb_id", ""), name, "Absent from Supplementary Table S2"))
    summary = {
        "strategy": "fixed_s2",
        "source": "Supplementary Table S2",
        "source_file": "supplementary data/Supplementary Table S2.xlsx",
        "source_sha256": sha256(destination),
        "source_csv_sha256": sha256(args.csv),
        "source_sheet": "Sheet 1",
        "counts": {split: len(names) for split, names in retained.items()},
        "total_records": sum(map(len, retained.values())),
        "unique_protein_ids": {split: len(group_ids[split]["uniprot_id"]) for split in SPLITS},
        "manifest_format": "graph_basename",
        "manifest_sha256": {split: sha256(output / f"{split}_graphs.txt") for split in SPLITS},
        "membership_file": "splits/s2_membership.csv",
        "membership_sha256": sha256(output / "splits/s2_membership.csv"),
        "excluded_graphs_file": "splits/s2_excluded_graphs.csv",
        "removed_from_previous_manifests": {split: len(names) for split, names in removed.items()},
        "recovered_refseq_rows": sum(display in {"NP", "WP", "XP"} for rows in s2.values() for _, display in rows),
        "resolution_method": "Exact display-ID multiplicity and ordered correspondence to previous graph manifests, followed by exact graph-file matching to source CSV. Partial repeated-ID retention is rejected.",
        "group_identifier_overlap_counts": overlaps,
        "label_coverage": coverage,
    }
    (output / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("counts", "total_records", "unique_protein_ids", "removed_from_previous_manifests", "recovered_refseq_rows", "label_coverage")}, indent=2))


if __name__ == "__main__":
    main()
