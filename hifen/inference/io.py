from __future__ import annotations

from pathlib import Path


def read_fasta(path: str | Path) -> tuple[str, str]:
    name = Path(path).stem
    seq_parts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0] or name
            else:
                seq_parts.append(line)
    return name, "".join(seq_parts)
