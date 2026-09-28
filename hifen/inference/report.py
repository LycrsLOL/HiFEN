from __future__ import annotations

from pathlib import Path


def write_inference_report(path: str | Path, result: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# HiFEN Inference Report: {result.get('protein_id', 'unknown')}",
        "",
        f"- Sequence length: {result.get('sequence_length')}",
        f"- Structure available: {result.get('structure_available')}",
        "",
        "## EC Predictions",
    ]
    for level in ("ec1", "ec2", "ec3", "ec4"):
        lines.append(f"### {level.upper()}")
        for row in result.get("predicted_ec", {}).get(level, []):
            lines.append(f"- {row['rank']}. {row['label']}: {row['probability']:.4f}")
    lines.extend(
        [
            "",
            "## Predicted Functional Sites",
            "Sites listed here are model predictions, not curated annotations.",
        ]
    )
    for key in ("active_sites", "binding_sites", "catalytic_sites"):
        lines.append(f"- {key}: {result.get('predicted_sites', {}).get(key, [])}")
    warnings = result.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {warning}" for warning in warnings])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
