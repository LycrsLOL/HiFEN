from __future__ import annotations

from pathlib import Path

import torch

from .confidence import topk_labels
from ..data.graph_builder import toy_thin_graph
from ..models import HiFENModel
from .report import write_inference_report
from ..utils import write_json


def run_mock_inference(config: dict, sequence: str, output_dir: str | None = None) -> dict[str, object]:
    output_dir = output_dir or (config.get("runtime_outputs", {}) or {}).get("inference_result_dir", "inference_results")
    protein_id = "mock_sequence"
    num_classes = {"ec1": 7, "ec2": 12, "ec3": 16, "ec4": 20}
    data = toy_thin_graph(num_nodes=min(max(len(sequence), 4), 32), embedding_dim=32)
    model_options = dict(config.get("model", {}) or {})
    model_options.pop("type", None)
    model_options.update(
        input_dim=32, hidden_dim=32, num_classes=num_classes,
        annotation_dim=12, num_layers=2, heads=4, dropout=0.0,
        use_annotation_features=bool(
            (config.get("features", {}) or {}).get("use_annotation_features", True)
        ),
    )
    model = HiFENModel(**model_options)
    model.eval()
    with torch.no_grad():
        outputs = model(data)
    predicted_ec = {}
    for level, n in num_classes.items():
        labels = [f"{level}_{idx}" for idx in range(n)]
        probs = torch.sigmoid(outputs[level])
        predicted_ec[level] = topk_labels(probs, labels, k=min(5, n))
    result = {
        "protein_id": protein_id,
        "sequence_length": len(sequence),
        "structure_available": False,
        "predicted_ec": predicted_ec,
        "predicted_sites": {
            "active_sites": [],
            "binding_sites": [],
            "catalytic_sites": [],
        },
        "warnings": ["Mock inference uses random weights and is for smoke testing only."],
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{protein_id}.json", result)
    write_inference_report(out_dir / f"{protein_id}_report.md", result)
    return result
