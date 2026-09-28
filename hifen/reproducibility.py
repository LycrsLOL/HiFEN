from __future__ import annotations

import platform
from pathlib import Path

from .utils import write_json


def export_reproducibility_metadata(config: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "project": "HiFEN",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_path": config.get("_config_path"),
        "ai_use": "AI assistance may be used for code and documentation drafts; manuscript text remains researcher-authored.",
    }
    out_path = out_dir / "reproducibility_metadata.json"
    write_json(out_path, payload)
    return out_path
