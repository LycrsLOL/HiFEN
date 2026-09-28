from __future__ import annotations

import copy
import csv
import filecmp
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils import write_json


REQUIRED_TRAIN_COLUMNS = ("uniprot_id", "sequence", "ec_numbers")
FAILED_COUNTERS = {
    "embedding_failed",
    "embedding_invalid",
    "embedding_skipped_no_sequence",
    "embedding_skipped_no_uniprot",
    "completion_failed",
    "completion_missing_predicted",
    "completion_pipeline_graph_failed",
    "completion_skipped_no_uniprot",
    "failed",
    "missing_embedding",
    "missing_structure",
    "mismatch",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inspect_train_csv(path: str | Path) -> dict[str, Any]:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise ValueError(f"train CSV does not exist: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"train CSV is not a file: {csv_path}")

    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            missing = [name for name in REQUIRED_TRAIN_COLUMNS if name not in columns]
            if missing:
                raise ValueError(
                    f"train CSV is missing required column(s): {', '.join(missing)}; "
                    f"available columns: {', '.join(columns) or '<none>'}"
                )
            row_count = sum(1 for _ in reader)
    except (OSError, csv.Error) as exc:
        raise ValueError(f"could not read train CSV {csv_path}: {exc}") from exc

    if row_count == 0:
        raise ValueError(f"train CSV contains no data rows: {csv_path}")
    return {
        "path": str(csv_path),
        "row_count": row_count,
        "columns": columns,
    }


def configure_one_click_deployment(
    config: dict[str, Any],
    *,
    train_csv: str | Path,
    output_dir: str | Path,
    esm_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Return a train-only deployment config rooted below ``output_dir``.

    The normal YAML/profile mode remains unchanged.  This overlay deliberately
    keeps embeddings and raw structures in addition to graphs so the requested
    output directory is a complete, inspectable dataset deployment.
    """

    csv_info = inspect_train_csv(train_csv)
    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"output directory points to a file: {output_root}")

    dataset_root = output_root / "train"
    result = copy.deepcopy(config)
    storage = result.setdefault("storage", {})
    storage.update(
        {
            "save_dir": str(output_root),
            "project_artifact_dir": str(dataset_root),
            "manifest_dir": str(dataset_root / "manifests"),
            "graph_dir": str(dataset_root / "graph"),
            "checkpoint_dir": str(dataset_root / "checkpoints"),
            "label_map_dir": str(dataset_root / "label_maps"),
            "split_dir": str(dataset_root / "splits"),
            "metrics_dir": str(dataset_root / "metrics"),
            "reproducibility_dir": str(dataset_root / "reproducibility"),
            "graph_cache_mode": "thin",
            "save_embedding_in_graph": False,
            "skip_existing": True,
            "overwrite_existing": False,
        }
    )

    result["datasets"] = {
        "train": {
            "csv": csv_info["path"],
            "output_subdir": "train",
            "role": "train",
        }
    }

    runtime_outputs = result.setdefault("runtime_outputs", {})
    runtime_outputs.update(
        {
            "log_dir": str(dataset_root / "logs"),
            "inference_result_dir": str(dataset_root / "inference_results"),
        }
    )

    deployment = result.setdefault("deployment", {})
    deployment.update(
        {
            "allow_download": True,
            "allow_completion": True,
            "allow_generate_embedding": True,
            "allow_recompute_embedding": False,
            "download_predicted_structures": True,
            "download_predicted_structure_datasets": ["train"],
            "complete_structure_datasets": ["train"],
            "pipeline_graph_after_completion": True,
            "skip_resource_repair_for_existing_graph": False,
            "skip_structure_completion_for_existing_graph": False,
            "compress_structures_after_graph": False,
            "delete_raw_structures_after_compress": False,
            "recursive_index": True,
        }
    )

    structure_download = result.setdefault("structure_download", {})
    structure_download["local_alphafold_work_dir"] = str(dataset_root / "local_alphafold_predictions")
    if esm_checkpoint is not None:
        result.setdefault("esm", {})["checkpoint"] = str(Path(esm_checkpoint).expanduser().resolve())

    layout = {
        "output_root": str(output_root),
        "dataset_root": str(dataset_root),
        "train_csv": str(dataset_root / "train_dataset.csv"),
        "embedding_dir": str(dataset_root / "embedding"),
        "graph_dir": str(dataset_root / "graph"),
        "crystal_dir": str(dataset_root / "crystal"),
        "predicted_dir": str(dataset_root / "predicted"),
        "complete_dir": str(dataset_root / "complete"),
        "manifest_dir": str(dataset_root / "manifests"),
        "status_path": str(dataset_root / "deployment" / "deployment_status.json"),
    }
    result["_one_click_deployment"] = {
        "enabled": True,
        "train_csv": csv_info,
        "layout": layout,
    }
    return result


def stage_train_csv(config: dict[str, Any], *, dry_run: bool) -> Path:
    """Copy the source CSV into the deployment without silently replacing data."""

    metadata = config.get("_one_click_deployment", {}) or {}
    if not metadata.get("enabled", False):
        return Path(config["datasets"]["train"]["csv"])
    source = Path((metadata.get("train_csv", {}) or {})["path"])
    target = Path((metadata.get("layout", {}) or {})["train_csv"])
    if dry_run:
        return source
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            same_file = source.samefile(target)
        except OSError:
            same_file = False
        if not same_file and not filecmp.cmp(source, target, shallow=False):
            raise RuntimeError(
                f"refusing to overwrite a different deployed train CSV: {target}; "
                "choose another --output-dir or reconcile the existing deployment first"
            )
    else:
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(source, tmp_path)
        tmp_path.replace(target)
    config["datasets"]["train"]["csv"] = str(target)
    return target


class DeploymentStatus:
    """Durable progress metadata for resumable one-click deployments."""

    def __init__(self, config: dict[str, Any], *, dry_run: bool) -> None:
        metadata = config.get("_one_click_deployment", {}) or {}
        self.enabled = bool(metadata.get("enabled", False))
        self.path = Path((metadata.get("layout", {}) or {}).get("status_path", "."))
        self.payload: dict[str, Any] = {}
        if not self.enabled:
            return
        self.payload = {
            "status": "running",
            "current_stage": "initializing",
            "started_at": _utc_now(),
            "updated_at": _utc_now(),
            "dry_run": bool(dry_run),
            "source": metadata.get("train_csv", {}),
            "layout": metadata.get("layout", {}),
            "stages": {},
            "resume_hint": "Run the same deploy-data command again; existing valid artifacts will be reused.",
        }
        self._write()

    def _write(self) -> None:
        if not self.enabled:
            return
        self.payload["updated_at"] = _utc_now()
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        write_json(tmp_path, self.payload)
        tmp_path.replace(self.path)

    def record(self, stage: str, stats: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        self.payload["current_stage"] = stage
        if stats is not None:
            self.payload.setdefault("stages", {})[stage] = stats
        self._write()

    def complete(self, stats: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        if stats is not None:
            self.payload.setdefault("stages", {})["final"] = stats
        failure_counts = _collect_failure_counts(stats or {})
        status = "completed_with_errors" if failure_counts else "completed"
        self.payload["status"] = status
        self.payload["current_stage"] = status
        if failure_counts:
            self.payload["failure_counts"] = failure_counts
        self.payload["completed_at"] = _utc_now()
        self._write()

    def fail(self, exc: BaseException) -> None:
        if not self.enabled:
            return
        interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
        self.payload["status"] = "interrupted" if interrupted else "failed"
        self.payload["current_stage"] = self.payload.get("current_stage", "unknown")
        self.payload["error"] = f"{type(exc).__name__}: {exc}"
        self.payload["finished_at"] = _utc_now()
        self._write()


def _collect_failure_counts(stats: dict[str, Any], prefix: str = "") -> dict[str, int]:
    failures: dict[str, int] = {}
    for key, value in stats.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            failures.update(_collect_failure_counts(value, prefix=path))
        elif key in FAILED_COUNTERS and isinstance(value, (int, float)) and value > 0:
            failures[path] = int(value)
    return failures
