from __future__ import annotations

import importlib
import logging
import math
import os
from pathlib import Path
from typing import Any, Mapping


_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}
_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}


def _enabled(config: Mapping[str, Any]) -> bool:
    override = os.environ.get("HIFEN_SWANLAB_ENABLED")
    if override is not None:
        normalized = override.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
        raise ValueError(
            "HIFEN_SWANLAB_ENABLED must be one of "
            f"{sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {override!r}"
        )
    return bool((config.get("swanlab", {}) or {}).get("enabled", False))


def _serializable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _metric_name(name: str) -> str:
    if name == "train_loss":
        return "train/loss"
    if name == "val_loss":
        return "val/loss"
    if name.startswith("train_"):
        return f"train/{name.removeprefix('train_')}"
    if name.startswith("val_"):
        return f"val/{name.removeprefix('val_')}"
    if name.startswith("test_"):
        return f"test/{name.removeprefix('test_')}"
    return f"metrics/{name}"


def _dashboard_metric_name(name: str) -> str | None:
    for split in ("val", "test"):
        split_prefix = f"{split}_"
        if not name.startswith(split_prefix):
            continue
        metric_name = name.removeprefix(split_prefix)
        if metric_name.startswith("core_") or metric_name == "selection_score":
            return f"{split}/selection/{metric_name.removeprefix('core_')}"
        if metric_name.startswith("mean_"):
            return f"{split}/summary/{metric_name}"
        for level in ("ec1", "ec2", "ec3", "ec4"):
            level_prefix = f"{level}_"
            if not metric_name.startswith(level_prefix):
                continue
            level_metric = metric_name.removeprefix(level_prefix)
            for bucket in ("low", "high"):
                bucket_prefix = f"{bucket}_"
                if level_metric.startswith(bucket_prefix):
                    return f"{split}/{level}/{bucket}/{level_metric.removeprefix(bucket_prefix)}"
            if level_metric.startswith("single_label_"):
                return f"{split}/{level}/single_label/{level_metric.removeprefix('single_label_')}"
            if level_metric.startswith("threshold_"):
                return f"{split}/{level}/threshold/{level_metric.removeprefix('threshold_')}"
            if level_metric in {
                "observed_class_count",
                "unobserved_class_count",
                "class_coverage",
                "unobserved_label_fp_count",
                "unobserved_label_fp_rate",
                "unobserved_label_prediction_per_sample",
            }:
                return f"{split}/{level}/coverage/{level_metric}"
            return f"{split}/{level}/overall/{level_metric}"
    return None


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for raw_name, raw_value in metrics.items():
        name = str(raw_name)
        if name == "epoch":
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            payload[_metric_name(name)] = value
            dashboard_name = _dashboard_metric_name(name)
            if dashboard_name is not None:
                payload[dashboard_name] = value
    return payload


class SwanLabTracker:
    """Small, fail-open SwanLab adapter for the training loop."""

    def __init__(self, run: Any = None, *, fail_on_error: bool = False) -> None:
        self._run = run
        self._fail_on_error = fail_on_error
        self._finished = False
        self._warned_log_error = False

    @property
    def enabled(self) -> bool:
        return self._run is not None and not self._finished

    @classmethod
    def start(
        cls,
        config: Mapping[str, Any],
        *,
        job_type: str | None = None,
    ) -> "SwanLabTracker":
        tracking = config.get("swanlab", {}) or {}
        fail_on_error = bool(tracking.get("fail_on_error", False))
        if not _enabled(config):
            return cls(fail_on_error=fail_on_error)

        try:
            credential_root_value = (
                os.environ.get("HIFEN_SWANLAB_ROOT")
                or tracking.get("credential_root")
            )
            credential_root: Path | None = None
            if credential_root_value:
                credential_root = Path(str(credential_root_value)).expanduser()
                credential_root.mkdir(parents=True, exist_ok=True)
                credential_root.chmod(0o700)
                credential_file = credential_root / ".netrc"
                if credential_file.exists():
                    credential_file.chmod(0o600)
                os.environ["SWANLAB_ROOT"] = str(credential_root)

            swanlab = importlib.import_module("swanlab")
            artifact_dir = Path(
                str(
                    (config.get("run", {}) or {}).get("artifact_dir")
                    or (config.get("storage", {}) or {}).get("metrics_dir")
                    or "."
                )
            )
            logdir_value = tracking.get("log_dir")
            logdir = Path(str(logdir_value)) if logdir_value else artifact_dir / "swanlab"
            logdir.mkdir(parents=True, exist_ok=True)

            run_config = _serializable(dict(config))
            run_meta = config.get("run", {}) or {}
            profile = str(config.get("_profile") or "default")
            resolved_job_type = str(
                job_type
                or tracking.get("job_type")
                or config.get("mode")
                or "train"
            ).strip().lower()
            artifact_tag = str(run_meta.get("artifact_tag") or "").strip()
            experiment_name = str(tracking.get("experiment_name") or artifact_tag or profile)
            run_id = str(tracking.get("run_id") or artifact_tag).strip()
            group = str(tracking.get("group") or profile).strip()

            configured_tags = tracking.get("tags", []) or []
            if isinstance(configured_tags, str):
                configured_tags = [configured_tags]
            auto_tags = [
                resolved_job_type,
                profile,
                str((config.get("model", {}) or {}).get("type") or "model"),
            ]
            tags = list(
                dict.fromkeys(
                    str(tag).strip()[:20]
                    for tag in [*configured_tags, *auto_tags]
                    if str(tag).strip()
                )
            )

            init_kwargs: dict[str, Any] = {
                "project": str(tracking.get("project") or "HiFEN"),
                "workspace": str(tracking.get("workspace") or ""),
                "experiment_name": experiment_name,
                "description": str(
                    tracking.get("description")
                    or f"HiFEN {resolved_job_type} run for profile {profile}"
                ),
                "job_type": resolved_job_type,
                "group": group or None,
                "tags": tags,
                "config": run_config,
                "logdir": str(logdir),
                "mode": str(tracking.get("mode") or "online"),
                "public": bool(tracking.get("public", False)),
            }
            if run_id:
                init_kwargs["id"] = run_id
                init_kwargs["resume"] = "allow"

            run = swanlab.init(**init_kwargs)
            experiment_url = ""
            public = getattr(run, "public", None)
            cloud = getattr(public, "cloud", None)
            if cloud is not None:
                experiment_url = str(getattr(cloud, "experiment_url", "") or "")
            logging.info(
                "SwanLab tracking initialized | job_type=%s project=%s workspace=%s experiment=%s credential_root=%s url=%s",
                resolved_job_type,
                init_kwargs["project"],
                init_kwargs["workspace"],
                experiment_name,
                credential_root or "<default>",
                experiment_url or "<unavailable>",
            )
            return cls(run, fail_on_error=fail_on_error)
        except Exception:
            if fail_on_error:
                raise
            logging.exception("SwanLab initialization failed; run will continue without tracking")
            return cls(fail_on_error=fail_on_error)

    def log_epoch(
        self,
        metrics: Mapping[str, Any],
        *,
        epoch: int,
        extra_metrics: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        payload = _scalar_metrics(metrics)
        payload["progress/epoch"] = float(epoch)
        for name, raw_value in (extra_metrics or {}).items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                payload[str(name)] = value
        self._log(payload, step=int(epoch))

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        payload: dict[str, float] = {}
        for name, raw_value in summary.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                payload[f"summary/{name}"] = value
        self._log(payload)

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
    ) -> None:
        """Log a scalar metric mapping outside the epoch-based training loop."""
        if not self.enabled:
            return
        self._log(_scalar_metrics(metrics), step=step)

    def log_progress(
        self,
        stage: str,
        *,
        completed: int,
        total: int,
        step: int | None = None,
    ) -> None:
        """Publish a bounded stage progress update for evaluation/deployment jobs."""
        if not self.enabled:
            return
        safe_stage = str(stage).strip().replace(" ", "_") or "run"
        safe_total = max(int(total), 1)
        safe_completed = min(max(int(completed), 0), safe_total)
        self._log(
            {
                f"progress/{safe_stage}_completed": float(safe_completed),
                f"progress/{safe_stage}_total": float(safe_total),
                f"progress/{safe_stage}_percent": 100.0 * safe_completed / safe_total,
            },
            step=step,
        )

    def log_best(self, metrics: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        payload: dict[str, float] = {}
        for raw_name, raw_value in metrics.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                metric_name = str(raw_name)
                payload[f"best/{metric_name.removeprefix('val_')}"] = value
                dashboard_name = _dashboard_metric_name(metric_name)
                if dashboard_name is not None:
                    payload[f"best/{dashboard_name.removeprefix('val/')}"] = value
        self._log(payload)

    def _log(self, payload: Mapping[str, float], *, step: int | None = None) -> None:
        if not self.enabled or not payload:
            return
        try:
            self._run.log(dict(payload), step=step)
        except Exception:
            if self._fail_on_error:
                raise
            if not self._warned_log_error:
                logging.exception("SwanLab metric upload failed; training will continue")
                self._warned_log_error = True

    def finish(self, *, status: str) -> None:
        if not self.enabled:
            return
        status_codes = {
            "completed": 0.0,
            "early_stopped": 1.0,
            "interrupted": 2.0,
            "failed": 3.0,
            "dry_run": 4.0,
        }
        try:
            self._log({"run/status_code": status_codes.get(status, 3.0)})
            self._run.finish()
        except Exception:
            if self._fail_on_error:
                raise
            logging.exception("SwanLab finalization failed")
        finally:
            self._finished = True
