from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch

from ..config import ensure_dirs
from ..data.ec_labels import LEVELS
from ..data.label_repair import graph_path_for_row, read_csv_rows
from .trainer import (
    DistributedContext,
    _collect_validation_outputs,
    _compute_validation_metrics,
    _configure_cpu_threads,
    _format_metric,
    _label_supports_from_csv,
    _load_label_maps,
    _make_loader,
    _make_loss,
    _make_model,
    _metric_config,
    _resolve_device,
    _thresholds_from_payload,
    load_or_create_splits,
)
from .swanlab_tracker import SwanLabTracker


FORMAL_TEST_PROTOCOL = "formal_test_v1"


def _external_graph_paths(config: dict, dataset: str) -> list[str]:
    rows = read_csv_rows(config["datasets"][dataset]["csv"])
    paths: list[str] = []
    for row in rows:
        graph_path = graph_path_for_row(config, dataset, row)
        if graph_path is not None and graph_path.exists():
            paths.append(str(graph_path))
    return paths


def _evaluation_paths(config: dict, dataset: str) -> list[str]:
    if dataset in {"train", "val", "test"}:
        return load_or_create_splits(config)[dataset]
    return _external_graph_paths(config, dataset)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_list_sha256(paths: list[str]) -> str:
    payload = "\n".join(sorted(str(Path(path)) for path in paths)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rename_validation_metrics(metrics: dict[str, float], dataset: str) -> dict[str, float]:
    prefix = f"{dataset}_"
    return {
        (prefix + key.removeprefix("val_")) if key.startswith("val_") else key: value
        for key, value in metrics.items()
    }


def _comparison_views(metrics: dict[str, float], metric_prefix: str) -> dict[str, object]:
    single_label_levels: dict[str, dict[str, float]] = {}
    multilabel_levels: dict[str, dict[str, float]] = {}
    for level in LEVELS:
        level_prefix = f"{metric_prefix}_{level}_"
        single_prefix = f"{level_prefix}single_label_"
        single_label_levels[level] = {
            key.removeprefix(single_prefix): float(value)
            for key, value in metrics.items()
            if key.startswith(single_prefix)
        }
        multilabel_levels[level] = {
            key.removeprefix(level_prefix): float(value)
            for key, value in metrics.items()
            if key.startswith(level_prefix) and not key.startswith(single_prefix)
        }
    return {
        "single_label": {
            "primary_level": "ec4",
            "eligibility_rule": "exactly_one_reference_label_at_each_reported_level",
            "prediction_rule": "argmax_without_thresholding",
            "levels": single_label_levels,
        },
        "multilabel": {
            "primary_level": "ec4",
            "eligibility_rule": "all_formal_test_samples",
            "prediction_rule": "validation_calibrated_thresholds",
            "levels": multilabel_levels,
        },
    }


def _metric_prefix(result: dict[str, object], metrics: dict[str, object]) -> str:
    configured = str(result.get("metric_prefix") or "").strip()
    if configured:
        return configured
    dataset = str(result.get("dataset") or "").strip()
    if dataset and any(str(key).startswith(f"{dataset}_") for key in metrics):
        return dataset
    return "val"


def format_evaluation_result(result: dict[str, object]) -> str:
    """Format the CLI evaluation result as a compact, grouped report."""
    status = str(result.get("status", "completed")).replace("_", " ")
    class_counts = result.get("num_classes", {})
    if isinstance(class_counts, dict):
        classes = " ".join(f"{level}={class_counts.get(level, 0)}" for level in LEVELS)
    else:
        classes = str(class_counts)
    lines = [
        f"Evaluation {status}",
        f"  dataset    | {result.get('dataset', '-')}",
        f"  graphs     | {result.get('num_graphs', 0)}",
        f"  classes    | {classes}",
        f"  checkpoint | {result.get('checkpoint', '-')}",
    ]
    protocol = result.get("protocol")
    if isinstance(protocol, dict):
        lines.append(
            "  protocol   | "
            f"{protocol.get('name', '-')} "
            f"thresholds={protocol.get('threshold_source', '-')} "
            f"test_for_selection={protocol.get('test_used_for_model_selection', '-')}"
        )
        if protocol.get("comparison_views"):
            lines.append(
                "  views      | single-label=single-label(argmax) "
                "multi-label=multi-label(validation-thresholds)"
            )

    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        metric_prefix = _metric_prefix(result, metrics)
        lines.append("Metrics")
        for level in LEVELS:
            prefix = f"{metric_prefix}_{level}"
            if f"{prefix}_f1_micro" not in metrics:
                continue
            lines.append(
                f"  {level.upper():<4} ML | "
                f"f1 micro/macro/observed="
                f"{_format_metric(metrics.get(f'{prefix}_f1_micro'))}/"
                f"{_format_metric(metrics.get(f'{prefix}_f1_macro'))}/"
                f"{_format_metric(metrics.get(f'{prefix}_f1_macro_observed'))} "
                f"p/r micro="
                f"{_format_metric(metrics.get(f'{prefix}_precision_micro'))}/"
                f"{_format_metric(metrics.get(f'{prefix}_recall_micro'))} "
                f"auprc={_format_metric(metrics.get(f'{prefix}_auprc_micro'))} "
                f"fmax={_format_metric(metrics.get(f'{prefix}_fmax'))} "
                f"threshold={_format_metric(metrics.get(f'{prefix}_threshold_mean'))}"
            )
            if f"{prefix}_single_label_sample_count" in metrics:
                sample_fraction = float(metrics.get(f"{prefix}_single_label_sample_fraction", 0.0))
                lines.append(
                    "       SL | "
                    f"n={int(metrics.get(f'{prefix}_single_label_sample_count', 0))} "
                    f"({sample_fraction:.1%}) "
                    f"accuracy/balanced="
                    f"{_format_metric(metrics.get(f'{prefix}_single_label_accuracy'))}/"
                    f"{_format_metric(metrics.get(f'{prefix}_single_label_balanced_accuracy'))} "
                    f"f1 macro/weighted="
                    f"{_format_metric(metrics.get(f'{prefix}_single_label_f1_macro'))}/"
                    f"{_format_metric(metrics.get(f'{prefix}_single_label_f1_weighted'))} "
                    f"mcc={_format_metric(metrics.get(f'{prefix}_single_label_mcc'))}"
                )
                topk = [
                    f"{k}={_format_metric(metrics.get(f'{prefix}_single_label_top{k}_accuracy'))}"
                    for k in (1, 3, 5, 10)
                    if f"{prefix}_single_label_top{k}_accuracy" in metrics
                ]
                lines.append(
                    "          | "
                    f"p/r macro="
                    f"{_format_metric(metrics.get(f'{prefix}_single_label_precision_macro'))}/"
                    f"{_format_metric(metrics.get(f'{prefix}_single_label_recall_macro'))} "
                    f"top-k {' '.join(topk)}"
                )

        lines.append(
            "  mean      | "
            f"f1 micro/macro="
            f"{_format_metric(metrics.get(f'{metric_prefix}_mean_f1_micro'))}/"
            f"{_format_metric(metrics.get(f'{metric_prefix}_mean_f1_macro'))} "
            f"fmax={_format_metric(metrics.get(f'{metric_prefix}_mean_fmax'))} "
            f"selection={_format_metric(metrics.get(f'{metric_prefix}_selection_score'))}"
        )
        if f"{metric_prefix}_hierarchical_f1" in metrics:
            lines.append(
                "  hierarchy | "
                f"p/r/f1="
                f"{_format_metric(metrics.get(f'{metric_prefix}_hierarchical_precision'))}/"
                f"{_format_metric(metrics.get(f'{metric_prefix}_hierarchical_recall'))}/"
                f"{_format_metric(metrics.get(f'{metric_prefix}_hierarchical_f1'))} "
                f"violation={_format_metric(metrics.get(f'{metric_prefix}_hierarchy_violation_rate'))} "
                f"consistency={_format_metric(metrics.get(f'{metric_prefix}_hierarchy_consistency'))}"
            )
        site_metrics = [
            f"{key.removeprefix(f'{metric_prefix}_site_')}={_format_metric(value)}"
            for key, value in sorted(metrics.items())
            if key.startswith(f"{metric_prefix}_site_") and key.endswith("_f1")
        ]
        if site_metrics:
            lines.append("  sites     | " + " ".join(site_metrics))
    return "\n".join(lines)


def run_evaluate(config: dict, checkpoint: str | None = None, dry_run: bool = False) -> dict[str, object]:
    ensure_dirs(config)
    eval_config = config.get("evaluate", {}) or {}
    train_config = config.get("train", {}) or {}
    _configure_cpu_threads(train_config)
    dataset = str(eval_config.get("dataset", "test"))
    protocol_name = str(eval_config.get("protocol", "standard_evaluation"))
    formal_test = protocol_name == FORMAL_TEST_PROTOCOL
    checkpoint_path = Path(checkpoint or eval_config.get("checkpoint") or Path(config["storage"]["checkpoint_dir"]) / "best.pt")
    label_maps, parent_child_maps = _load_label_maps(config)
    paths = _evaluation_paths(config, dataset)
    if not paths:
        raise RuntimeError(f"No graph files found for evaluation dataset: {dataset}")
    expected_num_graphs = eval_config.get("expected_num_graphs")
    if formal_test:
        if dataset != "test":
            raise ValueError(f"{FORMAL_TEST_PROTOCOL} requires evaluate.dataset=test, got {dataset!r}")
        if expected_num_graphs is None:
            raise ValueError(f"{FORMAL_TEST_PROTOCOL} requires evaluate.expected_num_graphs")
    if expected_num_graphs is not None and len(paths) != int(expected_num_graphs):
        raise RuntimeError(
            f"Evaluation split size mismatch for {dataset}: expected {int(expected_num_graphs)}, found {len(paths)}"
        )
    protocol = {
        "name": protocol_name,
        "expected_num_graphs": int(expected_num_graphs) if expected_num_graphs is not None else None,
        "threshold_source": "validation" if dataset != "val" else "evaluation_dataset",
        "model_selection_source": "validation",
        "test_used_for_model_selection": False,
        "comparison_views": ["single_label", "multilabel"],
    }
    result_base = {
        "status": "dry_run" if dry_run else "completed",
        "dataset": dataset,
        "checkpoint": str(checkpoint_path),
        "num_graphs": len(paths),
        "num_classes": {level: len(label_maps[level]) for level in LEVELS},
        "metric_prefix": dataset if formal_test else "val",
        "protocol": protocol,
        "split_path_sha256": _path_list_sha256(paths),
    }
    logging.info("Evaluation selected: dataset=%s graphs=%s checkpoint=%s", dataset, len(paths), checkpoint_path)
    if dry_run:
        logging.info("%s", format_evaluation_result(result_base))
        return result_base
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    default_output_filename = (
        f"evaluation_{dataset}_{checkpoint_path.stem}.json"
        if formal_test
        else f"evaluation_{dataset}.json"
    )
    output_filename = str(eval_config.get("output_filename") or default_output_filename)
    if Path(output_filename).name != output_filename:
        raise ValueError("evaluate.output_filename must be a file name without directories")
    out_path = Path(config["storage"]["metrics_dir"]) / output_filename
    allow_overwrite = bool(eval_config.get("allow_overwrite", not formal_test))
    if out_path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Evaluation output already exists and allow_overwrite=false: {out_path}"
        )

    distributed = DistributedContext(enabled=False)
    device = _resolve_device(train_config, eval_config.get("device"), distributed)
    split_paths = load_or_create_splits(config)
    class_supports = _label_supports_from_csv(config, label_maps, split_paths["train"])
    criterion = _make_loss(config, class_supports)
    model = _make_model(config, label_maps).to(device)
    checkpoint_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint_payload.get("model_state_dict", checkpoint_payload)
    model.load_state_dict(state_dict)
    metric_config = _metric_config(config)
    tracker = SwanLabTracker.start(config, job_type="evaluate")
    tracking_status = "failed"
    swanlab_log_interval = max(int(eval_config.get("swanlab_log_interval", 25)), 1)

    def progress_callback(stage: str):
        def update(completed: int, total: int) -> None:
            if completed == 1 or completed == total or completed % swanlab_log_interval == 0:
                tracker.log_progress(stage, completed=completed, total=total)

        return update

    threshold_overrides = None
    calibration_count = 0
    try:
        tracker.log_progress("setup", completed=1, total=1)
        if dataset != "val" and str(metric_config.get("threshold_mode", "fixed")).lower() in {"auto", "per_class", "per_class_f1"}:
            calibration_loader, _ = _make_loader(
                split_paths["val"],
                config,
                shuffle=False,
                distributed=distributed,
                label_maps=label_maps,
            )
            calibration_payload = _collect_validation_outputs(
                model,
                calibration_loader,
                criterion,
                device,
                distributed,
                progress_label="Validation threshold calibration",
                progress_callback=progress_callback("threshold_calibration"),
            )
            calibration_metric_config = dict(metric_config)
            calibration_metric_config["threshold_calibration_fraction"] = 0.0
            threshold_overrides, calibration_count = _thresholds_from_payload(
                calibration_payload,
                parent_child_maps,
                calibration_metric_config,
                class_supports,
                progress_label="Validation threshold calibration",
            )
            logging.info("Evaluation thresholds calibrated from full validation split: graphs=%s", calibration_count)
        if formal_test and threshold_overrides is None:
            raise RuntimeError(
                f"{FORMAL_TEST_PROTOCOL} requires validation-calibrated auto/per-class thresholds"
            )
        loader, _ = _make_loader(
            paths,
            config,
            shuffle=False,
            distributed=distributed,
            label_maps=label_maps,
        )
        evaluation_payload = _collect_validation_outputs(
            model,
            loader,
            criterion,
            device,
            distributed,
            progress_label=f"Formal {dataset} inference" if formal_test else f"{dataset} inference",
            progress_callback=progress_callback("test_inference" if dataset == "test" else "inference"),
        )
        raw_metrics = _compute_validation_metrics(
            evaluation_payload,
            parent_child_maps,
            class_supports,
            metric_config,
            threshold_overrides=threshold_overrides,
            progress_label=f"Formal {dataset} metrics" if formal_test else f"{dataset} metrics",
        )
        metrics = (
            _rename_validation_metrics(raw_metrics, dataset)
            if formal_test
            else raw_metrics
        )
        protocol["threshold_calibration_sample_count"] = int(calibration_count)
        protocol["validation_split_path_sha256"] = _path_list_sha256(split_paths["val"])
        output = {
            **result_base,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "metrics": metrics,
            "comparison_views": _comparison_views(
                metrics,
                dataset if formal_test else "val",
            ),
            "output_path": str(out_path),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        tracker.log_metrics(metrics)
        tracker.log_summary(
            {
                "num_graphs": len(paths),
                "threshold_calibration_sample_count": calibration_count,
                "formal_test": float(formal_test),
            }
        )
        tracking_status = "completed"
        logging.info("%s", format_evaluation_result(output))
        return output
    finally:
        tracker.finish(status=tracking_status)

