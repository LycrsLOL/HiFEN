from __future__ import annotations

import csv
import fnmatch
import json
import logging
import math
import os
import random
import shutil
import sys
import warnings
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import average_precision_score
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch_geometric.loader import DataLoader
except Exception:  # pragma: no cover
    DataLoader = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable

from ..config import ensure_dirs
from ..data.dataset import ThinGraphDataset
from ..data.fixed_splits import fixed_split_directory, load_fixed_splits
from ..data.ec_labels import LEVELS, encode_ec_values, parse_ec_list, split_ec_tokens
from ..data.graph_builder import SITE_TASKS
from ..data.label_repair import build_current_label_maps, graph_path_for_row, read_csv_rows, repair_graph_labels, save_label_maps
from ..deployment.pdb_quality import rank_pdb_candidate_groups
from .losses import HierarchicalECLoss
from .metrics import (
    apply_hierarchy_probability_gate,
    apply_hierarchy_threshold_gate,
    bootstrap_ci,
    binary_histogram_counts,
    binary_metrics_from_histogram,
    build_metric_cache,
    class_frequency_bucket_metrics,
    closed_set_metrics,
    hierarchical_metrics,
    multilabel_metrics,
    optimize_global_threshold,
    optimize_per_class_thresholds,
    single_label_metrics,
    topk_metrics,
)
from ..models import HiFENModel
from ..run_artifacts import (
    artifact_filename,
    synchronize_training_artifact_config,
    training_artifact_context,
    write_training_run_inputs,
)
from ..utils import write_json
from .swanlab_tracker import SwanLabTracker


SPLIT_NAMES = ("train", "val", "test")
ECLabel = tuple[str, str]


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = ""

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class WeightedEpochSampler(Sampler[int]):
    def __init__(
        self,
        weights: torch.Tensor,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        replacement: bool = True,
        seed: int = 0,
    ) -> None:
        if weights.dim() != 1 or weights.numel() == 0:
            raise ValueError("WeightedEpochSampler requires a non-empty 1D weight tensor")
        self.weights = weights.detach().cpu().double().clamp_min(0.0)
        if float(self.weights.sum()) <= 0.0:
            raise ValueError("WeightedEpochSampler weights must contain at least one positive value")
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(self.weights.numel() / self.num_replicas))

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * self.num_replicas + self.rank)
        yield from torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement,
            generator=generator,
        ).tolist()

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_cpu_threads(train_config: dict) -> None:
    cpu_threads = int(train_config.get("cpu_threads", 0) or 0)
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
    cpu_interop_threads = int(train_config.get("cpu_interop_threads", cpu_threads) or 0)
    if cpu_interop_threads > 0:
        try:
            torch.set_num_interop_threads(cpu_interop_threads)
        except RuntimeError:
            logging.debug("Torch interop thread count was already initialized; keeping the existing value.")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _distributed_requested(train_config: dict, distributed_override: bool | None) -> bool:
    if distributed_override is not None:
        return distributed_override
    configured = train_config.get("distributed", "auto")
    if isinstance(configured, str) and configured.lower() == "auto":
        return _env_int("WORLD_SIZE", 1) > 1
    return bool(configured)


def _init_distributed(train_config: dict, distributed_override: bool | None) -> DistributedContext:
    if not _distributed_requested(train_config, distributed_override):
        return DistributedContext(enabled=False)

    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", rank)
    world_size = _env_int("WORLD_SIZE", 1)
    backend = str(train_config.get("dist_backend") or ("nccl" if torch.cuda.is_available() else "gloo"))
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"
    if torch.cuda.is_available():
        visible_device_count = torch.cuda.device_count()
        if local_rank >= visible_device_count:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<all>")
            raise RuntimeError(
                "Invalid distributed CUDA launch: "
                f"LOCAL_RANK={local_rank}, but PyTorch sees only {visible_device_count} visible CUDA device(s) "
                f"(CUDA_VISIBLE_DEVICES={visible}). "
                "Set torchrun --nproc_per_node to the number of visible GPUs, or expose more GPUs."
            )
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        timeout_seconds = float(train_config.get("dist_timeout_seconds", 7200))
        dist.init_process_group(
            backend=backend,
            init_method=str(train_config.get("dist_url", "env://")),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_seconds),
        )
    return DistributedContext(
        enabled=True,
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
        backend=backend,
    )


def _barrier(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        if distributed.backend == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[distributed.local_rank])
        else:
            dist.barrier()


def _cleanup_distributed(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _log_main(distributed: DistributedContext, message: str, *args) -> None:
    if distributed.is_main:
        logging.info(message, *args)


def _terminal_main(distributed: DistributedContext, message: str, *args) -> None:
    if not (distributed.is_main and sys.stderr.isatty()):
        return
    text = message % args if args else message
    if hasattr(tqdm, "write"):
        tqdm.write(text, file=sys.stderr)
    else:
        print(text, file=sys.stderr, flush=True)


def _progress_bar_enabled(distributed: DistributedContext) -> bool:
    progress_flag = os.environ.get("HIFEN_PROGRESS")
    if progress_flag is None:
        enabled = True
    else:
        enabled = str(progress_flag).strip().lower() in {"1", "true", "yes", "on"}
    return enabled and distributed.is_main and sys.stderr.isatty()


def _progress_mode(distributed: DistributedContext, log_interval: int) -> str:
    if _progress_bar_enabled(distributed):
        return "tty-bar"
    if log_interval > 0:
        return f"log-every-{log_interval}-batches"
    return "epoch-only"


class _NoopProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def set_description_str(self, description: str) -> None:
        return None

    def update(self, n: int = 1) -> None:
        return None


def _metric_progress_enabled(metric_config: dict[str, object]) -> bool:
    progress_flag = metric_config.get("progress", True)
    if isinstance(progress_flag, str):
        progress_flag = progress_flag.strip().lower() in {"1", "true", "yes", "on"}
    return bool(progress_flag) and sys.stderr.isatty()


def _make_metric_progress(metric_config: dict[str, object], *, label: str, total: int):
    if total <= 0 or not _metric_progress_enabled(metric_config):
        return _NoopProgress()
    try:
        return tqdm(
            total=total,
            desc=label,
            unit="step",
            dynamic_ncols=True,
            leave=True,
            file=sys.stderr,
        )
    except TypeError:
        return _NoopProgress()


def _set_metric_progress(progress, description: str) -> None:
    if hasattr(progress, "set_description_str"):
        progress.set_description_str(description)


def _safe_ratio(numerator: float | int | np.integer, denominator: float | int | np.integer) -> float:
    denominator = float(denominator)
    return 0.0 if denominator == 0.0 else float(numerator) / denominator


def _safe_average_precision_torch(y_true: np.ndarray, y_prob: np.ndarray, device: torch.device) -> float:
    true_tensor = torch.as_tensor(y_true, device=device).reshape(-1) > 0
    score_tensor = torch.as_tensor(y_prob, dtype=torch.float32, device=device).reshape(-1)
    score_tensor = torch.nan_to_num(score_tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
    if score_tensor.numel() == 0:
        return 0.0
    positive_count = float(true_tensor.sum().item())
    if positive_count <= 0:
        return 0.0

    order = torch.argsort(score_tensor, descending=True)
    sorted_scores = score_tensor[order]
    sorted_true = true_tensor[order].to(torch.float64)

    threshold_ends = torch.ones(sorted_scores.numel(), dtype=torch.bool, device=device)
    threshold_ends[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    threshold_idxs = torch.nonzero(threshold_ends, as_tuple=False).flatten()

    true_positives = torch.cumsum(sorted_true, dim=0)[threshold_idxs]
    precision = true_positives / (threshold_idxs.to(torch.float64) + 1.0)
    recall = true_positives / positive_count
    previous_recall = torch.cat((torch.zeros(1, dtype=torch.float64, device=device), recall[:-1]))
    value = torch.sum((recall - previous_recall) * precision)
    numeric = float(value.detach().cpu())
    return numeric if math.isfinite(numeric) else 0.0


def _safe_average_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    device: torch.device | None = None,
) -> float:
    y_true = (np.asarray(y_true) > 0).astype(np.int64)
    y_prob = np.nan_to_num(np.asarray(y_prob, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    if y_true.size == 0 or int(y_true.sum()) == 0:
        return 0.0
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        try:
            return _safe_average_precision_torch(y_true, y_prob, device)
        except Exception as exc:
            logging.warning("Falling back to CPU average precision after CUDA metric failure: %s", exc)
    try:
        value = float(average_precision_score(y_true.ravel(), y_prob.ravel()))
    except Exception:
        return 0.0
    return value if math.isfinite(value) else 0.0


def _threshold_values_for_metrics(threshold: float | np.ndarray, n_classes: int) -> np.ndarray:
    values = np.asarray(threshold, dtype=np.float64)
    if values.ndim == 0 or values.size == 1:
        return np.full((1, int(n_classes)), float(values.reshape(-1)[0]), dtype=np.float64)
    values = values.reshape(-1)
    if values.size != int(n_classes):
        raise ValueError(f"threshold has {values.size} value(s), expected 1 or {n_classes}")
    return values.reshape(1, int(n_classes))


def _core_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | np.ndarray,
    *,
    device: torch.device,
    y_pred_override: np.ndarray | None = None,
    class_support: np.ndarray | None = None,
    low_max_support: int = 50,
    progress=None,
    progress_label: str | None = None,
) -> dict[str, float]:
    progress_label = progress_label or "core ML f1"
    _set_metric_progress(progress, progress_label)
    y_true = (np.asarray(y_true) > 0).astype(np.int64)
    y_prob = np.nan_to_num(np.asarray(y_prob, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    if y_prob.ndim != 2 or y_prob.shape[1] == 0:
        if progress is not None:
            progress.update(2)
        return {
            "f1_macro": 0.0,
            "f1_macro_observed": 0.0,
            "observed_class_count": 0.0,
            "f1_micro": 0.0,
            "precision_macro": 0.0,
            "precision_macro_observed": 0.0,
            "precision_micro": 0.0,
            "recall_macro": 0.0,
            "recall_macro_observed": 0.0,
            "recall_micro": 0.0,
            "auprc_micro": 0.0,
        }
    y_pred = (
        (y_prob >= _threshold_values_for_metrics(threshold, y_prob.shape[1])).astype(np.int64)
        if y_pred_override is None
        else (np.asarray(y_pred_override) > 0).astype(np.int64)
    )
    true_bool = y_true.astype(bool)
    pred_bool = y_pred.astype(bool)
    tp = np.count_nonzero(true_bool & pred_bool, axis=0)
    fp = np.count_nonzero(~true_bool & pred_bool, axis=0)
    fn = np.count_nonzero(true_bool & ~pred_bool, axis=0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fn) != 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
    tp_total = int(tp.sum())
    fp_total = int(fp.sum())
    fn_total = int(fn.sum())
    precision_micro = _safe_ratio(tp_total, tp_total + fp_total)
    recall_micro = _safe_ratio(tp_total, tp_total + fn_total)
    precision_macro = float(precision.mean()) if precision.size else 0.0
    recall_macro = float(recall.mean()) if recall.size else 0.0
    f1_macro = float(f1.mean()) if f1.size else 0.0
    f1_micro = _safe_ratio(2.0 * precision_micro * recall_micro, precision_micro + recall_micro)
    observed = y_true.sum(axis=0) > 0
    unobserved = ~observed
    unobserved_label_fp_count = int(y_pred[:, unobserved].sum()) if unobserved.any() else 0
    predicted_positive_count = int(y_pred.sum())
    support = np.asarray(class_support if class_support is not None else y_true.sum(axis=0), dtype=np.float64)
    buckets = {
        "low": (support > 0) & (support <= int(low_max_support)),
        "high": support > int(low_max_support),
    }
    if progress is not None:
        progress.update(1)
    _set_metric_progress(progress, progress_label.replace("ML f1", "AUPRC"))
    auprc_micro = _safe_average_precision(y_true, y_prob, device=device)
    if progress is not None:
        progress.update(1)
    metrics = {
        "f1_macro": f1_macro,
        "f1_macro_observed": float(f1[observed].mean()) if observed.any() else 0.0,
        "observed_class_count": float(observed.sum()),
        "unobserved_class_count": float(unobserved.sum()),
        "class_coverage": _safe_ratio(observed.sum(), y_true.shape[1]),
        "unobserved_label_fp_count": float(unobserved_label_fp_count),
        "unobserved_label_fp_rate": _safe_ratio(unobserved_label_fp_count, predicted_positive_count),
        "unobserved_label_prediction_per_sample": _safe_ratio(unobserved_label_fp_count, y_true.shape[0]),
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "precision_macro_observed": float(precision[observed].mean()) if observed.any() else 0.0,
        "precision_micro": precision_micro,
        "recall_macro": recall_macro,
        "recall_macro_observed": float(recall[observed].mean()) if observed.any() else 0.0,
        "recall_micro": recall_micro,
        "auprc_micro": auprc_micro,
    }
    for name, mask in buckets.items():
        class_count = int(mask.sum())
        observed_mask = mask & observed
        observed_class_count = int(observed_mask.sum())
        bucket_tp = int(tp[mask].sum())
        bucket_fp = int(fp[mask].sum())
        bucket_fn = int(fn[mask].sum())
        bucket_precision = _safe_ratio(bucket_tp, bucket_tp + bucket_fp)
        bucket_recall = _safe_ratio(bucket_tp, bucket_tp + bucket_fn)
        metrics[f"{name}_class_count"] = float(class_count)
        metrics[f"{name}_observed_class_count"] = float(observed_class_count)
        metrics[f"{name}_class_coverage"] = _safe_ratio(observed_class_count, class_count)
        metrics[f"{name}_f1_macro"] = float(f1[mask].mean()) if mask.any() else 0.0
        metrics[f"{name}_f1_macro_observed"] = (
            float(f1[observed_mask].mean()) if observed_mask.any() else 0.0
        )
        metrics[f"{name}_precision_macro"] = float(precision[mask].mean()) if mask.any() else 0.0
        metrics[f"{name}_precision_macro_observed"] = (
            float(precision[observed_mask].mean()) if observed_mask.any() else 0.0
        )
        metrics[f"{name}_precision_micro"] = bucket_precision
        metrics[f"{name}_recall_macro"] = float(recall[mask].mean()) if mask.any() else 0.0
        metrics[f"{name}_recall_macro_observed"] = (
            float(recall[observed_mask].mean()) if observed_mask.any() else 0.0
        )
        metrics[f"{name}_recall_micro"] = bucket_recall
        metrics[f"{name}_f1_micro"] = _safe_ratio(
            2.0 * bucket_precision * bucket_recall,
            bucket_precision + bucket_recall,
        )
        metrics[f"{name}_auprc_micro"] = (
            _safe_average_precision(y_true[:, mask], y_prob[:, mask], device=device)
            if mask.any()
            else 0.0
        )
    return metrics


def _core_single_label_f1_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    progress=None,
    progress_label: str | None = None,
) -> dict[str, float]:
    progress_label = progress_label or "core SL f1"
    _set_metric_progress(progress, progress_label)
    y_true = (np.asarray(y_true) > 0).astype(np.int64)
    y_prob = np.nan_to_num(np.asarray(y_prob, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    sample_count = int(y_true.shape[0]) if y_true.ndim == 2 else 0
    eligible = y_true.sum(axis=1) == 1 if y_true.ndim == 2 else np.zeros(0, dtype=bool)
    single_count = int(eligible.sum())
    metrics = {
        "single_label_sample_count": float(single_count),
        "single_label_accuracy": 0.0,
        "single_label_precision_macro": 0.0,
        "single_label_precision_micro": 0.0,
        "single_label_precision_weighted": 0.0,
        "single_label_recall_macro": 0.0,
        "single_label_recall_micro": 0.0,
        "single_label_recall_weighted": 0.0,
        "single_label_f1_macro": 0.0,
        "single_label_f1_micro": 0.0,
        "single_label_f1_weighted": 0.0,
    }
    if single_count > 0 and y_prob.ndim == 2 and y_prob.shape[1] > 0:
        true_index = np.argmax(y_true[eligible], axis=1)
        predicted_index = np.argmax(y_prob[eligible], axis=1)
        class_count = int(y_prob.shape[1])
        correct = true_index == predicted_index
        true_counts = np.bincount(true_index, minlength=class_count).astype(np.float64)
        pred_counts = np.bincount(predicted_index, minlength=class_count).astype(np.float64)
        true_positive = np.bincount(true_index[correct], minlength=class_count).astype(np.float64)
        precision = np.divide(true_positive, pred_counts, out=np.zeros_like(true_positive), where=pred_counts != 0)
        recall = np.divide(true_positive, true_counts, out=np.zeros_like(true_positive), where=true_counts != 0)
        f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
        observed_labels = np.unique(true_index)
        accuracy = float(np.mean(correct))
        support = true_counts[observed_labels] if observed_labels.size else np.array([], dtype=np.float64)
        support_total = float(support.sum())
        metrics["single_label_accuracy"] = accuracy
        metrics["single_label_precision_macro"] = float(precision[observed_labels].mean()) if observed_labels.size else 0.0
        metrics["single_label_precision_micro"] = accuracy
        metrics["single_label_precision_weighted"] = (
            float(np.average(precision[observed_labels], weights=support)) if support_total > 0.0 else 0.0
        )
        metrics["single_label_recall_macro"] = float(recall[observed_labels].mean()) if observed_labels.size else 0.0
        metrics["single_label_recall_micro"] = accuracy
        metrics["single_label_recall_weighted"] = (
            float(np.average(recall[observed_labels], weights=support)) if support_total > 0.0 else 0.0
        )
        metrics["single_label_f1_macro"] = float(f1[observed_labels].mean()) if observed_labels.size else 0.0
        metrics["single_label_f1_micro"] = accuracy
        metrics["single_label_f1_weighted"] = (
            float(np.average(f1[observed_labels], weights=support)) if support_total > 0.0 else 0.0
        )
    if progress is not None:
        progress.update(1)
    return metrics


def _format_metric(value: object, *, precision: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(numeric):
        return "-"
    return f"{numeric:.{precision}f}"


def _format_epoch_metrics(
    metrics: dict[str, float],
    *,
    monitor_metric: str,
    source: str,
) -> str:
    epoch = int(metrics.get("epoch", 0))
    is_core = str(metrics.get("metrics_status", "")).lower() == "core"
    lines = [
        f"Epoch {epoch:03d} metrics [{source}]",
        "  loss      | "
        f"train={_format_metric(metrics.get('train_loss'))} "
        f"val={_format_metric(metrics.get('val_loss'))}",
    ]
    if not is_core:
        lines.append(
            "  selection | "
            f"{monitor_metric}={_format_metric(metrics.get(monitor_metric))} "
            f"ec4_auprc={_format_metric(metrics.get('val_ec4_auprc_micro'))} "
            f"mean_f1={_format_metric(metrics.get('val_mean_f1_micro'))} "
            f"mean_fmax={_format_metric(metrics.get('val_mean_fmax'))}"
        )
    lines.append(
        "  samples   | "
        f"calibration={int(metrics.get('val_calibration_sample_count', 0))} "
        f"evaluation={int(metrics.get('val_evaluation_sample_count', 0))} "
        f"threshold_cv_folds={int(metrics.get('val_threshold_cv_folds', 0))}"
    )
    if is_core:
        lines.append(
            "  selection | "
            f"{monitor_metric}={_format_metric(metrics.get(monitor_metric))} "
            f"core_select={_format_metric(metrics.get('val_core_selection_score'))} "
            f"profile={metrics.get('metrics_profile', '-')}"
        )
        for level in LEVELS:
            prefix = f"val_{level}"
            if f"{prefix}_f1_macro" not in metrics:
                continue
            lines.append(
                f"  {level:<9} | "
                f"auprc={_format_metric(metrics.get(f'{prefix}_auprc_micro'))} "
                f"ML f1={_format_metric(metrics.get(f'{prefix}_f1_micro'))}/"
                f"{_format_metric(metrics.get(f'{prefix}_f1_macro'))} "
                f"p/r={_format_metric(metrics.get(f'{prefix}_precision_micro'))}/"
                f"{_format_metric(metrics.get(f'{prefix}_recall_micro'))} "
                f"observed_macro={_format_metric(metrics.get(f'{prefix}_f1_macro_observed'))} "
                f"SL f1={_format_metric(metrics.get(f'{prefix}_single_label_f1_micro'))}/"
                f"{_format_metric(metrics.get(f'{prefix}_single_label_f1_macro'))} "
                f"p/r={_format_metric(metrics.get(f'{prefix}_single_label_precision_macro'))}/"
                f"{_format_metric(metrics.get(f'{prefix}_single_label_recall_macro'))}"
            )
            lines.append(
                "             | "
                f"low/high observed_macro="
                f"{_format_metric(metrics.get(f'{prefix}_low_f1_macro_observed'))}/"
                f"{_format_metric(metrics.get(f'{prefix}_high_f1_macro_observed'))} "
                f"micro={_format_metric(metrics.get(f'{prefix}_low_f1_micro'))}/"
                f"{_format_metric(metrics.get(f'{prefix}_high_f1_micro'))} "
                f"low_p/r={_format_metric(metrics.get(f'{prefix}_low_precision_micro'))}/"
                f"{_format_metric(metrics.get(f'{prefix}_low_recall_micro'))}"
            )
            lines.append(
                "             | "
                f"coverage(low/high)={_format_metric(metrics.get(f'{prefix}_low_class_coverage'))}/"
                f"{_format_metric(metrics.get(f'{prefix}_high_class_coverage'))} "
                f"unobserved_fp_rate={_format_metric(metrics.get(f'{prefix}_unobserved_label_fp_rate'))}"
            )
        return "\n".join(lines)
    for level in LEVELS:
        prefix = f"val_{level}"
        lines.append(
            f"  {level:<9} | "
            f"ML f1={_format_metric(metrics.get(f'{prefix}_f1_micro'))}/"
            f"{_format_metric(metrics.get(f'{prefix}_f1_macro'))} "
            f"obs_macro={_format_metric(metrics.get(f'{prefix}_f1_macro_observed'))} "
            f"p/r={_format_metric(metrics.get(f'{prefix}_precision_micro'))}/"
            f"{_format_metric(metrics.get(f'{prefix}_recall_micro'))} "
            f"auprc={_format_metric(metrics.get(f'{prefix}_auprc_micro'))} "
            f"fmax={_format_metric(metrics.get(f'{prefix}_fmax'))} "
            f"thr={_format_metric(metrics.get(f'{prefix}_threshold_mean'))}"
        )
        if f"{prefix}_single_label_sample_count" in metrics:
            lines.append(
                "             | "
                f"SL n={int(metrics.get(f'{prefix}_single_label_sample_count', 0))} "
                f"acc={_format_metric(metrics.get(f'{prefix}_single_label_accuracy'))} "
                f"f1={_format_metric(metrics.get(f'{prefix}_single_label_f1_macro'))}/"
                f"{_format_metric(metrics.get(f'{prefix}_single_label_f1_weighted'))} "
                f"mcc={_format_metric(metrics.get(f'{prefix}_single_label_mcc'))} "
                f"top1={_format_metric(metrics.get(f'{prefix}_single_label_top1_accuracy'))}"
            )
        lines.append(
            "             | "
            f"low/high observed_macro="
            f"{_format_metric(metrics.get(f'{prefix}_low_f1_macro_observed'))}/"
            f"{_format_metric(metrics.get(f'{prefix}_high_f1_macro_observed'))} "
            f"micro={_format_metric(metrics.get(f'{prefix}_low_f1_micro'))}/"
            f"{_format_metric(metrics.get(f'{prefix}_high_f1_micro'))} "
            f"low_p/r={_format_metric(metrics.get(f'{prefix}_low_precision_micro'))}/"
            f"{_format_metric(metrics.get(f'{prefix}_low_recall_micro'))}"
        )
        lines.append(
            "             | "
            f"coverage(low/high)={_format_metric(metrics.get(f'{prefix}_low_class_coverage'))}/"
            f"{_format_metric(metrics.get(f'{prefix}_high_class_coverage'))} "
            f"unobserved_fp_rate={_format_metric(metrics.get(f'{prefix}_unobserved_label_fp_rate'))}"
        )
    lines.append(
        "  hierarchy | "
        f"f1={_format_metric(metrics.get('val_hierarchical_f1'))} "
        f"violation={_format_metric(metrics.get('val_hierarchy_violation_rate'))} "
        f"raw_violation={_format_metric(metrics.get('val_raw_hierarchy_violation_rate'))}"
    )
    site_metrics = [
        f"{task}={_format_metric(metrics.get(f'val_site_{task}_f1'))}"
        for task in SITE_TASKS
        if f"val_site_{task}_f1" in metrics
    ]
    if site_metrics:
        lines.append("  sites     | f1 " + " ".join(site_metrics))
    intermediate_site_metrics = [
        f"{task}={_format_metric(metrics.get(f'val_intermediate_site_{task}_f1'))}"
        for task in SITE_TASKS
        if f"val_intermediate_site_{task}_f1" in metrics
    ]
    if intermediate_site_metrics:
        lines.append("  inter-site| f1 " + " ".join(intermediate_site_metrics))
    hypergraph_metrics = [
        f"{name}={_format_metric(value)}"
        for key, value in sorted(metrics.items())
        if key.startswith("val_hypergraph_")
        for name in [key.removeprefix("val_hypergraph_")]
    ]
    if hypergraph_metrics:
        lines.append("  hypergraph| " + " ".join(hypergraph_metrics))
    tool_annotation_metrics = [
        f"{name}={_format_metric(value)}"
        for key, value in sorted(metrics.items())
        if key.startswith("val_tool_annotation_")
        for name in [key.removeprefix("val_tool_annotation_")]
    ]
    if tool_annotation_metrics:
        lines.append("  tool-anno | " + " ".join(tool_annotation_metrics))
    return "\n".join(lines)


def _log_epoch_metrics(metrics: dict[str, float], *, monitor_metric: str, source: str) -> None:
    logging.info("%s", _format_epoch_metrics(metrics, monitor_metric=monitor_metric, source=source))


def _configure_warning_filters(metric_config: dict[str, object]) -> None:
    if bool(metric_config.get("suppress_sklearn_warnings", True)):
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
            module=r"sklearn\.metrics\._classification",
        )


def _resolve_device(train_config: dict, device_override: str | None, distributed: DistributedContext) -> torch.device:
    device_name = str(device_override or train_config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")).strip()
    if distributed.enabled and torch.cuda.is_available() and device_name.startswith("cuda"):
        return torch.device(f"cuda:{distributed.local_rank}")
    return torch.device(device_name)


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pdb_graph_selection_config(config: dict) -> dict[str, Any]:
    raw = (config.get("train", {}) or {}).get("pdb_graph_selection", {})
    if isinstance(raw, bool):
        return {"enabled": raw, "mode": "quality", "max_graphs_per_uniprot": 1}
    if not isinstance(raw, dict):
        return {"enabled": False}
    return {
        **raw,
        "enabled": bool(raw.get("enabled", False)),
        "mode": str(raw.get("mode", "quality")).strip().lower(),
        "max_graphs_per_uniprot": int(raw.get("max_graphs_per_uniprot", 1) or 1),
    }


def _select_pdb_graph_rows(
    config: dict,
    candidates: list[tuple[dict[str, str], Path]],
) -> list[tuple[dict[str, str], Path]]:
    selection = _pdb_graph_selection_config(config)
    if not bool(selection.get("enabled", False)):
        return candidates
    max_graphs = int(selection.get("max_graphs_per_uniprot", 1) or 1)
    if max_graphs <= 0:
        return candidates

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    path_by_row_id: dict[int, Path] = {}
    pdb_candidate_count = 0
    for row, path in candidates:
        uniprot_id = str(row.get("uniprot_id") or "").strip()
        pdb_id = str(row.get("pdb_id") or "").strip()
        if uniprot_id and pdb_id and pdb_id.lower() not in {"nan", "none", "null"}:
            groups[uniprot_id].append(row)
            path_by_row_id[id(row)] = path
            pdb_candidate_count += 1

    if not groups:
        return candidates

    mode = str(selection.get("mode", "quality")).strip().lower()
    if mode == "quality":
        deployment = config.get("deployment", {}) or {}
        cache_path = selection.get("quality_cache_path") or deployment.get("pdb_quality_cache_path")
        if not cache_path:
            cache_path = Path(config["storage"]["manifest_dir"]) / "pdb_candidate_quality.json"
        ranked_groups = rank_pdb_candidate_groups(
            groups,
            cache_path=cache_path,
            api_url=str(selection.get("quality_api_url") or deployment.get("pdb_quality_api_url", "https://data.rcsb.org/graphql")),
            batch_size=int(selection.get("quality_batch_size") or deployment.get("pdb_quality_batch_size", 100) or 100),
            timeout=float(selection.get("quality_timeout_seconds") or deployment.get("pdb_quality_timeout_seconds", 60.0)),
            retries=int(selection.get("quality_retries") or deployment.get("pdb_quality_retries", 3) or 3),
            retry_backoff_seconds=float(
                selection.get("quality_retry_backoff_seconds")
                or deployment.get("pdb_quality_retry_backoff_seconds", 2.0)
            ),
        )
    elif mode == "csv_order":
        ranked_groups = groups
    elif mode in {"all", "none", "off", "false"}:
        return candidates
    else:
        raise ValueError(
            f"Unsupported train.pdb_graph_selection.mode={mode!r}; expected 'quality', 'csv_order', or 'all'"
        )

    selected_row_ids: set[int] = set()
    for rows in ranked_groups.values():
        selected_row_ids.update(id(row) for row in rows[:max_graphs])

    selected = [
        (row, path)
        for row, path in candidates
        if not _norm_pdb_id(row.get("pdb_id")) or id(row) in selected_row_ids
    ]
    logging.info(
        "Training PDB graph selection | mode=%s max_graphs_per_uniprot=%d groups=%d "
        "pdb_candidates=%d selected=%d",
        mode,
        max_graphs,
        len(groups),
        pdb_candidate_count,
        sum(1 for row, _ in selected if _norm_pdb_id(row.get("pdb_id"))),
    )
    return selected


def _norm_pdb_id(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _graph_rows(config: dict, limit: int | None = None) -> list[tuple[dict[str, str], Path]]:
    rows = read_csv_rows(config["datasets"]["train"]["csv"])
    graph_rows: list[tuple[dict[str, str], Path]] = []
    seen_paths: set[str] = set()
    for row in rows:
        graph_path = graph_path_for_row(config, "train", row)
        graph_path_text = str(graph_path) if graph_path is not None else ""
        if graph_path is not None and graph_path.exists() and graph_path_text not in seen_paths:
            seen_paths.add(graph_path_text)
            graph_rows.append((row, graph_path))
    graph_rows = _select_pdb_graph_rows(config, graph_rows)
    if limit is not None:
        return graph_rows[:limit]
    return graph_rows



def _cached_splits_match_graph_rows(
    split_paths: dict[str, list[str]],
    graph_rows: list[tuple[dict[str, str], Path]],
) -> bool:
    split_sets = {name: set(split_paths.get(name, [])) for name in SPLIT_NAMES}
    if any(len(split_sets[name]) != len(split_paths.get(name, [])) for name in SPLIT_NAMES):
        return False
    if (split_sets["train"] & split_sets["val"]) or (split_sets["train"] & split_sets["test"]) or (split_sets["val"] & split_sets["test"]):
        return False
    expected = {str(path) for _, path in graph_rows}
    return set().union(*split_sets.values()) == expected


def _cluster_group_columns(cluster_column: str) -> list[str]:
    return list(dict.fromkeys((cluster_column, "homology_cluster", "sequence_cluster", "uniprot_id", "source_record_id")))


def _cluster_keys(row: dict[str, str], cluster_column: str) -> list[str]:
    keys: list[str] = []
    for key in _cluster_group_columns(cluster_column):
        value = row.get(key)
        if value is not None and str(value).strip() and str(value).strip().lower() not in {"nan", "none", "null"}:
            keys.append(f"{key}:{str(value).strip()}")
    return keys or [f"row:{id(row)}"]


def _connected_cluster_ids(
    graph_rows: list[tuple[dict[str, str], Path]],
    *,
    cluster_column: str,
) -> dict[str, str]:
    parents: dict[str, str] = {}
    first_path_by_key: dict[str, str] = {}

    def find(path: str) -> str:
        parents.setdefault(path, path)
        while parents[path] != path:
            parents[path] = parents[parents[path]]
            path = parents[path]
        return path

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for row, graph_path in graph_rows:
        path = str(graph_path)
        find(path)
        for key in _cluster_keys(row, cluster_column):
            previous = first_path_by_key.setdefault(key, path)
            union(path, previous)
    return {path: find(path) for path in parents}


def _assign_cluster_splits(
    graph_rows: list[tuple[dict[str, str], Path]],
    *,
    cluster_column: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    path_clusters = _connected_cluster_ids(graph_rows, cluster_column=cluster_column)
    clusters: dict[str, list[str]] = defaultdict(list)
    for row, graph_path in graph_rows:
        path = str(graph_path)
        clusters[path_clusters[path]].append(path)

    items = list(clusters.items())
    rng = random.Random(seed)
    rng.shuffle(items)
    total = sum(len(paths) for _, paths in items)
    target = {
        "val": int(round(total * val_fraction)),
        "test": int(round(total * test_fraction)),
    }
    target["train"] = max(total - target["val"] - target["test"], 0)
    split_paths = {"train": [], "val": [], "test": []}

    # Greedy assignment keeps each homology cluster intact while tracking target sample counts.
    for _, paths in sorted(items, key=lambda item: len(item[1]), reverse=True):
        remaining = {key: target[key] - len(split_paths[key]) for key in split_paths}
        split = max(remaining, key=remaining.get)
        split_paths[split].extend(paths)

    for paths in split_paths.values():
        rng.shuffle(paths)
    return split_paths


def _closed_set_split_config(train_config: dict) -> dict[str, object]:
    raw_config = train_config.get("closed_set_split", {})
    if isinstance(raw_config, bool):
        raw: dict[str, object] = {}
        enabled = raw_config
    else:
        raw = raw_config if isinstance(raw_config, dict) else {}
        enabled = bool(raw.get("enabled", True))
    return {
        "enabled": enabled,
        "rebalance": bool(raw.get("rebalance", True)),
    }


def _ec_labels_for_row(row: dict[str, str]) -> set[ECLabel]:
    labels: set[ECLabel] = set()
    for parsed in parse_ec_list(_supervised_ec_value(row)):
        for level in LEVELS:
            label = getattr(parsed, level)
            if label:
                labels.add((level, label))
    return labels


def _supervised_ec_value(row: dict[str, str]) -> object:
    """Return the active positives while preserving raw EC metadata separately."""
    if "ec_numbers_supervised" in row:
        return row.get("ec_numbers_supervised")
    return row.get("ec_numbers")


def _build_ec_index(
    graph_rows: list[tuple[dict[str, str], Path]],
) -> tuple[Counter[ECLabel], dict[str, set[ECLabel]]]:
    supports: Counter[ECLabel] = Counter()
    graph_labels: dict[str, set[ECLabel]] = {}
    for row, graph_path in graph_rows:
        labels = _ec_labels_for_row(row)
        supports.update(labels)
        graph_labels[str(graph_path)] = labels
    return supports, graph_labels


def _build_cluster_index(
    graph_rows: list[tuple[dict[str, str], Path]],
    *,
    cluster_column: str,
) -> tuple[dict[str, list[str]], dict[str, set[ECLabel]], dict[str, str]]:
    path_clusters = _connected_cluster_ids(graph_rows, cluster_column=cluster_column)
    clusters: dict[str, list[str]] = defaultdict(list)
    cluster_labels: dict[str, set[ECLabel]] = defaultdict(set)
    for row, graph_path in graph_rows:
        path = str(graph_path)
        cluster = path_clusters[path]
        clusters[cluster].append(path)
        cluster_labels[cluster].update(_ec_labels_for_row(row))
    return dict(clusters), dict(cluster_labels), path_clusters


def _cluster_split_assignments(
    split_paths: dict[str, list[str]],
    path_clusters: dict[str, str],
) -> tuple[dict[str, str], set[str]]:
    assignments: dict[str, str] = {}
    overlaps: set[str] = set()
    for split in SPLIT_NAMES:
        for path in split_paths.get(split, []):
            cluster = path_clusters[path]
            previous = assignments.setdefault(cluster, split)
            if previous != split:
                overlaps.add(cluster)
    return assignments, overlaps


def _missing_train_ec_labels(
    split_sets: dict[str, set[str]],
    graph_labels: dict[str, set[ECLabel]],
) -> set[ECLabel]:
    train_labels = set().union(*(graph_labels.get(path, set()) for path in split_sets["train"]))
    evaluation_paths = split_sets["val"] | split_sets["test"]
    evaluation_labels = set().union(*(graph_labels.get(path, set()) for path in evaluation_paths))
    return evaluation_labels - train_labels


def _closed_set_level_summary(
    split_sets: dict[str, set[str]],
    graph_labels: dict[str, set[ECLabel]],
) -> dict[str, dict[str, int]]:
    labels_by_split = {
        split: set().union(*(graph_labels.get(path, set()) for path in split_sets[split]))
        for split in SPLIT_NAMES
    }
    return {
        level: {
            "train_labels": sum(label[0] == level for label in labels_by_split["train"]),
            "val_labels": sum(label[0] == level for label in labels_by_split["val"]),
            "test_labels": sum(label[0] == level for label in labels_by_split["test"]),
            "val_unseen_in_train": sum(label[0] == level for label in labels_by_split["val"] - labels_by_split["train"]),
            "test_unseen_in_train": sum(label[0] == level for label in labels_by_split["test"] - labels_by_split["train"]),
        }
        for level in LEVELS
    }


def _repair_train_ec_coverage(
    split_paths: dict[str, list[str]],
    graph_rows: list[tuple[dict[str, str], Path]],
    train_config: dict,
) -> tuple[dict[str, list[str]], dict[str, object]]:
    config = _closed_set_split_config(train_config)
    if not config["enabled"]:
        return split_paths, {"enabled": False}
    cluster_column = str(train_config.get("homology_cluster_column", "homology_cluster"))
    clusters, cluster_labels, path_clusters = _build_cluster_index(graph_rows, cluster_column=cluster_column)
    split_sets = {split: set(split_paths.get(split, [])) for split in SPLIT_NAMES}
    target_sizes = {split: len(split_paths.get(split, [])) for split in SPLIT_NAMES}
    cluster_split, overlaps = _cluster_split_assignments(split_paths, path_clusters)
    if overlaps:
        raise RuntimeError(f"Initial split contains {len(overlaps)} cross-split connected data cluster(s)")

    _, graph_labels = _build_ec_index(graph_rows)
    missing = _missing_train_ec_labels(split_sets, graph_labels)
    initial_missing = len(missing)
    moved_to_train: list[tuple[str, str]] = []
    while missing:
        candidates = [
            (
                -len(cluster_labels[cluster] & missing),
                len(clusters[cluster]),
                cluster,
                donor,
            )
            for cluster, donor in cluster_split.items()
            if donor != "train" and cluster_labels[cluster] & missing
        ]
        if not candidates:
            raise RuntimeError(f"Unable to move {len(missing)} evaluation EC label(s) into the training split")
        _, _, cluster, donor = min(candidates)
        for path in clusters[cluster]:
            split_sets[donor].remove(path)
            split_sets["train"].add(path)
        cluster_split[cluster] = "train"
        moved_to_train.append((cluster, donor))
        missing = _missing_train_ec_labels(split_sets, graph_labels)

    rebalanced: list[tuple[str, str]] = []
    if config["rebalance"]:
        train_cluster_counts: Counter[ECLabel] = Counter()
        for cluster, split in cluster_split.items():
            if split == "train":
                train_cluster_counts.update(cluster_labels[cluster])
        while True:
            current_penalty = sum(abs(len(split_sets[split]) - target_sizes[split]) for split in SPLIT_NAMES)
            best_move: tuple[int, int, str, str] | None = None
            for recipient in ("val", "test"):
                if len(split_sets[recipient]) >= target_sizes[recipient]:
                    continue
                for cluster, donor in cluster_split.items():
                    if donor != "train":
                        continue
                    labels = cluster_labels[cluster]
                    if any(train_cluster_counts[label] <= 1 for label in labels):
                        continue
                    cluster_size = len(clusters[cluster])
                    next_penalty = (
                        abs((len(split_sets["train"]) - cluster_size) - target_sizes["train"])
                        + abs((len(split_sets[recipient]) + cluster_size) - target_sizes[recipient])
                        + sum(
                            abs(len(split_sets[split]) - target_sizes[split])
                            for split in SPLIT_NAMES
                            if split not in {"train", recipient}
                        )
                    )
                    score = (next_penalty, cluster_size, cluster, recipient)
                    if next_penalty < current_penalty and (best_move is None or score < best_move):
                        best_move = score
            if best_move is None:
                break
            _, _, cluster, recipient = best_move
            for path in clusters[cluster]:
                split_sets["train"].remove(path)
                split_sets[recipient].add(path)
            cluster_split[cluster] = recipient
            train_cluster_counts.subtract(cluster_labels[cluster])
            rebalanced.append((cluster, recipient))

    rng = random.Random(int(train_config.get("seed", 42)))
    repaired = {split: list(split_sets[split]) for split in SPLIT_NAMES}
    for paths in repaired.values():
        rng.shuffle(paths)
    _, overlaps = _cluster_split_assignments(repaired, path_clusters)
    missing = _missing_train_ec_labels(split_sets, graph_labels)
    if overlaps or missing:
        raise RuntimeError("Closed-set split repair failed to preserve cluster isolation and train EC coverage")
    return repaired, {
        "enabled": True,
        "cluster_column": cluster_column,
        "group_columns": _cluster_group_columns(cluster_column),
        "initial_unseen_evaluation_labels": initial_missing,
        "moved_clusters_to_train": len(moved_to_train),
        "rebalanced_clusters": len(rebalanced),
        "cross_split_cluster_count": 0,
        "remaining_unseen_evaluation_labels": 0,
        "level_summary": _closed_set_level_summary(split_sets, graph_labels),
    }


def _closed_set_coverage_satisfied(
    split_paths: dict[str, list[str]],
    graph_rows: list[tuple[dict[str, str], Path]],
    train_config: dict,
) -> tuple[bool, dict[str, object]]:
    config = _closed_set_split_config(train_config)
    if not config["enabled"]:
        return True, {"enabled": False}
    cluster_column = str(train_config.get("homology_cluster_column", "homology_cluster"))
    _, _, path_clusters = _build_cluster_index(graph_rows, cluster_column=cluster_column)
    _, graph_labels = _build_ec_index(graph_rows)
    split_sets = {split: set(split_paths.get(split, [])) for split in SPLIT_NAMES}
    _, overlaps = _cluster_split_assignments(split_paths, path_clusters)
    missing = _missing_train_ec_labels(split_sets, graph_labels)
    summary = {
        "enabled": True,
        "cluster_column": cluster_column,
        "group_columns": _cluster_group_columns(cluster_column),
        "cross_split_cluster_count": len(overlaps),
        "unseen_evaluation_label_count": len(missing),
        "level_summary": _closed_set_level_summary(split_sets, graph_labels),
    }
    return not overlaps and not missing, summary


def load_or_create_splits(config: dict, limit: int | None = None) -> dict[str, list[str]]:
    train_config = config.get("train", {}) or {}
    split_dir = Path(config["storage"]["split_dir"])
    split_dir.mkdir(parents=True, exist_ok=True)
    split_files = {name: split_dir / f"{name}_graphs.txt" for name in ("train", "val", "test")}
    if fixed_split_directory(config) is not None:
        fixed_paths, fixed_summary = load_fixed_splits(config, limit=limit)
        if limit is None:
            for name, paths in fixed_paths.items():
                if not split_files[name].exists() or _read_lines(split_files[name]) != paths:
                    _write_lines(split_files[name], paths)
            summary_path = split_dir / "split_summary.json"
            if not summary_path.exists() or _load_json(summary_path) != fixed_summary:
                write_json(summary_path, fixed_summary)
        return fixed_paths
    force_resplit = bool(train_config.get("force_resplit", False))
    graph_rows: list[tuple[dict[str, str], Path]] | None = None

    if limit is None and not force_resplit and all(path.exists() for path in split_files.values()):
        loaded = {name: _read_lines(path) for name, path in split_files.items()}
        if all(loaded.values()) and all(Path(path).exists() for paths in loaded.values() for path in paths):
            graph_rows = _graph_rows(config, limit=limit)
            if not _cached_splits_match_graph_rows(loaded, graph_rows):
                logging.info("Existing split files are stale relative to the current graph-backed CSV rows; regenerating splits.")
            else:
                closed_set_ok, closed_set_summary = _closed_set_coverage_satisfied(loaded, graph_rows, train_config)
                if closed_set_ok:
                    return loaded
                logging.info("Existing split files are not strict closed-set splits; regenerating: %s", closed_set_summary)

    if graph_rows is None:
        graph_rows = _graph_rows(config, limit=limit)
    if not graph_rows:
        raise RuntimeError("No prepared training graphs found. Run deploy-data before training.")
    split_paths = _assign_cluster_splits(
        graph_rows,
        cluster_column=str(train_config.get("homology_cluster_column", "homology_cluster")),
        val_fraction=float(train_config.get("val_fraction", 0.1)),
        test_fraction=float(train_config.get("test_fraction", 0.1)),
        seed=int(train_config.get("seed", 42)),
    )
    split_paths, closed_set_summary = _repair_train_ec_coverage(split_paths, graph_rows, train_config)
    if limit is None:
        for name, paths in split_paths.items():
            _write_lines(split_files[name], paths)
        write_json(
            split_dir / "split_summary.json",
            {
                "strategy": "connected_closed_set",
                "counts": {name: len(paths) for name, paths in split_paths.items()},
                "cluster_column": str(train_config.get("homology_cluster_column", "homology_cluster")),
                "group_columns": _cluster_group_columns(str(train_config.get("homology_cluster_column", "homology_cluster"))),
                "seed": int(train_config.get("seed", 42)),
                "closed_set_split": closed_set_summary,
            },
        )
    return split_paths


def _load_label_maps(config: dict) -> tuple[dict[str, dict[str, int]], dict[str, list[tuple[int, int]]]]:
    label_dir = Path(config["storage"]["label_map_dir"])
    label_path = label_dir / "ec_label_maps.json"
    parent_path = label_dir / "parent_child_maps.json"
    if label_path.exists() and parent_path.exists():
        label_maps = _load_json(label_path)
        parent_child_maps = _load_json(parent_path)
        return label_maps, parent_child_maps
    label_maps, parent_child_maps = build_current_label_maps(config)
    save_label_maps(config, label_maps, parent_child_maps)
    return label_maps, parent_child_maps


def _label_supports_from_csv(
    config: dict,
    label_maps: dict[str, dict[str, int]],
    graph_paths: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    supports = {level: np.zeros(len(label_maps[level]), dtype=np.float64) for level in LEVELS}
    selected_paths = set(graph_paths) if graph_paths is not None else None
    for row, graph_path in _graph_rows(config):
        if selected_paths is not None and str(graph_path) not in selected_paths:
            continue
        encoded = encode_ec_values(_supervised_ec_value(row), label_maps)
        for level in LEVELS:
            supports[level] += encoded[level].detach().cpu().numpy()
    return supports


def _class_balanced_pos_weights(
    class_supports: dict[str, np.ndarray] | None,
    loss_config: dict,
) -> dict[str, torch.Tensor] | None:
    mode = str(loss_config.get("class_balance", "none")).lower()
    if class_supports is None or mode in {"", "none", "false", "off"}:
        return None
    beta = float(loss_config.get("class_balance_beta", 0.9999))
    power = float(loss_config.get("class_balance_power", 1.0))
    min_weight = float(loss_config.get("class_balance_min_weight", 0.25))
    max_weight = float(loss_config.get("class_balance_max_weight", loss_config.get("pos_weight_max", 20.0)))
    pos_weights: dict[str, torch.Tensor] = {}
    for level, support_values in class_supports.items():
        support = np.asarray(support_values, dtype=np.float64)
        positive = support > 0
        weights = np.ones_like(support, dtype=np.float64)
        if mode in {"effective_number", "effective", "true"}:
            clipped_support = np.maximum(support, 1.0)
            weights = (1.0 - beta) / np.maximum(1.0 - np.power(beta, clipped_support), 1e-12)
            if positive.any():
                weights = weights / max(float(weights[positive].mean()), 1e-12)
            weights[~positive] = 0.0
        elif mode in {"inverse", "inverse_support"}:
            weights = 1.0 / np.maximum(support, 1.0)
            if positive.any():
                weights = weights / max(float(weights[positive].mean()), 1e-12)
            weights[~positive] = 0.0
        else:
            raise ValueError(f"Unsupported loss.class_balance mode: {mode}")
        weights = np.power(weights, power)
        weights = np.clip(weights, min_weight, max_weight)
        weights[~positive] = 0.0
        pos_weights[level] = torch.as_tensor(weights, dtype=torch.float32)
    return pos_weights


def _tail_sampler_config(train_config: dict) -> dict[str, object]:
    raw = train_config.get("tail_sampler", {})
    if isinstance(raw, bool):
        return {"enabled": raw}
    if not isinstance(raw, dict):
        return {"enabled": False}
    return raw


def _node_feature_mask_config(train_config: dict) -> dict[str, object]:
    augmentation = train_config.get("data_augmentation", {})
    if not isinstance(augmentation, dict):
        return {"enabled": False}
    raw = augmentation.get("node_feature_mask", {})
    if isinstance(raw, bool):
        return {"enabled": raw}
    if not isinstance(raw, dict):
        return {"enabled": False}
    return raw


def _tail_sample_weights(paths: list[str], config: dict, train_config: dict) -> torch.Tensor | None:
    sampler_config = _tail_sampler_config(train_config)
    if not bool(sampler_config.get("enabled", False)):
        return None
    level = str(sampler_config.get("level", "ec4"))
    if level not in LEVELS:
        raise ValueError(f"train.tail_sampler.level must be one of {LEVELS}, got {level!r}")
    selected_paths = {str(path) for path in paths}
    path_labels: dict[str, set[str]] = {}
    support: Counter[str] = Counter()
    for row, graph_path in _graph_rows(config):
        graph_path_text = str(graph_path)
        if graph_path_text not in selected_paths:
            continue
        labels = {
            label
            for parsed in parse_ec_list(_supervised_ec_value(row))
            for label in [getattr(parsed, level)]
            if label is not None
        }
        path_labels[graph_path_text] = labels
        support.update(labels)
    if not support:
        return None

    tail_max_support = max(int(sampler_config.get("tail_max_support", 10)), 1)
    power = float(sampler_config.get("power", 0.5))
    base_weight = float(sampler_config.get("base_weight", 1.0))
    tail_min_weight = float(sampler_config.get("tail_min_weight", base_weight))
    empty_weight = float(sampler_config.get("empty_weight", 0.35))
    min_weight = float(sampler_config.get("min_weight", 0.25))
    max_weight = float(sampler_config.get("max_weight", 5.0))
    weights: list[float] = []
    for path in paths:
        labels = path_labels.get(str(path), set())
        if not labels:
            weight = empty_weight
        else:
            weight = base_weight
            for label in labels:
                label_support = max(int(support[label]), 1)
                if label_support <= tail_max_support:
                    rarity_multiplier = (tail_max_support / label_support) ** power
                    weight = max(weight, tail_min_weight * rarity_multiplier)
        weights.append(float(np.clip(weight, min_weight, max_weight)))
    return torch.as_tensor(weights, dtype=torch.double)


def _metric_config(config: dict) -> dict[str, object]:
    train_config = config.get("train", {}) or {}
    metric_config = config.get("metrics", {}) or {}
    return {
        "threshold": float(metric_config.get("threshold", 0.5)),
        "threshold_mode": str(metric_config.get("threshold_mode", train_config.get("threshold_mode", "fixed"))).lower(),
        "core_threshold_mode": str(
            metric_config.get("core_threshold_mode", metric_config.get("threshold_mode", train_config.get("threshold_mode", "fixed")))
        ).lower(),
        "threshold_steps": int(metric_config.get("threshold_steps", metric_config.get("fmax_steps", 101))),
        "threshold_min": float(metric_config.get("threshold_min", 0.0)),
        "threshold_max": float(metric_config.get("threshold_max", 1.0)),
        "threshold_calibration_fraction": float(metric_config.get("threshold_calibration_fraction", 0.0)),
        "core_threshold_calibration_fraction": float(
            metric_config.get("core_threshold_calibration_fraction", metric_config.get("threshold_calibration_fraction", 0.0))
        ),
        "threshold_unseen_fallback": str(metric_config.get("threshold_unseen_fallback", "one")).lower(),
        "threshold_fbeta": float(metric_config.get("threshold_fbeta", 1.0)),
        "threshold_low_fbeta": float(metric_config.get("threshold_low_fbeta", metric_config.get("threshold_fbeta", 1.0))),
        "threshold_high_fbeta": float(metric_config.get("threshold_high_fbeta", metric_config.get("threshold_fbeta", 1.0))),
        "core_threshold_cv_folds": int(metric_config.get("core_threshold_cv_folds", 0)),
        "threshold_calibration_seed": int(metric_config.get("threshold_calibration_seed", train_config.get("seed", 42))),
        "hierarchy_postprocess": str(metric_config.get("hierarchy_postprocess", "none")).lower(),
        "topk": [int(value) for value in metric_config.get("topk", [1, 3, 5, 10])],
        "fmax_steps": int(metric_config.get("fmax_steps", 101)),
        "calibration_bins": int(metric_config.get("calibration_bins", 15)),
        "frequency_bucket_schema": str(metric_config.get("frequency_bucket_schema", "low_high_v1")),
        "low_max_support": int(metric_config.get("low_max_support", 50)),
        "bootstrap_resamples": int(metric_config.get("bootstrap_resamples", 0)),
        "bootstrap_seed": int(metric_config.get("bootstrap_seed", train_config.get("seed", 42))),
        "async": bool(metric_config.get("async", False)),
        "profile": str(metric_config.get("profile", "full")).strip().lower(),
        "progress": metric_config.get("progress", True),
        "full_interval": int(metric_config.get("full_interval", metric_config.get("interval", 1))),
        "full_first_epoch": bool(metric_config.get("full_first_epoch", True)),
        "full_after_training_best": bool(metric_config.get("full_after_training_best", True)),
        "async_workers": int(metric_config.get("async_workers", 2)),
        "max_pending": int(metric_config.get("max_pending", 8)),
        "cache_compressed": bool(metric_config.get("cache_compressed", False)),
        "keep_prediction_cache": bool(metric_config.get("keep_prediction_cache", False)),
        "keep_epoch_checkpoints": bool(metric_config.get("keep_epoch_checkpoints", False)),
        "recover_pending_on_start": bool(metric_config.get("recover_pending_on_start", False)),
        "suppress_sklearn_warnings": bool(metric_config.get("suppress_sklearn_warnings", True)),
        "core_levels": metric_config.get("core_levels", ["ec4"]),
        "core_selection_score_weights": metric_config.get("core_selection_score_weights"),
    }


def _validate_label_shapes(graph_paths: list[str], label_maps: dict[str, dict[str, int]], max_checks: int = 256) -> None:
    for graph_path in graph_paths[:max_checks]:
        data = torch.load(graph_path, map_location="cpu", weights_only=False)
        for level in LEVELS:
            target = getattr(data, f"y_{level}", None)
            expected = len(label_maps[level])
            if target is None or int(target.numel()) != expected:
                raise RuntimeError(
                    f"{graph_path} has y_{level} size "
                    f"{None if target is None else int(target.numel())}, expected {expected}. "
                    "Run scripts/repair_graph_labels.py before training."
                )


def _make_model(config: dict, label_maps: dict[str, dict[str, int]]) -> torch.nn.Module:
    model_config = dict(config.get("model", {}) or {})
    model_type = str(model_config.pop("type", "hifen")).lower()
    if model_type != "hifen":
        raise ValueError(f"Unsupported model.type: {model_type!r}")
    model_config.setdefault("input_dim", int(config.get("esm", {}).get("embedding_dim", 1536)))
    model_config.setdefault("hidden_dim", 512)
    model_config["num_classes"] = {level: len(label_maps[level]) for level in LEVELS}
    model_config["use_annotation_features"] = bool(
        (config.get("features", {}) or {}).get("use_annotation_features", True)
    )
    model_config["edge_cutoff"] = float((config.get("graph", {}) or {}).get("cutoff", 10.0))
    return HiFENModel(**model_config)


def _configure_trainable_parameters(
    model: torch.nn.Module,
    patterns: Iterable[str] | None,
) -> dict[str, object]:
    wrapped = getattr(model, "module", model)
    selected_patterns = tuple(str(pattern) for pattern in (patterns or []) if str(pattern).strip())
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in wrapped.named_parameters():
        trainable = not selected_patterns or any(fnmatch.fnmatchcase(name, pattern) for pattern in selected_patterns)
        parameter.requires_grad_(trainable)
        (trainable_names if trainable else frozen_names).append(name)
    if not trainable_names:
        raise ValueError(f"train.trainable_parameter_patterns matched no parameters: {selected_patterns}")
    trainable_count = sum(parameter.numel() for parameter in wrapped.parameters() if parameter.requires_grad)
    total_count = sum(parameter.numel() for parameter in wrapped.parameters())
    return {
        "patterns": selected_patterns,
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
        "trainable_count": int(trainable_count),
        "frozen_count": int(total_count - trainable_count),
        "total_count": int(total_count),
    }


def _make_loss(
    config: dict,
    class_supports: dict[str, np.ndarray] | None = None,
    train_sample_count: int = 0,
) -> HierarchicalECLoss:
    loss_config = config.get("loss", {}) or {}
    pos_weights = _class_balanced_pos_weights(class_supports, loss_config)
    return HierarchicalECLoss(
        ec_level_weights=loss_config.get("ec_level_weights"),
        focal_gamma=float(loss_config.get("focal_gamma", 2.0)),
        pos_weights=pos_weights,
        auxiliary_site_weight=float(loss_config.get("auxiliary_site_weight", 0.0)),
        intermediate_site_weight=float(loss_config.get("intermediate_site_weight", 0.0)),
        site_focal_gamma=float(loss_config.get("site_focal_gamma", loss_config.get("focal_gamma", 2.0))),
        site_task_weights=loss_config.get("site_task_weights"),
        site_positive_weights=loss_config.get("site_positive_weights"),
        ignore_empty_label_levels=sorted(
            {"ec2", "ec3", "ec4"}
            | {str(level) for level in loss_config.get("ignore_empty_label_levels", [])}
        ),
        primary_loss=str(loss_config.get("primary_loss", "focal_bce")),
        class_supports=class_supports,
        train_sample_count=train_sample_count,
        db_rebalance_alpha=float(loss_config.get("db_rebalance_alpha", 0.1)),
        db_rebalance_beta=float(loss_config.get("db_rebalance_beta", 10.0)),
        db_rebalance_gamma=float(loss_config.get("db_rebalance_gamma", 0.2)),
        db_negative_scale=float(loss_config.get("db_negative_scale", 2.0)),
        db_init_bias_scale=float(loss_config.get("db_init_bias_scale", 0.05)),
    )


def _make_loader(
    paths: list[str],
    config: dict,
    *,
    shuffle: bool,
    distributed: DistributedContext,
    distributed_train: bool = False,
    class_supports: dict[str, np.ndarray] | None = None,
    label_maps: dict[str, dict[str, int]] | None = None,
) -> tuple[DataLoader, DistributedSampler | None]:
    if DataLoader is None:
        raise RuntimeError("torch_geometric is required for training")
    train_config = config.get("train", {}) or {}
    node_mask_config = _node_feature_mask_config(train_config)
    node_mask_support = None
    if distributed_train and shuffle and bool(node_mask_config.get("enabled", False)):
        level = str(node_mask_config.get("level", "ec4"))
        if not class_supports or level not in class_supports:
            raise ValueError(f"Node feature masking requires class supports for level {level!r}")
        node_mask_support = class_supports[level]
        if distributed.is_main:
            support = np.asarray(node_mask_support)
            tail_max_support = max(int(node_mask_config.get("tail_max_support", 10)), 1)
            logging.info(
                "Using tail-directed node feature masking | level=%s tail_classes=%s "
                "mask_ratio=%.3f probability=%.3f protect_sites=%s protected_hops=%s fill=%s",
                level,
                int(((support > 0) & (support <= tail_max_support)).sum()),
                float(node_mask_config.get("mask_ratio", 0.1)),
                float(node_mask_config.get("apply_probability", 1.0)),
                bool(node_mask_config.get("protect_site_positive", True)),
                int(node_mask_config.get("protected_hops", 1)),
                str(node_mask_config.get("fill_value", "zero")),
            )
    dynamic_label_encoding = bool(train_config.get("dynamic_label_encoding", False))
    ec_values_by_path: dict[str, str] = {}
    ignored_ec_values_by_path: dict[str, str] = {}
    if dynamic_label_encoding:
        wanted_paths = {str(path) for path in paths}
        labels_by_path: defaultdict[str, set[str]] = defaultdict(set)
        ignored_labels_by_path: defaultdict[str, set[str]] = defaultdict(set)
        for dataset_name, dataset_config in (config.get("datasets", {}) or {}).items():
            csv_path = dataset_config.get("csv")
            if not csv_path or not Path(csv_path).exists():
                continue
            for row in read_csv_rows(csv_path):
                graph_path = graph_path_for_row(config, dataset_name, row)
                path = str(graph_path) if graph_path is not None else ""
                if path in wanted_paths:
                    labels_by_path[path].update(split_ec_tokens(_supervised_ec_value(row)))
                    ignored_labels_by_path[path].update(split_ec_tokens(row.get("ec_numbers_ignored")))
        missing_paths = sorted(wanted_paths - set(labels_by_path))
        if missing_paths:
            raise RuntimeError(
                f"Dynamic label encoding could not map {len(missing_paths)} graph path(s) "
                f"to configured CSV rows; first missing path: {missing_paths[0]}"
            )
        ec_values_by_path = {
            path: ";".join(sorted(values)) for path, values in labels_by_path.items()
        }
        ignored_ec_values_by_path = {
            path: ";".join(sorted(ignored_labels_by_path.get(path, set())))
            for path in labels_by_path
        }
        if distributed.is_main:
            partial_count = sum(bool(value) for value in ignored_ec_values_by_path.values())
            logging.info(
                "Dynamic EC label encoding enabled for %s graph(s), including %s partial-supervision graph(s)",
                len(ec_values_by_path),
                partial_count,
            )
    dataset = ThinGraphDataset(
        paths,
        node_feature_mask_config=node_mask_config if node_mask_support is not None else None,
        node_feature_mask_support=node_mask_support,
        ignored_label_indices={
            level: sorted(
                int(index)
                for label, index in level_map.items()
                if "-" in str(label).split(".")
            )
            for level, level_map in (label_maps or {}).items()
        },
        ec_values_by_path=ec_values_by_path,
        ignored_ec_values_by_path=ignored_ec_values_by_path,
        label_maps=label_maps if dynamic_label_encoding else None,
    )
    weighted_sampler = None
    if distributed_train and shuffle:
        weights = _tail_sample_weights(paths, config, train_config)
        if weights is not None:
            weighted_sampler = WeightedEpochSampler(
                weights,
                num_replicas=distributed.world_size if distributed.enabled else 1,
                rank=distributed.rank if distributed.enabled else 0,
                replacement=bool(_tail_sampler_config(train_config).get("replacement", True)),
                seed=int(train_config.get("seed", 42)),
            )
            if distributed.is_main:
                logging.info(
                    "Using %s tail sampler | samples_per_rank=%s weight_range=%.3f..%.3f",
                    str(_tail_sampler_config(train_config).get("level", "ec4")),
                    len(weighted_sampler),
                    float(weights.min().item()),
                    float(weights.max().item()),
                )
    sampler = weighted_sampler
    if sampler is None and distributed.enabled and distributed_train:
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=shuffle,
            drop_last=bool(train_config.get("distributed_drop_last", False)),
        )
    num_workers = int(train_config.get("num_workers", 4))
    loader_kwargs = {
        "batch_size": int(train_config.get("batch_size", 32)),
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(train_config.get("pin_memory", torch.cuda.is_available())),
        "persistent_workers": bool(train_config.get("persistent_workers", False)) and num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(train_config.get("prefetch_factor", 2))
    return DataLoader(dataset, **loader_kwargs), sampler


def _batch_tensor_size(batch, name: str, dim: int | None = None) -> int:
    value = getattr(batch, name, None)
    if value is None or not hasattr(value, "size"):
        return 0
    if dim is None:
        return int(value.numel())
    if value.dim() <= dim:
        return 0
    return int(value.size(dim))


def _batch_diagnostics(batch, device: torch.device, distributed: DistributedContext) -> dict[str, float]:
    augmentation_eligible = getattr(batch, "augmentation_eligible", None)
    augmentation_applied = getattr(batch, "augmentation_applied", None)
    augmentation_masked_nodes = getattr(batch, "augmentation_masked_nodes", None)
    augmentation_protected_nodes = getattr(batch, "augmentation_protected_nodes", None)
    local = torch.tensor(
        [
            float(getattr(batch, "num_graphs", 1)),
            float(getattr(batch, "num_nodes", 0) or _batch_tensor_size(batch, "x", 0)),
            float(_batch_tensor_size(batch, "edge_index", 1)),
            float(augmentation_eligible.sum().item()) if augmentation_eligible is not None else 0.0,
            float(augmentation_applied.sum().item()) if augmentation_applied is not None else 0.0,
            float(augmentation_masked_nodes.sum().item()) if augmentation_masked_nodes is not None else 0.0,
            float(augmentation_protected_nodes.sum().item()) if augmentation_protected_nodes is not None else 0.0,
        ],
        dtype=torch.float64,
        device=device,
    )
    if not distributed.enabled:
        return {
            "graphs_avg": float(local[0].item()),
            "graphs_max": float(local[0].item()),
            "nodes_avg": float(local[1].item()),
            "nodes_max": float(local[1].item()),
            "edges_avg": float(local[2].item()),
            "edges_max": float(local[2].item()),
            "augmentation_eligible": float(local[3].item()),
            "augmentation_applied": float(local[4].item()),
            "augmentation_masked_nodes": float(local[5].item()),
            "augmentation_protected_nodes": float(local[6].item()),
        }
    summed = local.clone()
    maximum = local.clone()
    dist.all_reduce(summed, op=dist.ReduceOp.SUM)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return {
        "graphs_avg": float(summed[0].item() / max(distributed.world_size, 1)),
        "graphs_max": float(maximum[0].item()),
        "nodes_avg": float(summed[1].item() / max(distributed.world_size, 1)),
        "nodes_max": float(maximum[1].item()),
        "edges_avg": float(summed[2].item() / max(distributed.world_size, 1)),
        "edges_max": float(maximum[2].item()),
        "augmentation_eligible": float(summed[3].item()),
        "augmentation_applied": float(summed[4].item()),
        "augmentation_masked_nodes": float(summed[5].item()),
        "augmentation_protected_nodes": float(summed[6].item()),
    }


def _target_for(batch, level: str, shape: torch.Size, device: torch.device) -> torch.Tensor:
    target = getattr(batch, f"y_{level}")
    target = target.to(device=device, dtype=torch.float32)
    if target.dim() == 1:
        target = target.view(shape)
    return target


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class ParameterNormGuard:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        patterns: Iterable[str],
        max_growth_factor: float,
    ):
        self.patterns = tuple(str(pattern) for pattern in patterns)
        self.max_growth_factor = float(max_growth_factor)
        if self.max_growth_factor <= 1.0:
            raise ValueError("parameter_norm_guard.max_growth_factor must be greater than 1")
        wrapped = getattr(model, "module", model)
        self.reference_norms = {
            name: float(parameter.detach().float().norm().cpu())
            for name, parameter in wrapped.named_parameters()
            if parameter.requires_grad
            and any(fnmatch.fnmatchcase(name, pattern) for pattern in self.patterns)
            and float(parameter.detach().float().norm().cpu()) > 0.0
        }

    def apply(self, model: torch.nn.Module) -> list[tuple[str, float, float]]:
        wrapped = getattr(model, "module", model)
        clamped: list[tuple[str, float, float]] = []
        nonfinite: list[str] = []
        with torch.no_grad():
            for name, parameter in wrapped.named_parameters():
                reference = self.reference_norms.get(name)
                if reference is None:
                    continue
                norm = parameter.detach().float().norm()
                if not bool(torch.isfinite(norm)):
                    nonfinite.append(name)
                    continue
                value = float(norm.cpu())
                limit = reference * self.max_growth_factor
                if value > limit:
                    parameter.mul_(limit / max(value, torch.finfo(torch.float32).tiny))
                    clamped.append((name, value, limit))
        if nonfinite:
            raise FloatingPointError(f"Non-finite guarded parameter(s): {nonfinite[:20]}")
        return clamped


def _make_parameter_norm_guard(model: torch.nn.Module, train_config: dict) -> ParameterNormGuard | None:
    raw = train_config.get("parameter_norm_guard", {})
    if isinstance(raw, bool):
        enabled = raw
        config: dict[str, object] = {}
    else:
        config = raw if isinstance(raw, dict) else {}
        enabled = bool(config.get("enabled", False))
    if not enabled:
        return None
    return ParameterNormGuard(
        model,
        patterns=config.get("patterns", ["backbone.convs.*.lin_r.weight", "backbone.convs.*.lin_r.bias"]),
        max_growth_factor=float(config.get("max_growth_factor", 32.0)),
    )


def _autocast(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _nonfinite_output_paths(value, prefix: str = "outputs") -> list[str]:
    paths: list[str] = []
    if torch.is_tensor(value):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            paths.append(prefix)
        return paths
    if isinstance(value, dict):
        for key, nested in value.items():
            paths.extend(_nonfinite_output_paths(nested, f"{prefix}.{key}"))
    return paths


def _distributed_any(
    value: bool,
    device: torch.device,
    distributed: DistributedContext | None,
) -> bool:
    """Return whether any rank reports ``value`` without splitting DDP control flow."""
    if distributed is None or not distributed.enabled:
        return value
    packed = torch.tensor([int(value)], dtype=torch.int32, device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.MAX)
    return bool(packed.item())


def _forward_loss_with_amp_fallback(
    model: torch.nn.Module,
    criterion: HierarchicalECLoss,
    batch,
    device: torch.device,
    *,
    amp: bool,
    distributed: DistributedContext | None = None,
) -> tuple[dict, torch.Tensor, float, bool, list[str]]:
    with _autocast(device, amp):
        outputs = model(batch)
        loss = criterion(outputs, batch)
    loss_value = float(loss.detach().cpu())
    if not amp:
        return outputs, loss, loss_value, False, []

    local_amp_nonfinite = not math.isfinite(loss_value)
    retry_in_fp32 = _distributed_any(local_amp_nonfinite, device, distributed)
    if not retry_in_fp32:
        return outputs, loss, loss_value, False, []

    amp_nonfinite_outputs = _nonfinite_output_paths(outputs) if local_amp_nonfinite else []
    del outputs, loss
    with _autocast(device, False):
        outputs = model(batch)
        loss = criterion(outputs, batch)
    loss_value = float(loss.detach().cpu())
    if _distributed_any(not math.isfinite(loss_value), device, distributed):
        # Force every rank down the same pre-backward failure path even when
        # only one rank still has a non-finite loss after the FP32 retry.
        loss_value = float("nan")
    return outputs, loss, loss_value, True, amp_nonfinite_outputs


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: HierarchicalECLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp: bool,
    scaler,
    parameter_norm_guard: ParameterNormGuard | None,
    grad_clip_norm: float | None,
    log_interval: int,
    epoch: int,
    total_epochs: int,
    distributed: DistributedContext,
) -> float:
    model.train()
    running = 0.0
    seen = 0
    total_steps = len(loader)
    show_progress_bar = _progress_bar_enabled(distributed)
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:03d}/{total_epochs:03d} train",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress_bar,
    )
    skipped_nonfinite_gradients = 0
    recovered_amp_forward_overflows = 0
    parameter_clamp_events = 0
    for step, batch in enumerate(progress, start=1):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs, loss, loss_value, amp_retried, amp_nonfinite_outputs = _forward_loss_with_amp_fallback(
            model,
            criterion,
            batch,
            device,
            amp=amp,
            distributed=distributed,
        )
        graph_paths = getattr(batch, "graph_path", [])
        if isinstance(graph_paths, str):
            graph_paths = [graph_paths]
        if amp_retried and math.isfinite(loss_value):
            recovered_amp_forward_overflows += 1
            if distributed.is_main and recovered_amp_forward_overflows <= 3:
                logging.warning(
                    "Recovered AMP non-finite forward with FP32 retry | epoch=%03d step=%05d "
                    "amp_nonfinite_outputs=%s graphs=%s",
                    epoch,
                    step,
                    amp_nonfinite_outputs[:20],
                    list(graph_paths)[:8],
                )
        if not math.isfinite(loss_value):
            nonfinite_outputs = _nonfinite_output_paths(outputs)
            raise FloatingPointError(
                f"Non-finite train loss at epoch {epoch}, step {step}: {loss_value}; "
                f"rank={distributed.rank}; nonfinite_outputs={nonfinite_outputs[:20]}; "
                f"amp_retried={amp_retried}; amp_nonfinite_outputs={amp_nonfinite_outputs[:20]}; "
                f"graphs={list(graph_paths)[:8]}"
            )
        scaler.scale(loss).backward()
        skip_optimizer_step = False
        if grad_clip_norm is not None and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, error_if_nonfinite=False)
            grad_norm_value = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
            if not math.isfinite(grad_norm_value):
                if not amp:
                    raise FloatingPointError(
                        f"Non-finite gradient norm at epoch {epoch}, step {step}: {grad_norm_value}"
                    )
                skipped_nonfinite_gradients += 1
                skip_optimizer_step = True
                optimizer.zero_grad(set_to_none=True)
                if distributed.is_main and skipped_nonfinite_gradients <= 3:
                    logging.debug(
                        "Skipping optimizer step after AMP non-finite gradient overflow | epoch=%03d step=%05d grad_norm=%s",
                        epoch,
                        step,
                        grad_norm_value,
                    )
        if skip_optimizer_step:
            scaler.update()
            continue
        scaler.step(optimizer)
        scaler.update()
        if parameter_norm_guard is not None:
            clamped = parameter_norm_guard.apply(model)
            if clamped:
                parameter_clamp_events += 1
                if distributed.is_main and parameter_clamp_events <= 3:
                    logging.info(
                        "Applied parameter norm guard | epoch=%03d step=%05d tensors=%s",
                        epoch,
                        step,
                        ", ".join(f"{name}:{value:.2f}->{limit:.2f}" for name, value, limit in clamped[:8]),
                    )
        batch_graphs = int(getattr(batch, "num_graphs", 1))
        running += loss_value * batch_graphs
        seen += batch_graphs
        if log_interval and (step % log_interval == 0 or step == total_steps):
            average_loss = running / max(seen, 1)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            batch_stats = _batch_diagnostics(batch, device, distributed)
            if not distributed.is_main:
                continue
            if show_progress_bar:
                progress.set_postfix(
                    loss=f"{average_loss:.6f}",
                    lr=f"{learning_rate:.3e}",
                    aug=f"{batch_stats['augmentation_applied']:.0f}/{batch_stats['augmentation_eligible']:.0f}",
                )
            else:
                logging.info(
                    "Epoch %03d/%03d | train %05d/%05d (%5.1f%%) | loss=%.6f | lr=%.3e | "
                    "batch graphs=%.1f/%.0f nodes=%.0f/%.0f edges=%.0f/%.0f | "
                    "augmentation eligible=%.0f applied=%.0f masked_nodes=%.0f protected_nodes=%.0f",
                    epoch,
                    total_epochs,
                    step,
                    total_steps,
                    100.0 * step / max(total_steps, 1),
                    average_loss,
                    learning_rate,
                    batch_stats["graphs_avg"],
                    batch_stats["graphs_max"],
                    batch_stats["nodes_avg"],
                    batch_stats["nodes_max"],
                    batch_stats["edges_avg"],
                    batch_stats["edges_max"],
                    batch_stats["augmentation_eligible"],
                    batch_stats["augmentation_applied"],
                    batch_stats["augmentation_masked_nodes"],
                    batch_stats["augmentation_protected_nodes"],
                )
    if distributed.is_main and recovered_amp_forward_overflows:
        logging.warning(
            "Recovered AMP non-finite forwards during epoch | epoch=%03d count=%d",
            epoch,
            recovered_amp_forward_overflows,
        )
    if distributed.enabled:
        packed = torch.tensor([running, float(seen)], dtype=torch.float64, device=device)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        running = float(packed[0].item())
        seen = int(packed[1].item())
    if seen <= 0:
        raise FloatingPointError(f"All training batches were skipped at epoch {epoch} due to non-finite gradients")
    return running / max(seen, 1)


@torch.no_grad()
def _collect_validation_outputs(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: HierarchicalECLoss,
    device: torch.device,
    distributed: DistributedContext,
    *,
    progress_label: str = "Validation",
    log_interval: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    seen = 0
    probs_by_level: dict[str, list[np.ndarray]] = {level: [] for level in LEVELS}
    targets_by_level: dict[str, list[np.ndarray]] = {level: [] for level in LEVELS}
    site_histograms = {
        task: {
            "positive": np.zeros(200, dtype=np.int64),
            "negative": np.zeros(200, dtype=np.int64),
            "count": 0,
        }
        for task in SITE_TASKS
    }
    intermediate_site_histograms = {
        task: {
            "positive": np.zeros(200, dtype=np.int64),
            "negative": np.zeros(200, dtype=np.int64),
            "count": 0,
        }
        for task in SITE_TASKS
    }
    hypergraph_diagnostic_sums: dict[str, float] = {}
    hypergraph_diagnostic_counts: dict[str, int] = {}
    tool_annotation_diagnostic_sums: dict[str, float] = {}
    tool_annotation_diagnostic_counts: dict[str, int] = {}
    progress = tqdm(
        loader,
        desc=progress_label,
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        disable=not _progress_bar_enabled(distributed),
    )
    for step, batch in enumerate(progress, start=1):
        batch = batch.to(device, non_blocking=True)
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite validation loss during {progress_label}, step {step}: {loss_value}")
        batch_graphs = int(getattr(batch, "num_graphs", 1))
        total_loss += loss_value * batch_graphs
        seen += batch_graphs
        for level in LEVELS:
            probs = torch.sigmoid(outputs[level]).detach().cpu().numpy()
            target = _target_for(batch, level, outputs[level].shape, device).detach().cpu().numpy()
            probs_by_level[level].append(probs)
            targets_by_level[level].append(target)
        site_targets = getattr(batch, "y_site", None)
        site_target_mask = getattr(batch, "y_site_mask", None)
        if site_targets is not None and site_target_mask is not None:
            site_targets = site_targets.detach().cpu().numpy()
            site_target_mask = site_target_mask.detach().cpu().numpy()
            for index, task in enumerate(SITE_TASKS):
                if task not in outputs:
                    continue
                valid = site_target_mask[:, index] > 0
                if not np.any(valid):
                    continue
                histogram = binary_histogram_counts(
                    site_targets[:, index][valid],
                    torch.sigmoid(outputs[task]).detach().cpu().numpy()[valid],
                    bins=200,
                )
                site_histograms[task]["positive"] += histogram["positive"]
                site_histograms[task]["negative"] += histogram["negative"]
                site_histograms[task]["count"] += int(histogram["count"])
                intermediate_sites = outputs.get("intermediate_sites") or {}
                if task in intermediate_sites:
                    intermediate_histogram = binary_histogram_counts(
                        site_targets[:, index][valid],
                        torch.sigmoid(intermediate_sites[task]).detach().cpu().numpy()[valid],
                        bins=200,
                    )
                    intermediate_site_histograms[task]["positive"] += intermediate_histogram["positive"]
                    intermediate_site_histograms[task]["negative"] += intermediate_histogram["negative"]
                    intermediate_site_histograms[task]["count"] += int(intermediate_histogram["count"])
        for name, values in (outputs.get("hypergraph_diagnostics") or {}).items():
            values = values.detach().float()
            finite = torch.isfinite(values)
            if bool(finite.any()):
                hypergraph_diagnostic_sums[str(name)] = (
                    hypergraph_diagnostic_sums.get(str(name), 0.0)
                    + float(values[finite].sum().cpu())
                )
                hypergraph_diagnostic_counts[str(name)] = (
                    hypergraph_diagnostic_counts.get(str(name), 0)
                    + int(finite.sum().cpu())
                )
        for name, values in (outputs.get("tool_annotation_diagnostics") or {}).items():
            values = values.detach().float()
            finite = torch.isfinite(values)
            if bool(finite.any()):
                tool_annotation_diagnostic_sums[str(name)] = (
                    tool_annotation_diagnostic_sums.get(str(name), 0.0)
                    + float(values[finite].sum().cpu())
                )
                tool_annotation_diagnostic_counts[str(name)] = (
                    tool_annotation_diagnostic_counts.get(str(name), 0)
                    + int(finite.sum().cpu())
                )
        if distributed.is_main and log_interval and (step % log_interval == 0 or step == len(loader)):
            logging.info(
                "Progress | stage=validation | label=%s | batch=%05d/%05d | percent=%5.1f%%",
                progress_label,
                step,
                len(loader),
                100.0 * step / max(len(loader), 1),
            )
        if distributed.is_main and progress_callback is not None:
            progress_callback(step, len(loader))
    return {
        "val_loss": total_loss / max(seen, 1),
        "probs": {level: np.concatenate(probs_by_level[level], axis=0) for level in LEVELS},
        "targets": {level: np.concatenate(targets_by_level[level], axis=0) for level in LEVELS},
        "site_histograms": site_histograms,
        "intermediate_site_histograms": intermediate_site_histograms,
        "hypergraph_diagnostic_sums": hypergraph_diagnostic_sums,
        "hypergraph_diagnostic_counts": hypergraph_diagnostic_counts,
        "tool_annotation_diagnostic_sums": tool_annotation_diagnostic_sums,
        "tool_annotation_diagnostic_counts": tool_annotation_diagnostic_counts,
    }


def _thresholds_from_payload(
    validation_payload: dict[str, object],
    parent_child_maps: dict[str, list[tuple[int, int]]],
    metric_config: dict[str, object],
    class_supports: dict[str, np.ndarray] | None = None,
    *,
    progress=None,
    progress_label: str = "Metrics",
) -> tuple[dict[str, float | np.ndarray], int]:
    hierarchy_postprocess = str(metric_config.get("hierarchy_postprocess", "none")).lower()
    raw_probs_by_level = validation_payload["probs"]
    probs_by_level = (
        apply_hierarchy_probability_gate(raw_probs_by_level, parent_child_maps, mode=hierarchy_postprocess)
        if hierarchy_postprocess not in {"", "none", "off", "false"}
        else raw_probs_by_level
    )
    targets_by_level = validation_payload["targets"]
    sample_count = int(next(iter(targets_by_level.values())).shape[0])
    calibration_indices, _ = _validation_calibration_indices(sample_count, metric_config)
    threshold_by_level = {}
    for level in LEVELS:
        _set_metric_progress(progress, f"{progress_label} threshold {level}")
        level_calibration_indices = _known_level_indices(
            targets_by_level[level],
            calibration_indices,
        )
        threshold_by_level[level] = _threshold_for_level(
            targets_by_level[level][level_calibration_indices],
            probs_by_level[level][level_calibration_indices],
            metric_config,
            None if class_supports is None else class_supports[level],
        )
        if progress is not None:
            progress.update(1)
    if hierarchy_postprocess not in {"", "none", "off", "false"}:
        threshold_by_level = apply_hierarchy_threshold_gate(threshold_by_level, parent_child_maps)
    return threshold_by_level, int(calibration_indices.size)


def _compute_validation_metrics(
    validation_payload: dict[str, object],
    parent_child_maps: dict[str, list[tuple[int, int]]],
    class_supports: dict[str, np.ndarray],
    metric_config: dict[str, object],
    threshold_overrides: dict[str, float | np.ndarray] | None = None,
    progress_label: str | None = None,
) -> dict[str, float]:
    threshold = float(metric_config["threshold"])
    hierarchy_postprocess = str(metric_config.get("hierarchy_postprocess", "none")).lower()
    topk_values = list(metric_config["topk"])
    fmax_steps = int(metric_config["fmax_steps"])
    calibration_bins = int(metric_config["calibration_bins"])
    low_max_support = int(metric_config["low_max_support"])
    bootstrap_resamples = int(metric_config["bootstrap_resamples"])
    bootstrap_seed = int(metric_config["bootstrap_seed"])

    raw_probs_by_level = validation_payload["probs"]
    probs_by_level = (
        apply_hierarchy_probability_gate(raw_probs_by_level, parent_child_maps, mode=hierarchy_postprocess)
        if hierarchy_postprocess not in {"", "none", "off", "false"}
        else raw_probs_by_level
    )
    targets_by_level = validation_payload["targets"]
    metrics: dict[str, float] = {
        "val_loss": float(validation_payload["val_loss"]),
        "frequency_bucket_schema": str(metric_config.get("frequency_bucket_schema", "low_high_v1")),
    }
    sample_count = int(next(iter(targets_by_level.values())).shape[0])
    site_histograms = validation_payload.get("site_histograms", {})
    intermediate_site_histograms = validation_payload.get("intermediate_site_histograms", {})
    hypergraph_diagnostic_counts = validation_payload.get("hypergraph_diagnostic_counts", {})
    tool_annotation_diagnostic_counts = validation_payload.get(
        "tool_annotation_diagnostic_counts",
        {},
    )
    progress_label = progress_label or "Validation metrics"
    progress_total = (
        (len(LEVELS) if threshold_overrides is None else 0)
        + len(LEVELS) * 6
        + (len(LEVELS) if bootstrap_resamples > 0 else 0)
        + 2
        + sum(1 for histogram in site_histograms.values() if int(histogram.get("count", 0)) > 0)
        + sum(
            1
            for histogram in intermediate_site_histograms.values()
            if int(histogram.get("count", 0)) > 0
        )
        + (1 if hypergraph_diagnostic_counts else 0)
        + (1 if tool_annotation_diagnostic_counts else 0)
    )
    with _make_metric_progress(metric_config, label=progress_label, total=progress_total) as progress:
        if threshold_overrides is None:
            threshold_by_level, calibration_count = _thresholds_from_payload(
                validation_payload,
                parent_child_maps,
                metric_config,
                class_supports,
                progress=progress,
                progress_label=progress_label,
            )
            calibration_indices, evaluation_indices = _validation_calibration_indices(
                sample_count,
                metric_config,
            )
        else:
            threshold_by_level = threshold_overrides
            calibration_count = 0
            calibration_indices = np.empty(0, dtype=np.int64)
            evaluation_indices = np.arange(sample_count, dtype=np.int64)
        metrics["val_calibration_sample_count"] = float(calibration_count)
        metrics["val_evaluation_sample_count"] = float(evaluation_indices.size)
        consistency_inputs: dict[str, np.ndarray] = {}
        raw_consistency_inputs: dict[str, np.ndarray] = {}
        target_inputs: dict[str, np.ndarray] = {}
        known_level_masks: dict[str, np.ndarray] = {}
        micro_scores = []
        fmax_scores = []
        macro_scores = []
        observed_macro_scores = []
        for level in LEVELS:
            level_evaluation_indices = _known_level_indices(
                targets_by_level[level],
                evaluation_indices,
            )
            level_calibration_indices = _known_level_indices(
                targets_by_level[level],
                calibration_indices,
            )
            y_prob = probs_by_level[level][level_evaluation_indices]
            raw_y_prob = raw_probs_by_level[level][level_evaluation_indices]
            y_true = targets_by_level[level][level_evaluation_indices]
            level_threshold = threshold_by_level[level]
            hierarchy_targets = targets_by_level[level][evaluation_indices]
            eligible_sample_count = _known_level_indices(targets_by_level[level]).size
            metrics[f"val_{level}_eligible_sample_count"] = float(eligible_sample_count)
            metrics[f"val_{level}_excluded_unknown_sample_count"] = float(
                sample_count - eligible_sample_count
            )
            metrics[f"val_{level}_calibration_sample_count"] = float(
                level_calibration_indices.size
            )
            metrics[f"val_{level}_evaluation_sample_count"] = float(
                level_evaluation_indices.size
            )
            metrics.update(_threshold_summary(f"val_{level}", level_threshold))
            consistency_inputs[level] = probs_by_level[level][evaluation_indices]
            raw_consistency_inputs[level] = raw_probs_by_level[level][evaluation_indices]
            target_inputs[level] = hierarchy_targets
            known_level_masks[level] = hierarchy_targets.sum(axis=1) > 0

            _set_metric_progress(progress, f"{progress_label} {level} cache")
            metric_cache = build_metric_cache(y_true, y_prob, threshold=level_threshold)
            progress.update(1)

            _set_metric_progress(progress, f"{progress_label} {level} multilabel")
            level_metrics = multilabel_metrics(
                y_true,
                y_prob,
                threshold=level_threshold,
                fmax_steps=fmax_steps,
                calibration_bins=calibration_bins,
                metric_cache=metric_cache,
            )
            for key, value in level_metrics.items():
                metrics[f"val_{level}_{key}"] = value
            progress.update(1)

            _set_metric_progress(progress, f"{progress_label} {level} top-k")
            for key, value in topk_metrics(y_true, y_prob, ks=topk_values).items():
                metrics[f"val_{level}_{key}"] = value
            progress.update(1)

            _set_metric_progress(progress, f"{progress_label} {level} single-label")
            for key, value in single_label_metrics(y_true, y_prob, ks=topk_values).items():
                metrics[f"val_{level}_{key}"] = value
            progress.update(1)

            _set_metric_progress(progress, f"{progress_label} {level} closed-set")
            for key, value in closed_set_metrics(
                y_true,
                y_prob,
                class_supports[level],
                threshold=level_threshold,
                fmax_steps=fmax_steps,
                calibration_bins=calibration_bins,
            ).items():
                metrics[f"val_{level}_{key}"] = value
            progress.update(1)

            _set_metric_progress(progress, f"{progress_label} {level} frequency-buckets")
            for key, value in class_frequency_bucket_metrics(
                y_true,
                y_prob,
                class_support=class_supports[level],
                threshold=level_threshold,
                low_max_support=low_max_support,
                metric_cache=metric_cache,
            ).items():
                metrics[f"val_{level}_{key}"] = value
            progress.update(1)

            if bootstrap_resamples > 0:
                _set_metric_progress(progress, f"{progress_label} {level} bootstrap")
                f1_low, f1_high = bootstrap_ci(
                    y_true,
                    y_prob,
                    lambda true, prob: multilabel_metrics(true, prob, threshold=level_threshold)["f1_micro"],
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed,
                )
                auprc_low, auprc_high = bootstrap_ci(
                    y_true,
                    y_prob,
                    lambda true, prob: multilabel_metrics(true, prob, threshold=threshold)["auprc_micro"],
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed + 17,
                )
                metrics[f"val_{level}_f1_micro_ci95_low"] = f1_low
                metrics[f"val_{level}_f1_micro_ci95_high"] = f1_high
                metrics[f"val_{level}_auprc_micro_ci95_low"] = auprc_low
                metrics[f"val_{level}_auprc_micro_ci95_high"] = auprc_high
            micro_scores.append(level_metrics["f1_micro"])
            fmax_scores.append(level_metrics["fmax"])
            macro_scores.append(level_metrics["f1_macro"])
            observed_macro_scores.append(level_metrics["f1_macro_observed"])
        metrics["val_mean_f1_micro"] = float(np.mean(micro_scores)) if micro_scores else 0.0
        metrics["val_mean_f1_macro"] = float(np.mean(macro_scores)) if macro_scores else 0.0
        metrics["val_mean_f1_macro_observed"] = (
            float(np.mean(observed_macro_scores)) if observed_macro_scores else 0.0
        )
        metrics["val_mean_fmax"] = float(np.mean(fmax_scores)) if fmax_scores else 0.0

        _set_metric_progress(progress, f"{progress_label} hierarchy gated")
        for key, value in hierarchical_metrics(
            target_inputs,
            consistency_inputs,
            parent_child_maps,
            threshold=threshold_by_level,
            known_level_masks=known_level_masks,
        ).items():
            metrics[f"val_{key}"] = value
        progress.update(1)

        _set_metric_progress(progress, f"{progress_label} hierarchy raw")
        for key, value in hierarchical_metrics(
            target_inputs,
            raw_consistency_inputs,
            parent_child_maps,
            threshold=threshold_by_level,
            known_level_masks=known_level_masks,
        ).items():
            metrics[f"val_raw_{key}"] = value
        progress.update(1)

        metrics["val_selection_score"] = float(
            0.30 * metrics.get("val_mean_f1_macro_observed", 0.0)
            + 0.25 * metrics.get("val_ec4_f1_macro_observed", 0.0)
            + 0.25 * metrics.get("val_ec4_single_label_f1_macro", 0.0)
            + 0.10 * metrics.get("val_mean_fmax", 0.0)
            + 0.10 * metrics.get("val_hierarchical_f1", 0.0)
        )
        for task, histogram in site_histograms.items():
            if int(histogram.get("count", 0)) <= 0:
                continue
            _set_metric_progress(progress, f"{progress_label} site {task}")
            for key, value in binary_metrics_from_histogram(
                histogram["positive"],
                histogram["negative"],
                threshold=threshold,
            ).items():
                metrics[f"val_site_{task}_{key}"] = float(value)
            progress.update(1)
        for task, histogram in intermediate_site_histograms.items():
            if int(histogram.get("count", 0)) <= 0:
                continue
            _set_metric_progress(progress, f"{progress_label} intermediate site {task}")
            for key, value in binary_metrics_from_histogram(
                histogram["positive"],
                histogram["negative"],
                threshold=threshold,
            ).items():
                metrics[f"val_intermediate_site_{task}_{key}"] = float(value)
            progress.update(1)
        if hypergraph_diagnostic_counts:
            _set_metric_progress(progress, f"{progress_label} hypergraph diagnostics")
            diagnostic_sums = validation_payload.get("hypergraph_diagnostic_sums", {})
            for name, count in hypergraph_diagnostic_counts.items():
                if int(count) > 0:
                    metrics[f"val_hypergraph_{name}"] = float(diagnostic_sums[name]) / int(count)
            progress.update(1)
        if tool_annotation_diagnostic_counts:
            _set_metric_progress(progress, f"{progress_label} tool annotation diagnostics")
            diagnostic_sums = validation_payload.get("tool_annotation_diagnostic_sums", {})
            for name, count in tool_annotation_diagnostic_counts.items():
                if int(count) > 0:
                    metrics[f"val_tool_annotation_{name}"] = (
                        float(diagnostic_sums[name]) / int(count)
                    )
            progress.update(1)
    return metrics


def _configured_core_levels(metric_config: dict[str, object]) -> list[str]:
    raw_levels = metric_config.get("core_levels", ["ec4"])
    if isinstance(raw_levels, str):
        raw_levels = [part.strip() for part in raw_levels.replace(",", " ").split()]
    levels: list[str] = []
    for raw_level in raw_levels if isinstance(raw_levels, Iterable) else []:
        level = str(raw_level).strip().lower()
        if not level:
            continue
        if level not in LEVELS:
            raise ValueError(f"metrics.core_levels contains unsupported EC level: {raw_level}")
        if level not in levels:
            levels.append(level)
    return levels or ["ec4"]


def _core_selection_score(metrics: dict[str, float], metric_config: dict[str, object]) -> float:
    weights = metric_config.get("core_selection_score_weights")
    if isinstance(weights, dict) and weights:
        score = 0.0
        for key, raw_weight in weights.items():
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                continue
            score += weight * float(metrics.get(str(key), 0.0))
        return float(score)
    return float(
        0.45 * metrics.get("val_ec4_f1_macro_observed", 0.0)
        + 0.35 * metrics.get("val_ec4_single_label_f1_macro", 0.0)
        + 0.20 * metrics.get("val_ec4_auprc_micro", 0.0)
    )


def _compute_core_validation_metrics(
    validation_payload: dict[str, object],
    parent_child_maps: dict[str, list[tuple[int, int]]],
    class_supports: dict[str, np.ndarray],
    metric_config: dict[str, object],
    *,
    device: torch.device,
    progress_label: str | None = None,
) -> dict[str, float]:
    core_metric_config = dict(metric_config)
    core_metric_config["threshold_mode"] = str(metric_config.get("core_threshold_mode", metric_config.get("threshold_mode", "fixed"))).lower()
    core_metric_config["threshold_calibration_fraction"] = float(
        metric_config.get("core_threshold_calibration_fraction", metric_config.get("threshold_calibration_fraction", 0.0))
    )
    threshold = float(core_metric_config.get("threshold", 0.5))
    hierarchy_postprocess = str(metric_config.get("hierarchy_postprocess", "none")).lower()
    raw_probs_by_level = validation_payload["probs"]
    probs_by_level = (
        apply_hierarchy_probability_gate(raw_probs_by_level, parent_child_maps, mode=hierarchy_postprocess)
        if hierarchy_postprocess not in {"", "none", "off", "false"}
        else raw_probs_by_level
    )
    targets_by_level = validation_payload["targets"]
    core_levels = _configured_core_levels(metric_config)
    sample_count = int(targets_by_level[core_levels[0]].shape[0])
    calibration_counts: list[int] = []
    evaluation_counts: list[int] = []
    threshold_cv_fold_values: list[int] = []
    threshold_cv_folds = int(core_metric_config.get("core_threshold_cv_folds", 0))
    progress_label = progress_label or "Core validation metrics"
    metrics: dict[str, float] = {
        "val_loss": float(validation_payload["val_loss"]),
        "metrics_status": "core",
        "metrics_profile": "core_ec4_observed_macro_selection" if core_levels == ["ec4"] else "core_balanced_selection",
        "frequency_bucket_schema": str(metric_config.get("frequency_bucket_schema", "low_high_v1")),
    }
    with _make_metric_progress(metric_config, label=progress_label, total=3 * len(core_levels)) as progress:
        for level in core_levels:
            level_prob = probs_by_level[level]
            level_true = targets_by_level[level]
            known_indices = _known_level_indices(level_true)
            metrics[f"val_{level}_eligible_sample_count"] = float(known_indices.size)
            metrics[f"val_{level}_excluded_unknown_sample_count"] = float(
                sample_count - known_indices.size
            )
            level_prob = level_prob[known_indices]
            level_true = level_true[known_indices]
            level_sample_count = int(known_indices.size)
            threshold_predictions = None
            level_threshold = threshold
            if (
                core_metric_config["threshold_mode"] in {"auto", "per_class", "per_class_f1"}
                and threshold_cv_folds >= 2
                and level_sample_count >= 2
            ):
                threshold_predictions, level_threshold, calibration_count = _cross_fitted_threshold_predictions(
                    level_true,
                    level_prob,
                    core_metric_config,
                    class_supports[level],
                )
                calibration_indices = np.arange(level_sample_count, dtype=np.int64)
                evaluation_indices = calibration_indices
            elif core_metric_config["threshold_mode"] in {"auto", "per_class", "per_class_f1"}:
                calibration_indices, evaluation_indices = _validation_calibration_indices(level_sample_count, core_metric_config)
                level_threshold = _threshold_for_level(
                    level_true[calibration_indices],
                    level_prob[calibration_indices],
                    core_metric_config,
                    class_supports[level],
                )
                calibration_count = int(
                    calibration_indices.size
                    if evaluation_indices.size != level_sample_count
                    else 0
                )
            else:
                calibration_indices = np.arange(level_sample_count, dtype=np.int64)
                evaluation_indices = calibration_indices
                calibration_count = 0
            eval_true = level_true[evaluation_indices]
            eval_prob = level_prob[evaluation_indices]
            calibration_counts.append(int(calibration_count))
            evaluation_counts.append(int(evaluation_indices.size))
            threshold_cv_fold_values.append(int(threshold_cv_folds if threshold_predictions is not None else 0))
            metrics[f"val_{level}_calibration_sample_count"] = float(calibration_count)
            metrics[f"val_{level}_evaluation_sample_count"] = float(evaluation_indices.size)
            metrics[f"val_{level}_threshold_cv_folds"] = float(threshold_cv_folds if threshold_predictions is not None else 0)
            metrics.update(_threshold_summary(f"val_{level}", level_threshold))
            for key, value in _core_multilabel_metrics(
                eval_true,
                eval_prob,
                level_threshold,
                device=device,
                y_pred_override=threshold_predictions,
                class_support=class_supports[level],
                low_max_support=int(metric_config.get("low_max_support", 50)),
                progress=progress,
                progress_label=f"{progress_label} {level} ML f1",
            ).items():
                metrics[f"val_{level}_{key}"] = value
            for key, value in _core_single_label_f1_metrics(
                eval_true,
                eval_prob,
                progress=progress,
                progress_label=f"{progress_label} {level} SL f1",
            ).items():
                metrics[f"val_{level}_{key}"] = value
    metrics["val_calibration_sample_count"] = float(max(calibration_counts) if calibration_counts else 0)
    metrics["val_evaluation_sample_count"] = float(max(evaluation_counts) if evaluation_counts else sample_count)
    metrics["val_threshold_cv_folds"] = float(max(threshold_cv_fold_values) if threshold_cv_fold_values else 0)
    metrics["val_core_ec4_selection_score"] = float(
        0.45 * metrics.get("val_ec4_f1_macro_observed", 0.0)
        + 0.35 * metrics.get("val_ec4_single_label_f1_macro", 0.0)
        + 0.20 * metrics.get("val_ec4_auprc_micro", 0.0)
    )
    metrics["val_core_selection_score"] = _core_selection_score(metrics, metric_config)
    return metrics


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: HierarchicalECLoss,
    device: torch.device,
    parent_child_maps: dict[str, list[tuple[int, int]]],
    distributed: DistributedContext,
    class_supports: dict[str, np.ndarray],
    metric_config: dict[str, object],
    threshold_overrides: dict[str, float | np.ndarray] | None = None,
) -> dict[str, float]:
    payload = _collect_validation_outputs(model, loader, criterion, device, distributed)
    return _compute_validation_metrics(
        payload,
        parent_child_maps,
        class_supports,
        metric_config,
        threshold_overrides=threshold_overrides,
        progress_label="Validation metrics",
    )


def _save_validation_payload(
    path: Path,
    validation_payload: dict[str, object],
    *,
    compressed: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"val_loss": np.asarray([validation_payload["val_loss"]], dtype=np.float64)}
    probs = validation_payload["probs"]
    targets = validation_payload["targets"]
    for level in LEVELS:
        arrays[f"prob_{level}"] = np.asarray(probs[level], dtype=np.float32)
        arrays[f"target_{level}"] = np.asarray(targets[level], dtype=np.float32)
    for task, histogram in validation_payload.get("site_histograms", {}).items():
        arrays[f"site_positive_{task}"] = np.asarray(histogram["positive"], dtype=np.int64)
        arrays[f"site_negative_{task}"] = np.asarray(histogram["negative"], dtype=np.int64)
        arrays[f"site_count_{task}"] = np.asarray([histogram["count"]], dtype=np.int64)
    for task, histogram in validation_payload.get("intermediate_site_histograms", {}).items():
        arrays[f"intermediate_site_positive_{task}"] = np.asarray(histogram["positive"], dtype=np.int64)
        arrays[f"intermediate_site_negative_{task}"] = np.asarray(histogram["negative"], dtype=np.int64)
        arrays[f"intermediate_site_count_{task}"] = np.asarray([histogram["count"]], dtype=np.int64)
    diagnostic_sums = validation_payload.get("hypergraph_diagnostic_sums", {})
    diagnostic_counts = validation_payload.get("hypergraph_diagnostic_counts", {})
    diagnostic_names = list(diagnostic_sums)
    arrays["hypergraph_diagnostic_names"] = np.asarray(diagnostic_names, dtype=np.str_)
    arrays["hypergraph_diagnostic_sums"] = np.asarray(
        [diagnostic_sums[name] for name in diagnostic_names],
        dtype=np.float64,
    )
    arrays["hypergraph_diagnostic_counts"] = np.asarray(
        [diagnostic_counts.get(name, 0) for name in diagnostic_names],
        dtype=np.int64,
    )
    tool_diagnostic_sums = validation_payload.get("tool_annotation_diagnostic_sums", {})
    tool_diagnostic_counts = validation_payload.get("tool_annotation_diagnostic_counts", {})
    tool_diagnostic_names = list(tool_diagnostic_sums)
    arrays["tool_annotation_diagnostic_names"] = np.asarray(tool_diagnostic_names, dtype=np.str_)
    arrays["tool_annotation_diagnostic_sums"] = np.asarray(
        [tool_diagnostic_sums[name] for name in tool_diagnostic_names],
        dtype=np.float64,
    )
    arrays["tool_annotation_diagnostic_counts"] = np.asarray(
        [tool_diagnostic_counts.get(name, 0) for name in tool_diagnostic_names],
        dtype=np.int64,
    )
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if compressed:
        np.savez_compressed(tmp_path, **arrays)
    else:
        np.savez(tmp_path, **arrays)
    saved_tmp = tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
    saved_tmp.replace(path)


def _load_validation_payload(path: str | Path) -> dict[str, object]:
    with np.load(path) as payload:
        site_histograms = {
            task: {
                "positive": payload[f"site_positive_{task}"].copy(),
                "negative": payload[f"site_negative_{task}"].copy(),
                "count": int(payload[f"site_count_{task}"][0]),
            }
            for task in SITE_TASKS
            if f"site_positive_{task}" in payload
        }
        intermediate_site_histograms = {
            task: {
                "positive": payload[f"intermediate_site_positive_{task}"].copy(),
                "negative": payload[f"intermediate_site_negative_{task}"].copy(),
                "count": int(payload[f"intermediate_site_count_{task}"][0]),
            }
            for task in SITE_TASKS
            if f"intermediate_site_positive_{task}" in payload
        }
        diagnostic_names = (
            payload["hypergraph_diagnostic_names"].tolist()
            if "hypergraph_diagnostic_names" in payload
            else []
        )
        diagnostic_sums = (
            payload["hypergraph_diagnostic_sums"].tolist()
            if "hypergraph_diagnostic_sums" in payload
            else []
        )
        diagnostic_counts = (
            payload["hypergraph_diagnostic_counts"].tolist()
            if "hypergraph_diagnostic_counts" in payload
            else []
        )
        tool_diagnostic_names = (
            payload["tool_annotation_diagnostic_names"].tolist()
            if "tool_annotation_diagnostic_names" in payload
            else []
        )
        tool_diagnostic_sums = (
            payload["tool_annotation_diagnostic_sums"].tolist()
            if "tool_annotation_diagnostic_sums" in payload
            else []
        )
        tool_diagnostic_counts = (
            payload["tool_annotation_diagnostic_counts"].tolist()
            if "tool_annotation_diagnostic_counts" in payload
            else []
        )
        return {
            "val_loss": float(payload["val_loss"][0]),
            "probs": {level: payload[f"prob_{level}"].copy() for level in LEVELS},
            "targets": {level: payload[f"target_{level}"].copy() for level in LEVELS},
            "site_histograms": site_histograms,
            "intermediate_site_histograms": intermediate_site_histograms,
            "hypergraph_diagnostic_sums": dict(zip(diagnostic_names, diagnostic_sums)),
            "hypergraph_diagnostic_counts": dict(zip(diagnostic_names, diagnostic_counts)),
            "tool_annotation_diagnostic_sums": dict(
                zip(tool_diagnostic_names, tool_diagnostic_sums)
            ),
            "tool_annotation_diagnostic_counts": dict(
                zip(tool_diagnostic_names, tool_diagnostic_counts)
            ),
        }


def _threshold_fbeta_values(
    metric_config: dict[str, object],
    class_support: np.ndarray | None,
    class_count: int,
) -> np.ndarray:
    default_beta = float(metric_config.get("threshold_fbeta", 1.0))
    values = np.full(int(class_count), default_beta, dtype=np.float64)
    if class_support is not None:
        support = np.asarray(class_support, dtype=np.float64).reshape(-1)
        if support.size != int(class_count):
            raise ValueError(f"class_support has {support.size} value(s), expected {class_count}")
        low_max_support = int(metric_config.get("low_max_support", 50))
        values[(support > 0) & (support <= low_max_support)] = float(
            metric_config.get("threshold_low_fbeta", default_beta)
        )
        values[support > low_max_support] = float(
            metric_config.get("threshold_high_fbeta", default_beta)
        )
    if np.any(values <= 0.0):
        raise ValueError("metrics threshold F-beta values must be positive")
    return values


def _threshold_for_level(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_config: dict[str, object],
    class_support: np.ndarray | None = None,
) -> float | np.ndarray:
    mode = str(metric_config.get("threshold_mode", "fixed")).lower()
    default_threshold = float(metric_config.get("threshold", 0.5))
    if int(y_true.shape[0]) == 0:
        if mode in {"auto", "per_class", "per_class_f1"}:
            return np.full(int(y_prob.shape[1]), default_threshold, dtype=np.float64)
        return default_threshold
    if mode in {"auto", "per_class", "per_class_f1"}:
        beta_values = _threshold_fbeta_values(metric_config, class_support, y_prob.shape[1])
        fallback_mode = str(metric_config.get("threshold_unseen_fallback", "one")).lower()
        unseen_fallback_threshold = None
        if fallback_mode in {"global", "global_f1"}:
            observed_classes = y_true.sum(axis=0) > 0
            global_mask = observed_classes if observed_classes.any() else np.ones(y_prob.shape[1], dtype=bool)
            global_fallback = optimize_global_threshold(
                y_true[:, global_mask],
                y_prob[:, global_mask],
                steps=int(metric_config.get("threshold_steps", metric_config.get("fmax_steps", 101))),
                default_threshold=default_threshold,
                min_threshold=float(metric_config.get("threshold_min", 0.0)),
                max_threshold=float(metric_config.get("threshold_max", 1.0)),
                beta=float(metric_config.get("threshold_fbeta", 1.0)),
            )
            unseen_fallback_threshold = np.full(y_prob.shape[1], global_fallback, dtype=np.float64)
            if class_support is not None and np.unique(beta_values).size > 1:
                support = np.asarray(class_support, dtype=np.float64).reshape(-1)
                low_max_support = int(metric_config.get("low_max_support", 50))
                bucket_masks = (
                    (support > 0) & (support <= low_max_support),
                    support > low_max_support,
                )
                for bucket_mask in bucket_masks:
                    calibration_mask = bucket_mask & observed_classes
                    if not calibration_mask.any():
                        continue
                    bucket_beta = float(beta_values[calibration_mask][0])
                    unseen_fallback_threshold[bucket_mask] = optimize_global_threshold(
                        y_true[:, calibration_mask],
                        y_prob[:, calibration_mask],
                        steps=int(metric_config.get("threshold_steps", metric_config.get("fmax_steps", 101))),
                        default_threshold=default_threshold,
                        min_threshold=float(metric_config.get("threshold_min", 0.0)),
                        max_threshold=float(metric_config.get("threshold_max", 1.0)),
                        beta=bucket_beta,
                    )
        elif fallback_mode in {"default", "fixed"}:
            unseen_fallback_threshold = default_threshold
        elif fallback_mode not in {"one", "1", "legacy"}:
            raise ValueError(f"Unsupported metrics.threshold_unseen_fallback: {fallback_mode}")
        return optimize_per_class_thresholds(
            y_true,
            y_prob,
            steps=int(metric_config.get("threshold_steps", metric_config.get("fmax_steps", 101))),
            default_threshold=default_threshold,
            min_threshold=float(metric_config.get("threshold_min", 0.0)),
            max_threshold=float(metric_config.get("threshold_max", 1.0)),
            unseen_fallback_threshold=unseen_fallback_threshold,
            beta=beta_values,
        )
    return default_threshold


def _validation_calibration_indices(
    sample_count: int,
    metric_config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(int(sample_count), dtype=np.int64)
    fraction = float(metric_config.get("threshold_calibration_fraction", 0.0))
    if fraction <= 0.0 or sample_count < 2:
        return indices, indices
    if fraction >= 1.0:
        raise ValueError("metrics.threshold_calibration_fraction must be less than 1.0")
    rng = np.random.default_rng(int(metric_config.get("threshold_calibration_seed", 42)))
    shuffled = rng.permutation(indices)
    calibration_count = min(max(int(round(sample_count * fraction)), 1), sample_count - 1)
    return np.sort(shuffled[:calibration_count]), np.sort(shuffled[calibration_count:])


def _known_level_indices(
    targets: np.ndarray,
    candidate_indices: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(targets)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D EC target matrix, got shape {values.shape}")
    indices = (
        np.arange(values.shape[0], dtype=np.int64)
        if candidate_indices is None
        else np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    )
    if indices.size == 0:
        return indices
    return indices[values[indices].sum(axis=1) > 0]


def _cross_fitted_threshold_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_config: dict[str, object],
    class_support: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    sample_count = int(y_true.shape[0])
    folds = min(max(int(metric_config.get("core_threshold_cv_folds", 0)), 2), sample_count)
    rng = np.random.default_rng(int(metric_config.get("threshold_calibration_seed", 42)))
    shuffled = rng.permutation(np.arange(sample_count, dtype=np.int64))
    predictions = np.zeros_like(y_true, dtype=np.int64)
    threshold_values: list[np.ndarray] = []
    calibration_sizes: list[int] = []
    all_indices = np.arange(sample_count, dtype=np.int64)
    for evaluation_indices in np.array_split(shuffled, folds):
        calibration_mask = np.ones(sample_count, dtype=bool)
        calibration_mask[evaluation_indices] = False
        calibration_indices = all_indices[calibration_mask]
        threshold = _threshold_for_level(
            y_true[calibration_indices],
            y_prob[calibration_indices],
            metric_config,
            class_support,
        )
        threshold_array = _threshold_values_for_metrics(threshold, y_prob.shape[1])
        predictions[evaluation_indices] = (y_prob[evaluation_indices] >= threshold_array).astype(np.int64)
        threshold_values.append(threshold_array.reshape(-1))
        calibration_sizes.append(int(calibration_indices.size))
    return predictions, np.mean(np.stack(threshold_values, axis=0), axis=0), int(round(np.mean(calibration_sizes)))


def _threshold_summary(prefix: str, threshold: float | np.ndarray) -> dict[str, float]:
    values = np.asarray(threshold, dtype=np.float64).reshape(-1)
    return {
        f"{prefix}_threshold_mean": float(values.mean()) if values.size else 0.0,
        f"{prefix}_threshold_min": float(values.min()) if values.size else 0.0,
        f"{prefix}_threshold_max": float(values.max()) if values.size else 0.0,
    }


def _compute_metrics_from_cache(
    payload_path: str,
    parent_child_maps: dict[str, list[tuple[int, int]]],
    class_supports: dict[str, np.ndarray],
    metric_config: dict[str, object],
    *,
    epoch: int,
    train_loss: float,
    cleanup_payload: bool,
) -> dict[str, float]:
    validation_payload = _load_validation_payload(payload_path)
    metrics = _compute_validation_metrics(
        validation_payload,
        parent_child_maps,
        class_supports,
        metric_config,
        progress_label=f"Epoch {epoch:03d} metrics",
    )
    metrics["epoch"] = float(epoch)
    metrics["train_loss"] = float(train_loss)
    metrics["metrics_status"] = "full"
    metrics["metrics_profile"] = "full"
    if cleanup_payload:
        Path(payload_path).unlink(missing_ok=True)
    return metrics


@dataclass
class PendingMetricJob:
    epoch: int
    future: Future
    checkpoint_path: Path


class AsyncMetricScheduler:
    def __init__(
        self,
        *,
        metrics_dir: Path,
        parent_child_maps: dict[str, list[tuple[int, int]]],
        class_supports: dict[str, np.ndarray],
        metric_config: dict[str, object],
    ):
        self.metrics_dir = metrics_dir
        self.parent_child_maps = parent_child_maps
        self.class_supports = class_supports
        self.metric_config = metric_config
        self.cache_dir = metrics_dir / "validation_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max(int(metric_config.get("async_workers", 2)), 1))
        self.pending: list[PendingMetricJob] = []
        self.max_pending = max(int(metric_config.get("max_pending", 8)), 1)
        self._last_backlog_notice = 0

    def submit(
        self,
        *,
        epoch: int,
        train_loss: float,
        validation_payload: dict[str, object],
        checkpoint_path: Path,
    ) -> list[tuple[dict[str, float], Path]]:
        completed = self.drain(block=False)
        if len(self.pending) >= self.max_pending and len(self.pending) != self._last_backlog_notice:
            self._last_backlog_notice = len(self.pending)
            logging.info(
                "Async metric backlog has %s pending job(s), exceeding metrics.max_pending=%s. "
                "Waiting for one metric job before queuing epoch %s.",
                len(self.pending),
                self.max_pending,
                epoch,
            )
        if len(self.pending) >= self.max_pending:
            completed.extend(self.drain(block=True, count=1))
        payload_path = self.cache_dir / f"epoch_{epoch:05d}.npz"
        _save_validation_payload(
            payload_path,
            validation_payload,
            compressed=bool(self.metric_config.get("cache_compressed", False)),
        )
        future = self.executor.submit(
            _compute_metrics_from_cache,
            str(payload_path),
            self.parent_child_maps,
            self.class_supports,
            self.metric_config,
            epoch=epoch,
            train_loss=train_loss,
            cleanup_payload=not bool(self.metric_config.get("keep_prediction_cache", False)),
        )
        self.pending.append(PendingMetricJob(epoch=epoch, future=future, checkpoint_path=checkpoint_path))
        return completed

    def drain(self, *, block: bool = False, count: int | None = None) -> list[tuple[dict[str, float], Path]]:
        completed: list[tuple[dict[str, float], Path]] = []
        if not block:
            remaining: list[PendingMetricJob] = []
            for index, job in enumerate(self.pending):
                if job.future.done():
                    metrics = job.future.result()
                    completed.append((metrics, job.checkpoint_path))
                    if count is not None and len(completed) >= count:
                        remaining.extend(self.pending[index + 1 :])
                        break
                else:
                    remaining.append(job)
            self.pending = remaining
            return completed
        while self.pending:
            job = self.pending.pop(0)
            metrics = job.future.result()
            completed.append((metrics, job.checkpoint_path))
            if count is not None and len(completed) >= count:
                break
        return completed

    def close(self) -> list[tuple[dict[str, float], Path]]:
        completed = self.drain(block=True)
        self.executor.shutdown(wait=True)
        return completed


def _is_better(value: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    return value < best if mode == "min" else value > best


def _monitor_uses_core_metrics(monitor_metric: str) -> bool:
    core_metric_names = {
        "val_core_selection_score",
        "val_core_ec4_selection_score",
        "val_ec4_auprc_micro",
        "val_ec4_f1_macro",
        "val_ec4_single_label_f1_macro",
    }
    if monitor_metric in core_metric_names:
        return True
    core_suffixes = (
        "_auprc_micro",
        "_f1_macro",
        "_f1_macro_observed",
        "_f1_micro",
        "_precision_macro",
        "_precision_macro_observed",
        "_precision_micro",
        "_recall_macro",
        "_recall_macro_observed",
        "_recall_micro",
        "_single_label_accuracy",
        "_single_label_f1_macro",
        "_single_label_f1_micro",
        "_single_label_f1_weighted",
        "_single_label_precision_macro",
        "_single_label_precision_micro",
        "_single_label_precision_weighted",
        "_single_label_recall_macro",
        "_single_label_recall_micro",
        "_single_label_recall_weighted",
    )
    return any(
        monitor_metric.startswith(f"val_{level}_") and monitor_metric.endswith(core_suffixes)
        for level in LEVELS
    )


def _metrics_can_update_monitor(metrics: dict[str, float], monitor_metric: str) -> bool:
    status = str(metrics.get("metrics_status", "full")).lower()
    if status in {"pending", "skipped"}:
        return False
    if _monitor_uses_core_metrics(monitor_metric) and status != "core":
        return False
    return monitor_metric in metrics


def _staged_lr_multiplier(train_config: dict, epoch: int) -> float:
    staged = train_config.get("staged_training", {}) or {}
    if not bool(staged.get("enabled", False)):
        return 1.0
    end_to_end_start_epoch = max(int(staged.get("end_to_end_start_epoch", 1)), 1)
    end_to_end_lr_scale = float(staged.get("end_to_end_lr_scale", 1.0))
    if not 0.0 < end_to_end_lr_scale <= 1.0:
        raise ValueError(
            "train.staged_training.end_to_end_lr_scale must satisfy 0 < value <= 1"
        )
    return end_to_end_lr_scale if int(epoch) >= end_to_end_start_epoch else 1.0


def _make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    train_config: dict,
    *,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    mode = str(train_config.get("lr_scheduler", "none")).lower()
    if mode in {"", "none", "off", "false"}:
        return None
    if mode not in {"warmup_cosine", "cosine"}:
        raise ValueError(f"Unsupported train.lr_scheduler: {mode}")
    warmup_epochs = int(train_config.get("warmup_epochs", 0)) if mode == "warmup_cosine" else 0
    min_lr_ratio = float(train_config.get("min_lr_ratio", 0.05))
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("train.warmup_epochs must satisfy 0 <= warmup_epochs < train.epochs")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("train.min_lr_ratio must satisfy 0 <= min_lr_ratio <= 1")

    def scale(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            base_scale = float(epoch_index + 1) / float(warmup_epochs)
        else:
            cosine_epochs = max(epochs - warmup_epochs - 1, 1)
            progress = min(max((epoch_index - warmup_epochs) / cosine_epochs, 0.0), 1.0)
            base_scale = min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )
        return base_scale * _staged_lr_multiplier(train_config, epoch_index + 1)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def _model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    wrapped = getattr(model, "module", model)
    return wrapped.state_dict()


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    epoch: int,
    metrics: dict[str, float],
    config: dict,
    label_maps: dict[str, dict[str, int]],
    parent_child_maps: dict[str, list[tuple[int, int]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": _model_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "metrics": metrics,
            "config": config,
            "label_maps": label_maps,
            "parent_child_maps": parent_child_maps,
        },
        path,
    )


def _copy_checkpoint_with_metrics(src: Path, dst: Path, metrics: dict[str, float]) -> None:
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    checkpoint["metrics"] = metrics
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, dst)


def _resolve_resume_checkpoint(resume_checkpoint: str | None, last_path: Path) -> Path | None:
    if resume_checkpoint is None or str(resume_checkpoint).strip() == "":
        return None
    value = str(resume_checkpoint).strip()
    if value.lower() in {"1", "true", "yes", "last", "auto"}:
        return last_path
    return Path(value).expanduser()


def _sync_scheduler_to_completed_epoch(
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    completed_epoch: int,
) -> None:
    if lr_scheduler is None:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(max(int(completed_epoch), 0)):
            lr_scheduler.step()


def _load_training_history(
    metrics_jsonl: Path,
    *,
    monitor_metric: str,
    monitor_mode: str,
    monitor_start_epoch: int = 1,
) -> tuple[list[dict[str, float]], int, float | None, int]:
    history: list[dict[str, float]] = []
    if not metrics_jsonl.exists():
        return history, 0, None, 0
    for line in metrics_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            history.append(json.loads(line))
        except json.JSONDecodeError:
            logging.warning("Skipping malformed metric history line in %s", metrics_jsonl)
    best_epoch = 0
    best_value: float | None = None
    epochs_without_improvement = 0
    for metrics in history:
        if int(metrics.get("epoch", 0)) < int(monitor_start_epoch):
            continue
        if not _metrics_can_update_monitor(metrics, monitor_metric):
            continue
        value = float(metrics[monitor_metric])
        if _is_better(value, best_value, monitor_mode):
            best_value = value
            best_epoch = int(metrics.get("epoch", 0))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
    return history, best_epoch, best_value, epochs_without_improvement


def _archive_fresh_start_metric_history(
    metrics_dir: Path,
    metrics_jsonl: Path,
    summary_path: Path,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = metrics_dir / "archived_metric_history"
    archive_dir.mkdir(parents=True, exist_ok=True)
    if metrics_jsonl.exists():
        archive_path = archive_dir / f"{metrics_jsonl.stem}_{timestamp}{metrics_jsonl.suffix}"
        metrics_jsonl.replace(archive_path)
        logging.info("Archived existing metric history for fresh training start: %s", archive_path)
    if summary_path.exists():
        archive_path = archive_dir / f"{summary_path.stem}_{timestamp}{summary_path.suffix}"
        summary_path.replace(archive_path)
        logging.info("Archived existing training summary for fresh training start: %s", archive_path)


def _metric_full_interval(metric_config: dict[str, object]) -> int:
    value = metric_config.get("full_interval", metric_config.get("interval", 1))
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 1


def _should_compute_full_metrics(
    epoch: int,
    total_epochs: int,
    *,
    interval: int,
    first_epoch: bool,
) -> bool:
    if interval <= 0:
        return False
    if interval <= 1:
        return True
    return epoch == total_epochs or (first_epoch and epoch == 1) or (epoch % interval == 0)


def _load_training_checkpoint(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    load_optimizer_state: bool = True,
) -> int:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    wrapped_model = getattr(model, "module", model)
    wrapped_model.load_state_dict(checkpoint["model_state_dict"])
    if load_optimizer_state:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    completed_epoch = int(checkpoint.get("epoch", 0))
    _sync_scheduler_to_completed_epoch(lr_scheduler, completed_epoch)
    return completed_epoch


def _load_model_checkpoint(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    device: torch.device,
    strict: bool = True,
    target_label_maps: dict[str, dict[str, int]] | None = None,
    remap_label_heads: bool = False,
) -> dict[str, object]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    wrapped_model = getattr(model, "module", model)
    checkpoint_state = checkpoint["model_state_dict"]
    if strict:
        wrapped_model.load_state_dict(checkpoint_state)
        return checkpoint
    model_state = wrapped_model.state_dict()
    remapped_rows: dict[str, int] = {}
    source_label_maps = checkpoint.get("label_maps") or {}
    if remap_label_heads and source_label_maps and target_label_maps:
        checkpoint_state = dict(checkpoint_state)
        output_keys = {
            level: (f"heads.{level}.3.weight", f"heads.{level}.3.bias")
            for level in LEVELS
        }
        for level, keys in output_keys.items():
            source_map = source_label_maps.get(level) or {}
            target_map = target_label_maps.get(level) or {}
            matching_labels = sorted(set(source_map) & set(target_map))
            for name in keys:
                source_value = checkpoint_state.get(name)
                target_value = model_state.get(name)
                if source_value is None or target_value is None:
                    continue
                if tuple(source_value.shape[1:]) != tuple(target_value.shape[1:]):
                    continue
                remapped_value = target_value.clone()
                copied = 0
                for label in matching_labels:
                    source_index = int(source_map[label])
                    target_index = int(target_map[label])
                    if source_index >= source_value.shape[0] or target_index >= target_value.shape[0]:
                        continue
                    remapped_value[target_index].copy_(source_value[source_index])
                    copied += 1
                if copied:
                    checkpoint_state[name] = remapped_value
                    remapped_rows[name] = copied
    compatible_state = {}
    skipped_keys: dict[str, str] = {}
    for name, value in checkpoint_state.items():
        if remap_label_heads and name.endswith("_parent_index"):
            skipped_keys[name] = "retained target parent-child map"
            continue
        if name not in model_state:
            skipped_keys[name] = "unexpected"
            continue
        if tuple(model_state[name].shape) != tuple(value.shape):
            skipped_keys[name] = f"shape {tuple(value.shape)} -> {tuple(model_state[name].shape)}"
            continue
        compatible_state[name] = value
    load_result = wrapped_model.load_state_dict(compatible_state, strict=False)
    checkpoint["loaded_model_state_key_count"] = len(compatible_state)
    checkpoint["skipped_model_state_keys"] = skipped_keys
    checkpoint["missing_model_state_keys"] = list(load_result.missing_keys)
    checkpoint["unexpected_model_state_keys"] = list(load_result.unexpected_keys)
    checkpoint["remapped_model_state_rows"] = remapped_rows
    return checkpoint


def run_train(
    config: dict,
    dry_run: bool = False,
    *,
    limit: int | None = None,
    epochs_override: int | None = None,
    device_override: str | None = None,
    repair_labels: bool = False,
    distributed: bool | None = None,
    resume_checkpoint: str | None = None,
    init_checkpoint: str | None = None,
) -> dict[str, object]:
    ensure_dirs(config)
    train_config = config.get("train", {}) or {}
    configured_init_checkpoint = train_config.get("init_checkpoint")
    if resume_checkpoint is None and init_checkpoint is None and configured_init_checkpoint:
        init_checkpoint = str(configured_init_checkpoint)
    if resume_checkpoint and init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    _configure_cpu_threads(train_config)
    distributed_context = _init_distributed(train_config, distributed)
    swanlab_tracker: SwanLabTracker | None = None
    tracking_status = "failed"
    try:
        artifact_context = training_artifact_context(config) if distributed_context.is_main else None
        if distributed_context.enabled:
            payload = [artifact_context]
            dist.broadcast_object_list(payload, src=0)
            artifact_context = payload[0]
        synchronize_training_artifact_config(config, artifact_context)
        if distributed_context.is_main:
            ensure_dirs(config)
        _barrier(distributed_context)

        _seed_everything(int(train_config.get("seed", 42)) + distributed_context.rank)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        _log_main(distributed_context, "Stage | name=setup | status=started | action=prepare splits and labels")
        if repair_labels and distributed_context.is_main:
            repair_stats = repair_graph_labels(config, dataset_names=["train", "external_price149"])
            logging.info("Graph label repair before training: %s", repair_stats)
        _barrier(distributed_context)

        if distributed_context.enabled:
            if distributed_context.is_main:
                load_or_create_splits(config, limit=limit)
                _load_label_maps(config)
            _barrier(distributed_context)
        split_paths = load_or_create_splits(config, limit=limit)
        label_maps, parent_child_maps = _load_label_maps(config)
        metric_config = _metric_config(config)
        _configure_warning_filters(metric_config)
        if not bool(train_config.get("dynamic_label_encoding", False)):
            _validate_label_shapes(split_paths["train"] + split_paths["val"], label_maps)
        if distributed_context.is_main:
            artifact_inputs = write_training_run_inputs(
                config,
                split_paths=split_paths,
                label_maps=label_maps,
            )
            if artifact_inputs:
                logging.info("Tagged training inputs saved: %s", artifact_inputs)
        _barrier(distributed_context)
        _log_main(
            distributed_context,
            "Stage | name=setup | status=labels_ready | train=%s val=%s test=%s",
            len(split_paths["train"]),
            len(split_paths["val"]),
            len(split_paths["test"]),
        )
        if dry_run:
            _log_main(distributed_context, "Stage | name=dry_run | status=completed | action=no model training started")
            tracking_status = "dry_run"
            return {
                "status": "dry_run",
                "distributed": distributed_context.enabled,
                "rank": distributed_context.rank,
                "world_size": distributed_context.world_size,
                "splits": {key: len(value) for key, value in split_paths.items()},
                "num_classes": {level: len(label_maps[level]) for level in LEVELS},
            }

        if distributed_context.is_main:
            swanlab_tracker = SwanLabTracker.start(config)

        _log_main(distributed_context, "Stage | name=setup | status=started | action=calculate class supports")
        class_supports = _label_supports_from_csv(config, label_maps, split_paths["train"])
        device = _resolve_device(train_config, device_override, distributed_context)
        _log_main(distributed_context, "Stage | name=setup | status=started | action=build model | device=%s", device)
        model = _make_model(config, label_maps).to(device)
        if init_checkpoint:
            init_path = Path(str(init_checkpoint)).expanduser()
            init_strict = bool(train_config.get("init_checkpoint_strict", True))
            init_remap_label_heads = bool(
                train_config.get("init_checkpoint_remap_label_heads", False)
            )
            _log_main(
                distributed_context,
                "Stage | name=initialization | status=started | action=load model weights | checkpoint=%s | strict=%s",
                init_path,
                init_strict,
            )
            init_payload = _load_model_checkpoint(
                init_path,
                model=model,
                device=device,
                strict=init_strict,
                target_label_maps=label_maps,
                remap_label_heads=init_remap_label_heads,
            )
            _log_main(
                distributed_context,
                "Stage | name=initialization | status=completed | source_epoch=%s | checkpoint=%s | loaded_keys=%s | skipped_keys=%s",
                int(init_payload.get("epoch", 0)),
                init_path,
                init_payload.get("loaded_model_state_key_count", "all"),
                len(init_payload.get("skipped_model_state_keys", {})),
            )
            remapped_rows = init_payload.get("remapped_model_state_rows", {})
            if remapped_rows:
                _log_main(
                    distributed_context,
                    "Label-aware initialization | remapped=%s",
                    ", ".join(f"{name}:{count}" for name, count in sorted(remapped_rows.items())),
                )
        trainable_summary = _configure_trainable_parameters(model, train_config.get("trainable_parameter_patterns"))
        _log_main(
            distributed_context,
            "Stage | name=parameter_selection | status=completed | trainable=%s frozen=%s total=%s patterns=%s",
            trainable_summary["trainable_count"],
            trainable_summary["frozen_count"],
            trainable_summary["total_count"],
            ",".join(trainable_summary["patterns"]) if trainable_summary["patterns"] else "<all>",
        )
        if distributed_context.enabled:
            ddp_kwargs = {
                "find_unused_parameters": bool(train_config.get("find_unused_parameters", True)),
                "broadcast_buffers": bool(train_config.get("broadcast_buffers", True)),
            }
            if device.type == "cuda":
                ddp_kwargs["device_ids"] = [distributed_context.local_rank]
                ddp_kwargs["output_device"] = distributed_context.local_rank
            model = DistributedDataParallel(model, **ddp_kwargs)
        criterion = _make_loss(
            config,
            class_supports,
            train_sample_count=len(split_paths["train"]),
        )
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(train_config.get("learning_rate", 1e-3)),
            weight_decay=float(train_config.get("weight_decay", 5e-4)),
        )
        _log_main(
            distributed_context,
            "Stage | name=setup | status=started | action=build train and validation loaders | loss=%s",
            str((config.get("loss", {}) or {}).get("primary_loss", "focal_bce")),
        )
        train_loader, train_sampler = _make_loader(
            split_paths["train"],
            config,
            shuffle=True,
            distributed=distributed_context,
            distributed_train=True,
            class_supports=class_supports,
            label_maps=label_maps,
        )
        val_loader, _ = _make_loader(
            split_paths["val"],
            config,
            shuffle=False,
            distributed=distributed_context,
            distributed_train=False,
            class_supports=class_supports,
            label_maps=label_maps,
        )
        _log_main(
            distributed_context,
            "Stage | name=setup | status=completed | train_batches=%s val_batches=%s",
            len(train_loader),
            len(val_loader),
        )
        epochs = int(epochs_override or train_config.get("epochs", 100))
        lr_scheduler = _make_lr_scheduler(optimizer, train_config, epochs=epochs)
        amp = bool(train_config.get("amp", torch.cuda.is_available())) and device.type == "cuda"
        scaler = _make_grad_scaler(amp)
        parameter_norm_guard = _make_parameter_norm_guard(model, train_config)
        grad_clip_norm = train_config.get("gradient_clip_norm", 1.0)
        grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        monitor_metric = str(train_config.get("monitor_metric", "val_mean_f1_micro"))
        monitor_mode = str(train_config.get("monitor_mode", "max")).lower()
        if monitor_mode not in {"min", "max"}:
            raise ValueError("train.monitor_mode must be 'min' or 'max'")
        monitor_start_epoch = max(int(train_config.get("monitor_start_epoch", 1)), 1)
        if monitor_start_epoch > epochs:
            raise ValueError("train.monitor_start_epoch must not exceed train.epochs")
        log_interval = int(train_config.get("log_interval", 0))
        early_stopping_patience = train_config.get("early_stopping_patience")
        early_stopping_patience = None if early_stopping_patience in {None, 0, "0"} else int(early_stopping_patience)

        checkpoint_dir = Path(config["storage"]["checkpoint_dir"])
        metrics_dir = Path(config["storage"]["metrics_dir"])
        if distributed_context.is_main:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir.mkdir(parents=True, exist_ok=True)
        _barrier(distributed_context)
        metrics_jsonl = metrics_dir / artifact_filename(config, "training_metrics.jsonl")
        summary_path = metrics_dir / artifact_filename(config, "training_summary.json")
        best_path = checkpoint_dir / artifact_filename(config, "best.pt")
        last_path = checkpoint_dir / artifact_filename(config, "last.pt")
        async_metrics = bool(metric_config.get("async", False))
        metric_full_interval = _metric_full_interval(metric_config)
        metric_full_first_epoch = bool(metric_config.get("full_first_epoch", True))
        metric_profile = str(metric_config.get("profile", "full")).strip().lower()
        core_metrics_enabled = metric_profile in {"fast", "core", "core_full", "hybrid"}
        pending_checkpoint_dir = checkpoint_dir / artifact_filename(config, "pending_metrics")
        metric_scheduler = (
            AsyncMetricScheduler(
                metrics_dir=metrics_dir,
                parent_child_maps=parent_child_maps,
                class_supports=class_supports,
                metric_config=metric_config,
            )
            if async_metrics and metric_full_interval > 0 and distributed_context.is_main
            else None
        )
        if metric_scheduler is not None:
            pending_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_value: float | None = None
        best_epoch = 0
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []
        start_epoch = 1
        resume_path = _resolve_resume_checkpoint(resume_checkpoint, last_path)
        if distributed_context.is_main:
            if resume_path is not None:
                history, best_epoch, best_value, epochs_without_improvement = _load_training_history(
                    metrics_jsonl,
                    monitor_metric=monitor_metric,
                    monitor_mode=monitor_mode,
                    monitor_start_epoch=monitor_start_epoch,
                )
            elif bool(train_config.get("archive_metric_history_on_fresh_start", True)):
                _archive_fresh_start_metric_history(metrics_dir, metrics_jsonl, summary_path)
        if resume_path is not None:
            reset_optimizer_on_resume = bool(train_config.get("reset_optimizer_on_resume", False))
            completed_epoch = _load_training_checkpoint(
                resume_path,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                device=device,
                load_optimizer_state=not reset_optimizer_on_resume,
            )
            clamped_on_resume = parameter_norm_guard.apply(model) if parameter_norm_guard is not None else []
            start_epoch = completed_epoch + 1
            _log_main(
                distributed_context,
                "Resumed training\n"
                "  checkpoint | %s\n"
                "  epochs     | completed=%s next=%s target=%s\n"
                "  optimizer  | %s\n"
                "  norm_guard | clamped=%s",
                resume_path,
                completed_epoch,
                start_epoch,
                epochs,
                "reset from config" if reset_optimizer_on_resume else "restored from checkpoint",
                len(clamped_on_resume),
            )
        _barrier(distributed_context)

        def record_completed_metrics(completed: list[tuple[dict[str, float], Path]]) -> None:
            nonlocal best_value, best_epoch, epochs_without_improvement
            if not distributed_context.is_main:
                return
            for completed_metrics, checkpoint_path in completed:
                history.append(completed_metrics)
                with metrics_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(completed_metrics, sort_keys=True) + "\n")
                epoch_value = int(completed_metrics.get("epoch", 0))
                _log_epoch_metrics(completed_metrics, monitor_metric=monitor_metric, source="async")
                if swanlab_tracker is not None:
                    swanlab_tracker.log_epoch(completed_metrics, epoch=epoch_value)
                if (
                    epoch_value >= monitor_start_epoch
                    and _metrics_can_update_monitor(completed_metrics, monitor_metric)
                ):
                    value = float(completed_metrics[monitor_metric])
                    if _is_better(value, best_value, monitor_mode):
                        best_value = value
                        best_epoch = epoch_value
                        epochs_without_improvement = 0
                        _copy_checkpoint_with_metrics(checkpoint_path, best_path, completed_metrics)
                    else:
                        epochs_without_improvement += 1
                if not bool(metric_config.get("keep_epoch_checkpoints", False)):
                    checkpoint_path.unlink(missing_ok=True)

        def recover_orphaned_pending_metrics() -> None:
            if metric_scheduler is None or not bool(metric_config.get("recover_pending_on_start", False)):
                return
            cache_dir = metrics_dir / "validation_cache"
            if not cache_dir.exists():
                return
            recorded_epochs = {int(metrics.get("epoch", -1)) for metrics in history}
            recovered: list[tuple[dict[str, float], Path]] = []
            for cache_path in sorted(cache_dir.glob("epoch_*.npz")):
                try:
                    epoch_value = int(cache_path.stem.split("_")[-1])
                except ValueError:
                    continue
                if epoch_value in recorded_epochs:
                    continue
                checkpoint_path = pending_checkpoint_dir / f"epoch_{epoch_value:05d}.pt"
                if not checkpoint_path.exists():
                    continue
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                train_loss = float((checkpoint.get("metrics") or {}).get("train_loss", 0.0))
                recovered.append(
                    (
                        _compute_metrics_from_cache(
                            str(cache_path),
                            parent_child_maps,
                            class_supports,
                            metric_config,
                            epoch=epoch_value,
                            train_loss=train_loss,
                            cleanup_payload=not bool(metric_config.get("keep_prediction_cache", False)),
                        ),
                        checkpoint_path,
                    )
                )
                recorded_epochs.add(epoch_value)
            if recovered:
                logging.info("Recovered %s pending async metric job(s) from previous interrupted run.", len(recovered))
                record_completed_metrics(recovered)

        recover_orphaned_pending_metrics()
        _barrier(distributed_context)

        _log_main(
            distributed_context,
            "Starting training\n"
            "  splits    | train=%s val=%s test=%s\n"
            "  runtime   | device=%s epochs=%s distributed=%s world_size=%s\n"
            "  optimize  | loss=%s trainable=%s/%s init_checkpoint=%s\n"
            "  metrics   | profile=%s async=%s monitor=%s monitor_start=%s full_interval=%s progress=%s",
            len(split_paths["train"]),
            len(split_paths["val"]),
            len(split_paths["test"]),
            device,
            epochs,
            distributed_context.enabled,
            distributed_context.world_size,
            str((config.get("loss", {}) or {}).get("primary_loss", "focal_bce")),
            trainable_summary["trainable_count"],
            trainable_summary["total_count"],
            init_checkpoint or "<none>",
            metric_profile,
            async_metrics,
            monitor_metric,
            monitor_start_epoch,
            metric_full_interval,
            _progress_mode(distributed_context, log_interval),
        )
        early_stopped = False
        hypergraph_feedback_scale = 0.0
        tool_annotation_scale = 0.0
        training_stage = 2
        hypergraph_site_probabilities_detached = False
        for epoch in range(start_epoch, epochs + 1):
            unwrapped_model = getattr(model, "module", model)
            if hasattr(unwrapped_model, "set_training_epoch"):
                hypergraph_feedback_scale = float(unwrapped_model.set_training_epoch(epoch))
                tool_annotation_scale = float(
                    getattr(unwrapped_model, "tool_annotation_scale", 0.0)
                )
                training_stage = int(getattr(unwrapped_model, "training_stage", 2))
                training_stage_name = str(
                    getattr(unwrapped_model, "training_stage_name", "end_to_end")
                )
                hypergraph_site_probabilities_detached = bool(
                    getattr(unwrapped_model, "detach_hypergraph_site_probabilities", False)
                )
                _log_main(
                    distributed_context,
                    "Feature curricula | epoch=%03d stage=%s:%s hypergraph_scale=%.3f tool_scale=%.3f hypergraph_sites_detached=%s lr_multiplier=%.3f",
                    epoch,
                    training_stage,
                    training_stage_name,
                    hypergraph_feedback_scale,
                    tool_annotation_scale,
                    hypergraph_site_probabilities_detached,
                    _staged_lr_multiplier(train_config, epoch),
                )
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if metric_scheduler is not None:
                record_completed_metrics(metric_scheduler.drain(block=False))
            _log_main(
                distributed_context,
                "Stage | name=train | status=started | epoch=%03d/%03d | batches=%s",
                epoch,
                epochs,
                len(train_loader),
            )
            train_loss = _train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                amp=amp,
                scaler=scaler,
                parameter_norm_guard=parameter_norm_guard,
                grad_clip_norm=grad_clip_norm,
                log_interval=log_interval,
                epoch=epoch,
                total_epochs=epochs,
                distributed=distributed_context,
            )
            _log_main(
                distributed_context,
                "Stage | name=train | status=completed | epoch=%03d/%03d | train_loss=%.6f",
                epoch,
                epochs,
                train_loss,
            )
            _log_main(
                distributed_context,
                "Stage | name=validation | status=started | epoch=%03d/%03d | batches=%s",
                epoch,
                epochs,
                len(val_loader),
            )
            validation_payload = _collect_validation_outputs(
                model,
                val_loader,
                criterion,
                device,
                distributed_context,
                progress_label=f"Epoch {epoch:03d}/{epochs:03d} val",
                log_interval=log_interval,
            )
            _log_main(
                distributed_context,
                "Stage | name=validation | status=completed | epoch=%03d/%03d | train_loss=%.6f | val_loss=%.6f",
                epoch,
                epochs,
                train_loss,
                float(validation_payload["val_loss"]),
            )
            epoch_tracking_metrics: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(validation_payload["val_loss"]),
            }
            should_stop = False
            compute_full_metrics = _should_compute_full_metrics(
                epoch,
                epochs,
                interval=metric_full_interval,
                first_epoch=metric_full_first_epoch,
            )
            core_metrics: dict[str, float] | None = None
            if core_metrics_enabled and distributed_context.is_main:
                core_level_label = ",".join(_configured_core_levels(metric_config))
                logging.info(
                    "Stage | name=core_metrics | status=started | epoch=%03d/%03d | action=threshold calibration and scoring | levels=%s",
                    epoch,
                    epochs,
                    core_level_label,
                )
                core_metrics = _compute_core_validation_metrics(
                    validation_payload,
                    parent_child_maps,
                    class_supports,
                    metric_config,
                    device=device,
                    progress_label=f"Epoch {epoch:03d}/{epochs:03d} core metrics",
                )
                logging.info(
                    "Stage | name=core_metrics | status=completed | epoch=%03d/%03d",
                    epoch,
                    epochs,
                )
                core_metrics["epoch"] = float(epoch)
                core_metrics["train_loss"] = float(train_loss)
                history.append(core_metrics)
                with metrics_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(core_metrics, sort_keys=True) + "\n")
                _log_epoch_metrics(core_metrics, monitor_metric=monitor_metric, source="core")
                epoch_tracking_metrics.update(core_metrics)
                _save_checkpoint(
                    last_path,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    epoch=epoch,
                    metrics=core_metrics,
                    config=config,
                    label_maps=label_maps,
                    parent_child_maps=parent_child_maps,
                )
                if (
                    epoch >= monitor_start_epoch
                    and _metrics_can_update_monitor(core_metrics, monitor_metric)
                ):
                    value = float(core_metrics[monitor_metric])
                    if _is_better(value, best_value, monitor_mode):
                        best_value = value
                        best_epoch = epoch
                        epochs_without_improvement = 0
                        _save_checkpoint(
                            best_path,
                            model=model,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            epoch=epoch,
                            metrics=core_metrics,
                            config=config,
                            label_maps=label_maps,
                            parent_child_maps=parent_child_maps,
                        )
                    else:
                        epochs_without_improvement += 1
            if not compute_full_metrics and core_metrics_enabled:
                if metric_full_interval > 0:
                    _log_main(
                        distributed_context,
                        "Epoch %03d/%03d | full metrics deferred | next scheduled by metrics.full_interval=%s",
                        epoch,
                        epochs,
                        metric_full_interval,
                    )
            elif not compute_full_metrics:
                checkpoint_metrics = {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(validation_payload["val_loss"]),
                    "metrics_status": "skipped",
                    "metric_full_interval": float(metric_full_interval),
                }
                if distributed_context.is_main:
                    _save_checkpoint(
                        last_path,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        epoch=epoch,
                        metrics=checkpoint_metrics,
                        config=config,
                        label_maps=label_maps,
                        parent_child_maps=parent_child_maps,
                    )
                    logging.info(
                        "Epoch %03d/%03d | full metrics skipped | next scheduled by metrics.full_interval=%s",
                        epoch,
                        epochs,
                        metric_full_interval,
                    )
            elif async_metrics:
                checkpoint_metrics = {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(validation_payload["val_loss"]),
                    "metrics_status": "pending",
                }
                if distributed_context.is_main:
                    if not core_metrics_enabled:
                        _save_checkpoint(
                            last_path,
                            model=model,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            epoch=epoch,
                            metrics=checkpoint_metrics,
                            config=config,
                            label_maps=label_maps,
                            parent_child_maps=parent_child_maps,
                        )
                    epoch_checkpoint = pending_checkpoint_dir / f"epoch_{epoch:05d}.pt"
                    _save_checkpoint(
                        epoch_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        epoch=epoch,
                        metrics=checkpoint_metrics,
                        config=config,
                        label_maps=label_maps,
                        parent_child_maps=parent_child_maps,
                    )
                    completed = metric_scheduler.submit(
                        epoch=epoch,
                        train_loss=train_loss,
                        validation_payload=validation_payload,
                        checkpoint_path=epoch_checkpoint,
                    )
                    record_completed_metrics(completed)
                    record_completed_metrics(metric_scheduler.drain(block=False))
                    logging.info("Epoch %03d/%03d | metrics queued for async summary", epoch, epochs)
            else:
                _log_main(
                    distributed_context,
                    "Stage | name=full_metrics | status=started | epoch=%03d/%03d | action=complete validation metric suite",
                    epoch,
                    epochs,
                )
                metrics = _compute_validation_metrics(
                    validation_payload,
                    parent_child_maps,
                    class_supports,
                    metric_config,
                    progress_label=f"Epoch {epoch:03d}/{epochs:03d} metrics",
                )
                _log_main(
                    distributed_context,
                    "Stage | name=full_metrics | status=completed | epoch=%03d/%03d",
                    epoch,
                    epochs,
                )
                metrics["epoch"] = float(epoch)
                metrics["train_loss"] = float(train_loss)
                metrics["metrics_status"] = "full"
                metrics["metrics_profile"] = "full"
                history.append(metrics)
                epoch_tracking_metrics.update(metrics)
                if distributed_context.is_main:
                    with metrics_jsonl.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                    _log_epoch_metrics(metrics, monitor_metric=monitor_metric, source="sync")

                    _save_checkpoint(
                        last_path,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        epoch=epoch,
                        metrics=metrics,
                        config=config,
                        label_maps=label_maps,
                        parent_child_maps=parent_child_maps,
                    )
                if (
                    epoch >= monitor_start_epoch
                    and _metrics_can_update_monitor(metrics, monitor_metric)
                ):
                    value = float(metrics[monitor_metric])
                    if _is_better(value, best_value, monitor_mode):
                        best_value = value
                        best_epoch = epoch
                        epochs_without_improvement = 0
                        if distributed_context.is_main:
                            _save_checkpoint(
                                best_path,
                                model=model,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                epoch=epoch,
                                metrics=metrics,
                                config=config,
                                label_maps=label_maps,
                                parent_child_maps=parent_child_maps,
                            )
                    else:
                        epochs_without_improvement += 1
            if swanlab_tracker is not None:
                tracking_extras = {
                    "metrics/frequency_bucket_schema_version": 3.0,
                    "metrics/low_max_support": float(metric_config.get("low_max_support", 50)),
                    "optimizer/learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "optimizer/stage_lr_multiplier": _staged_lr_multiplier(train_config, epoch),
                    "curriculum/stage": float(training_stage),
                    "curriculum/hypergraph_scale": hypergraph_feedback_scale,
                    "curriculum/tool_annotation_scale": tool_annotation_scale,
                    "curriculum/hypergraph_site_probabilities_detached": float(
                        hypergraph_site_probabilities_detached
                    ),
                    "progress/total_epochs": float(epochs),
                    "progress/epoch_percent": 100.0 * float(epoch) / max(float(epochs), 1.0),
                    "progress/monitor_active": float(epoch >= monitor_start_epoch),
                    "progress/epochs_without_improvement": float(epochs_without_improvement),
                    "progress/best_epoch": float(best_epoch),
                }
                if best_value is not None:
                    tracking_extras["progress/best_value"] = float(best_value)
                swanlab_tracker.log_epoch(
                    epoch_tracking_metrics,
                    epoch=epoch,
                    extra_metrics=tracking_extras,
                )
            if (
                epoch >= monitor_start_epoch
                and early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                should_stop = True
                _log_main(distributed_context, "Early stopping at epoch %s", epoch)
            if distributed_context.enabled:
                stop_tensor = torch.tensor([1 if should_stop else 0], dtype=torch.int64, device=device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())
            if lr_scheduler is not None:
                lr_scheduler.step()
            if should_stop:
                early_stopped = True
                break

        if metric_scheduler is not None:
            record_completed_metrics(metric_scheduler.close())

        best_full_metrics_computed = False
        if bool(metric_config.get("full_after_training_best", True)):
            _barrier(distributed_context)
            if not best_path.exists():
                _log_main(distributed_context, "Best checkpoint does not exist; cannot compute final full metrics: %s", best_path)
            else:
                _log_main(
                    distributed_context,
                    "Stage | name=best_checkpoint_full_evaluation | status=started | training_status=%s | checkpoint=%s",
                    "early_stopped" if early_stopped else "completed",
                    best_path,
                )
                best_checkpoint = _load_model_checkpoint(best_path, model=model, device=device)
                best_checkpoint_metrics = best_checkpoint.get("metrics") or {}
                best_full_epoch = int(best_checkpoint.get("epoch", best_epoch))
                best_full_train_loss = float(best_checkpoint_metrics.get("train_loss", 0.0))
                best_validation_payload = _collect_validation_outputs(
                    model,
                    val_loader,
                    criterion,
                    device,
                    distributed_context,
                    progress_label=f"Epoch {best_full_epoch:03d}/{epochs:03d} best full val",
                    log_interval=log_interval,
                )
                _log_main(
                    distributed_context,
                    "Stage | name=best_checkpoint_full_evaluation | status=validation_complete | epoch=%03d/%03d | val_loss=%.6f",
                    best_full_epoch,
                    epochs,
                    float(best_validation_payload["val_loss"]),
                )
                best_full_metrics = _compute_validation_metrics(
                    best_validation_payload,
                    parent_child_maps,
                    class_supports,
                    metric_config,
                    progress_label=f"Epoch {best_full_epoch:03d}/{epochs:03d} best full metrics",
                )
                best_full_metrics["epoch"] = float(best_full_epoch)
                best_full_metrics["train_loss"] = best_full_train_loss
                best_full_metrics["metrics_status"] = "full"
                best_full_metrics["metrics_profile"] = (
                    "full_after_early_stop_best" if early_stopped else "full_after_training_best"
                )
                if best_value is not None:
                    best_full_metrics["selection_best_value"] = float(best_value)
                history.append(best_full_metrics)
                best_full_metrics_computed = True
                if distributed_context.is_main:
                    with metrics_jsonl.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(best_full_metrics, sort_keys=True) + "\n")
                    _log_epoch_metrics(best_full_metrics, monitor_metric=monitor_metric, source="best-full")
                    if swanlab_tracker is not None:
                        swanlab_tracker.log_best(best_full_metrics)
                    _copy_checkpoint_with_metrics(best_path, best_path, best_full_metrics)
                    logging.info(
                        "Stage | name=best_checkpoint_full_evaluation | status=completed | epoch=%03d/%03d",
                        best_full_epoch,
                        epochs,
                    )
            _barrier(distributed_context)

        summary = {
            "status": "early_stopped" if early_stopped else "completed",
            "distributed": distributed_context.enabled,
            "world_size": distributed_context.world_size,
            "async_metrics": async_metrics,
            "metric_profile": metric_profile,
            "metric_full_interval": metric_full_interval,
            "best_full_metrics_computed": best_full_metrics_computed,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "best_epoch": best_epoch,
            "best_metric": monitor_metric,
            "best_value": best_value,
            "splits": {key: len(value) for key, value in split_paths.items()},
        }
        if distributed_context.is_main:
            write_json(summary_path, summary)
        tracking_status = str(summary["status"])
        if swanlab_tracker is not None:
            swanlab_tracker.log_summary(summary)
        return summary
    except KeyboardInterrupt:
        tracking_status = "interrupted"
        raise
    finally:
        try:
            _cleanup_distributed(distributed_context)
        finally:
            if swanlab_tracker is not None:
                swanlab_tracker.finish(status=tracking_status)
