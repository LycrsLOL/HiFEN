from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

from ..data.graph_builder import SITE_TASKS


class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = float(gamma)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        targets = targets.float()
        if target_mask is not None:
            target_mask = target_mask.to(device=logits.device, dtype=logits.dtype).view(logits.shape)
        if sample_mask is not None:
            mask = sample_mask.to(device=logits.device, dtype=torch.bool)
            if mask.dim() != 1:
                mask = mask.view(mask.size(0), -1).any(dim=1)
            if not bool(mask.any()):
                return torch.zeros((), device=logits.device, dtype=logits.dtype)
            logits = logits[mask]
            targets = targets[mask]
            if target_mask is not None:
                target_mask = target_mask[mask]
        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        loss = ((1.0 - pt) ** self.gamma) * bce
        if target_mask is None:
            return loss.mean()
        denominator = target_mask.sum()
        if not bool(denominator > 0):
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        return (loss * target_mask).sum() / denominator


class DistributionBalancedLoss(nn.Module):
    """Distribution-balanced loss for multi-label long-tailed classification."""

    def __init__(
        self,
        class_support: torch.Tensor,
        train_sample_count: int,
        *,
        focal_gamma: float = 2.0,
        rebalance_alpha: float = 0.1,
        rebalance_beta: float = 10.0,
        rebalance_gamma: float = 0.2,
        negative_scale: float = 2.0,
        init_bias_scale: float = 0.05,
    ):
        super().__init__()
        support = class_support.float().clamp_min(1.0)
        sample_count = max(int(train_sample_count), 1)
        self.focal_gamma = float(focal_gamma)
        self.rebalance_alpha = float(rebalance_alpha)
        self.rebalance_beta = float(rebalance_beta)
        self.rebalance_gamma = float(rebalance_gamma)
        self.negative_scale = float(negative_scale)
        if self.negative_scale <= 0:
            raise ValueError("DistributionBalancedLoss.negative_scale must be positive")
        inverse_frequency = support.reciprocal()
        prior = (support / float(sample_count)).clamp(1e-6, 1.0 - 1e-6)
        self.register_buffer("inverse_frequency", inverse_frequency)
        self.register_buffer("init_bias", torch.logit(prior) * float(init_bias_scale))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        targets = targets.float()
        if target_mask is not None:
            target_mask = target_mask.to(device=logits.device, dtype=logits.dtype).view(logits.shape)
        if sample_mask is not None:
            mask = sample_mask.to(device=logits.device, dtype=torch.bool)
            if mask.dim() != 1:
                mask = mask.view(mask.size(0), -1).any(dim=1)
            if not bool(mask.any()):
                return torch.zeros((), device=logits.device, dtype=logits.dtype)
            logits = logits[mask]
            targets = targets[mask]
            if target_mask is not None:
                target_mask = target_mask[mask]

        inverse_frequency = self.inverse_frequency.to(device=logits.device, dtype=logits.dtype)
        repeat_rate = (targets * inverse_frequency.view(1, -1)).sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(logits.dtype).tiny
        )
        positive_weight = inverse_frequency.view(1, -1) / repeat_rate
        weight = torch.sigmoid(self.rebalance_beta * (positive_weight - self.rebalance_alpha)) + self.rebalance_gamma

        shifted_logits = logits + self.init_bias.to(device=logits.device, dtype=logits.dtype).view(1, -1)
        scaled_logits = shifted_logits * targets + shifted_logits * (1.0 - targets) * self.negative_scale
        weight = weight * targets + weight * (1.0 - targets) / self.negative_scale
        bce = F.binary_cross_entropy_with_logits(scaled_logits, targets, reduction="none")
        probabilities = torch.sigmoid(scaled_logits)
        pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        loss = weight * ((1.0 - pt) ** self.focal_gamma) * bce
        if target_mask is None:
            return loss.mean()
        denominator = target_mask.sum()
        if not bool(denominator > 0):
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        return (loss * target_mask).sum() / denominator


def site_auxiliary_loss(
    outputs: dict[str, torch.Tensor],
    batch,
    *,
    gamma: float = 2.0,
    task_weights: dict[str, float] | None = None,
    positive_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    device = outputs["ec1"].device
    targets = getattr(batch, "y_site", None)
    target_mask = getattr(batch, "y_site_mask", None)
    if targets is None or target_mask is None:
        return torch.zeros((), device=device)
    targets = targets.to(device=device, dtype=outputs["ec1"].dtype)
    target_mask = target_mask.to(device=device, dtype=outputs["ec1"].dtype)
    if targets.dim() != 2 or target_mask.shape != targets.shape or targets.size(1) != len(SITE_TASKS):
        raise ValueError(f"Expected y_site and y_site_mask with shape [num_nodes, {len(SITE_TASKS)}]")

    losses = []
    weights = task_weights or {}
    pos_weights = positive_weights or {}
    for index, task in enumerate(SITE_TASKS):
        logits = outputs.get(task)
        valid = target_mask[:, index] > 0
        if logits is None or not bool(valid.any()):
            continue
        selected_logits = logits.view(-1)[valid]
        selected_targets = targets[:, index][valid]
        pos_weight = torch.as_tensor(float(pos_weights.get(task, 1.0)), dtype=selected_logits.dtype, device=selected_logits.device)
        bce = F.binary_cross_entropy_with_logits(selected_logits, selected_targets, reduction="none", pos_weight=pos_weight)
        probabilities = torch.sigmoid(selected_logits)
        pt = probabilities * selected_targets + (1.0 - probabilities) * (1.0 - selected_targets)
        losses.append(float(weights.get(task, 1.0)) * (((1.0 - pt) ** float(gamma)) * bce).mean())
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).sum()


class HierarchicalECLoss(nn.Module):
    """EC classification loss with functional-hyperedge site supervision."""
    def __init__(
        self,
        ec_level_weights: dict[str, float] | None = None,
        focal_gamma: float = 2.0,
        pos_weights: dict[str, torch.Tensor] | None = None,
        auxiliary_site_weight: float = 0.0,
        intermediate_site_weight: float = 0.0,
        site_focal_gamma: float = 2.0,
        site_task_weights: dict[str, float] | None = None,
        site_positive_weights: dict[str, float] | None = None,
        ignore_empty_label_levels: Iterable[str] | None = None,
        primary_loss: str = "focal_bce",
        class_supports: dict[str, torch.Tensor] | None = None,
        train_sample_count: int = 0,
        db_rebalance_alpha: float = 0.1,
        db_rebalance_beta: float = 10.0,
        db_rebalance_gamma: float = 0.2,
        db_negative_scale: float = 2.0,
        db_init_bias_scale: float = 0.05,
    ):
        super().__init__()
        self.level_weights = ec_level_weights or {"ec1": 1.0, "ec2": 1.0, "ec3": 1.0, "ec4": 1.0}
        self.primary_loss = str(primary_loss).strip().lower()
        if self.primary_loss not in {"focal_bce", "focal", "distribution_balanced", "db"}:
            raise ValueError(f"Unsupported loss.primary_loss: {primary_loss!r}")
        self.focal = FocalBCELoss(gamma=focal_gamma)
        self.level_focal = nn.ModuleDict()
        for level, pos_weight in (pos_weights or {}).items():
            self.level_focal[level] = FocalBCELoss(gamma=focal_gamma, pos_weight=pos_weight)
        self.level_db = nn.ModuleDict()
        if self.primary_loss in {"distribution_balanced", "db"}:
            if not class_supports or train_sample_count <= 0:
                raise ValueError("Distribution-balanced loss requires class_supports and train_sample_count")
            for level, support in class_supports.items():
                self.level_db[level] = DistributionBalancedLoss(
                    torch.as_tensor(support, dtype=torch.float32),
                    train_sample_count,
                    focal_gamma=focal_gamma,
                    rebalance_alpha=db_rebalance_alpha,
                    rebalance_beta=db_rebalance_beta,
                    rebalance_gamma=db_rebalance_gamma,
                    negative_scale=db_negative_scale,
                    init_bias_scale=db_init_bias_scale,
                )
        self.auxiliary_site_weight = float(auxiliary_site_weight)
        self.intermediate_site_weight = float(intermediate_site_weight)
        self.site_focal_gamma = float(site_focal_gamma)
        self.site_task_weights = site_task_weights or {}
        self.site_positive_weights = site_positive_weights or {}
        self.ignore_empty_label_levels = {str(level) for level in (ignore_empty_label_levels or [])}

    def forward(self, outputs: dict[str, torch.Tensor], batch) -> torch.Tensor:
        total = torch.zeros((), device=outputs["ec1"].device)
        for level in ("ec1", "ec2", "ec3", "ec4"):
            target = getattr(batch, f"y_{level}", None)
            if target is None:
                continue
            if target.dim() == 1:
                target = target.view(outputs[level].shape)
            target_mask = getattr(batch, f"y_{level}_mask", None)
            if target_mask is not None:
                target_mask = target_mask.view(outputs[level].shape)
            sample_mask = target.sum(dim=1) > 0 if level in self.ignore_empty_label_levels else None
            if self.primary_loss in {"distribution_balanced", "db"}:
                primary = self.level_db[level]
            else:
                primary = self.level_focal[level] if level in self.level_focal else self.focal
            total = total + float(self.level_weights.get(level, 1.0)) * primary(
                outputs[level],
                target,
                sample_mask=sample_mask,
                target_mask=target_mask,
            )
        if self.auxiliary_site_weight > 0:
            total = total + self.auxiliary_site_weight * site_auxiliary_loss(
                outputs,
                batch,
                gamma=self.site_focal_gamma,
                task_weights=self.site_task_weights,
                positive_weights=self.site_positive_weights,
            )
        if self.intermediate_site_weight > 0 and outputs.get("intermediate_sites"):
            intermediate_outputs = {**outputs, **outputs["intermediate_sites"]}
            total = total + self.intermediate_site_weight * site_auxiliary_loss(
                intermediate_outputs,
                batch,
                gamma=self.site_focal_gamma,
                task_weights=self.site_task_weights,
                positive_weights=self.site_positive_weights,
            )
        return total
