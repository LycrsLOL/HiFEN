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


def hierarchy_consistency_loss(outputs: dict[str, torch.Tensor], parent_child_maps: dict[str, list[tuple[int, int]]]) -> torch.Tensor:
    probs = {key: torch.sigmoid(outputs[key]) for key in ("ec1", "ec2", "ec3", "ec4") if key in outputs}
    losses = []
    for parent_level, child_level, key in [
        ("ec1", "ec2", "ec1_to_ec2"),
        ("ec2", "ec3", "ec2_to_ec3"),
        ("ec3", "ec4", "ec3_to_ec4"),
    ]:
        if parent_level not in probs or child_level not in probs:
            continue
        pairs = parent_child_maps.get(key, [])
        if not pairs:
            continue
        idx = torch.tensor(pairs, dtype=torch.long, device=probs[parent_level].device)
        parent_probs = probs[parent_level][:, idx[:, 0]]
        child_probs = probs[child_level][:, idx[:, 1]]
        losses.append(F.relu(child_probs - parent_probs).pow(2).mean())
    if not losses:
        device = next(iter(outputs.values())).device
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def modality_balance_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    weights = outputs.get("modality_attention")
    if weights is None or weights.numel() == 0 or weights.size(-1) <= 1:
        return torch.zeros((), device=outputs["ec1"].device)
    availability = outputs.get("modality_availability")
    if availability is None:
        availability = torch.ones_like(weights, dtype=torch.bool)
    else:
        availability = availability.to(device=weights.device, dtype=torch.bool)
        if availability.shape != weights.shape:
            raise ValueError("modality_availability must match modality_attention")
    available_count = availability.sum(dim=-1, keepdim=True).clamp_min(1).to(weights.dtype)
    safe_weights = weights.clamp_min(torch.finfo(weights.dtype).eps)
    terms = safe_weights * torch.log(safe_weights * available_count)
    return terms.masked_fill(~availability, 0.0).sum(dim=-1).mean()


def structure_auxiliary_loss(
    outputs: dict[str, torch.Tensor],
    batch,
    contact_cutoff: float = 8.0,
    distance_weight: float = 1.0,
    contact_weight: float = 1.0,
) -> torch.Tensor:
    device = outputs["ec1"].device
    edge_attr = getattr(batch, "edge_attr", None)
    if edge_attr is None or edge_attr.numel() == 0:
        return torch.zeros((), device=device)

    distances = edge_attr[:, 0].to(device=device, dtype=outputs["ec1"].dtype)
    losses = []
    if "edge_distance" in outputs:
        predicted_distance = outputs["edge_distance"].view(-1)
        if predicted_distance.numel() == distances.numel():
            losses.append(float(distance_weight) * F.smooth_l1_loss(predicted_distance, distances))
    if "edge_contact" in outputs:
        predicted_contact = outputs["edge_contact"].view(-1)
        if predicted_contact.numel() == distances.numel():
            contact_target = (distances <= float(contact_cutoff)).to(dtype=predicted_contact.dtype)
            losses.append(float(contact_weight) * F.binary_cross_entropy_with_logits(predicted_contact, contact_target))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).sum()


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


def conditional_hierarchy_loss(
    outputs: dict[str, torch.Tensor],
    batch,
    parent_child_maps: dict[str, list[tuple[int, int]]],
    *,
    gamma: float = 2.0,
) -> torch.Tensor:
    conditional_logits = outputs.get("conditional_logits")
    if not conditional_logits:
        return torch.zeros((), device=outputs["ec1"].device)
    losses = []
    for parent_level, child_level in (("ec1", "ec2"), ("ec2", "ec3"), ("ec3", "ec4")):
        logits = conditional_logits.get(child_level)
        parent_target = getattr(batch, f"y_{parent_level}", None)
        child_target = getattr(batch, f"y_{child_level}", None)
        parent_target_mask = getattr(batch, f"y_{parent_level}_mask", None)
        child_target_mask = getattr(batch, f"y_{child_level}_mask", None)
        if logits is None or parent_target is None or child_target is None:
            continue
        parent_target = parent_target.view(logits.size(0), -1).to(device=logits.device, dtype=logits.dtype)
        child_target = child_target.view(logits.shape).to(device=logits.device, dtype=logits.dtype)
        if parent_target_mask is not None:
            parent_target_mask = parent_target_mask.view(parent_target.shape).to(device=logits.device, dtype=torch.bool)
        if child_target_mask is not None:
            child_target_mask = child_target_mask.view(child_target.shape).to(device=logits.device, dtype=torch.bool)
        parent_index = torch.full((logits.size(1),), -1, dtype=torch.long, device=logits.device)
        for parent, child in parent_child_maps.get(f"{parent_level}_to_{child_level}", []):
            if 0 <= int(child) < parent_index.numel():
                parent_index[int(child)] = int(parent)
        valid_children = parent_index >= 0
        if not bool(valid_children.any()):
            continue
        selected_logits = logits[:, valid_children]
        selected_targets = child_target[:, valid_children]
        selected_parents = parent_index[valid_children]
        child_known = child_target.sum(dim=1) > 0
        valid_entries = (parent_target[:, selected_parents] > 0) & child_known.unsqueeze(1)
        if parent_target_mask is not None:
            valid_entries &= parent_target_mask[:, selected_parents]
        if child_target_mask is not None:
            valid_entries &= child_target_mask[:, valid_children]
        if not bool(valid_entries.any()):
            continue
        bce = F.binary_cross_entropy_with_logits(selected_logits, selected_targets, reduction="none")
        probabilities = torch.sigmoid(selected_logits)
        pt = probabilities * selected_targets + (1.0 - probabilities) * (1.0 - selected_targets)
        losses.append((((1.0 - pt) ** float(gamma)) * bce)[valid_entries].mean())
    if not losses:
        return torch.zeros((), device=outputs["ec1"].device)
    return torch.stack(losses).mean()


def single_label_auxiliary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weight: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if targets.dim() == 1:
        targets = targets.view(logits.shape)
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    if target_mask is not None:
        target_mask = target_mask.view(logits.shape).to(device=logits.device, dtype=torch.bool)
    eligible = targets.sum(dim=1) == 1
    if not bool(eligible.any()):
        return torch.zeros((), device=logits.device)
    true_index = targets[eligible].argmax(dim=1)
    weight = None
    if class_weight is not None:
        weight = class_weight.to(device=logits.device, dtype=logits.dtype)
        if float(weight[true_index].sum().detach().cpu()) <= 0.0:
            weight = None
    selected_logits = logits[eligible]
    if target_mask is not None:
        selected_mask = target_mask[eligible]
        selected_mask.scatter_(1, true_index.unsqueeze(1), True)
        selected_logits = selected_logits.masked_fill(~selected_mask, torch.finfo(selected_logits.dtype).min)
    return F.cross_entropy(selected_logits, true_index, weight=weight)


class HierarchicalECLoss(nn.Module):
    def __init__(
        self,
        ec_level_weights: dict[str, float] | None = None,
        focal_gamma: float = 2.0,
        pos_weights: dict[str, torch.Tensor] | None = None,
        hierarchy_weight: float = 0.0,
        parent_child_maps: dict[str, list[tuple[int, int]]] | None = None,
        structure_aux_weight: float = 0.0,
        structure_distance_weight: float = 1.0,
        structure_contact_weight: float = 1.0,
        structure_contact_cutoff: float = 8.0,
        auxiliary_site_weight: float = 0.0,
        intermediate_site_weight: float = 0.0,
        hypergraph_auxiliary_weight: float = 0.0,
        site_focal_gamma: float = 2.0,
        site_task_weights: dict[str, float] | None = None,
        site_positive_weights: dict[str, float] | None = None,
        modality_balance_weight: float = 0.0,
        ignore_empty_label_levels: Iterable[str] | None = None,
        single_label_auxiliary_weight: float = 0.0,
        single_label_auxiliary_levels: Iterable[str] | None = None,
        single_label_auxiliary_class_weight: bool = True,
        conditional_hierarchy_weight: float = 0.0,
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
            if single_label_auxiliary_class_weight:
                self.register_buffer(f"{level}_single_label_weight", pos_weight.float())
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
        self.hierarchy_weight = float(hierarchy_weight)
        self.parent_child_maps = parent_child_maps or {}
        self.structure_aux_weight = float(structure_aux_weight)
        self.structure_distance_weight = float(structure_distance_weight)
        self.structure_contact_weight = float(structure_contact_weight)
        self.structure_contact_cutoff = float(structure_contact_cutoff)
        self.auxiliary_site_weight = float(auxiliary_site_weight)
        self.intermediate_site_weight = float(intermediate_site_weight)
        self.hypergraph_auxiliary_weight = float(hypergraph_auxiliary_weight)
        self.site_focal_gamma = float(site_focal_gamma)
        self.site_task_weights = site_task_weights or {}
        self.site_positive_weights = site_positive_weights or {}
        self.modality_balance_weight = float(modality_balance_weight)
        self.ignore_empty_label_levels = {str(level) for level in (ignore_empty_label_levels or [])}
        self.single_label_auxiliary_weight = float(single_label_auxiliary_weight)
        self.single_label_auxiliary_levels = {str(level) for level in (single_label_auxiliary_levels or [])}
        self.conditional_hierarchy_weight = float(conditional_hierarchy_weight)

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
            if self.single_label_auxiliary_weight > 0 and level in self.single_label_auxiliary_levels:
                class_weight = getattr(self, f"{level}_single_label_weight", None)
                total = total + self.single_label_auxiliary_weight * single_label_auxiliary_loss(
                    outputs[level],
                    target,
                    class_weight=class_weight,
                    target_mask=target_mask,
                )
        hypergraph_aux_logits = outputs.get("hypergraph_aux_ec4")
        hypergraph_aux_target = getattr(batch, "y_ec4", None)
        if (
            self.hypergraph_auxiliary_weight > 0
            and hypergraph_aux_logits is not None
            and hypergraph_aux_target is not None
        ):
            if hypergraph_aux_target.dim() == 1:
                hypergraph_aux_target = hypergraph_aux_target.view(
                    hypergraph_aux_logits.shape
                )
            hypergraph_target_mask = getattr(batch, "y_ec4_mask", None)
            if hypergraph_target_mask is not None:
                hypergraph_target_mask = hypergraph_target_mask.view(hypergraph_aux_logits.shape)
            hypergraph_sample_mask = (
                hypergraph_aux_target.sum(dim=1) > 0
                if "ec4" in self.ignore_empty_label_levels
                else None
            )
            if self.primary_loss in {"distribution_balanced", "db"}:
                hypergraph_primary = self.level_db["ec4"]
            else:
                hypergraph_primary = (
                    self.level_focal["ec4"]
                    if "ec4" in self.level_focal
                    else self.focal
                )
            total = total + self.hypergraph_auxiliary_weight * hypergraph_primary(
                hypergraph_aux_logits,
                hypergraph_aux_target,
                sample_mask=hypergraph_sample_mask,
                target_mask=hypergraph_target_mask,
            )
        if self.hierarchy_weight > 0:
            total = total + self.hierarchy_weight * hierarchy_consistency_loss(outputs, self.parent_child_maps)
        if self.conditional_hierarchy_weight > 0:
            total = total + self.conditional_hierarchy_weight * conditional_hierarchy_loss(
                outputs,
                batch,
                self.parent_child_maps,
                gamma=self.focal.gamma,
            )
        if self.modality_balance_weight > 0:
            total = total + self.modality_balance_weight * modality_balance_loss(outputs)
        if self.structure_aux_weight > 0:
            total = total + self.structure_aux_weight * structure_auxiliary_loss(
                outputs,
                batch,
                contact_cutoff=self.structure_contact_cutoff,
                distance_weight=self.structure_distance_weight,
                contact_weight=self.structure_contact_weight,
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

