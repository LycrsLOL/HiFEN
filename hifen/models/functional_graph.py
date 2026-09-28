from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sparsemax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparse probability projection with the same shape contract as softmax."""
    shifted = scores - scores.max(dim=dim, keepdim=True).values
    sorted_scores = shifted.sort(dim=dim, descending=True).values
    cumulative = sorted_scores.cumsum(dim)
    size = scores.size(dim)
    ranks_shape = [1] * scores.dim()
    ranks_shape[dim] = size
    ranks = torch.arange(1, size + 1, device=scores.device, dtype=scores.dtype).view(ranks_shape)
    support = 1.0 + ranks * sorted_scores > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold = (cumulative.gather(dim, support_size - 1) - 1.0) / support_size.to(scores.dtype)
    return torch.clamp(shifted - threshold, min=0.0)


class SoftFunctionalHypergraph(nn.Module):
    """Propagate residue features through latent, task-specific soft hyperedges."""

    def __init__(
        self,
        hidden_dim: int,
        num_tasks: int,
        slots_per_task: int = 4,
        dropout: float = 0.0,
        gate_init: float = 0.1,
        assignment_mode: str = "softmax",
        assignment_temperature: float = 1.0,
        site_probability_threshold: float = 0.0,
        min_edge_mass: float = 0.0,
        preserve_site_confidence: bool = False,
        context_mode: str = "normalized",
        confidence_power: float = 1.0,
        contrastive_context: bool = False,
        min_feedback_gate: float = 0.0,
        max_feedback_gate: float = 1.0,
    ):
        super().__init__()
        if num_tasks < 1 or slots_per_task < 1:
            raise ValueError("num_tasks and slots_per_task must be positive")
        assignment_mode = str(assignment_mode).strip().lower()
        if assignment_mode not in {"softmax", "sparsemax"}:
            raise ValueError("assignment_mode must be 'softmax' or 'sparsemax'")
        if float(assignment_temperature) <= 0.0:
            raise ValueError("assignment_temperature must be positive")
        if not 0.0 <= float(site_probability_threshold) < 1.0:
            raise ValueError("site_probability_threshold must be in [0, 1)")
        if float(min_edge_mass) < 0.0:
            raise ValueError("min_edge_mass must be non-negative")
        context_mode = str(context_mode).strip().lower()
        if context_mode not in {"normalized", "task_attention"}:
            raise ValueError("context_mode must be 'normalized' or 'task_attention'")
        if float(confidence_power) <= 0.0:
            raise ValueError("confidence_power must be positive")
        if not 0.0 <= float(min_feedback_gate) < 1.0:
            raise ValueError("min_feedback_gate must be in [0, 1)")
        if not 0.0 < float(max_feedback_gate) <= 1.0:
            raise ValueError("max_feedback_gate must be in (0, 1]")
        if float(max_feedback_gate) < float(min_feedback_gate):
            raise ValueError("max_feedback_gate must be >= min_feedback_gate")
        self.num_tasks = int(num_tasks)
        self.slots_per_task = int(slots_per_task)
        self.assignment_mode = assignment_mode
        self.assignment_temperature = float(assignment_temperature)
        self.site_probability_threshold = float(site_probability_threshold)
        self.min_edge_mass = float(min_edge_mass)
        self.preserve_site_confidence = bool(preserve_site_confidence)
        self.context_mode = context_mode
        self.confidence_power = float(confidence_power)
        self.contrastive_context = bool(contrastive_context)
        self.min_feedback_gate = float(min_feedback_gate)
        self.max_feedback_gate = float(max_feedback_gate)
        self.feedback_scale = 1.0
        self.slot_assignment = (
            nn.Linear(hidden_dim, self.num_tasks * self.slots_per_task)
            if self.slots_per_task > 1
            else None
        )
        self.hyperedge_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
        )
        self.task_context_gate = (
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1, max(hidden_dim // 2, 1)),
                nn.PReLU(),
                nn.Linear(max(hidden_dim // 2, 1), 1),
            )
            if self.context_mode == "task_attention"
            else None
        )
        gate_input_dim = hidden_dim * 2 + (3 if self.preserve_site_confidence else 0)
        self.feedback_gate = nn.Linear(gate_input_dim, 1)
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.feedback_gate.weight)
        nn.init.constant_(self.feedback_gate.bias, math.log(gate_init / (1.0 - gate_init)))
        self.output_dropout = nn.Dropout(dropout)

    def set_feedback_scale(self, value: float) -> None:
        """Set a non-persistent warm-up multiplier without changing checkpoints."""
        self.feedback_scale = min(max(float(value), 0.0), 1.0)

    def forward(
        self,
        x: torch.Tensor,
        site_probabilities: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if site_probabilities.shape != (x.size(0), self.num_tasks):
            raise ValueError(
                f"site_probabilities must have shape {(x.size(0), self.num_tasks)}, "
                f"got {tuple(site_probabilities.shape)}"
            )
        if batch.dim() != 1 or batch.numel() != x.size(0):
            raise ValueError("batch must contain one graph index per residue")

        # Hyperedge reductions can sum thousands of residues and overflow FP16.
        # Keep the full soft-assignment and aggregation path in FP32 under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_float = x.float()
            site_probabilities_float = site_probabilities.float().clamp(0.0, 1.0)
            if self.site_probability_threshold > 0.0:
                site_confidence = (
                    (site_probabilities_float - self.site_probability_threshold)
                    / (1.0 - self.site_probability_threshold)
                ).clamp(0.0, 1.0)
            else:
                site_confidence = site_probabilities_float
            if self.slot_assignment is None:
                slot_weights = x_float.new_ones((x.size(0), self.num_tasks, 1))
            else:
                slot_scores = self.slot_assignment(x_float).view(
                    -1,
                    self.num_tasks,
                    self.slots_per_task,
                )
                scaled_slot_scores = slot_scores / self.assignment_temperature
                if self.assignment_mode == "sparsemax":
                    slot_weights = _sparsemax(scaled_slot_scores, dim=-1)
                else:
                    slot_weights = torch.softmax(scaled_slot_scores, dim=-1)
            memberships = site_confidence.unsqueeze(-1) * slot_weights

            num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
            edges_per_graph = self.num_tasks * self.slots_per_task
            local_edge_ids = torch.arange(edges_per_graph, device=x.device).view(1, -1)
            edge_ids = batch.view(-1, 1) * edges_per_graph + local_edge_ids
            flat_memberships = memberships.view(x.size(0), edges_per_graph)

            edge_sums = x_float.new_zeros((num_graphs * edges_per_graph, x.size(-1)))
            edge_mass = x_float.new_zeros((num_graphs * edges_per_graph, 1))
            repeated_x = x_float.unsqueeze(1).expand(-1, edges_per_graph, -1)
            edge_sums.index_add_(
                0,
                edge_ids.reshape(-1),
                (repeated_x * flat_memberships.unsqueeze(-1)).reshape(-1, x.size(-1)),
            )
            edge_mass.index_add_(0, edge_ids.reshape(-1), flat_memberships.reshape(-1, 1))
            edge_inputs = edge_sums / edge_mass.clamp_min(1e-6)
            if self.contrastive_context:
                graph_sums = x_float.new_zeros((num_graphs, x.size(-1)))
                graph_counts = x_float.new_zeros((num_graphs, 1))
                graph_sums.index_add_(0, batch, x_float)
                graph_counts.index_add_(
                    0,
                    batch,
                    torch.ones((x.size(0), 1), device=x.device, dtype=x_float.dtype),
                )
                graph_means = graph_sums / graph_counts.clamp_min(1.0)
                edge_background = graph_means.unsqueeze(1).expand(
                    -1,
                    edges_per_graph,
                    -1,
                ).reshape_as(edge_inputs)
                edge_inputs = edge_inputs - edge_background
            hyperedges = self.hyperedge_encoder(edge_inputs)
            if self.min_edge_mass > 0.0:
                valid_edges = edge_mass >= self.min_edge_mass
                hyperedges = hyperedges * valid_edges.to(dtype=hyperedges.dtype)
            else:
                valid_edges = edge_mass > 0.0
            hyperedges = hyperedges.view(num_graphs, self.num_tasks, self.slots_per_task, x.size(-1))

            node_hyperedges = hyperedges[batch]
            membership_mass = memberships.sum(dim=(1, 2), keepdim=False).unsqueeze(-1)
            node_valid_edges = valid_edges.view(
                num_graphs,
                self.num_tasks,
                self.slots_per_task,
                1,
            )[batch].squeeze(-1)
            if self.context_mode == "task_attention":
                task_context = (node_hyperedges * slot_weights.unsqueeze(-1)).sum(dim=2)
                task_edge_support = (
                    slot_weights * node_valid_edges.to(dtype=slot_weights.dtype)
                ).sum(dim=-1)
                task_available = (site_confidence > 0.0) & (task_edge_support > 0.0)
                expanded_x = x_float.unsqueeze(1).expand(-1, self.num_tasks, -1)
                task_logits = self.task_context_gate(
                    torch.cat([expanded_x, task_context, site_confidence.unsqueeze(-1)], dim=-1)
                ).squeeze(-1)
                task_logits = task_logits + site_confidence.clamp_min(1e-6).log()
                task_logits = task_logits.masked_fill(
                    ~task_available,
                    torch.finfo(task_logits.dtype).min,
                )
                has_available_task = task_available.any(dim=1, keepdim=True)
                task_attention = torch.softmax(task_logits, dim=1)
                task_attention = torch.where(
                    has_available_task,
                    task_attention,
                    torch.zeros_like(task_attention),
                )
                context = (task_context * task_attention.unsqueeze(-1)).sum(dim=1)
            else:
                context = (node_hyperedges * memberships.unsqueeze(-1)).sum(dim=(1, 2))
                context = context / membership_mass.clamp_min(1e-6)
                task_attention = None
            context = self.context_encoder(context)

            residue_site_confidence = site_confidence.max(dim=1, keepdim=True).values
            if self.slots_per_task > 1:
                slot_entropy = -(slot_weights.clamp_min(1e-12).log() * slot_weights).sum(dim=-1)
                slot_entropy = slot_entropy / math.log(self.slots_per_task)
                task_mass = site_confidence.sum(dim=1, keepdim=True).clamp_min(1e-6)
                assignment_entropy = (slot_entropy * site_confidence).sum(dim=1, keepdim=True) / task_mass
            else:
                assignment_entropy = torch.zeros_like(residue_site_confidence)
            edge_support = (
                (memberships * node_valid_edges.to(dtype=memberships.dtype)).sum(dim=(1, 2), keepdim=False).unsqueeze(-1)
                / membership_mass.clamp_min(1e-6)
            )
            if task_attention is not None and self.num_tasks > 1:
                task_attention_entropy = -(
                    task_attention.clamp_min(1e-12).log() * task_attention
                ).sum(dim=1, keepdim=True) / math.log(self.num_tasks)
            else:
                task_attention_entropy = torch.zeros_like(residue_site_confidence)

            gate_inputs = [x_float, context]
            if self.preserve_site_confidence:
                gate_inputs.extend([residue_site_confidence, 1.0 - assignment_entropy, edge_support])
            raw_gate = torch.sigmoid(self.feedback_gate(torch.cat(gate_inputs, dim=-1)))
            gate = self.min_feedback_gate + (
                self.max_feedback_gate - self.min_feedback_gate
            ) * raw_gate
            if self.preserve_site_confidence:
                if self.confidence_power < 1.0:
                    # A fractional power has an infinite derivative at zero. The
                    # probability threshold above deliberately creates exact zeros,
                    # so the naive expression can yield finite outputs but NaN
                    # gradients once a site logit crosses the threshold. A small
                    # positive shift keeps both endpoints (0 -> 0, 1 -> 1) while
                    # retaining the intended amplification of weak confidence.
                    confidence_epsilon = 1.0e-3
                    confidence_offset = confidence_epsilon**self.confidence_power
                    confidence_normalizer = (
                        (1.0 + confidence_epsilon) ** self.confidence_power
                        - confidence_offset
                    )
                    confidence_scale = (
                        (residue_site_confidence + confidence_epsilon).pow(
                            self.confidence_power
                        )
                        - confidence_offset
                    ) / confidence_normalizer
                else:
                    confidence_scale = residue_site_confidence.pow(self.confidence_power)
            else:
                confidence_scale = 1.0
            feedback = (
                float(self.feedback_scale)
                * gate
                * confidence_scale
                * self.output_dropout(context)
            )
            supported = (residue_site_confidence > 0.0) & (edge_support > 0.0)
            nan_values = torch.full_like(gate, float("nan"))
            context_norm = context.norm(dim=-1, keepdim=True) / math.sqrt(context.size(-1))
            feedback_norm = feedback.norm(dim=-1, keepdim=True) / math.sqrt(feedback.size(-1))
            diagnostics = {
                "site_confidence": residue_site_confidence,
                "assignment_entropy": assignment_entropy,
                "edge_support": edge_support,
                "task_attention_entropy": task_attention_entropy,
                "gate": gate,
                "gate_on_supported": torch.where(supported, gate, nan_values),
                "context_norm": context_norm,
                "context_norm_on_supported": torch.where(supported, context_norm, nan_values),
                "feedback_norm": feedback_norm,
                "feedback_norm_on_supported": torch.where(supported, feedback_norm, nan_values),
                "supported_fraction": supported.to(dtype=gate.dtype),
                "feedback_scale": torch.full_like(residue_site_confidence, float(self.feedback_scale)),
            }
        return x + feedback.to(dtype=x.dtype), memberships, gate, diagnostics
