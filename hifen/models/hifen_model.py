from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.graph_builder import SITE_TASKS
from .backbone import ResidualGATBackbone
from .edge_encoder import RBFEdgeEncoder
from .functional_graph import SoftFunctionalHypergraph
from .pooling import LevelAttentionPool


EC_LEVELS = ("ec1", "ec2", "ec3", "ec4")
PARENT_LEVEL = {"ec2": "ec1", "ec3": "ec2", "ec4": "ec3"}


def _feature_curriculum_scale(epoch: int, start_epoch: int, warmup_epochs: int) -> float:
    """Return a zero-to-one feature scale without changing legacy warmup semantics."""
    if int(epoch) < int(start_epoch):
        return 0.0
    if int(warmup_epochs) <= 1:
        return 1.0
    progress = float(int(epoch) - int(start_epoch)) / float(int(warmup_epochs) - 1)
    return min(max(progress, 0.0), 1.0)


class LayerwiseRepresentationFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, 1), 1),
        )

    def forward(self, layer_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if not layer_outputs:
            raise ValueError("layer_outputs must not be empty")
        if len(layer_outputs) == 1:
            weights = torch.ones(
                (layer_outputs[0].size(0), 1),
                dtype=layer_outputs[0].dtype,
                device=layer_outputs[0].device,
            )
            return layer_outputs[0], weights
        stacked = torch.stack(layer_outputs, dim=1)
        scores = self.gate(stacked).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return (stacked * weights.unsqueeze(-1)).sum(dim=1), weights


class ResidueModalityFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_modalities: int,
        dropout: float = 0.0,
        modality_dropout: float = 0.0,
        sequence_modality_dropout: float = 0.0,
        modality_dropout_probabilities: list[float] | None = None,
        fusion_mode: str = "learned",
    ):
        super().__init__()
        num_modalities = int(num_modalities)
        fusion_mode = str(fusion_mode).strip().lower()
        if fusion_mode not in {"learned", "uniform"}:
            raise ValueError("fusion_mode must be 'learned' or 'uniform'")
        self.fusion_mode = fusion_mode
        if modality_dropout_probabilities is None:
            modality_dropout_probabilities = [
                float(sequence_modality_dropout),
                *([float(modality_dropout)] * max(num_modalities - 1, 0)),
            ]
        if len(modality_dropout_probabilities) != num_modalities:
            raise ValueError("modality_dropout_probabilities must contain one value per modality")
        if any(not 0.0 <= float(value) < 1.0 for value in modality_dropout_probabilities):
            raise ValueError("modality dropout probabilities must be in [0, 1)")
        self.modality_dropout_probabilities = tuple(float(value) for value in modality_dropout_probabilities)
        self.gate = (
            nn.Sequential(
                nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
                nn.PReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(hidden_dim // 2, 1), 1),
            )
            if self.fusion_mode == "learned"
            else None
        )
        self.modality_bias = (
            nn.Parameter(torch.zeros(num_modalities))
            if self.fusion_mode == "learned"
            else None
        )
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        modalities: list[torch.Tensor],
        residual: torch.Tensor,
        *,
        availability: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not modalities:
            raise ValueError("modalities must not be empty")
        stacked = torch.stack(modalities, dim=1)
        scores = (
            self.gate(stacked).squeeze(-1) + self.modality_bias
            if self.gate is not None and self.modality_bias is not None
            else stacked.new_zeros(stacked.shape[:2])
        )
        if availability is None:
            availability = torch.ones_like(scores, dtype=torch.bool)
        else:
            if availability.shape != scores.shape:
                raise ValueError(
                    f"availability must have shape {tuple(scores.shape)}, got {tuple(availability.shape)}"
                )
            availability = availability.to(device=scores.device, dtype=torch.bool)
        if not bool(availability.any(dim=1).all()):
            raise ValueError("every residue must have at least one available modality")

        active = availability
        if self.training and any(value > 0.0 for value in self.modality_dropout_probabilities) and stacked.size(1) > 1:
            if batch is None:
                batch = torch.zeros(stacked.size(0), dtype=torch.long, device=stacked.device)
            else:
                batch = batch.to(device=stacked.device, dtype=torch.long)
            num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
            drop_probabilities = scores.new_tensor(self.modality_dropout_probabilities)
            graph_dropped = torch.rand(
                (num_graphs, stacked.size(1)),
                device=stacked.device,
            ) < drop_probabilities.view(1, -1)
            active = availability & ~graph_dropped[batch]
            no_active = ~active.any(dim=1)
            if bool(no_active.any()):
                # Rare all-dropped rows fall back to their genuinely available inputs.
                active = active.clone()
                active[no_active] = availability[no_active]
        if self.fusion_mode == "uniform":
            weights = active.to(dtype=stacked.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            scores = scores.masked_fill(~active, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return residual + self.output_dropout(fused), weights


def _node_confidence(data, x: torch.Tensor) -> torch.Tensor | None:
    for name in ("node_confidence", "residue_confidence", "plddt", "x_confidence"):
        confidence = getattr(data, name, None)
        if confidence is None:
            continue
        confidence = confidence.to(device=x.device, dtype=x.dtype)
        if confidence.dim() == 1:
            confidence = confidence.view(-1, 1)
        if confidence.size(0) != x.size(0):
            continue
        confidence = confidence[:, :1]
        if confidence.numel() and float(confidence.detach().max()) > 1.5:
            confidence = confidence / 100.0
        return confidence.clamp(0.0, 1.0)
    return None


def _node_confidence_mask(data, confidence: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mask = getattr(data, "node_confidence_mask", None)
    if mask is None:
        return torch.ones_like(confidence)
    mask = mask.to(device=x.device, dtype=x.dtype)
    if mask.dim() == 1:
        mask = mask.view(-1, 1)
    if mask.size(0) != x.size(0):
        return torch.ones_like(confidence)
    return mask[:, :1].clamp(0.0, 1.0)


class HiFENModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: dict[str, int],
        annotation_dim: int = 0,
        num_layers: int = 4,
        heads: int = 8,
        dropout: float = 0.3,
        edge_rbf_dim: int = 16,
        edge_cutoff: float = 10.0,
        use_annotation_features: bool = False,
        site_heads: bool = True,
        use_layerwise_fusion: bool = True,
        use_parent_conditioning: bool = True,
        detach_parent_conditioning: bool = False,
        use_confidence_features: bool = True,
        use_structure_aux_heads: bool = True,
        use_predicted_site_feedback: bool = True,
        detach_predicted_site_feedback: bool = False,
        predicted_site_feedback_start_epoch: int = 1,
        predicted_site_feedback_warmup_epochs: int = 0,
        predicted_site_feedback_detach_until_epoch: int = 0,
        predicted_site_fusion_mode: str = "softmax",
        predicted_site_gate_init: float = 0.1,
        predicted_site_residual_scale: float = 0.5,
        detach_hypergraph_site_probabilities: bool = False,
        hypergraph_site_probabilities_detach_until_epoch: int = 0,
        include_structure_in_modality_fusion: bool = True,
        modality_fusion_mode: str = "learned",
        modality_dropout: float = 0.1,
        sequence_modality_dropout: float = 0.0,
        tool_annotation_dropout: float | None = None,
        tool_annotation_fusion_mode: str = "softmax",
        tool_annotation_gate_init: float = 0.2,
        tool_annotation_residual_scale: float = 1.0,
        tool_annotation_start_epoch: int = 1,
        tool_annotation_warmup_epochs: int = 0,
        use_function_hypergraph: bool = True,
        hypergraph_tasks: list[str] | tuple[str, ...] | None = None,
        hypergraph_feedback_layer: int = 2,
        hypergraph_slots_per_task: int = 4,
        hypergraph_gate_init: float = 0.1,
        hypergraph_assignment_mode: str = "softmax",
        hypergraph_assignment_temperature: float = 1.0,
        hypergraph_site_probability_threshold: float = 0.0,
        hypergraph_min_edge_mass: float = 0.0,
        hypergraph_preserve_site_confidence: bool = False,
        hypergraph_context_mode: str = "normalized",
        hypergraph_confidence_power: float = 1.0,
        hypergraph_contrastive_context: bool = False,
        hypergraph_min_feedback_gate: float = 0.0,
        hypergraph_max_feedback_gate: float = 1.0,
        hypergraph_start_epoch: int = 1,
        hypergraph_warmup_epochs: int = 0,
        hypergraph_auxiliary_head: bool = False,
        use_hypergraph_main_feedback: bool = True,
        use_level_specific_pooling: bool = True,
        pooling_mode: str | None = None,
        hybrid_pool_level_init: float = 0.75,
        use_conditional_hierarchy: bool = True,
        parent_conditioning_mode: str = "concat",
        parent_residual_gate_init: float = 0.1,
        conditional_hierarchy_mode: str = "hard_product",
        conditional_hierarchy_gate_init: float = 0.1,
        conditional_hierarchy_gate_max: float = 1.0,
        parent_child_maps: dict[str, list[tuple[int, int]]] | None = None,
    ):
        super().__init__()
        self.use_annotation_features = bool(use_annotation_features and annotation_dim > 0)
        self.annotation_dim = int(annotation_dim)
        self.edge_cutoff = float(edge_cutoff)
        self.use_parent_conditioning = bool(use_parent_conditioning)
        self.detach_parent_conditioning = bool(detach_parent_conditioning)
        self.use_structure_aux_heads = bool(use_structure_aux_heads)
        self.use_predicted_site_feedback = bool(use_predicted_site_feedback and site_heads)
        self.base_detach_predicted_site_feedback = bool(
            detach_predicted_site_feedback and self.use_predicted_site_feedback
        )
        self.detach_predicted_site_feedback = self.base_detach_predicted_site_feedback
        self.predicted_site_feedback_start_epoch = max(
            int(predicted_site_feedback_start_epoch), 1
        )
        self.predicted_site_feedback_warmup_epochs = max(
            int(predicted_site_feedback_warmup_epochs), 0
        )
        self.predicted_site_feedback_detach_until_epoch = max(
            int(predicted_site_feedback_detach_until_epoch), 0
        )
        self.predicted_site_feedback_scale = 1.0
        predicted_site_fusion_mode = str(predicted_site_fusion_mode).strip().lower()
        if predicted_site_fusion_mode not in {"softmax", "residual_gate"}:
            raise ValueError(
                "predicted_site_fusion_mode must be 'softmax' or 'residual_gate'"
            )
        self.predicted_site_fusion_mode = predicted_site_fusion_mode
        self.predicted_site_residual_scale = float(predicted_site_residual_scale)
        if self.predicted_site_residual_scale < 0.0:
            raise ValueError("predicted_site_residual_scale must be non-negative")
        self.include_structure_in_modality_fusion = bool(include_structure_in_modality_fusion)
        self.modality_fusion_mode = str(modality_fusion_mode).strip().lower()
        if self.modality_fusion_mode not in {"learned", "uniform"}:
            raise ValueError("modality_fusion_mode must be 'learned' or 'uniform'")
        if pooling_mode is None:
            pooling_mode = "level_specific" if use_level_specific_pooling else "shared"
        pooling_mode = str(pooling_mode).strip().lower()
        if pooling_mode not in {"level_specific", "shared", "hybrid"}:
            raise ValueError("pooling_mode must be 'level_specific', 'shared', or 'hybrid'")
        self.pooling_mode = pooling_mode
        self.use_level_specific_pooling = pooling_mode == "level_specific"
        tool_annotation_fusion_mode = str(tool_annotation_fusion_mode).strip().lower()
        if tool_annotation_fusion_mode not in {"softmax", "residual_gate"}:
            raise ValueError("tool_annotation_fusion_mode must be 'softmax' or 'residual_gate'")
        self.tool_annotation_fusion_mode = tool_annotation_fusion_mode
        self.tool_annotation_residual_scale = float(tool_annotation_residual_scale)
        if self.tool_annotation_residual_scale < 0.0:
            raise ValueError("tool_annotation_residual_scale must be non-negative")
        self.tool_annotation_start_epoch = max(int(tool_annotation_start_epoch), 1)
        self.tool_annotation_warmup_epochs = max(int(tool_annotation_warmup_epochs), 0)
        # Evaluation without an explicit training epoch uses the fully enabled
        # feature path. Training calls set_training_epoch before every epoch.
        self.tool_annotation_scale = 1.0
        self.use_function_hypergraph = bool(use_function_hypergraph and site_heads and num_layers > 1)
        self.base_detach_hypergraph_site_probabilities = bool(
            detach_hypergraph_site_probabilities and self.use_function_hypergraph
        )
        self.detach_hypergraph_site_probabilities = (
            self.base_detach_hypergraph_site_probabilities
        )
        self.hypergraph_site_probabilities_detach_until_epoch = max(
            int(hypergraph_site_probabilities_detach_until_epoch), 0
        )
        selected_hypergraph_tasks = tuple(hypergraph_tasks or SITE_TASKS)
        invalid_hypergraph_tasks = sorted(set(selected_hypergraph_tasks) - set(SITE_TASKS))
        if invalid_hypergraph_tasks:
            raise ValueError(f"Unsupported hypergraph tasks: {invalid_hypergraph_tasks}")
        if len(set(selected_hypergraph_tasks)) != len(selected_hypergraph_tasks):
            raise ValueError("hypergraph_tasks must not contain duplicates")
        if self.use_function_hypergraph and not selected_hypergraph_tasks:
            raise ValueError("hypergraph_tasks must not be empty when the hypergraph is enabled")
        self.hypergraph_tasks = selected_hypergraph_tasks if self.use_function_hypergraph else ()
        self.hypergraph_feedback_layer = min(max(int(hypergraph_feedback_layer), 0), max(int(num_layers) - 1, 0))
        self.hypergraph_start_epoch = max(int(hypergraph_start_epoch), 1)
        self.hypergraph_warmup_epochs = max(int(hypergraph_warmup_epochs), 0)
        self.hypergraph_feedback_scale = 1.0
        self.training_stage = 4
        self.training_stage_name = "end_to_end"
        self.use_hypergraph_auxiliary_head = bool(
            self.use_function_hypergraph and hypergraph_auxiliary_head
        )
        self.use_hypergraph_main_feedback = bool(
            self.use_function_hypergraph and use_hypergraph_main_feedback
        )
        self.use_conditional_hierarchy = bool(use_conditional_hierarchy and use_parent_conditioning and parent_child_maps)
        parent_conditioning_mode = str(parent_conditioning_mode).strip().lower()
        if parent_conditioning_mode not in {"concat", "residual_gate"}:
            raise ValueError(
                "parent_conditioning_mode must be 'concat' or 'residual_gate'"
            )
        self.parent_conditioning_mode = parent_conditioning_mode
        conditional_hierarchy_mode = str(conditional_hierarchy_mode).strip().lower()
        if conditional_hierarchy_mode not in {"hard_product", "adaptive_residual"}:
            raise ValueError(
                "conditional_hierarchy_mode must be 'hard_product' or 'adaptive_residual'"
            )
        self.conditional_hierarchy_mode = conditional_hierarchy_mode
        self.conditional_hierarchy_gate_max = float(conditional_hierarchy_gate_max)
        if not 0.0 < self.conditional_hierarchy_gate_max <= 1.0:
            raise ValueError("conditional_hierarchy_gate_max must be in (0, 1]")
        conditional_hierarchy_gate_init = float(conditional_hierarchy_gate_init)
        if not 0.0 < conditional_hierarchy_gate_init < self.conditional_hierarchy_gate_max:
            raise ValueError(
                "conditional_hierarchy_gate_init must be in (0, conditional_hierarchy_gate_max)"
            )
        self.edge_encoder = RBFEdgeEncoder(edge_rbf_dim, cutoff=edge_cutoff, extra_dim=1)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
        )
        if self.use_annotation_features:
            self.annotation_encoder = nn.Sequential(
                nn.Linear(annotation_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout),
            )
            if self.tool_annotation_fusion_mode == "residual_gate":
                self.annotation_graph_encoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                )
                self.annotation_residual_adapter = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                nn.init.zeros_(self.annotation_residual_adapter[-1].weight)
                nn.init.zeros_(self.annotation_residual_adapter[-1].bias)
                self.annotation_residual_gate = nn.Linear(hidden_dim * 3 + 1, 1)
                tool_annotation_gate_init = min(
                    max(float(tool_annotation_gate_init), 1e-4),
                    1.0 - 1e-4,
                )
                nn.init.zeros_(self.annotation_residual_gate.weight)
                nn.init.constant_(
                    self.annotation_residual_gate.bias,
                    math.log(tool_annotation_gate_init / (1.0 - tool_annotation_gate_init)),
                )
            else:
                self.annotation_graph_encoder = None
                self.annotation_residual_adapter = None
                self.annotation_residual_gate = None
        else:
            self.annotation_encoder = None
            self.annotation_graph_encoder = None
            self.annotation_residual_adapter = None
            self.annotation_residual_gate = None
        self.backbone = ResidualGATBackbone(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            edge_dim=self.edge_encoder.out_dim,
            dropout=dropout,
        )
        if self.use_function_hypergraph:
            self.intermediate_site_heads = nn.ModuleDict(
                {task: nn.Linear(hidden_dim, 1) for task in self.hypergraph_tasks}
            )
            self.function_hypergraph = SoftFunctionalHypergraph(
                hidden_dim=hidden_dim,
                num_tasks=len(self.hypergraph_tasks),
                slots_per_task=hypergraph_slots_per_task,
                dropout=dropout,
                gate_init=hypergraph_gate_init,
                assignment_mode=hypergraph_assignment_mode,
                assignment_temperature=hypergraph_assignment_temperature,
                site_probability_threshold=hypergraph_site_probability_threshold,
                min_edge_mass=hypergraph_min_edge_mass,
                preserve_site_confidence=hypergraph_preserve_site_confidence,
                context_mode=hypergraph_context_mode,
                confidence_power=hypergraph_confidence_power,
                contrastive_context=hypergraph_contrastive_context,
                min_feedback_gate=hypergraph_min_feedback_gate,
                max_feedback_gate=hypergraph_max_feedback_gate,
            )
        else:
            self.intermediate_site_heads = nn.ModuleDict()
            self.function_hypergraph = None
        if self.use_hypergraph_auxiliary_head:
            self.hypergraph_aux_pool = LevelAttentionPool(hidden_dim, dropout=dropout)
            self.hypergraph_aux_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, int(num_classes["ec4"])),
            )
        else:
            self.hypergraph_aux_pool = None
            self.hypergraph_aux_head = None
        self.layer_fusion = LayerwiseRepresentationFusion(hidden_dim, dropout=dropout) if use_layerwise_fusion else None
        self.confidence_gate = (
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Sigmoid(),
            )
            if use_confidence_features
            else None
        )
        if self.pooling_mode == "level_specific":
            pools = {level: LevelAttentionPool(hidden_dim, dropout=dropout) for level in EC_LEVELS}
        elif self.pooling_mode == "shared":
            pools = {"shared": LevelAttentionPool(hidden_dim, dropout=dropout)}
        else:
            pools = {
                "shared": LevelAttentionPool(hidden_dim, dropout=dropout),
                **{level: LevelAttentionPool(hidden_dim, dropout=dropout) for level in EC_LEVELS},
            }
        self.level_pools = nn.ModuleDict(pools)
        if self.pooling_mode == "hybrid":
            hybrid_pool_level_init = min(max(float(hybrid_pool_level_init), 1e-4), 1.0 - 1e-4)
            self.pool_fusion_logits = nn.Parameter(
                torch.full(
                    (len(EC_LEVELS),),
                    math.log(hybrid_pool_level_init / (1.0 - hybrid_pool_level_init)),
                )
            )
        else:
            self.register_parameter("pool_fusion_logits", None)
        self.parent_context = nn.ModuleDict()
        self.parent_residual_heads = nn.ModuleDict()
        self.parent_residual_gates = nn.ModuleDict()
        self.conditional_hierarchy_gate_logits = nn.ParameterDict()
        self.heads = nn.ModuleDict()
        parent_child_maps = parent_child_maps or {}
        for level in EC_LEVELS:
            head_input_dim = hidden_dim
            parent_level = PARENT_LEVEL.get(level)
            if self.use_parent_conditioning and parent_level:
                self.parent_context[level] = nn.Sequential(
                    nn.Linear(int(num_classes[parent_level]), hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.PReLU(),
                )
                if self.parent_conditioning_mode == "concat":
                    head_input_dim += hidden_dim
                else:
                    self.parent_residual_heads[level] = nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.PReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, int(num_classes[level])),
                    )
                    nn.init.zeros_(self.parent_residual_heads[level][-1].weight)
                    nn.init.zeros_(self.parent_residual_heads[level][-1].bias)
                    self.parent_residual_gates[level] = nn.Linear(hidden_dim * 2, 1)
                    parent_residual_gate_init = min(
                        max(float(parent_residual_gate_init), 1e-4),
                        1.0 - 1e-4,
                    )
                    nn.init.zeros_(self.parent_residual_gates[level].weight)
                    nn.init.constant_(
                        self.parent_residual_gates[level].bias,
                        math.log(
                            parent_residual_gate_init
                            / (1.0 - parent_residual_gate_init)
                        ),
                    )
            self.heads[level] = nn.Sequential(
                nn.Linear(head_input_dim, hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, int(num_classes[level])),
            )
            if self.use_conditional_hierarchy and parent_level:
                parent_index = torch.full((int(num_classes[level]),), -1, dtype=torch.long)
                for parent, child in parent_child_maps.get(f"{parent_level}_to_{level}", []):
                    if 0 <= int(child) < parent_index.numel():
                        parent_index[int(child)] = int(parent)
                self.register_buffer(f"{level}_parent_index", parent_index)
                if self.conditional_hierarchy_mode == "adaptive_residual":
                    normalized_gate_init = (
                        conditional_hierarchy_gate_init
                        / self.conditional_hierarchy_gate_max
                    )
                    self.conditional_hierarchy_gate_logits[level] = nn.Parameter(
                        torch.full(
                            (int(num_classes[level]),),
                            math.log(
                                normalized_gate_init
                                / (1.0 - normalized_gate_init)
                            ),
                        )
                    )
        self.site_heads = nn.ModuleDict()
        if site_heads:
            for task in SITE_TASKS:
                self.site_heads[task] = nn.Linear(hidden_dim, 1)
        if self.use_predicted_site_feedback:
            self.site_feedback_encoder = nn.Sequential(
                nn.Linear(len(SITE_TASKS), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout),
            )
            if self.predicted_site_fusion_mode == "residual_gate":
                self.site_feedback_residual_adapter = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                nn.init.zeros_(self.site_feedback_residual_adapter[-1].weight)
                nn.init.zeros_(self.site_feedback_residual_adapter[-1].bias)
                self.site_feedback_residual_gate = nn.Linear(hidden_dim * 2, 1)
                predicted_site_gate_init = min(
                    max(float(predicted_site_gate_init), 1e-4),
                    1.0 - 1e-4,
                )
                nn.init.zeros_(self.site_feedback_residual_gate.weight)
                nn.init.constant_(
                    self.site_feedback_residual_gate.bias,
                    math.log(predicted_site_gate_init / (1.0 - predicted_site_gate_init)),
                )
            else:
                self.site_feedback_residual_adapter = None
                self.site_feedback_residual_gate = None
        else:
            self.site_feedback_encoder = None
            self.site_feedback_residual_adapter = None
            self.site_feedback_residual_gate = None
        modality_names = ["sequence"]
        if self.include_structure_in_modality_fusion:
            modality_names.append("structure")
        if (
            self.use_predicted_site_feedback
            and self.predicted_site_fusion_mode == "softmax"
        ):
            modality_names.append("predicted_sites")
        if self.use_annotation_features and self.tool_annotation_fusion_mode == "softmax":
            modality_names.append("tool_annotations")
        self.modality_names = tuple(modality_names)
        tool_annotation_dropout = (
            float(modality_dropout)
            if tool_annotation_dropout is None
            else float(tool_annotation_dropout)
        )
        if not 0.0 <= tool_annotation_dropout < 1.0:
            raise ValueError("tool_annotation_dropout must be in [0, 1)")
        self.tool_annotation_dropout = tool_annotation_dropout
        modality_dropout_probabilities = []
        for name in self.modality_names:
            if name == "sequence":
                modality_dropout_probabilities.append(float(sequence_modality_dropout))
            elif name == "tool_annotations":
                modality_dropout_probabilities.append(tool_annotation_dropout)
            else:
                modality_dropout_probabilities.append(float(modality_dropout))
        self.modality_fusion = ResidueModalityFusion(
            hidden_dim,
            num_modalities=len(self.modality_names),
            dropout=dropout,
            modality_dropout=modality_dropout,
            sequence_modality_dropout=sequence_modality_dropout,
            modality_dropout_probabilities=modality_dropout_probabilities,
            fusion_mode=self.modality_fusion_mode,
        )
        if self.use_structure_aux_heads:
            edge_pair_dim = hidden_dim * 4
            self.edge_distance_head = nn.Sequential(
                nn.Linear(edge_pair_dim, hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.edge_contact_head = nn.Sequential(
                nn.Linear(edge_pair_dim, hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.edge_distance_head = None
            self.edge_contact_head = None

    def set_training_epoch(self, epoch: int) -> float:
        """Apply staged feature curricula and gradient-isolation transitions."""
        epoch = int(epoch)
        if self.site_feedback_encoder is None:
            self.predicted_site_feedback_scale = 0.0
        else:
            self.predicted_site_feedback_scale = _feature_curriculum_scale(
                epoch,
                self.predicted_site_feedback_start_epoch,
                self.predicted_site_feedback_warmup_epochs,
            )
        self.detach_predicted_site_feedback = bool(
            self.base_detach_predicted_site_feedback
            or (
                self.predicted_site_feedback_detach_until_epoch > 0
                and epoch <= self.predicted_site_feedback_detach_until_epoch
            )
        )

        if self.function_hypergraph is None:
            scale = 0.0
        else:
            scale = _feature_curriculum_scale(
                epoch,
                self.hypergraph_start_epoch,
                self.hypergraph_warmup_epochs,
            )
        self.hypergraph_feedback_scale = scale
        self.detach_hypergraph_site_probabilities = bool(
            self.base_detach_hypergraph_site_probabilities
            or (
                self.hypergraph_site_probabilities_detach_until_epoch > 0
                and epoch <= self.hypergraph_site_probabilities_detach_until_epoch
            )
        )
        if self.function_hypergraph is not None:
            self.function_hypergraph.set_feedback_scale(scale)
        if not self.use_annotation_features:
            self.tool_annotation_scale = 0.0
        else:
            self.tool_annotation_scale = _feature_curriculum_scale(
                epoch,
                self.tool_annotation_start_epoch,
                self.tool_annotation_warmup_epochs,
            )

        downstream_starts = []
        if self.function_hypergraph is not None:
            downstream_starts.append(self.hypergraph_start_epoch)
        if self.use_annotation_features:
            downstream_starts.append(self.tool_annotation_start_epoch)
        downstream_start = min(
            downstream_starts,
            default=self.predicted_site_feedback_start_epoch,
        )
        final_stage_epoch = max(
            self.predicted_site_feedback_detach_until_epoch,
            self.hypergraph_site_probabilities_detach_until_epoch,
            downstream_start - 1,
        ) + 1
        if epoch < self.predicted_site_feedback_start_epoch:
            self.training_stage = 1
            self.training_stage_name = "backbone_site_supervision"
        elif epoch < downstream_start:
            self.training_stage = 2
            self.training_stage_name = "detached_site_feedback"
        elif epoch < final_stage_epoch:
            self.training_stage = 3
            self.training_stage_name = "detached_multimodal_curriculum"
        else:
            self.training_stage = 4
            self.training_stage_name = "end_to_end"
        return scale

    def forward(self, data):
        sequence_repr = self.input_proj(data.x.float())
        confidence = _node_confidence(data, sequence_repr) if self.confidence_gate is not None else None
        if confidence is not None:
            confidence_mask = _node_confidence_mask(data, confidence, sequence_repr)
            gated_scale = 0.5 + self.confidence_gate(confidence)
            sequence_repr = sequence_repr * (1.0 + confidence_mask * (gated_scale - 1.0))

        edge_index = data.edge_index.to(device=sequence_repr.device, dtype=torch.long)
        raw_edge_attr = data.edge_attr.to(device=sequence_repr.device, dtype=sequence_repr.dtype)
        edge_attr = self.edge_encoder(raw_edge_attr)
        batch = getattr(data, "batch", torch.zeros(sequence_repr.size(0), dtype=torch.long, device=sequence_repr.device))
        batch = batch.to(device=sequence_repr.device, dtype=torch.long)
        intermediate_sites = None
        hypergraph_memberships = None
        hypergraph_gate = None
        hypergraph_diagnostics = None
        hypergraph_aux_logits = None
        hypergraph_aux_attention = None
        if self.function_hypergraph is not None:
            early_outputs = []
            intermediate_repr = sequence_repr
            if self.hypergraph_feedback_layer > 0:
                early_outputs = self.backbone.forward_layers(
                    sequence_repr,
                    edge_index,
                    edge_attr,
                    batch,
                    end=self.hypergraph_feedback_layer,
                )
                intermediate_repr = early_outputs[-1]
            intermediate_sites = {
                task: head(intermediate_repr).view(-1)
                for task, head in self.intermediate_site_heads.items()
            }
            intermediate_probabilities = torch.stack(
                [torch.sigmoid(intermediate_sites[task]) for task in self.hypergraph_tasks],
                dim=-1,
            )
            hypergraph_site_probabilities = (
                intermediate_probabilities.detach()
                if self.detach_hypergraph_site_probabilities
                else intermediate_probabilities
            )
            if self.hypergraph_feedback_scale <= 0.0:
                # Keep the intermediate site heads directly supervised while
                # functionally freezing the random hypergraph branch.
                feedback_repr = intermediate_repr
            else:
                (
                    feedback_repr,
                    hypergraph_memberships,
                    hypergraph_gate,
                    hypergraph_diagnostics,
                ) = self.function_hypergraph(
                    intermediate_repr,
                    hypergraph_site_probabilities,
                    batch,
                )
            if (
                self.hypergraph_feedback_scale > 0.0
                and self.hypergraph_aux_pool is not None
                and self.hypergraph_aux_head is not None
            ):
                # Supervise the actual hypergraph correction rather than the base
                # representation. This prevents the auxiliary classifier from
                # bypassing the functional context and gives the hyperedges a
                # direct EC4 learning signal even when the main residual gate is
                # initially conservative.
                hypergraph_delta = feedback_repr - intermediate_repr
                hypergraph_aux_pooled, hypergraph_aux_attention = self.hypergraph_aux_pool(
                    hypergraph_delta,
                    batch,
                )
                hypergraph_aux_logits = self.hypergraph_aux_head(hypergraph_aux_pooled)
            main_feedback_repr = (
                feedback_repr if self.use_hypergraph_main_feedback else intermediate_repr
            )
            late_outputs = self.backbone.forward_layers(
                main_feedback_repr,
                edge_index,
                edge_attr,
                batch,
                start=self.hypergraph_feedback_layer,
            )
            layer_outputs = early_outputs + late_outputs
        else:
            layer_outputs = self.backbone(sequence_repr, edge_index, edge_attr, batch)
        if self.layer_fusion is None:
            structure_repr = layer_outputs[-1]
            layer_weights = None
        else:
            structure_repr, layer_weights = self.layer_fusion(layer_outputs)

        output = {}
        if intermediate_sites is not None:
            output["intermediate_sites"] = intermediate_sites
            output["hypergraph_memberships"] = hypergraph_memberships
            output["hypergraph_gate"] = hypergraph_gate
            output["hypergraph_diagnostics"] = hypergraph_diagnostics
            output["hypergraph_task_names"] = self.hypergraph_tasks
            if hypergraph_aux_logits is not None:
                output["hypergraph_aux_ec4"] = hypergraph_aux_logits
                output["hypergraph_aux_attention"] = hypergraph_aux_attention
        for name, head in self.site_heads.items():
            output[name] = head(structure_repr).view(-1)
        modalities = [sequence_repr]
        site_feedback_repr = None
        site_feedback_active = False
        modality_availability = [
            torch.ones((sequence_repr.size(0), 1), dtype=torch.bool, device=sequence_repr.device),
        ]
        if self.include_structure_in_modality_fusion:
            modalities.append(structure_repr)
            modality_availability.append(
                torch.ones((sequence_repr.size(0), 1), dtype=torch.bool, device=sequence_repr.device)
            )
        if self.site_feedback_encoder is not None:
            site_feedback_scale = float(self.predicted_site_feedback_scale)
            site_feedback_active = site_feedback_scale > 0.0
            if site_feedback_active:
                site_probabilities = torch.stack(
                    [torch.sigmoid(output[task]) for task in SITE_TASKS],
                    dim=-1,
                )
                if self.detach_predicted_site_feedback:
                    site_probabilities = site_probabilities.detach()
                site_feedback_repr = (
                    self.site_feedback_encoder(site_probabilities) * site_feedback_scale
                )
            else:
                # Do not execute the random encoder before its stage begins;
                # this also prevents AdamW decay through a zero-valued graph.
                site_feedback_repr = torch.zeros_like(sequence_repr)
            if self.predicted_site_fusion_mode == "softmax":
                modalities.append(site_feedback_repr)
                modality_availability.append(
                    torch.full(
                        (sequence_repr.size(0), 1),
                        site_feedback_active,
                        dtype=torch.bool,
                        device=sequence_repr.device,
                    )
                )
            output["predicted_site_feedback_scale"] = sequence_repr.new_full(
                (sequence_repr.size(0), 1),
                site_feedback_scale,
            )
        if self.use_annotation_features:
            annotation_x = getattr(data, "annotation_x", None)
            annotation_mask = getattr(data, "annotation_mask", None)
            if annotation_x is None:
                annotation_x = torch.zeros(
                    (sequence_repr.size(0), self.annotation_dim),
                    dtype=sequence_repr.dtype,
                    device=sequence_repr.device,
                )
            else:
                annotation_x = annotation_x.to(device=sequence_repr.device, dtype=sequence_repr.dtype)
            if annotation_mask is None:
                annotation_mask = torch.zeros_like(annotation_x)
            else:
                annotation_mask = annotation_mask.to(device=sequence_repr.device, dtype=sequence_repr.dtype)
            annotation_available = annotation_mask.ne(0).any(dim=-1, keepdim=True)
            tool_scale = float(self.tool_annotation_scale)
            if tool_scale > 0.0:
                annotation_repr = self.annotation_encoder(
                    torch.cat([annotation_x, annotation_mask], dim=-1)
                )
            else:
                # The tool branch is functionally frozen before its stage.
                annotation_repr = torch.zeros_like(sequence_repr)
            tool_scale_values = torch.full_like(
                annotation_available,
                tool_scale,
                dtype=annotation_repr.dtype,
            )
            tool_active = annotation_available & (tool_scale > 0.0)
            if self.tool_annotation_fusion_mode == "softmax":
                modalities.append(annotation_repr * tool_scale)
                modality_availability.append(tool_active)
            output["annotation_availability"] = annotation_available
            nan_values = torch.full_like(tool_scale_values, float("nan"))
            output["tool_annotation_diagnostics"] = {
                "availability": annotation_available.to(dtype=annotation_repr.dtype),
                "scale_on_available": torch.where(
                    annotation_available,
                    tool_scale_values,
                    nan_values,
                ),
            }
        modality_availability_tensor = torch.cat(modality_availability, dim=1)
        residue_repr, modality_weights = self.modality_fusion(
            modalities,
            residual=structure_repr,
            availability=modality_availability_tensor,
            batch=batch,
        )
        if (
            self.site_feedback_residual_adapter is not None
            and self.site_feedback_residual_gate is not None
        ):
            if site_feedback_active and site_feedback_repr is not None:
                site_gate = torch.sigmoid(
                    self.site_feedback_residual_gate(
                        torch.cat([structure_repr, site_feedback_repr], dim=-1)
                    )
                )
                site_correction = (
                    self.predicted_site_residual_scale
                    * float(self.predicted_site_feedback_scale)
                    * site_gate
                    * self.site_feedback_residual_adapter(site_feedback_repr)
                )
            else:
                site_gate = sequence_repr.new_zeros((sequence_repr.size(0), 1))
                site_correction = torch.zeros_like(sequence_repr)
            residue_repr = residue_repr + site_correction
            output["predicted_site_feedback_gate"] = site_gate
            output["predicted_site_feedback_diagnostics"] = {
                "gate": site_gate,
                "correction_norm": (
                    site_correction.norm(dim=-1, keepdim=True)
                    / math.sqrt(site_correction.size(-1))
                ),
            }
        if self.use_annotation_features and self.tool_annotation_fusion_mode == "residual_gate":
            annotation_active = tool_active
            num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
            if self.training and self.tool_annotation_dropout > 0.0:
                graph_dropped = torch.rand(num_graphs, device=batch.device) < self.tool_annotation_dropout
                annotation_active = annotation_active & ~graph_dropped[batch].unsqueeze(-1)
            active_float = annotation_active.to(dtype=annotation_repr.dtype)
            graph_annotation_sum = annotation_repr.new_zeros((num_graphs, annotation_repr.size(-1)))
            graph_annotation_count = annotation_repr.new_zeros((num_graphs, 1))
            graph_annotation_sum.index_add_(0, batch, annotation_repr * active_float)
            graph_annotation_count.index_add_(0, batch, active_float)
            graph_annotation_mean = graph_annotation_sum / graph_annotation_count.clamp_min(1.0)
            graph_annotation_repr = self.annotation_graph_encoder(graph_annotation_mean)[batch]
            tool_hit_signal = annotation_x[:, 3:5].amax(dim=-1, keepdim=True)
            tool_correction = self.annotation_residual_adapter(
                torch.cat([annotation_repr, graph_annotation_repr], dim=-1)
            )
            tool_gate = torch.sigmoid(
                self.annotation_residual_gate(
                    torch.cat(
                        [structure_repr, annotation_repr, graph_annotation_repr, tool_hit_signal],
                        dim=-1,
                    )
                )
            )
            tool_correction = (
                self.tool_annotation_residual_scale
                * tool_scale
                * active_float
                * tool_gate
                * tool_correction
            )
            residue_repr = residue_repr + tool_correction
            nan_values = torch.full_like(tool_gate, float("nan"))
            output["tool_annotation_gate"] = tool_gate * active_float
            output["tool_annotation_diagnostics"].update({
                "gate_on_available": torch.where(annotation_available, tool_gate, nan_values),
                "hit_signal_on_available": torch.where(
                    annotation_available,
                    tool_hit_signal,
                    nan_values,
                ),
                "correction_norm_on_available": torch.where(
                    annotation_available,
                    tool_correction.norm(dim=-1, keepdim=True) / math.sqrt(tool_correction.size(-1)),
                    nan_values,
                ),
            })
        attention = {}
        pooled_by_level = {}
        pooling_gate = None
        if self.pooling_mode == "level_specific":
            for level, pool in self.level_pools.items():
                pooled, weights = pool(residue_repr, batch)
                pooled_by_level[level] = pooled
                attention[level] = weights
        elif self.pooling_mode == "shared":
            pooled, weights = self.level_pools["shared"](residue_repr, batch)
            for level in EC_LEVELS:
                pooled_by_level[level] = pooled
                attention[level] = weights
        else:
            shared_pooled, shared_weights = self.level_pools["shared"](residue_repr, batch)
            pooling_gate = torch.sigmoid(self.pool_fusion_logits)
            for index, level in enumerate(EC_LEVELS):
                level_pooled, level_weights = self.level_pools[level](residue_repr, batch)
                level_gate = pooling_gate[index]
                pooled_by_level[level] = (
                    level_gate * level_pooled + (1.0 - level_gate) * shared_pooled
                )
                attention[level] = (
                    level_gate * level_weights + (1.0 - level_gate) * shared_weights
                )
        conditional_logits = {}
        parent_conditioning_gates = {}
        conditional_hierarchy_gates = {}
        for level in EC_LEVELS:
            head_input = pooled_by_level[level]
            parent_level = PARENT_LEVEL.get(level)
            parent_signal = None
            parent_context = None
            if self.use_parent_conditioning and parent_level:
                parent_signal = torch.sigmoid(output[parent_level])
                if self.detach_parent_conditioning:
                    parent_signal = parent_signal.detach()
                parent_context = self.parent_context[level](parent_signal)
                if self.parent_conditioning_mode == "concat":
                    head_input = torch.cat([head_input, parent_context], dim=-1)
            raw_logits = self.heads[level](head_input)
            if (
                parent_context is not None
                and self.parent_conditioning_mode == "residual_gate"
            ):
                residual_input = torch.cat(
                    [pooled_by_level[level], parent_context], dim=-1
                )
                parent_gate = torch.sigmoid(
                    self.parent_residual_gates[level](residual_input)
                )
                raw_logits = raw_logits + parent_gate * self.parent_residual_heads[level](
                    residual_input
                )
                parent_conditioning_gates[level] = parent_gate
            if self.use_conditional_hierarchy and parent_level and parent_signal is not None:
                parent_index = getattr(self, f"{level}_parent_index")
                valid = parent_index >= 0
                # In FP16, 1 - 1e-6 rounds to 1 and logit(1) becomes inf.
                # Keep probability composition and logit conversion in FP32.
                parent_probability = torch.ones_like(raw_logits, dtype=torch.float32)
                parent_probability[:, valid] = parent_signal[:, parent_index[valid]].float()
                conditional_probability = torch.sigmoid(raw_logits.float())
                if self.conditional_hierarchy_mode == "hard_product":
                    gated_probability = conditional_probability * parent_probability
                else:
                    hierarchy_gate = self.conditional_hierarchy_gate_max * torch.sigmoid(
                        self.conditional_hierarchy_gate_logits[level]
                    )
                    parent_factor = 1.0 - hierarchy_gate.view(1, -1) * (
                        1.0 - parent_probability
                    )
                    gated_probability = conditional_probability * parent_factor
                    conditional_hierarchy_gates[level] = hierarchy_gate
                gated_probability = gated_probability.clamp(1e-6, 1.0 - 1e-6)
                output[level] = torch.logit(gated_probability)
                conditional_logits[level] = raw_logits
            else:
                output[level] = raw_logits
        if conditional_logits:
            output["conditional_logits"] = conditional_logits
        if parent_conditioning_gates:
            output["parent_conditioning_gates"] = parent_conditioning_gates
        if conditional_hierarchy_gates:
            output["conditional_hierarchy_gates"] = conditional_hierarchy_gates
        if self.use_structure_aux_heads:
            src, dst = edge_index
            src_repr = structure_repr[src]
            dst_repr = structure_repr[dst]
            edge_pair_repr = torch.cat([src_repr, dst_repr, torch.abs(src_repr - dst_repr), src_repr * dst_repr], dim=-1)
            output["edge_distance"] = F.softplus(self.edge_distance_head(edge_pair_repr).float()).view(-1)
            output["edge_contact"] = self.edge_contact_head(edge_pair_repr).view(-1)
        output["attention"] = attention
        if pooling_gate is not None:
            output["pooling_gate"] = pooling_gate
        output["modality_attention"] = modality_weights
        output["modality_names"] = self.modality_names
        output["modality_availability"] = modality_availability_tensor
        if layer_weights is not None:
            output["layer_attention"] = layer_weights
        return output

