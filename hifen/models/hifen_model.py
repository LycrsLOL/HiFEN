from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..data.graph_builder import SITE_TASKS
from .backbone import ResidualGATBackbone
from .edge_encoder import RBFEdgeEncoder
from .functional_graph import SoftFunctionalHypergraph
from .pooling import LevelAttentionPool


EC_LEVELS = ("ec1", "ec2", "ec3", "ec4")
PARENT_LEVEL = {"ec2": "ec1", "ec3": "ec2", "ec4": "ec3"}


def _feature_curriculum_scale(epoch: int, start_epoch: int, warmup_epochs: int) -> float:
    """Return a zero-to-one feature scale for the module warm-up schedule."""
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


class HiFENModel(nn.Module):
    """EC predictor with layerwise fusion, parent conditioning and annotation encoding.

    Tool annotation encoding augments the residue representation. Structure
    completion runs during graph preparation, before this network is called.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: dict[str, int],
        annotation_dim: int = 12,
        num_layers: int = 4,
        heads: int = 8,
        dropout: float = 0.3,
        edge_rbf_dim: int = 16,
        edge_cutoff: float = 10.0,
        use_annotation_features: bool = True,
        use_layerwise_fusion: bool = True,
        use_parent_conditioning: bool = True,
        detach_parent_conditioning: bool = True,
        parent_conditioning_mode: str = "residual_gate",
        parent_residual_gate_init: float = 0.1,
        tool_annotation_dropout: float = 0.3,
        tool_annotation_gate_init: float = 0.1,
        tool_annotation_residual_scale: float = 0.5,
        tool_annotation_start_epoch: int = 1,
        tool_annotation_warmup_epochs: int = 5,
        use_function_hypergraph: bool = False,
        hypergraph_tasks: list[str] | tuple[str, ...] | None = None,
        hypergraph_feedback_layer: int = 2,
        hypergraph_slots_per_task: int = 1,
        hypergraph_gate_init: float = 0.1,
        hypergraph_assignment_mode: str = "softmax",
        hypergraph_assignment_temperature: float = 0.75,
        hypergraph_site_probability_threshold: float = 0.2,
        hypergraph_min_edge_mass: float = 0.1,
        hypergraph_preserve_site_confidence: bool = True,
        hypergraph_context_mode: str = "task_attention",
        hypergraph_confidence_power: float = 0.5,
        hypergraph_contrastive_context: bool = True,
        hypergraph_min_feedback_gate: float = 0.05,
        hypergraph_max_feedback_gate: float = 0.1,
        hypergraph_start_epoch: int = 8,
        hypergraph_warmup_epochs: int = 8,
        detach_hypergraph_site_probabilities: bool = True,
        hypergraph_site_probabilities_detach_until_epoch: int = 0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if any(int(num_classes.get(level, 0)) < 1 for level in EC_LEVELS):
            raise ValueError("num_classes must include a positive count for every EC level")
        if use_annotation_features and annotation_dim < 1:
            raise ValueError("annotation_dim must be positive when annotation encoding is enabled")
        self.annotation_dim = int(annotation_dim)
        self.use_annotation_features = bool(use_annotation_features)
        self.use_parent_conditioning = bool(use_parent_conditioning)
        self.detach_parent_conditioning = bool(detach_parent_conditioning)
        self.parent_conditioning_mode = str(parent_conditioning_mode).strip().lower()
        if self.parent_conditioning_mode not in {"concat", "residual_gate"}:
            raise ValueError("parent_conditioning_mode must be 'concat' or 'residual_gate'")
        self.tool_annotation_dropout = float(tool_annotation_dropout)
        if not 0.0 <= self.tool_annotation_dropout < 1.0:
            raise ValueError("tool_annotation_dropout must be in [0, 1)")
        self.tool_annotation_residual_scale = float(tool_annotation_residual_scale)
        if self.tool_annotation_residual_scale < 0.0:
            raise ValueError("tool_annotation_residual_scale must be non-negative")
        self.tool_annotation_start_epoch = max(int(tool_annotation_start_epoch), 1)
        self.tool_annotation_warmup_epochs = max(int(tool_annotation_warmup_epochs), 0)
        self.tool_annotation_scale = 1.0
        self.use_function_hypergraph = bool(use_function_hypergraph)
        selected_tasks = tuple(
            ("active_or_catalytic_site", "binding_site")
            if hypergraph_tasks is None else hypergraph_tasks
        )
        invalid_tasks = sorted(set(selected_tasks) - set(SITE_TASKS))
        if invalid_tasks:
            raise ValueError(f"Unsupported hypergraph tasks: {invalid_tasks}")
        if len(set(selected_tasks)) != len(selected_tasks):
            raise ValueError("hypergraph_tasks must not contain duplicates")
        if self.use_function_hypergraph and not selected_tasks:
            raise ValueError("hypergraph_tasks must not be empty when functional hyperedges are enabled")
        self.hypergraph_tasks = selected_tasks if self.use_function_hypergraph else ()
        self.hypergraph_feedback_layer = min(max(int(hypergraph_feedback_layer), 0), num_layers - 1)
        self.hypergraph_start_epoch = max(int(hypergraph_start_epoch), 1)
        self.hypergraph_warmup_epochs = max(int(hypergraph_warmup_epochs), 0)
        self.hypergraph_feedback_scale = 1.0
        self.base_detach_hypergraph_site_probabilities = bool(detach_hypergraph_site_probabilities)
        self.detach_hypergraph_site_probabilities = self.base_detach_hypergraph_site_probabilities
        self.hypergraph_site_probabilities_detach_until_epoch = max(
            int(hypergraph_site_probabilities_detach_until_epoch), 0
        )
        self.training_stage = 2
        self.training_stage_name = "end_to_end"
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
            gate_init = min(max(float(tool_annotation_gate_init), 1e-4), 1.0 - 1e-4)
            nn.init.zeros_(self.annotation_residual_gate.weight)
            nn.init.constant_(self.annotation_residual_gate.bias, math.log(gate_init / (1.0 - gate_init)))
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
        self.layer_fusion = LayerwiseRepresentationFusion(hidden_dim, dropout=dropout) if use_layerwise_fusion else None
        self.pool = LevelAttentionPool(hidden_dim, dropout=dropout)
        self.parent_context = nn.ModuleDict()
        self.parent_residual_heads = nn.ModuleDict()
        self.parent_residual_gates = nn.ModuleDict()
        self.heads = nn.ModuleDict()
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
        # Site predictors supply the soft memberships and their supervision.
        self.site_heads = nn.ModuleDict(
            {task: nn.Linear(hidden_dim, 1) for task in self.hypergraph_tasks}
        )

    def set_training_epoch(self, epoch: int) -> float:
        """Warm up functional hyperedges and tool annotation encoding."""
        self.hypergraph_feedback_scale = (
            _feature_curriculum_scale(epoch, self.hypergraph_start_epoch, self.hypergraph_warmup_epochs)
            if self.function_hypergraph is not None else 0.0
        )
        self.detach_hypergraph_site_probabilities = bool(
            self.base_detach_hypergraph_site_probabilities
            or epoch <= self.hypergraph_site_probabilities_detach_until_epoch
        )
        if self.function_hypergraph is not None:
            self.function_hypergraph.set_feedback_scale(self.hypergraph_feedback_scale)
        self.tool_annotation_scale = (
            _feature_curriculum_scale(epoch, self.tool_annotation_start_epoch, self.tool_annotation_warmup_epochs)
            if self.use_annotation_features else 0.0
        )
        warming_up = (
            (self.function_hypergraph is not None and self.hypergraph_feedback_scale < 1.0)
            or (self.use_annotation_features and self.tool_annotation_scale < 1.0)
        )
        self.training_stage = 1 if warming_up else 2
        self.training_stage_name = "feature_warmup" if warming_up else "end_to_end"
        return self.hypergraph_feedback_scale

    def _encode_annotations(self, data, residue_repr, batch):
        annotation_x = getattr(data, "annotation_x", None)
        annotation_mask = getattr(data, "annotation_mask", None)
        shape = (residue_repr.size(0), self.annotation_dim)
        annotation_x = (
            residue_repr.new_zeros(shape) if annotation_x is None
            else annotation_x.to(device=residue_repr.device, dtype=residue_repr.dtype)
        )
        annotation_mask = (
            torch.zeros_like(annotation_x) if annotation_mask is None
            else annotation_mask.to(device=residue_repr.device, dtype=residue_repr.dtype)
        )
        if annotation_x.shape != shape or annotation_mask.shape != shape:
            raise ValueError(f"annotation_x and annotation_mask must have shape {shape}")
        annotation_mask = annotation_mask.clamp(0.0, 1.0)
        annotation_x = torch.where(annotation_mask > 0, annotation_x, torch.zeros_like(annotation_x))
        available = annotation_mask.ne(0).any(dim=-1, keepdim=True)
        scale = float(self.tool_annotation_scale)
        diagnostics = {"availability": available.to(residue_repr.dtype)}
        if scale <= 0.0 or not bool(available.any()):
            return residue_repr, {
                "annotation_availability": available,
                "tool_annotation_gate": residue_repr.new_zeros((residue_repr.size(0), 1)),
                "tool_annotation_diagnostics": diagnostics,
            }
        annotation_repr = self.annotation_encoder(torch.cat([annotation_x, annotation_mask], dim=-1))
        active = available
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        if self.training and self.tool_annotation_dropout > 0.0:
            graph_dropped = torch.rand(num_graphs, device=batch.device) < self.tool_annotation_dropout
            active = active & ~graph_dropped[batch].unsqueeze(-1)
        active_float = active.to(annotation_repr.dtype)
        graph_sum = annotation_repr.new_zeros((num_graphs, annotation_repr.size(-1)))
        graph_count = annotation_repr.new_zeros((num_graphs, 1))
        graph_sum.index_add_(0, batch, annotation_repr * active_float)
        graph_count.index_add_(0, batch, active_float)
        graph_repr = self.annotation_graph_encoder(graph_sum / graph_count.clamp_min(1.0))[batch]
        hit_signal = (
            annotation_x[:, 3:5].amax(dim=-1, keepdim=True)
            if self.annotation_dim >= 5 else annotation_x.amax(dim=-1, keepdim=True)
        )
        gate = torch.sigmoid(self.annotation_residual_gate(
            torch.cat([residue_repr, annotation_repr, graph_repr, hit_signal], dim=-1)
        ))
        correction = (
            self.tool_annotation_residual_scale * scale * active_float * gate
            * self.annotation_residual_adapter(torch.cat([annotation_repr, graph_repr], dim=-1))
        )
        nan_values = torch.full_like(gate, float("nan"))
        diagnostics.update({
            "scale_on_available": torch.where(available, torch.full_like(gate, scale), nan_values),
            "gate_on_available": torch.where(available, gate, nan_values),
            "hit_signal_on_available": torch.where(available, hit_signal, nan_values),
            "correction_norm_on_available": torch.where(
                available, correction.norm(dim=-1, keepdim=True) / math.sqrt(correction.size(-1)), nan_values
            ),
        })
        return residue_repr + correction, {
            "annotation_availability": available,
            "tool_annotation_gate": gate * active_float,
            "tool_annotation_diagnostics": diagnostics,
        }

    def forward(self, data):
        sequence_repr = self.input_proj(data.x.float())
        edge_index = data.edge_index.to(device=sequence_repr.device, dtype=torch.long)
        raw_edge_attr = data.edge_attr.to(device=sequence_repr.device, dtype=sequence_repr.dtype)
        edge_attr = self.edge_encoder(raw_edge_attr)
        batch = getattr(data, "batch", torch.zeros(sequence_repr.size(0), dtype=torch.long, device=sequence_repr.device))
        batch = batch.to(device=sequence_repr.device, dtype=torch.long)
        intermediate_sites = None
        hypergraph_memberships = None
        hypergraph_gate = None
        hypergraph_diagnostics = None
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
            late_outputs = self.backbone.forward_layers(
                feedback_repr,
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
        for name, head in self.site_heads.items():
            output[name] = head(structure_repr).view(-1)
        residue_repr = structure_repr
        if self.use_annotation_features:
            residue_repr, annotation_outputs = self._encode_annotations(data, residue_repr, batch)
            output.update(annotation_outputs)
        pooled, weights = self.pool(residue_repr, batch)
        parent_conditioning_gates = {}
        for level in EC_LEVELS:
            head_input = pooled
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
                    [pooled, parent_context], dim=-1
                )
                parent_gate = torch.sigmoid(
                    self.parent_residual_gates[level](residual_input)
                )
                raw_logits = raw_logits + parent_gate * self.parent_residual_heads[level](
                    residual_input
                )
                parent_conditioning_gates[level] = parent_gate
            output[level] = raw_logits
        if parent_conditioning_gates:
            output["parent_conditioning_gates"] = parent_conditioning_gates
        output["attention"] = {level: weights for level in EC_LEVELS}
        if layer_weights is not None:
            output["layer_attention"] = layer_weights
        return output
