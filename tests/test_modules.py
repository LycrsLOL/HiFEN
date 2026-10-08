from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from hifen.cli import build_parser
from hifen.config import load_config
from hifen.data.graph_builder import SITE_TASKS, toy_thin_graph
from hifen.data.graph_preparation import _annotation_features, _direct_structure_path, _select_structure
from hifen.deployment.structure_fusion import complete_structure_if_needed
from hifen.deployment.workflow import configure_one_click_deployment
from hifen.models import HiFENModel
from hifen.models.functional_graph import SoftFunctionalHypergraph
from hifen.training.trainer import (
    DistributedContext,
    _collect_validation_outputs,
    _load_validation_payload,
    _make_loss,
    _make_model,
    _save_validation_payload,
    _train_one_epoch,
)


LEVELS = ("ec1", "ec2", "ec3", "ec4")
CONFIG = Path(__file__).resolve().parents[1] / "configs/hifen.yaml"
PROFILES = (
    "full", "without_layerwise_fusion",
    "without_parent_conditioning", "without_tool_annotations", "without_structure_completion",
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def small_config(profile="full"):
    config = load_config(CONFIG, profile=profile)
    config["model"].update(input_dim=16, hidden_dim=16, heads=4, num_layers=3, dropout=0.0)
    config["model"]["tool_annotation_dropout"] = 0.0
    return config


def label_maps():
    return {level: {f"{level}_{i}": i for i in range(3)} for level in LEVELS}


def graph_batch():
    graphs = []
    for i in range(2):
        graph = toy_thin_graph(num_nodes=6 + i, embedding_dim=16)
        graph.annotation_x[:, 3] = 1.0
        graph.annotation_mask[:, 3] = 1.0
        for level in LEVELS:
            setattr(graph, f"y_{level}", torch.tensor([1.0, 0.0, 0.0]))
            setattr(graph, f"y_{level}_mask", torch.ones(3))
        graph.y_site = torch.zeros((graph.num_nodes, len(SITE_TASKS)))
        graph.y_site[0, :2] = 1.0
        graph.y_site_mask = torch.ones_like(graph.y_site)
        graphs.append(graph)
    return Batch.from_data_list(graphs)


@pytest.mark.parametrize("profile", PROFILES)
def test_profiles_forward_backward_and_module_switches(profile):
    config = small_config(profile)
    model = _make_model(config, label_maps())
    model.set_training_epoch(20)
    outputs = model(graph_batch())
    assert all(outputs[level].shape == (2, 3) for level in LEVELS)
    assert all(torch.isfinite(outputs[level]).all() for level in LEVELS)
    loss = _make_loss(config)(outputs, graph_batch())
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert model.function_hypergraph is None
    assert len(model.intermediate_site_heads) == len(model.site_heads) == 0
    assert not any("hypergraph" in name or "_site_heads" in name for name, _ in model.named_parameters())
    assert "intermediate_site_logits" not in outputs
    assert not any(name in outputs for name in ("active_or_catalytic_site", "binding_site", "hypergraph_memberships"))
    assert config["loss"]["auxiliary_site_weight"] == config["loss"]["intermediate_site_weight"] == 0.0
    assert (model.layer_fusion is None) == (profile == "without_layerwise_fusion")
    assert model.use_parent_conditioning == (profile != "without_parent_conditioning")
    assert model.use_annotation_features == (profile != "without_tool_annotations")
    assert config["deployment"]["allow_completion"] == (profile != "without_structure_completion")


def test_complete_network_components():
    model = _make_model(small_config(), label_maps())
    assert set(dict(model.named_children())) == {
        "input_proj", "edge_encoder", "backbone", "pool", "heads",
        "intermediate_site_heads", "site_heads", "layer_fusion",
        "annotation_encoder", "annotation_graph_encoder", "annotation_residual_adapter",
        "annotation_residual_gate", "parent_context", "parent_residual_heads", "parent_residual_gates",
    }


def test_legacy_opt_in_functional_hyperedges_change_ec_predictions_and_receive_ec_gradients():
    config = small_config()
    config["model"]["use_function_hypergraph"] = True
    model = _make_model(config, label_maps()).eval()
    batch = graph_batch()
    with torch.no_grad():
        for head in model.intermediate_site_heads.values():
            head.weight.zero_()
            head.bias.fill_(1.0)
    enabled = model(batch)
    model.set_training_epoch(1)
    disabled = model(batch)
    assert not torch.allclose(enabled["ec4"], disabled["ec4"], atol=1e-7, rtol=1e-7)
    model.set_training_epoch(20)
    model(batch)["ec4"].sum().backward()
    grad = model.function_hypergraph.hyperedge_encoder[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0


@pytest.mark.parametrize("layers", [1, 2, 4])
def test_layerwise_attention_and_single_layer_execution(layers):
    config = small_config()
    config["model"]["num_layers"] = layers
    model = _make_model(config, label_maps()).eval()
    output = model(graph_batch())
    weights = output["layer_attention"]
    assert weights.shape == (13, layers)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(13))


@pytest.mark.parametrize("detach", [False, True])
def test_parent_conditioning_gradient_control(detach):
    config = small_config()
    config["model"].update(detach_parent_conditioning=detach, parent_conditioning_mode="concat")
    model = _make_model(config, label_maps()).eval()
    model(graph_batch())["ec4"].sum().backward()
    grad = model.heads["ec3"][-1].weight.grad
    if detach:
        assert grad is None
    else:
        assert grad is not None and grad.abs().sum() > 0


def test_annotation_encoding_handles_missing_and_masked_values():
    model = _make_model(small_config(), label_maps()).eval()
    with torch.no_grad():
        model.annotation_residual_adapter[-1].weight.normal_(std=0.1)
    batch = graph_batch()
    batch.annotation_mask.zero_()
    batch.annotation_x.normal_()
    missing = model(batch)["ec4"]
    batch.annotation_x.fill_(100.0)
    torch.testing.assert_close(missing, model(batch)["ec4"])
    del batch.annotation_x
    del batch.annotation_mask
    torch.testing.assert_close(missing, model(batch)["ec4"])


def test_tool_annotations_change_ec_predictions():
    model = _make_model(small_config(), label_maps()).eval()
    with torch.no_grad():
        model.annotation_residual_adapter[-1].weight.normal_(std=0.1)
    batch = graph_batch()
    with_tool = model(batch)["ec4"]
    batch.annotation_mask.zero_()
    assert not torch.allclose(with_tool, model(batch)["ec4"])


def test_tool_annotation_features_use_domain_and_motif_ranges():
    row = {
        "domain_ranges": "domain:2-4", "motif_ranges": "motif:5-6",
        "domain_source": "test_tool", "active_sites": "1", "binding_sites": "7",
    }
    values, mask = _annotation_features(row, 7)
    assert values[:, 3].tolist() == [0, 1, 1, 1, 0, 0, 0]
    assert values[:, 4].tolist() == [0, 0, 0, 0, 1, 1, 0]
    assert torch.count_nonzero(values[:, :3]) == 0
    assert torch.count_nonzero(mask[:, :3]) == 0


def test_feature_warmup():
    model = _make_model(small_config(), label_maps())
    assert model.set_training_epoch(1) == 0
    assert model.tool_annotation_scale == 0
    model.set_training_epoch(5)
    assert model.tool_annotation_scale == 1
    assert model.set_training_epoch(8) == 0
    assert model.set_training_epoch(15) == 0
    assert model.training_stage_name == "end_to_end"


def test_direct_constructor_defaults_to_four_module_model():
    model = HiFENModel(input_dim=16, hidden_dim=16, heads=4, num_classes={level: 3 for level in LEVELS})
    assert model.function_hypergraph is None
    assert not model.use_function_hypergraph
    assert len(model.site_heads) == len(model.intermediate_site_heads) == 0


def test_soft_hyperedges_zero_support_and_fractional_confidence_have_finite_gradients():
    module = SoftFunctionalHypergraph(
        8, 2, slots_per_task=2, preserve_site_confidence=True,
        confidence_power=0.5, site_probability_threshold=0.2,
    )
    x = torch.randn(5, 8, requires_grad=True)
    probabilities = torch.tensor([[0.0, 0.1], [0.2, 0.0], [0.7, 0.6], [0.3, 0.4], [0.9, 0.8]], requires_grad=True)
    output, memberships, _, _ = module(x, probabilities, torch.zeros(5, dtype=torch.long))
    torch.testing.assert_close(output[:2], x[:2])
    assert memberships[:2].count_nonzero() == 0
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(probabilities.grad).all()


def test_structure_selection_respects_completion_switch(tmp_path):
    paths = {}
    for resource in ("complete", "crystal", "predicted"):
        directory = tmp_path / resource
        directory.mkdir()
        path = directory / ("P00001_1abc.pdb" if resource != "predicted" else "P00001.pdb")
        path.write_text("HEADER test\n")
        paths[resource] = str(path)
    assert _direct_structure_path(tmp_path, "P00001", "1abc") == paths["complete"]
    assert _direct_structure_path(tmp_path, "P00001", "1abc", allow_completed=False) == paths["crystal"]
    exact = {("train", r, "P00001", "1abc"): [p] for r, p in paths.items()}
    assert _select_structure("train", "P00001", "1abc", exact, {}) == paths["complete"]
    assert _select_structure("train", "P00001", "1abc", exact, {}, allow_completed=False) == paths["crystal"]


def test_one_click_deployment_preserves_completion_profile(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("uniprot_id,sequence,ec_numbers\nP00001,AGSTV,1.1.1.1\n")
    config = configure_one_click_deployment(
        small_config("without_structure_completion"), train_csv=csv_path, output_dir=tmp_path / "out"
    )
    assert config["deployment"]["allow_completion"] is False
    assert config["deployment"]["pipeline_graph_after_completion"] is False


def test_structure_completion_fills_missing_residue(tmp_path):
    sequence = "AGSTV"
    names = ["ALA", "GLY", "SER", "THR", "VAL"]
    coords = [(0, 0, 0), (3, 0, 0), (3, 3, 0), (0, 3, 0), (0, 3, 3)]

    def write_pdb(path, positions):
        lines = []
        for serial, position in enumerate(positions, 1):
            x, y, z = coords[position - 1]
            lines.append(
                f"ATOM  {serial:5d}  CA  {names[position - 1]} A{position:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{90.0:6.2f}           C\n"
            )
        path.write_text("".join(lines) + "TER\nEND\n")

    crystal, predicted, output = (tmp_path / name for name in ("crystal.pdb", "predicted.pdb", "complete.pdb"))
    write_pdb(crystal, [1, 2, 3, 4])
    write_pdb(predicted, [1, 2, 3, 4, 5])
    result = complete_structure_if_needed(
        crystal, predicted, output, allow_completion=True,
        full_sequence=sequence, min_anchor_residues=3,
    )
    assert result is not None
    assert result.crystal_residues == 4 and result.predicted_residues == 1
    assert output.exists()


def test_training_and_validation_payload_roundtrip(tmp_path):
    config = small_config()
    model = _make_model(config, label_maps())
    model.set_training_epoch(20)
    criterion = _make_loss(config)
    batch = graph_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = _train_one_epoch(
        model, [batch], criterion, optimizer, torch.device("cpu"),
        scaler=torch.amp.GradScaler("cuda", enabled=False), amp=False,
        grad_clip_norm=0.5, parameter_norm_guard=None, distributed=DistributedContext(enabled=False),
        epoch=1, total_epochs=1, log_interval=0,
    )
    assert np.isfinite(loss)
    payload = _collect_validation_outputs(
        model, [batch], criterion, torch.device("cpu"), DistributedContext(enabled=False)
    )
    path = tmp_path / "validation.npz"
    _save_validation_payload(path, payload, compressed=False)
    loaded = _load_validation_payload(path)
    for level in LEVELS:
        np.testing.assert_allclose(loaded["probs"][level], payload["probs"][level])
    assert not loaded["hypergraph_diagnostic_counts"]
    assert loaded["tool_annotation_diagnostic_counts"]


@pytest.mark.parametrize("command", ["train", "deploy-data", "evaluate", "infer", "export-reproducibility"])
def test_cli_uses_shared_model_config(command):
    args = build_parser().parse_args([command])
    assert args.config == "configs/hifen.yaml"
