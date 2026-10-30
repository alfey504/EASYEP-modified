import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from moe_pruning.model.architecture import get_adapter
from moe_pruning.pruning.bias_expert import BiasExpert
from moe_pruning.pruning.replacement import replace_experts_with_biases
from moe_pruning.serialization.checkpoint import (
    MANIFEST_NAME,
    WEIGHTS_NAME,
    CheckpointError,
    load_pruned_model,
    save_pruned_model,
)
from moe_pruning.utils.paths import read_json, write_json
from tests.conftest import make_tiny_model

SELECTION = {"encoder.block.1": [0, 2], "decoder.block.3": [1]}


def _pruned_model():
    model = make_tiny_model(seed=7)
    adapter = get_adapter(model)
    layers = {layer.layer_id: layer for layer in adapter.discover_layers(model)}
    replace_experts_with_biases(
        model, adapter, layers,
        {lid: {e: torch.randn(16) for e in experts} for lid, experts in SELECTION.items()},
    )
    manifest = {
        "model_source": "tiny",
        "layers": {
            lid: {
                "expert_ids": experts,
                "expert_state_prefixes": {str(e): adapter.expert_state_prefix(layers[lid], e) for e in experts},
            }
            for lid, experts in SELECTION.items()
        },
    }
    return model, manifest


@pytest.fixture
def saved(tmp_path, tiny_tokenizer):
    model, manifest = _pruned_model()
    save_pruned_model(model, tiny_tokenizer, tmp_path / "ckpt", manifest, {"note": "test"})
    return model, tmp_path / "ckpt"


def test_removed_expert_weights_absent(saved):
    _, path = saved
    with safe_open(str(path / WEIGHTS_NAME), framework="pt") as fh:
        keys = set(fh.keys())
    for lid, experts in SELECTION.items():
        block, idx = lid.split(".block.")
        sub = 1 if block == "encoder" else 2
        for e in experts:
            prefix = f"{block}.block.{idx}.layer.{sub}.mlp.experts.expert_{e}."
            assert prefix + "bias" in keys
            assert not any(k.startswith(prefix) and not k.endswith(".bias") for k in keys)
    # A retained expert is stored.
    assert "encoder.block.1.layer.1.mlp.experts.expert_1.wi.weight" in keys
    assert "encoder.block.1.layer.1.mlp.router.classifier.weight" in keys


def test_loader_restores_architecture_and_outputs(saved, tiny_batch):
    model, path = saved
    reloaded, tokenizer, manifest = load_pruned_model(path, "cpu")
    adapter = get_adapter(reloaded)
    layers = {layer.layer_id: layer for layer in adapter.discover_layers(reloaded)}
    for lid, experts in SELECTION.items():
        for e in range(4):
            module = adapter.get_expert(reloaded, layers[lid], e)
            assert isinstance(module, BiasExpert) == (e in experts)
    torch.testing.assert_close(reloaded.lm_head.weight, reloaded.shared.weight)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(**tiny_batch).logits, model(**tiny_batch).logits, rtol=1e-5, atol=1e-6)
    assert manifest["tied_aliases"]
    assert tokenizer.pad_token == "<pad>"


def test_loader_rejects_unexpected_original_weights(saved):
    _, path = saved
    tensors = load_file(str(path / WEIGHTS_NAME))
    tensors["encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight"] = torch.zeros(32, 16)
    save_file(tensors, str(path / WEIGHTS_NAME))
    with pytest.raises(CheckpointError, match="unexpected"):
        load_pruned_model(path, "cpu")


def test_missing_weights_only_allowed_for_manifest_experts(saved):
    _, path = saved
    manifest = read_json(path / MANIFEST_NAME)
    # Pretend expert 2 was not replaced: its original weights are then required but absent.
    manifest["layers"]["encoder.block.1"]["expert_ids"] = [0]
    write_json(path / MANIFEST_NAME, manifest)
    with pytest.raises(CheckpointError, match="missing"):
        load_pruned_model(path, "cpu")


def test_save_refuses_original_weights_of_replaced_experts(tmp_path, tiny_tokenizer):
    model, manifest = _pruned_model()
    manifest["layers"]["encoder.block.1"]["expert_ids"].append(3)
    manifest["layers"]["encoder.block.1"]["expert_state_prefixes"]["3"] = (
        "encoder.block.1.layer.1.mlp.experts.expert_3."
    )
    with pytest.raises(CheckpointError):
        save_pruned_model(model, tiny_tokenizer, tmp_path / "bad", manifest, {})
