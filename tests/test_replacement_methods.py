"""Tests for the drop and reroute replacement methods."""

import pytest
import torch

from moe_pruning.model.architecture import get_adapter
from moe_pruning.pruning.replacement import replace_experts
from moe_pruning.pruning.zero_expert import ZeroExpert
from moe_pruning.serialization.checkpoint import load_pruned_model, save_pruned_model
from tests.conftest import make_tiny_model

LAYER = "encoder.block.1"
REMOVED = [0, 2]


def _setup(**overrides):
    model = make_tiny_model(seed=5, **overrides)
    adapter = get_adapter(model)
    layers = {layer.layer_id: layer for layer in adapter.discover_layers(model)}
    # Sharper router so that tokens spread over experts with distinct gates.
    with torch.no_grad():
        adapter.get_router(model, layers[LAYER]).classifier.weight.mul_(4.0)
    return model, adapter, layers


def _manual_routing(model, adapter, layer, x):
    probs = torch.softmax(adapter.get_router(model, layer).classifier(x).float(), dim=-1)
    return probs


def test_zero_expert():
    z = ZeroExpert(5)
    out = z(torch.randn(3, 7))
    assert out.shape == (3, 5) and torch.all(out == 0)
    assert sum(p.numel() for p in z.parameters()) == 0
    strict = ZeroExpert(5, must_be_unused=True)
    assert strict(torch.zeros(0, 7)).shape == (0, 5)
    with pytest.raises(RuntimeError):
        strict(torch.zeros(2, 7))


def test_drop_zeroes_removed_experts_only():
    model, adapter, layers = _setup()
    layer = layers[LAYER]
    x = torch.randn(2, 20, 16)
    with torch.no_grad():
        idx_before, gate_before = adapter.route(model, layer, x)
        y_before = adapter.run_moe_layer(model, layer, x)
    report = replace_experts(model, adapter, layers, {LAYER: REMOVED}, "drop")
    with torch.no_grad():
        idx_after, gate_after = adapter.route(model, layer, x)
        y_after = adapter.run_moe_layer(model, layer, x)
    torch.testing.assert_close(idx_after, idx_before)  # router untouched, no mask
    torch.testing.assert_close(gate_after, gate_before)
    removed = torch.isin(idx_before[..., 0], torch.tensor(REMOVED))
    assert removed.any() and (~removed).any()
    assert torch.all(y_after[removed] == 0)
    torch.testing.assert_close(y_after[~removed], y_before[~removed], rtol=0, atol=0)
    assert report.removed_parameters == 2 * 2 * 16 * 32 and report.added_parameters == 0
    assert "encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight" not in model.state_dict()


def test_reroute_sends_tokens_to_best_remaining_expert():
    model, adapter, layers = _setup()
    layer = layers[LAYER]
    x = torch.randn(2, 20, 16)
    with torch.no_grad():
        idx_before, gate_before = adapter.route(model, layer, x)
        y_before = adapter.run_moe_layer(model, layer, x)
        probs = _manual_routing(model, adapter, layer, x)
    replace_experts(model, adapter, layers, {LAYER: REMOVED}, "reroute")
    assert adapter.get_routing_mask(model, layer) == REMOVED
    with torch.no_grad():
        idx_after, gate_after = adapter.route(model, layer, x)
        y_after = adapter.run_moe_layer(model, layer, x)  # ZeroExpert(must_be_unused) would raise if reached
    i0, i1, g1 = idx_before[..., 0], idx_after[..., 0], gate_after[..., 0]
    removed = torch.isin(i0, torch.tensor(REMOVED))
    assert removed.any()
    assert not torch.isin(i1, torch.tensor(REMOVED)).any()
    # Unaffected tokens: identical expert, gate and output.
    torch.testing.assert_close(i1[~removed], i0[~removed])
    torch.testing.assert_close(g1[~removed], gate_before[..., 0][~removed], rtol=0, atol=0)
    # Same expert and gate; the expert now also sees rerouted tokens in its batch,
    # so the matmul may round differently at float32 precision.
    torch.testing.assert_close(y_after[~removed], y_before[~removed], rtol=1e-5, atol=1e-5)
    # Affected tokens: best remaining expert, gate = its original probability.
    masked = probs.clone()
    masked[..., REMOVED] = -1
    torch.testing.assert_close(i1[removed], masked.argmax(-1)[removed])
    torch.testing.assert_close(g1[removed], probs.gather(-1, i1.unsqueeze(-1))[..., 0][removed])
    assert torch.all(g1[removed] <= gate_before[..., 0][removed])
    # Ignoring the mask reproduces the original routing.
    with torch.no_grad():
        idx_orig, gate_orig = adapter.route(model, layer, x, apply_mask=False)
    torch.testing.assert_close(idx_orig, idx_before)
    torch.testing.assert_close(gate_orig, gate_before)


def test_reroute_respects_capacity():
    model, adapter, layers = _setup(expert_capacity=3)
    layer = layers[LAYER]
    replace_experts(model, adapter, layers, {LAYER: REMOVED}, "reroute")
    x = torch.randn(1, 16, 16)
    with torch.no_grad():
        idx, gate = adapter.route(model, layer, x)
        adapter.run_moe_layer(model, layer, x)
    counts = torch.bincount(idx[idx >= 0].flatten(), minlength=4)
    assert counts.max() <= 3 and counts[REMOVED].sum() == 0
    assert torch.all(gate[idx < 0] == 0)


def test_empty_mask_is_identity():
    model, adapter, layers = _setup()
    layer = layers[LAYER]
    x = torch.randn(2, 12, 16)
    with torch.no_grad():
        before = adapter.route(model, layer, x)
        adapter.set_routing_mask(model, layer, [1])
        adapter.set_routing_mask(model, layer, [])
        after = adapter.route(model, layer, x)
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["drop", "reroute"])
def test_checkpoint_roundtrip(method, tmp_path, tiny_tokenizer, tiny_batch):
    model, adapter, layers = _setup()
    replace_experts(model, adapter, layers, {LAYER: REMOVED, "decoder.block.3": [1]}, method)
    manifest = {
        "model_source": "tiny",
        "replacement_method": method,
        "layers": {
            lid: {
                "replacement_method": method,
                "expert_ids": ids,
                "expert_state_prefixes": {str(e): adapter.expert_state_prefix(layers[lid], e) for e in ids},
            }
            for lid, ids in {LAYER: REMOVED, "decoder.block.3": [1]}.items()
        },
    }
    save_pruned_model(model, tiny_tokenizer, tmp_path / "ckpt", manifest, {})
    reloaded, _, _ = load_pruned_model(tmp_path / "ckpt", "cpu")
    rl = {layer.layer_id: layer for layer in adapter.discover_layers(reloaded)}
    assert isinstance(adapter.get_expert(reloaded, rl[LAYER], 0), ZeroExpert)
    expected_mask = REMOVED if method == "reroute" else []
    assert adapter.get_routing_mask(reloaded, rl[LAYER]) == expected_mask
    with torch.no_grad():
        torch.testing.assert_close(reloaded(**tiny_batch).logits, model(**tiny_batch).logits, rtol=1e-5, atol=1e-6)
