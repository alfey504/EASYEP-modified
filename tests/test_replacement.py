import pytest
import torch

from moe_pruning.model.architecture import get_adapter
from moe_pruning.model.instrumentation import RecordCollector
from moe_pruning.pruning.bias_expert import BiasExpert
from moe_pruning.pruning.replacement import freeze_all_but_biases, replace_experts_with_biases
from tests.conftest import make_tiny_model


def _setup(dtype=torch.float32):
    model = make_tiny_model().to(dtype)
    adapter = get_adapter(model)
    layers = {layer.layer_id: layer for layer in adapter.discover_layers(model)}
    return model, adapter, layers


def _layer_records(model, adapter, layer, batch):
    collector = RecordCollector()
    with torch.no_grad(), adapter.instrument(model, [layer], collector):
        model(**batch)
    return collector.records[0]


def test_replacement_effects(tiny_batch):
    model, adapter, layers = _setup()
    layer = layers["encoder.block.1"]
    before = _layer_records(model, adapter, layer, tiny_batch)
    router_before = {k: v.clone() for k, v in adapter.get_router(model, layer).state_dict().items()}
    retained_before = {
        e: {k: v.clone() for k, v in adapter.get_expert(model, layer, e).state_dict().items()} for e in (1, 3)
    }
    b0, b2 = torch.randn(16), torch.randn(16)

    report = replace_experts_with_biases(model, adapter, layers, {layer.layer_id: {0: b0, 2: b2}})
    after = _layer_records(model, adapter, layer, tiny_batch)

    # Router unchanged and routing decisions identical (input of the first sparse layer is unchanged).
    for k, v in adapter.get_router(model, layer).state_dict().items():
        torch.testing.assert_close(v, router_before[k], rtol=0, atol=0)
    torch.testing.assert_close(after.expert_index, before.expert_index)
    torch.testing.assert_close(after.gate, before.gate)

    idx = after.expert_index[..., 0]
    g = after.gate[..., 0]
    # Replaced experts: output g * b (gate once), correct shape.
    for e, b in ((0, b0), (2, b2)):
        sel = idx == e
        assert sel.any()
        torch.testing.assert_close(after.moe_output[sel], g[sel, None] * b, rtol=1e-5, atol=1e-6)
        assert not torch.allclose(after.moe_output[sel], before.moe_output[sel])
    # Retained experts: outputs and weights unchanged.
    for e in (1, 3):
        sel = idx == e
        torch.testing.assert_close(after.moe_output[sel], before.moe_output[sel], rtol=0, atol=0)
        for k, v in adapter.get_expert(model, layer, e).state_dict().items():
            torch.testing.assert_close(v, retained_before[e][k], rtol=0, atol=0)
    assert after.moe_output.shape == before.moe_output.shape

    # Original parameters removed, slot count unchanged.
    assert report.removed_parameters == 2 * 2 * 16 * 32
    assert report.added_parameters == 32
    keys = model.state_dict().keys()
    assert "encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight" not in keys
    assert "encoder.block.1.layer.1.mlp.experts.expert_0.bias" in keys
    assert len(adapter.get_experts(model, layer)) == layer.num_experts


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bias_module_preserves_dtype_and_device(dtype):
    model, adapter, layers = _setup(dtype)
    layer = layers["decoder.block.1"]
    replace_experts_with_biases(model, adapter, layers, {layer.layer_id: {1: torch.randn(16)}})
    expert = adapter.get_expert(model, layer, 1)
    assert isinstance(expert, BiasExpert)
    assert expert.bias.dtype == dtype
    assert expert.bias.device == next(model.parameters()).device


def test_double_replacement_rejected():
    model, adapter, layers = _setup()
    replace_experts_with_biases(model, adapter, layers, {"encoder.block.1": {0: torch.zeros(16)}})
    with pytest.raises(ValueError):
        replace_experts_with_biases(model, adapter, layers, {"encoder.block.1": {0: torch.zeros(16)}})


def test_freeze_all_but_biases():
    model, adapter, layers = _setup()
    replace_experts_with_biases(
        model, adapter, layers, {"encoder.block.1": {0: torch.zeros(16)}, "decoder.block.3": {2: torch.zeros(16)}}
    )
    trainable = freeze_all_but_biases(model)
    assert sorted(trainable) == [
        "decoder.block.3.layer.2.mlp.experts.expert_2.bias",
        "encoder.block.1.layer.1.mlp.experts.expert_0.bias",
    ]
    assert {n for n, p in model.named_parameters() if p.requires_grad} == set(trainable)
