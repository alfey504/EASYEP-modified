import pytest
import torch

from moe_pruning.model.architecture import get_adapter, select_layers
from moe_pruning.model.instrumentation import RecordCollector, gate_consistency_error
from tests.conftest import REAL_MODEL_ID, make_tiny_model, requires_real_model


def test_discovers_encoder_and_decoder_layers(tiny_model):
    adapter = get_adapter(tiny_model)
    layers = adapter.discover_layers(tiny_model)
    assert [layer.layer_id for layer in layers] == [
        "encoder.block.1",
        "encoder.block.3",
        "decoder.block.1",
        "decoder.block.3",
    ]
    assert {layer.stack for layer in layers} == {"encoder", "decoder"}
    for layer in layers:
        assert layer.num_experts == 4
        assert layer.expert_input_dim == layer.expert_output_dim == 16
        assert layer.expert_intermediate_dim == 32
        assert layer.top_k == 1


def test_select_layers(tiny_model):
    layers = get_adapter(tiny_model).discover_layers(tiny_model)
    assert select_layers(layers, "first") == [layers[0]]
    assert select_layers(layers, "all") == layers
    assert select_layers(layers, "decoder.block.3,encoder.block.1") == [layers[0], layers[3]]
    with pytest.raises(ValueError):
        select_layers(layers, "encoder.block.2")


def test_records_match_layer_output(tiny_model, tiny_batch):
    adapter = get_adapter(tiny_model)
    layers = adapter.discover_layers(tiny_model)
    collector = RecordCollector()
    with torch.no_grad(), adapter.instrument(tiny_model, layers, collector):
        tiny_model(**tiny_batch)
    by_layer = collector.by_layer()
    assert set(by_layer) == {layer.layer_id for layer in layers}
    for records in by_layer.values():
        rec = records[0]
        assert gate_consistency_error(rec)["max_abs_error"] < 1e-6
        # MoE output is exactly the residual update of the feed-forward sub-layer
        assert rec.expert_output.shape[-2:] == (1, 16)


def test_expert_output_matches_direct_evaluation(tiny_model, tiny_batch):
    """Recorded E_e(x_t) must equal the expert applied directly to that token."""
    adapter = get_adapter(tiny_model)
    layer = adapter.discover_layers(tiny_model)[0]
    collector = RecordCollector()
    with torch.no_grad(), adapter.instrument(tiny_model, [layer], collector):
        tiny_model(**tiny_batch)
    rec = collector.records[0]
    x = rec.moe_input.reshape(-1, 16)
    idx = rec.expert_index.reshape(-1)
    out = rec.expert_output.reshape(-1, 16)
    with torch.no_grad():
        for t in range(x.shape[0]):
            if idx[t] >= 0:
                direct = adapter.get_expert(tiny_model, layer, int(idx[t]))(x[t : t + 1])
                torch.testing.assert_close(out[t : t + 1], direct)


def test_capacity_drop_is_recorded():
    model = make_tiny_model(expert_capacity=1)
    adapter = get_adapter(model)
    layer = adapter.discover_layers(model)[0]
    x = torch.randn(1, 12, 16)
    index, gate = adapter.route(model, layer, x)
    # 12 tokens, 4 experts, capacity 1 => at least 8 tokens dropped
    assert int((index < 0).sum()) >= 8
    assert torch.all(gate[index < 0] == 0)
    y = adapter.run_moe_layer(model, layer, x)
    dropped = (index[..., 0] < 0)[0]
    assert torch.all(y[0, dropped] == 0)


def test_set_expert_keeps_slot_and_router(tiny_model):
    adapter = get_adapter(tiny_model)
    layer = adapter.discover_layers(tiny_model)[0]
    router_before = {k: v.clone() for k, v in adapter.get_router(tiny_model, layer).state_dict().items()}
    replacement = torch.nn.Identity()
    old = adapter.set_expert(tiny_model, layer, 2, replacement)
    assert adapter.get_expert(tiny_model, layer, 2) is replacement
    assert old is not replacement
    for k, v in adapter.get_router(tiny_model, layer).state_dict().items():
        torch.testing.assert_close(v, router_before[k])
    assert adapter.expert_state_prefix(layer, 2) == "encoder.block.1.layer.1.mlp.experts.expert_2."


@pytest.mark.integration
@requires_real_model
def test_real_switch_base_8_architecture():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained(REAL_MODEL_ID).eval()
    tok = AutoTokenizer.from_pretrained(REAL_MODEL_ID)
    adapter = get_adapter(model)
    layers = adapter.discover_layers(model)
    assert len(layers) == 12
    assert sum(layer.stack == "encoder" for layer in layers) == 6
    assert all(layer.num_experts == 8 and layer.hidden_dim == 768 for layer in layers)
    assert all(layer.expert_intermediate_dim == 3072 for layer in layers)

    enc = tok("Where can I find the refund policy?", return_tensors="pt")
    labels = tok(text_target="On the help center page.", return_tensors="pt").input_ids
    collector = RecordCollector()
    with torch.no_grad(), adapter.instrument(model, layers, collector):
        model(**enc, labels=labels)
    for rec in collector.records:
        assert gate_consistency_error(rec)["max_rel_error"] < 1e-5
