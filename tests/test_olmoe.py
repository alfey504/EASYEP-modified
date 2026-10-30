"""OLMoE plugin: adapter, fused-expert pruning, causal task, serialization and training.

Uses a tiny random OLMoE (8 experts, top-2, transformers 5 fused experts), so
it runs on CPU without downloading allenai/OLMoE-1B-7B-0125.
"""

import pytest
import torch

from moe_pruning.data.calibration import CalibrationRecord, TokenizedPair
from moe_pruning.evaluation.evaluator import build_reference_probe, evaluate_model
from moe_pruning.model.architecture import get_adapter
from moe_pruning.model.instrumentation import RecordCollector, gate_consistency_error
from moe_pruning.model.loader import StateDictReader
from moe_pruning.models import get_plugin, plugin_for_model
from moe_pruning.models.olmoe.experts import FusedExpertView, PrunedFusedExperts
from moe_pruning.pruning.bias_expert import BiasExpert
from moe_pruning.pruning.replacement import freeze_all_but_biases, replace_experts
from moe_pruning.scoring.easyep import token_contribution
from moe_pruning.scoring.statistics import ExpertStatisticsAccumulator
from moe_pruning.serialization.checkpoint import (
    WEIGHTS_NAME,
    load_pruned_model,
    removed_weight_violations,
    save_pruned_model,
    stored_tensor_shapes,
)
from moe_pruning.tasks.causal_lm import CausalExample, CausalLMTask
from moe_pruning.training.bias_trainer import capture_layer_references, train_layer_biases
from tests.conftest import make_tiny_olmoe

LAYER = "layers.0"
REMOVED = [1, 4, 6]


def _setup(seed=0):
    model = make_tiny_olmoe(seed)
    adapter = get_adapter(model)
    layers = {layer.layer_id: layer for layer in adapter.discover_layers(model)}
    return model, adapter, layers


def _layer_io(model, adapter, layer, input_ids):
    collector = RecordCollector()
    with torch.no_grad(), adapter.instrument(model, [layer], collector):
        model(input_ids=input_ids)
    return collector.records[0]


def _manifest(adapter, layers, selection, method):
    return {
        "model_source": "tiny-olmoe",
        "replacement_method": method,
        "layers": {
            lid: {
                "replacement_method": method,
                "expert_ids": ids,
                "expert_state_prefixes": {str(e): adapter.expert_state_prefix(layers[lid], e) for e in ids},
            }
            for lid, ids in selection.items()
        },
    }


# ----------------------------------------------------------------------
# Plugin and adapter
# ----------------------------------------------------------------------
def test_plugin_registry(tiny_olmoe):
    plugin = plugin_for_model(tiny_olmoe)
    assert plugin.name == "olmoe" and isinstance(plugin.task(), CausalLMTask)
    assert get_plugin("switch_transformers").name == "switch_transformers"
    with pytest.raises(NotImplementedError, match="No model plugin"):
        get_plugin("not_a_model")


def test_discovery(tiny_olmoe):
    layers = get_adapter(tiny_olmoe).discover_layers(tiny_olmoe)
    assert [layer.layer_id for layer in layers] == ["layers.0", "layers.1"]
    first = layers[0]
    assert first.stack == "decoder" and first.top_k == 2 and first.num_experts == 8
    assert (first.hidden_dim, first.expert_intermediate_dim, first.expert_output_dim) == (16, 32, 16)
    assert first.expert_capacity is None


def test_instrumentation_matches_layer_output(tiny_olmoe):
    adapter = get_adapter(tiny_olmoe)
    layers = adapter.discover_layers(tiny_olmoe)
    collector = RecordCollector()
    with torch.no_grad(), adapter.instrument(tiny_olmoe, layers, collector):
        tiny_olmoe(input_ids=torch.randint(2, 96, (2, 9)))
    assert len(collector.records) == 2
    for rec in collector.records:
        assert rec.expert_index.shape == (2, 9, 2) and rec.expert_output.shape == (2, 9, 2, 16)
        assert gate_consistency_error(rec)["max_rel_error"] < 1e-5
        # Distinct experts per token, gates = original softmax probabilities (not renormalised).
        assert torch.all(rec.expert_index[..., 0] != rec.expert_index[..., 1])
        assert torch.all(rec.gate.sum(-1) < 1.0 + 1e-6)


def test_expert_view_matches_reference_weights(tiny_olmoe):
    adapter = get_adapter(tiny_olmoe)
    layer = adapter.discover_layers(tiny_olmoe)[0]
    experts = tiny_olmoe.model.layers[0].mlp.experts
    x = torch.randn(5, 16)
    view = adapter.get_expert(tiny_olmoe, layer, 3)
    assert isinstance(view, FusedExpertView)
    gate_up, down = experts.gate_up_proj[3], experts.down_proj[3]
    gate, up = (x @ gate_up.T).chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(gate) * up) @ down.T
    torch.testing.assert_close(view(x), expected)


@pytest.mark.parametrize("layout", ["per_expert", "fused"])
def test_load_original_expert_from_both_layouts(tiny_olmoe, layout):
    adapter = get_adapter(tiny_olmoe)
    layer = adapter.discover_layers(tiny_olmoe)[0]
    experts = tiny_olmoe.model.layers[0].mlp.experts
    if layout == "fused":
        state = {f"{layer.experts_path}.gate_up_proj": experts.gate_up_proj.detach(),
                 f"{layer.experts_path}.down_proj": experts.down_proj.detach()}
    else:  # hub layout: model.layers.0.mlp.experts.{e}.gate_proj/up_proj/down_proj.weight
        state = {}
        for e in range(8):
            gate, up = experts.gate_up_proj[e].detach().chunk(2, dim=0)
            prefix = f"{layer.module_path}.experts.{e}."
            state |= {prefix + "gate_proj.weight": gate, prefix + "up_proj.weight": up,
                      prefix + "down_proj.weight": experts.down_proj[e].detach()}
    module = adapter.load_original_expert(StateDictReader(state), tiny_olmoe, layer, 5)
    x = torch.randn(4, 16)
    with torch.no_grad():
        torch.testing.assert_close(module(x), adapter.get_expert(tiny_olmoe, layer, 5)(x))


# ----------------------------------------------------------------------
# Causal task
# ----------------------------------------------------------------------
def test_causal_task_tokenization_and_loss(tiny_olmoe, tiny_tokenizer):
    task = CausalLMTask()
    pairs = task.tokenize_calibration(
        [CalibrationRecord("w3 w4 w5", "w6 w7", 1), CalibrationRecord("w8", "w9 w10 w11", 2)], tiny_tokenizer, 64, 16
    )
    for pair in pairs:
        assert len(pair.input_ids) == len(pair.labels)
        n_prompt = sum(1 for t in pair.labels if t == -100)
        assert n_prompt > 0 and pair.labels[n_prompt:] == pair.input_ids[n_prompt:]
        assert pair.labels[-1] == tiny_tokenizer.eos_token_id
    batch = task.collate(pairs, tiny_tokenizer.pad_token_id)
    assert set(batch.masks) == {"decoder"}
    assert batch.masks["decoder"].sum() == sum(len(p.input_ids) for p in pairs)
    inputs = batch.model_inputs(torch.device("cpu"))
    with torch.no_grad():
        out = tiny_olmoe(**inputs)
    sums, counts = task.token_nll(out.logits, inputs["labels"])
    torch.testing.assert_close(sums.sum() / counts.sum(), out.loss, rtol=1e-5, atol=1e-6)
    assert [len(p) for p in task.predictions(out.logits, inputs["labels"])] == counts.tolist()


def test_statistics_with_top_k_and_padding(tiny_olmoe):
    """Padded causal batch gives the same EASY-EP sums as unpadded sequences; K=2 slots counted."""
    adapter = get_adapter(tiny_olmoe)
    layers = adapter.discover_layers(tiny_olmoe)
    task = CausalLMTask()
    items = [TokenizedPair(0, list(range(5, 15)), list(range(5, 15))), TokenizedPair(1, list(range(20, 26)), list(range(20, 26)))]

    def collect(batches):
        acc = ExpertStatisticsAccumulator(layers)
        current = {}
        with torch.no_grad(), adapter.instrument(tiny_olmoe, layers, lambda r: acc.update(r, current["m"]["decoder"])):
            for batch in batches:
                current["m"] = batch.token_masks()
                tiny_olmoe(**batch.model_inputs(torch.device("cpu")))
        return acc

    padded = collect([task.collate(items, 0)])
    single = collect([task.collate([it], 0) for it in items])
    for layer in layers:
        a, b = padded.stats[layer.layer_id], single.stats[layer.layer_id]
        assert a.layer["tokens"] == b.layer["tokens"] == 16
        assert int(a.expert["count"].sum()) == 16 * 2  # every token has two experts
        for name in ("count", "importance", "w_sum", "w_output_sum"):
            torch.testing.assert_close(a.expert[name], b.expert[name], rtol=1e-4, atol=1e-6)

    # Manual EASY-EP importance for one layer.
    rec = _layer_io(tiny_olmoe, adapter, layers[0], torch.tensor([items[0].input_ids]))
    s = token_contribution(rec.residual, rec.moe_output)
    manual = torch.zeros(8, dtype=torch.float64)
    for t in range(10):
        for k in range(2):
            e = int(rec.expert_index[0, t, k])
            manual[e] += float(rec.gate[0, t, k] * s[0, t] * rec.expert_output[0, t, k].norm())
    acc = ExpertStatisticsAccumulator(layers[:1])
    acc.update(rec, torch.ones(1, 10, dtype=torch.bool))
    torch.testing.assert_close(acc.stats[layers[0].layer_id].expert["importance"], manual, rtol=1e-5, atol=1e-8)


# ----------------------------------------------------------------------
# Replacement on fused experts
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ["bias", "drop", "reroute"])
def test_replacement_methods(method):
    model, adapter, layers = _setup()
    layer = layers[LAYER]
    ids = torch.randint(2, 96, (2, 12))
    before = _layer_io(model, adapter, layer, ids)
    biases = {LAYER: {e: torch.randn(16) for e in REMOVED}}
    report = replace_experts(model, adapter, layers, {LAYER: REMOVED}, method, biases)

    experts = model.model.layers[0].mlp.experts
    assert isinstance(experts, PrunedFusedExperts)
    assert experts.gate_up_proj.shape[0] == 8 - len(REMOVED)
    assert experts.retained_ids() == [0, 2, 3, 5, 7]
    per_expert = 2 * 32 * 16 + 16 * 32
    assert report.removed_parameters == per_expert * len(REMOVED)
    assert report.added_parameters == (16 * len(REMOVED) if method == "bias" else 0)
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    assert adapter.removed_weight_violations(shapes, layer, REMOVED) == []

    after = _layer_io(model, adapter, layer, ids)  # first layer: identical input
    removed = torch.tensor(REMOVED)
    affected = torch.isin(before.expert_index, removed).any(-1)
    assert affected.any() and (~affected).any()
    torch.testing.assert_close(after.moe_output[~affected], before.moe_output[~affected], rtol=1e-5, atol=1e-6)
    hit = torch.isin(before.expert_index, removed).unsqueeze(-1)  # [B,S,K,1]
    g = before.gate.unsqueeze(-1)
    if method == "reroute":
        assert not torch.isin(after.expert_index, removed).any()
        torch.testing.assert_close(after.expert_index[~affected], before.expert_index[~affected])
        torch.testing.assert_close(after.gate[~affected], before.gate[~affected])
        probs = torch.softmax(model.model.layers[0].mlp.gate.weight @ after.moe_input[affected].T, dim=0).T
        masked = probs.clone()
        masked[:, REMOVED] = -1
        torch.testing.assert_close(after.expert_index[affected], masked.topk(2, dim=-1).indices)
        torch.testing.assert_close(after.gate[affected], probs.gather(-1, after.expert_index[affected]))
    else:
        b = torch.zeros(8, 16)
        if method == "bias":
            for e in REMOVED:
                b[e] = biases[LAYER][e]
        # y_new = y_old + sum over removed slots of g * (replacement - E)
        replacement = b[before.expert_index]  # [B,S,K,D]
        delta = (g * (replacement - before.expert_output) * hit).sum(dim=2)
        torch.testing.assert_close(after.moe_output, before.moe_output + delta, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(after.expert_index, before.expert_index)


def test_bias_gate_applied_once_top_k():
    model, adapter, layers = _setup(seed=3)
    layer = layers[LAYER]
    replace_experts(model, adapter, layers, {LAYER: [2]}, "bias", {LAYER: {2: torch.ones(16)}})
    rec = _layer_io(model, adapter, layer, torch.randint(2, 96, (1, 30)))
    slots = rec.expert_index == 2
    assert slots.any()
    torch.testing.assert_close(rec.expert_output[slots], torch.ones(int(slots.sum()), 16))
    assert gate_consistency_error(rec)["max_rel_error"] < 1e-5


def test_compaction_twice_keeps_original_ids():
    model, adapter, layers = _setup()
    layer = layers[LAYER]
    replace_experts(model, adapter, layers, {LAYER: [1]}, "drop")
    replace_experts(model, adapter, layers, {LAYER: [6]}, "drop")
    experts = model.model.layers[0].mlp.experts
    assert experts.removed_ids() == [1, 6] and experts.retained_ids() == [0, 2, 3, 4, 5, 7]
    with pytest.raises(ValueError):
        replace_experts(model, adapter, layers, {LAYER: [1]}, "drop")


# ----------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ["bias", "drop", "reroute"])
def test_checkpoint_roundtrip(method, tmp_path, tiny_tokenizer):
    model, adapter, layers = _setup(seed=2)
    selection = {"layers.0": [1, 4], "layers.1": [0, 3, 7]}
    biases = {lid: {e: torch.randn(16) for e in ids} for lid, ids in selection.items()}
    replace_experts(model, adapter, layers, selection, method, biases)
    manifest = _manifest(adapter, layers, selection, method)
    save_pruned_model(model, tiny_tokenizer, tmp_path / "ckpt", manifest, {})

    shapes = stored_tensor_shapes(tmp_path / "ckpt" / WEIGHTS_NAME)
    assert shapes["model.layers.0.mlp.experts.gate_up_proj"][0] == 6
    assert shapes["model.layers.1.mlp.experts.down_proj"][0] == 5
    assert removed_weight_violations(adapter, model, shapes, manifest) == []

    reloaded, _, _ = load_pruned_model(tmp_path / "ckpt", "cpu")
    assert reloaded.model.layers[1].mlp.experts.retained_ids() == [1, 2, 4, 5, 6]
    if method == "reroute":
        rl = {layer.layer_id: layer for layer in adapter.discover_layers(reloaded)}
        assert adapter.get_routing_mask(reloaded, rl["layers.1"]) == [0, 3, 7]
    ids = torch.randint(2, 96, (2, 10))
    with torch.no_grad():
        torch.testing.assert_close(reloaded(input_ids=ids).logits, model(input_ids=ids).logits, rtol=1e-5, atol=1e-6)


def test_checkpoint_with_full_expert_tensor_is_rejected(tmp_path, tiny_tokenizer):
    model, adapter, layers = _setup()
    replace_experts(model, adapter, layers, {LAYER: [1]}, "drop")
    manifest = _manifest(adapter, layers, {LAYER: [1, 2]}, "drop")  # claims more removed than stored
    with pytest.raises(Exception, match="replaced experts"):
        save_pruned_model(model, tiny_tokenizer, tmp_path / "bad", manifest, {})


# ----------------------------------------------------------------------
# Probe and training
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ["bias", "drop", "reroute"])
def test_reference_probe_top_k(method):
    original, adapter, layers = _setup(seed=5)
    layer = layers[LAYER]
    reader = StateDictReader({k: v.detach().clone() for k, v in original.state_dict().items()})
    task = CausalLMTask()
    ids = [torch.randint(2, 96, (11,)).tolist() for _ in range(3)]
    examples = [CausalExample(i, x, list(x)) for i, x in enumerate(ids)]
    collector = RecordCollector()
    with torch.no_grad(), adapter.instrument(original, [layer], collector):
        evaluate_model(original, examples, torch.device("cpu"), 3, 0, task)
    y_orig = collector.records[0].moe_output

    pruned, _, _ = _setup(seed=5)
    replace_experts(pruned, adapter, layers, {LAYER: REMOVED}, method, {LAYER: {e: torch.randn(16) for e in REMOVED}})
    probe = build_reference_probe(pruned, adapter, layers, {LAYER: REMOVED}, reader, torch.device("cpu"))
    collector2 = RecordCollector()
    with torch.no_grad(), adapter.instrument(pruned, [layer], collector2):
        result = evaluate_model(pruned, examples, torch.device("cpu"), 3, 0, task, adapter=adapter, probe=probe)
    sq = float((y_orig - collector2.records[0].moe_output).square().sum())
    summary = result.layer_probe[LAYER]
    assert summary["tokens"] == 33
    assert abs(summary["mse_per_token"] - sq / 33) < 1e-4 * max(1.0, sq / 33)
    assert "perplexity" in result.metrics


def test_bias_training_on_olmoe():
    model, adapter, layers = _setup(seed=7)
    layer = layers[LAYER]
    task = CausalLMTask()
    items = [TokenizedPair(i, torch.randint(2, 96, (9,)).tolist(), [-100] * 9) for i in range(8)]
    refs = capture_layer_references(model, adapter, [layer], list(task.iter_batches(items, 2, 0)), torch.device("cpu"))
    # Start far from the optimum so the decrease is well above float noise.
    replace_experts(model, adapter, layers, {LAYER: REMOVED}, "bias", {LAYER: {e: torch.ones(16) for e in REMOVED}})
    trainable = freeze_all_but_biases(model)
    assert sorted(trainable) == [f"model.layers.0.mlp.experts.replaced.{e}.bias" for e in REMOVED]
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if n not in trainable}
    result = train_layer_biases(
        model, adapter, layer, REMOVED, refs[LAYER], list(range(6)), [6, 7], torch.device("cpu"),
        learning_rate=0.05, steps=60, batch_size=2, eval_interval=30, seed=0,
    )
    assert result.final_train_loss < result.initial_train_loss
    assert result.closed_form_train_loss is None  # top-k: biases are coupled
    for n, p in model.named_parameters():
        if n in frozen:
            assert torch.equal(p.detach(), frozen[n])
    assert all(isinstance(adapter.get_expert(model, layer, e), BiasExpert) for e in REMOVED)
