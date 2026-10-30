import pytest
import torch
import torch.nn.functional as F

from moe_pruning.model.architecture import get_adapter
from moe_pruning.model.instrumentation import RecordCollector
from moe_pruning.pruning.expert_selection import score_and_select
from moe_pruning.scoring.easyep import (
    output_aware_importance,
    rank_experts,
    select_lowest,
    token_contribution,
)
from moe_pruning.scoring.statistics import ExpertStatisticsAccumulator, LayerStatistics


def _collect(model, batch, layers=None):
    adapter = get_adapter(model)
    layers = layers or adapter.discover_layers(model)
    stack = {layer.layer_id: layer.stack for layer in layers}
    acc = ExpertStatisticsAccumulator(layers)
    masks = {"encoder": batch["attention_mask"].bool(), "decoder": batch["labels"] != -100}
    with torch.no_grad(), adapter.instrument(model, layers, lambda r: acc.update(r, masks[stack[r.layer_id]])):
        model(**batch)
    return acc, layers


def test_token_contribution_formula():
    h = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0, 0.0], [0.0, 1.0], [-2.0, -2.0]])
    s = token_contribution(h, y)
    expected = (1 - F.cosine_similarity(h, h + y, dim=-1)).clamp_min(0)
    torch.testing.assert_close(s, expected)
    assert s[0] == 0  # no update -> no contribution
    assert s[1] == pytest.approx(1 - 2**-0.5)
    assert s[2] == pytest.approx(2.0)  # opposite direction: 1 - (-1)


def test_output_aware_importance():
    g = torch.tensor([0.5, 0.25])
    out = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
    torch.testing.assert_close(output_aware_importance(g, out), torch.tensor([2.5, 2.5]))


def test_importance_matches_manual_computation(tiny_model, tiny_batch):
    adapter = get_adapter(tiny_model)
    layer = adapter.discover_layers(tiny_model)[0]
    collector = RecordCollector()
    with torch.no_grad(), adapter.instrument(tiny_model, [layer], collector):
        tiny_model(**tiny_batch)
    rec = collector.records[0]
    mask = tiny_batch["attention_mask"].bool().reshape(-1)
    s = token_contribution(rec.residual.reshape(-1, 16), rec.moe_output.reshape(-1, 16))
    idx = rec.expert_index.reshape(-1)
    g = rec.gate.reshape(-1)
    norm = rec.expert_output.reshape(-1, 16).norm(dim=-1)
    manual = torch.zeros(layer.num_experts, dtype=torch.float64)
    for t in range(idx.numel()):
        if mask[t] and idx[t] >= 0:
            manual[idx[t]] += float(g[t] * s[t] * norm[t])

    acc, _ = _collect(tiny_model, tiny_batch, [layer])
    torch.testing.assert_close(acc.stats[layer.layer_id].expert["importance"], manual, rtol=1e-5, atol=1e-8)
    assert acc.stats[layer.layer_id].layer["tokens"] == int(mask.sum())


def test_padding_tokens_are_excluded(tiny_model, tiny_batch):
    """A padded batch must give the same statistics as unpadded single sequences."""
    batched, layers = _collect(tiny_model, tiny_batch)
    single_acc = ExpertStatisticsAccumulator(layers)
    for i in range(2):
        enc_len = int(tiny_batch["attention_mask"][i].sum())
        dec_len = int((tiny_batch["labels"][i] != -100).sum())
        one = {
            "input_ids": tiny_batch["input_ids"][i : i + 1, :enc_len],
            "attention_mask": tiny_batch["attention_mask"][i : i + 1, :enc_len],
            "labels": tiny_batch["labels"][i : i + 1, :dec_len],
        }
        acc, _ = _collect(tiny_model, one)
        for layer_id, st in acc.stats.items():
            for name, value in st.expert.items():
                single_acc.stats[layer_id].expert[name] += value
            for name, value in st.layer.items():
                single_acc.stats[layer_id].layer[name] += value
    for layer_id, st in batched.stats.items():
        ref = single_acc.stats[layer_id]
        assert st.layer["tokens"] == ref.layer["tokens"]
        for name in ("count", "importance", "w_sum", "w_output_sum", "norm_sum"):
            torch.testing.assert_close(st.expert[name], ref.expert[name], rtol=1e-4, atol=1e-6)


def test_bias_mask_restricts_initialisation_sums(tiny_model, tiny_batch):
    adapter = get_adapter(tiny_model)
    layer = adapter.discover_layers(tiny_model)[0]
    acc = ExpertStatisticsAccumulator([layer])
    mask = tiny_batch["attention_mask"].bool()
    bias_mask = mask.clone()
    bias_mask[1] = False  # second row is "validation"
    with torch.no_grad(), adapter.instrument(tiny_model, [layer], lambda r: acc.update(r, mask, bias_mask)):
        tiny_model(**tiny_batch)
    st = acc.stats[layer.layer_id].expert
    assert int(st["count"].sum()) == int(mask.sum())
    assert int(st["bias_count"].sum()) == int(mask[0].sum())


def _stats(layer_id, importance, counts):
    st = LayerStatistics(layer_id, len(importance), 4)
    st.expert["importance"] = torch.tensor(importance, dtype=torch.float64)
    st.expert["count"] = torch.tensor(counts, dtype=torch.float64)
    st.layer["tokens"] = float(sum(counts))
    return st


def test_selection_is_independent_per_layer(tiny_model):
    layers = get_adapter(tiny_model).discover_layers(tiny_model)
    stats = {
        layers[0].layer_id: _stats(layers[0].layer_id, [1.0, 2.0, 30.0, 40.0], [5, 5, 5, 5]),
        layers[1].layer_id: _stats(layers[1].layer_id, [50.0, 60.0, 3.0, 4.0], [5, 5, 5, 5]),
        layers[2].layer_id: _stats(layers[2].layer_id, [9.0, 1.0, 8.0, 2.0], [5, 5, 5, 5]),
        layers[3].layer_id: _stats(layers[3].layer_id, [1.0, 1.0, 1.0, 1.0], [5, 5, 5, 5]),
    }
    rows, selection = score_and_select(stats, layers, layers, 2, allow_unobserved=False)
    assert selection == {
        layers[0].layer_id: [0, 1],
        layers[1].layer_id: [2, 3],
        layers[2].layer_id: [1, 3],
        layers[3].layer_id: [0, 1],  # ties broken by expert id
    }
    assert sum(r.selected_for_replacement for r in rows) == 8
    _, only_first = score_and_select(stats, layers, layers[:1], 2, allow_unobserved=False)
    assert list(only_first) == [layers[0].layer_id]


def test_unobserved_experts_excluded_by_default():
    importance = torch.tensor([0.0, 5.0, 1.0, 9.0], dtype=torch.float64)
    observed = torch.tensor([False, True, True, True])
    assert rank_experts(importance, observed) == [2, 1, 3, 0]
    assert select_lowest(importance, observed, 2) == [1, 2]
    assert select_lowest(importance, observed, 2, allow_unobserved=True) == [0, 2]
    with pytest.raises(ValueError):
        select_lowest(importance, torch.tensor([False, False, False, True]), 2)


def test_ranking_is_deterministic(tiny_model, tiny_batch):
    a, layers = _collect(tiny_model, tiny_batch)
    b, _ = _collect(tiny_model, tiny_batch)
    for layer in layers:
        ia, ib = a.stats[layer.layer_id], b.stats[layer.layer_id]
        assert rank_experts(ia.expert["importance"], ia.observed) == rank_experts(
            ib.expert["importance"], ib.observed
        )
