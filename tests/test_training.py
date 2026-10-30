import pytest
import torch

from moe_pruning.data.calibration import TokenizedPair
from moe_pruning.data.collators import iter_batches
from moe_pruning.model.architecture import get_adapter
from moe_pruning.pruning.replacement import freeze_all_but_biases, replace_experts_with_biases
from moe_pruning.training.bias_trainer import (
    capture_layer_references,
    closed_form_biases,
    evaluate_reconstruction,
    load_layer_reference,
    masked_sequence_batch,
    reconstruction_loss,
    save_layer_reference,
    train_layer_biases,
)
from tests.conftest import make_tiny_model

LAYER = "encoder.block.1"
REPLACED = [0, 2]


def _items(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    items = []
    for i in range(n):
        enc = torch.randint(3, 96, (int(torch.randint(5, 14, (1,), generator=g)),), generator=g).tolist() + [1]
        dec = torch.randint(3, 96, (int(torch.randint(3, 8, (1,), generator=g)),), generator=g).tolist() + [1]
        items.append(TokenizedPair(i, enc, dec))
    return items


@pytest.fixture
def setup():
    model = make_tiny_model(seed=4)
    adapter = get_adapter(model)
    layers = {layer.layer_id: layer for layer in adapter.discover_layers(model)}
    layer = layers[LAYER]
    items = _items()
    refs = capture_layer_references(model, adapter, [layer], list(iter_batches(items, 3, 0)), torch.device("cpu"))
    ref = refs[LAYER]
    originals = {e: adapter.get_expert(model, layer, e) for e in REPLACED}
    replace_experts_with_biases(model, adapter, layers, {LAYER: {e: torch.zeros(16) for e in REPLACED}})
    trainable = freeze_all_but_biases(model)
    return model, adapter, layer, ref, originals, trainable


def test_reference_cache_roundtrip(setup, tmp_path):
    _, _, _, ref, _, _ = setup
    assert len(ref.example_ids) == 12 and ref.offsets[-1] == ref.inputs.shape[0]
    save_layer_reference(ref, tmp_path, {"k": 1})
    loaded, meta = load_layer_reference(tmp_path, LAYER)
    assert meta == {"k": 1}
    torch.testing.assert_close(loaded.inputs, ref.inputs)
    assert loaded.example_ids == ref.example_ids


def test_only_bias_vectors_are_trainable(setup):
    model, adapter, layer, _, _, trainable = setup
    assert sorted(trainable) == [f"encoder.block.1.layer.1.mlp.experts.expert_{e}.bias" for e in REPLACED]
    assert {n for n, p in model.named_parameters() if p.requires_grad} == set(trainable)
    assert not any(p.requires_grad for p in adapter.get_router(model, layer).parameters())
    for e in (1, 3):
        assert not any(p.requires_grad for p in adapter.get_expert(model, layer, e).parameters())


def test_reconstruction_loss_matches_definition(setup):
    model, adapter, layer, ref, _, _ = setup
    rows = [0, 1, 2]
    x, y, mask = masked_sequence_batch(ref, rows, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        y_hat = adapter.run_moe_layer(model, layer, x)
    loss, n = reconstruction_loss(y, y_hat, mask)
    manual, count = 0.0, 0
    for i in rows:
        xi, yi = ref.sequence(i)
        with torch.no_grad():
            yh = adapter.run_moe_layer(model, layer, xi[None])[0]
        manual += float((yi - yh).square().sum())
        count += xi.shape[0]
    assert n == count
    assert float(loss) == pytest.approx(manual / count, rel=1e-5)


def test_closed_form_matches_brute_force(setup):
    model, adapter, layer, ref, originals, _ = setup
    rows = list(range(12))
    closed = closed_form_biases(model, adapter, layer, ref, rows, REPLACED, torch.device("cpu"), 4)
    num = {e: torch.zeros(16, dtype=torch.float64) for e in REPLACED}
    den = {e: 0.0 for e in REPLACED}
    with torch.no_grad():
        for i in rows:
            x, _ = ref.sequence(i)
            idx, g = adapter.route(model, layer, x[None])
            for t in range(x.shape[0]):
                e = int(idx[0, t, 0])
                if e in REPLACED:
                    gt = float(g[0, t, 0])
                    num[e] += gt**2 * originals[e](x[t : t + 1])[0].double()
                    den[e] += gt**2
    for e in REPLACED:
        torch.testing.assert_close(closed[e], num[e] / den[e], rtol=1e-4, atol=1e-5)


def test_training_changes_only_biases_and_reduces_loss(setup):
    model, adapter, layer, ref, _, trainable = setup
    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if n not in trainable}
    bias_before = {n: p.detach().clone() for n, p in trainable.items()}
    train_rows, val_rows = list(range(10)), [10, 11]
    result = train_layer_biases(
        model, adapter, layer, REPLACED, ref, train_rows, val_rows, torch.device("cpu"),
        learning_rate=0.1, steps=150, batch_size=2, eval_interval=50, seed=0,
    )
    for n, p in trainable.items():
        assert not torch.equal(p.detach(), bias_before[n]), f"{n} did not change"
    for n, p in model.named_parameters():
        if n in frozen_before:
            assert torch.equal(p.detach(), frozen_before[n]), f"frozen parameter {n} changed"
    assert result.final_train_loss < result.initial_train_loss
    assert result.trainable_parameters == 2 * 16
    # Loss values reported by the trainer agree with an independent evaluation.
    assert result.final_train_loss == pytest.approx(
        evaluate_reconstruction(model, adapter, layer, ref, train_rows, torch.device("cpu"), 4), rel=1e-5
    )
    # Training approaches the closed-form optimum of the same objective.
    assert result.final_train_loss <= result.initial_train_loss
    assert result.final_train_loss >= result.closed_form_train_loss - 1e-6
