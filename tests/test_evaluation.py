import pytest
import torch

from moe_pruning.data.span_corruption import CorruptedExample
from moe_pruning.evaluation.evaluator import build_reference_probe, evaluate_model
from moe_pruning.evaluation.metrics import prediction_agreement, summarize_losses
from moe_pruning.model.architecture import get_adapter
from moe_pruning.model.instrumentation import RecordCollector
from moe_pruning.model.loader import StateDictReader
from moe_pruning.pruning.replacement import replace_experts
from moe_pruning.tasks.seq2seq_denoising import Seq2SeqDenoisingTask
from tests.conftest import make_tiny_model


def test_token_nll_matches_hf_loss(tiny_model, tiny_batch):
    with torch.no_grad():
        out = tiny_model(**tiny_batch)
    sums, counts = Seq2SeqDenoisingTask().token_nll(out.logits, tiny_batch["labels"])
    assert counts.tolist() == [6, 4]
    torch.testing.assert_close(sums.sum() / counts.sum(), out.loss)
    summary = summarize_losses(sums.tolist(), counts.tolist())
    assert abs(summary["mean_token_loss"] - float(out.loss)) < 1e-5
    assert "perplexity" not in summary  # denoising loss is not perplexity


def test_prediction_agreement():
    assert prediction_agreement([[1, 2, 3], [4]], [[1, 0, 3], [4]]) == 0.75


@pytest.mark.parametrize("method", ["bias", "drop", "reroute"])
def test_reference_probe_measures_true_layer_error(method):
    original = make_tiny_model(seed=11)
    adapter = get_adapter(original)
    layers = {layer.layer_id: layer for layer in adapter.discover_layers(original)}
    layer = layers["encoder.block.1"]
    reader = StateDictReader({k: v.clone() for k, v in original.state_dict().items()})
    task = Seq2SeqDenoisingTask()
    examples = [
        CorruptedExample(i, [], torch.randint(3, 96, (12,)).tolist() + [1], torch.randint(3, 96, (5,)).tolist() + [1], [])
        for i in range(4)
    ]

    # Reference: original layer output on the same inputs.
    collector = RecordCollector()
    with torch.no_grad(), adapter.instrument(original, [layer], collector):
        evaluate_model(original, examples, torch.device("cpu"), 2, 0, task)
    y_orig = [r.moe_output for r in collector.records]

    pruned = make_tiny_model(seed=11)
    biases = {layer.layer_id: {0: torch.randn(16), 3: torch.randn(16)}}
    replace_experts(pruned, adapter, layers, {layer.layer_id: [0, 3]}, method, biases)
    probe = build_reference_probe(pruned, adapter, layers, {layer.layer_id: [0, 3]}, reader, torch.device("cpu"))
    collector2 = RecordCollector()
    with torch.no_grad(), adapter.instrument(pruned, [layer], collector2):
        result = evaluate_model(pruned, examples, torch.device("cpu"), 2, 0, task, adapter=adapter, probe=probe)
    y_pruned = [r.moe_output for r in collector2.records]

    # First sparse layer: identical input, so the true error is directly comparable.
    sq = sum(float((a - b).square().sum()) for a, b in zip(y_orig, y_pruned))
    tokens = sum(a.shape[0] * a.shape[1] for a in y_orig)  # no padding in these batches
    summary = result.layer_probe[layer.layer_id]
    assert summary["tokens"] == tokens
    assert abs(summary["mse_per_token"] - sq / tokens) < 1e-4 * max(1.0, sq / tokens)
    assert summary["tokens_routed_to_replaced_experts"] > 0
