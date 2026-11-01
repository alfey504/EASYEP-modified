"""Multi-device placement, delta checkpoints and float16 safety (Kaggle 2x T4 support).

A machine with one accelerator cannot run accelerate's multi-GPU dispatch for
real, so ``manual_shard`` places modules on two *different* devices (CPU and
MPS/CUDA) and moves inputs between them with pre-forward hooks - the same thing
accelerate's AlignDevicesHook does on 2 GPUs. This exercises every place where
the pipeline must follow a layer's device instead of assuming one device.
"""

import pytest
import torch
from torch import nn

from moe_pruning.config import ConfigError, load_config
from moe_pruning.data.calibration import TokenizedPair
from moe_pruning.evaluation.evaluator import build_reference_probe, evaluate_model
from moe_pruning.model.architecture import get_adapter
from moe_pruning.model.instrumentation import NonFiniteError, RecordCollector, assert_finite, gate_consistency_error
from moe_pruning.model.loader import StateDictReader
from moe_pruning.model.placement import (
    ModelPlacement,
    input_device,
    module_device,
    parse_device_map,
    parse_max_memory,
    place_model,
)
from moe_pruning.pruning.bias_expert import BiasExpert
from moe_pruning.pruning.replacement import freeze_all_but_biases, replace_experts
from moe_pruning.serialization.checkpoint import DELTA_WEIGHTS_NAME, WEIGHTS_NAME, load_pruned_model, save_pruned_model
from moe_pruning.tasks.causal_lm import CausalExample, CausalLMTask
from moe_pruning.training.bias_trainer import capture_layer_references, train_layer_biases
from tests.conftest import make_tiny_model, make_tiny_olmoe


def _second_device() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


SECOND = _second_device()
requires_two_devices = pytest.mark.skipif(SECOND is None, reason="needs CPU + one accelerator")


def _move(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return type(obj)(_move(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _move(v, device) for k, v in obj.items()}
    return obj


def manual_shard(model: nn.Module, placement: dict[str, torch.device]) -> nn.Module:
    """Put submodules on given devices; move each one's inputs to its device."""
    for name, device in placement.items():
        module = model.get_submodule(name)
        module.to(device)
        module.register_forward_pre_hook(
            lambda m, args, kwargs, d=device: (_move(args, d), _move(kwargs, d)), with_kwargs=True
        )
    return model


def _sharded_olmoe(seed=0):
    model = make_tiny_olmoe(seed)
    reference = make_tiny_olmoe(seed)
    manual_shard(model, {
        "model.embed_tokens": torch.device("cpu"),
        "model.rotary_emb": torch.device("cpu"),
        "model.layers.0": torch.device("cpu"),
        "model.layers.1": SECOND,
        "model.norm": SECOND,
        "lm_head": SECOND,
    })
    return model, reference


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
def test_parse_device_map_and_max_memory():
    assert parse_device_map("") is None
    assert parse_device_map("auto") == "auto"
    assert parse_device_map('{"model.layers.0": "cuda:0"}') == {"model.layers.0": "cuda:0"}
    with pytest.raises(ValueError):
        parse_device_map("everywhere")
    with pytest.raises(ValueError):
        parse_device_map("{not json")
    assert parse_max_memory("0:13GiB, 1:14GiB,cpu:20GiB") == {0: "13GiB", 1: "14GiB", "cpu": "20GiB"}
    assert parse_max_memory("") is None
    with pytest.raises(ValueError):
        parse_max_memory("13GiB")


def test_config_validates_placement(tmp_path, monkeypatch):
    for key in ("DEVICE_MAP", "MAX_MEMORY", "CHECKPOINT_FORMAT", "HF_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text("C4_SUPPORT_KEYWORDS=FAQ\n")
    cfg = load_config(env, {"DEVICE_MAP": "auto", "MAX_MEMORY": "0:13GiB,1:13GiB", "CHECKPOINT_FORMAT": "delta",
                            "HF_CACHE_DIR": str(tmp_path / "hf")})
    assert cfg.device_map == "auto" and cfg.checkpoint_format == "delta" and cfg.hf_cache_dir == tmp_path / "hf"
    with pytest.raises(ConfigError, match="DEVICE_MAP"):
        load_config(env, {"DEVICE_MAP": "sometimes"})
    with pytest.raises(ConfigError, match="CHECKPOINT_FORMAT"):
        load_config(env, {"CHECKPOINT_FORMAT": "zip"})


def test_kaggle_profiles_are_valid(monkeypatch):
    for key in ("HF_MODEL_ID", "MODEL_DTYPE", "DEVICE", "DEVICE_MAP", "CHECKPOINT_FORMAT", "MAX_MEMORY"):
        monkeypatch.delenv(key, raising=False)
    olmoe = load_config(profiles=["configs/kaggle/olmoe-1b-7b-t4x2.env"])
    assert olmoe.hf_model_id == "allenai/OLMoE-1B-7B-0125"
    assert olmoe.model_dtype == "float16" and olmoe.device_map == "auto" and olmoe.checkpoint_format == "delta"
    switch = load_config(profiles=["configs/kaggle/switch-base-8-t4.env"])
    assert switch.model_dtype == "float32" and switch.device_map == ""


# ----------------------------------------------------------------------
# Float16 safety
# ----------------------------------------------------------------------
def test_assert_finite_explains_float16_overflow():
    assert_finite(torch.ones(3), "ok", torch.float16)
    with pytest.raises(NonFiniteError, match="float16"):
        assert_finite(torch.tensor([1.0, float("inf")]), "logits", torch.float16)


# ----------------------------------------------------------------------
# Pipeline on a model spread over two devices
# ----------------------------------------------------------------------
@requires_two_devices
def test_sharded_forward_and_instrumentation():
    model, reference = _sharded_olmoe()
    adapter = get_adapter(model)
    layers = adapter.discover_layers(model)
    assert input_device(model).type == "cpu"
    assert module_device(adapter.get_moe_module(model, layers[1])).type == SECOND.type
    ids = torch.randint(2, 96, (2, 9))
    collector = RecordCollector(to_cpu=False)  # keep tensors where the layers produced them
    with torch.no_grad(), adapter.instrument(model, layers, collector):
        out = model(input_ids=ids).logits
    with torch.no_grad():
        torch.testing.assert_close(out.cpu(), reference(input_ids=ids).logits, rtol=1e-4, atol=1e-5)
    assert {r.moe_output.device.type for r in collector.records} == {"cpu", SECOND.type}
    for rec in collector.records:
        assert gate_consistency_error(rec)["max_rel_error"] < 1e-4


@requires_two_devices
@pytest.mark.parametrize("method", ["bias", "drop", "reroute"])
def test_sharded_replacement_probe_and_loss(method):
    model, reference = _sharded_olmoe(seed=1)
    adapter = get_adapter(model)
    layers = {layer.layer_id: layer for layer in adapter.discover_layers(model)}
    reader = StateDictReader({k: v.detach().cpu().clone() for k, v in reference.state_dict().items()})
    selection = {"layers.0": [1, 5], "layers.1": [2, 6]}
    biases = {lid: {e: torch.randn(16) for e in ids} for lid, ids in selection.items()}
    replace_experts(model, adapter, layers, selection, method, biases)
    for lid, ids in selection.items():
        layer_dev = module_device(adapter.get_moe_module(model, layers[lid]))
        for e in ids:
            module = adapter.get_expert(model, layers[lid], e)
            if isinstance(module, BiasExpert):
                assert module.bias.device.type == layer_dev.type

    task = CausalLMTask()
    examples = [CausalExample(i, x, list(x)) for i, x in enumerate(torch.randint(2, 96, (3, 10)).tolist())]
    probe = build_reference_probe(model, adapter, layers, selection, reader, torch.device("cpu"))
    result = evaluate_model(model, examples, torch.device("cpu"), 2, 0, task, adapter=adapter, probe=probe)
    assert set(result.layer_probe) == {"layers.0", "layers.1"}
    assert result.metrics["num_target_tokens"] == 3 * 9  # labels moved to the logits' device


@requires_two_devices
def test_sharded_bias_training_runs_on_layer_device():
    model, _ = _sharded_olmoe(seed=2)
    adapter = get_adapter(model)
    layers = {layer.layer_id: layer for layer in adapter.discover_layers(model)}
    layer = layers["layers.1"]
    task = CausalLMTask()
    items = [TokenizedPair(i, torch.randint(2, 96, (8,)).tolist(), [-100] * 8) for i in range(6)]
    refs = capture_layer_references(model, adapter, [layer], list(task.iter_batches(items, 2, 0)), torch.device("cpu"))
    replace_experts(model, adapter, layers, {"layers.1": [0, 4]}, "bias", {"layers.1": {0: torch.ones(16), 4: torch.ones(16)}})
    freeze_all_but_biases(model)
    result = train_layer_biases(
        model, adapter, layer, [0, 4], refs["layers.1"], [0, 1, 2, 3], [4, 5], torch.device("cpu"),
        learning_rate=0.05, steps=20, batch_size=2, eval_interval=10, seed=0,
    )
    assert result.final_train_loss < result.initial_train_loss
    assert adapter.get_expert(model, layer, 0).bias.device.type == SECOND.type


def test_place_model_with_accelerate_dispatch():
    model = make_tiny_olmoe()
    ids = torch.randint(2, 96, (1, 7))
    with torch.no_grad():
        expected = model(input_ids=ids).logits
    placed = place_model(model, ModelPlacement(torch.device("cpu"), {"": "cpu"}))
    with torch.no_grad():
        torch.testing.assert_close(placed(input_ids=ids).logits, expected)


# ----------------------------------------------------------------------
# Delta checkpoints
# ----------------------------------------------------------------------
def _manifest(adapter, layers, selection, method, source):
    return {
        "model_source": str(source),
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


@pytest.mark.parametrize("family", ["olmoe", "switch"])
@pytest.mark.parametrize("method", ["bias", "drop", "reroute"])
def test_delta_checkpoint_roundtrip(family, method, tmp_path, tiny_tokenizer):
    make = make_tiny_olmoe if family == "olmoe" else make_tiny_model
    base_dir = tmp_path / "base"
    make(seed=3).save_pretrained(base_dir)
    model = make(seed=3)
    adapter = get_adapter(model)
    layers = {layer.layer_id: layer for layer in adapter.discover_layers(model)}
    selection = {"layers.1": [0, 5]} if family == "olmoe" else {"decoder.block.3": [1, 2]}
    biases = {lid: {e: torch.randn(16) for e in ids} for lid, ids in selection.items()}
    replace_experts(model, adapter, layers, selection, method, biases)
    manifest = _manifest(adapter, layers, selection, method, base_dir)
    save_pruned_model(model, tiny_tokenizer, tmp_path / "delta", manifest, {}, checkpoint_format="delta")

    assert not (tmp_path / "delta" / WEIGHTS_NAME).exists()
    from moe_pruning.serialization.checkpoint import stored_tensor_shapes

    stored = stored_tensor_shapes(tmp_path / "delta" / DELTA_WEIGHTS_NAME)
    assert len(stored) == (2 if method == "bias" else 0)
    reloaded, _, loaded_manifest = load_pruned_model(tmp_path / "delta", "cpu")
    assert loaded_manifest["checkpoint_format"] == "delta"
    if family == "olmoe":
        ids = {"input_ids": torch.randint(2, 96, (2, 9))}
    else:
        ids = {"input_ids": torch.randint(2, 96, (2, 9)), "labels": torch.randint(2, 96, (2, 4))}
    with torch.no_grad():
        torch.testing.assert_close(reloaded(**ids).logits, model(**ids).logits, rtol=1e-5, atol=1e-6)


def test_delta_checkpoint_rejects_wrong_base_model(tmp_path, tiny_tokenizer):
    switch_dir = tmp_path / "switch"
    make_tiny_model().save_pretrained(switch_dir)
    model = make_tiny_olmoe()
    adapter = get_adapter(model)
    layers = {layer.layer_id: layer for layer in adapter.discover_layers(model)}
    replace_experts(model, adapter, layers, {"layers.0": [1]}, "drop")
    manifest = _manifest(adapter, layers, {"layers.0": [1]}, "drop", tmp_path / "missing")
    save_pruned_model(model, tiny_tokenizer, tmp_path / "delta", manifest, {}, checkpoint_format="delta")
    with pytest.raises(Exception, match="olmoe|switch"):
        load_pruned_model(tmp_path / "delta", "cpu", base_model_source=str(switch_dir))
