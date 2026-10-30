"""The plugin registry and model profiles."""

import sys
import textwrap

import pytest

from moe_pruning.config import load_config
from moe_pruning.models import ModelPlugin, available_plugins, get_plugin, load_plugins, register_plugin
from moe_pruning.models.switch_transformers.adapter import SwitchTransformersAdapter
from moe_pruning.tasks.seq2seq_denoising import Seq2SeqDenoisingTask


def test_builtin_plugins_are_discovered():
    plugins = available_plugins()
    assert plugins["switch_transformers"].name == "switch_transformers"
    assert plugins["olmoe"].name == "olmoe"


def test_external_plugin_module_is_loaded(tmp_path, monkeypatch):
    module = tmp_path / "my_moe_plugin.py"
    module.write_text(textwrap.dedent('''
        from transformers import AutoModelForSeq2SeqLM
        from moe_pruning.models import ModelPlugin, register_plugin
        from moe_pruning.models.switch_transformers.adapter import SwitchTransformersAdapter
        from moe_pruning.tasks.seq2seq_denoising import Seq2SeqDenoisingTask

        register_plugin(ModelPlugin(
            name="my_moe",
            model_types=("my_moe_type",),
            adapter_factory=SwitchTransformersAdapter,
            task_factory=Seq2SeqDenoisingTask,
            auto_model_class=AutoModelForSeq2SeqLM,
            description="test plugin",
        ))
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    load_plugins(["my_moe_plugin"])
    plugin = get_plugin("my_moe_type")
    assert plugin.name == "my_moe"
    assert isinstance(plugin.adapter(), SwitchTransformersAdapter)
    sys.modules.pop("my_moe_plugin", None)


def test_conflicting_registration_is_rejected():
    with pytest.raises(ValueError, match="already handled"):
        register_plugin(
            ModelPlugin("other", ("olmoe",), SwitchTransformersAdapter, Seq2SeqDenoisingTask, object)
        )


def test_profiles_are_layered_over_env(tmp_path, monkeypatch):
    for key in ("HF_MODEL_ID", "PRUNE_EXPERTS_PER_LAYER", "EXPERIMENT_NAME", "MODEL_DTYPE"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text("C4_SUPPORT_KEYWORDS=FAQ\nHF_MODEL_ID=google/switch-base-8\nPRUNE_EXPERTS_PER_LAYER=2\n")
    profile = tmp_path / "olmoe.env"
    profile.write_text("HF_MODEL_ID=allenai/OLMoE-1B-7B-0125\nPRUNE_EXPERTS_PER_LAYER=16\nMODEL_DTYPE=bfloat16\n")
    cfg = load_config(env, {"EXPERIMENT_NAME": "x"}, [profile])
    assert cfg.hf_model_id == "allenai/OLMoE-1B-7B-0125"
    assert cfg.prune_experts_per_layer == 16 and cfg.model_dtype == "bfloat16"
    assert cfg.experiment_name == "x"  # --set wins over the profile
    cfg = load_config(env, {"PRUNE_EXPERTS_PER_LAYER": "4"}, [profile])
    assert cfg.prune_experts_per_layer == 4


def test_shipped_profiles_are_valid(monkeypatch):
    for key in ("HF_MODEL_ID", "MODEL_DTYPE", "PRUNE_LAYERS", "PRUNE_EXPERTS_PER_LAYER"):
        monkeypatch.delenv(key, raising=False)
    switch = load_config(profiles=["configs/switch-base-8.env"])
    olmoe = load_config(profiles=["configs/olmoe-1b-7b-0125.env"])
    assert switch.hf_model_id == "google/switch-base-8"
    assert olmoe.hf_model_id == "allenai/OLMoE-1B-7B-0125" and olmoe.model_dtype == "bfloat16"
    assert switch.experiment_name != olmoe.experiment_name
