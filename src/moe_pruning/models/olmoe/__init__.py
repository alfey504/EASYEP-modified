"""OLMoE (allenai/OLMoE-1B-7B-0125, as used in the STEP repository): decoder-only, top-8 of 64 experts."""

from transformers import AutoModelForCausalLM

from moe_pruning.models import ModelPlugin, register_plugin
from moe_pruning.models.olmoe.adapter import OlmoeAdapter
from moe_pruning.tasks.causal_lm import CausalLMTask

PLUGIN = register_plugin(
    ModelPlugin(
        name="olmoe",
        model_types=("olmoe",),
        adapter_factory=OlmoeAdapter,
        task_factory=CausalLMTask,
        auto_model_class=AutoModelForCausalLM,
        description="OLMoE decoder-only MoE, softmax top-k routing over fused experts",
    )
)
