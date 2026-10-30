"""Switch Transformers (google/switch-base-8 and relatives): encoder-decoder, top-1 routing."""

from transformers import AutoModelForSeq2SeqLM

from moe_pruning.models import ModelPlugin, register_plugin
from moe_pruning.models.switch_transformers.adapter import SwitchTransformersAdapter
from moe_pruning.tasks.seq2seq_denoising import Seq2SeqDenoisingTask

PLUGIN = register_plugin(
    ModelPlugin(
        name="switch_transformers",
        model_types=("switch_transformers",),
        adapter_factory=SwitchTransformersAdapter,
        task_factory=Seq2SeqDenoisingTask,
        auto_model_class=AutoModelForSeq2SeqLM,
        description="Switch Transformers encoder-decoder MoE, top-1 routing with expert capacity",
    )
)
