"""Model plugins.

A model plugin bundles everything that is specific to one model family:

* an :class:`~moe_pruning.model.architecture.MoEAdapter` (how to find, route,
  instrument and replace experts),
* a :class:`~moe_pruning.tasks.base.TaskObjective` (how calibration and
  evaluation data become model inputs and how the loss is computed),
* the Hugging Face auto class used to load the model.

The pruning, scoring, initialisation, training, evaluation and serialization
code never imports a plugin directly; it looks the plugin up by the model's
``config.model_type``.

Adding a model
--------------
Create a module that calls :func:`register_plugin`, e.g.::

    # my_package/mixtral_plugin.py
    from moe_pruning.models import ModelPlugin, register_plugin
    from moe_pruning.tasks.causal_lm import CausalLMTask
    from transformers import AutoModelForCausalLM
    from my_package.mixtral_adapter import MixtralAdapter

    register_plugin(ModelPlugin(
        name="mixtral",
        model_types=("mixtral",),
        adapter_factory=MixtralAdapter,
        task_factory=CausalLMTask,
        auto_model_class=AutoModelForCausalLM,
        description="Mixtral 8x7B style sparse MoE",
    ))

and make it discoverable in one of three ways:

1. put it in ``src/moe_pruning/models/<name>/`` (built-in plugins are
   imported automatically);
2. list the module in ``MODEL_PLUGINS`` (comma-separated, ``.env`` or
   environment), e.g. ``MODEL_PLUGINS=my_package.mixtral_plugin``;
3. expose it as a Python entry point in the ``moe_pruning.models`` group.
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from moe_pruning.model.architecture import MoEAdapter
    from moe_pruning.tasks.base import TaskObjective

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "moe_pruning.models"
PLUGIN_ENV_VAR = "MODEL_PLUGINS"


@dataclass(frozen=True)
class ModelPlugin:
    name: str
    model_types: tuple[str, ...]  # HF config.model_type values handled by the plugin
    adapter_factory: Callable[[], "MoEAdapter"]
    task_factory: Callable[[], "TaskObjective"]
    auto_model_class: type  # e.g. AutoModelForSeq2SeqLM / AutoModelForCausalLM
    description: str = ""

    def adapter(self) -> "MoEAdapter":
        return self.adapter_factory()

    def task(self) -> "TaskObjective":
        return self.task_factory()


_PLUGINS: dict[str, ModelPlugin] = {}
_LOADED = False


def register_plugin(plugin: ModelPlugin) -> ModelPlugin:
    for model_type in plugin.model_types:
        existing = _PLUGINS.get(model_type)
        if existing is not None and existing.name != plugin.name:
            raise ValueError(
                f"model_type {model_type!r} is already handled by plugin {existing.name!r}; "
                f"cannot register {plugin.name!r}"
            )
        _PLUGINS[model_type] = plugin
    return plugin


def load_plugins(extra_modules: list[str] | None = None) -> None:
    """Import built-in plugins, entry-point plugins and MODEL_PLUGINS modules (once)."""
    global _LOADED
    modules = list(extra_modules or [])
    modules += [m.strip() for m in os.environ.get(PLUGIN_ENV_VAR, "").split(",") if m.strip()]
    if not _LOADED:
        for info in pkgutil.iter_modules(__path__, prefix=f"{__name__}."):
            importlib.import_module(info.name)
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            try:
                ep.load()
            except Exception as exc:  # a broken third-party plugin must not break the others
                logger.warning("Could not load model plugin entry point %s: %s", ep.name, exc)
        _LOADED = True
    for module in modules:
        importlib.import_module(module)


def available_plugins() -> dict[str, ModelPlugin]:
    load_plugins()
    return dict(_PLUGINS)


def get_plugin(model_type: str) -> ModelPlugin:
    load_plugins()
    try:
        return _PLUGINS[model_type]
    except KeyError:
        supported = sorted(_PLUGINS)
        raise NotImplementedError(
            f"No model plugin for model_type={model_type!r}. Supported: {supported}. "
            f"See moe_pruning/models/__init__.py for how to add one (or set {PLUGIN_ENV_VAR})."
        ) from None


def plugin_for_model(model) -> ModelPlugin:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return get_plugin(model_type)


def plugin_for_source(source: str, revision: str | None = None) -> ModelPlugin:
    """Plugin of a local model directory or Hub id (reads only config.json)."""
    from transformers import AutoConfig

    return get_plugin(AutoConfig.from_pretrained(source, revision=revision).model_type)
