"""Compact, reloadable checkpoints for models with replaced experts.

Format (directory)::

    config.json              original HF model configuration
    generation_config.json   (if present)
    tokenizer files
    model.safetensors        retained weights + BiasExpert biases (bias method); NO weights of replaced experts
    pruning_manifest.json    replaced experts and method per layer, bias details, tied-weight aliases
    experiment_config.json   configuration of the run that produced the checkpoint
    README.md

Tied tensors (e.g. shared embeddings / lm_head) are stored once; their aliases
are listed in the manifest and restored on load.

``load_pruned_model`` is required to load this format. A plain
``from_pretrained`` would re-create the original experts with random weights,
so it is deliberately not supported.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from moe_pruning.model.architecture import get_adapter
from moe_pruning.pruning.replacement import make_replacement_module
from moe_pruning.utils.paths import read_json, write_json

logger = logging.getLogger(__name__)

FORMAT_NAME = "moe-pruning/bias-replaced-experts"
FORMAT_VERSION = 1
WEIGHTS_NAME = "model.safetensors"
MANIFEST_NAME = "pruning_manifest.json"
EXPERIMENT_CONFIG_NAME = "experiment_config.json"


class CheckpointError(RuntimeError):
    pass


def _storage_key(t: torch.Tensor) -> tuple:
    return (t.device.type, t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), t.dtype)


def deduplicate_state_dict(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Keep one copy of tensors sharing storage; return (tensors, {alias: canonical})."""
    canonical_for: dict[tuple, str] = {}
    tensors: dict[str, torch.Tensor] = {}
    aliases: dict[str, str] = {}
    for name in sorted(state, key=lambda n: (n.count("."), n)):
        t = state[name]
        key = _storage_key(t)
        if key in canonical_for:
            aliases[name] = canonical_for[key]
        else:
            canonical_for[key] = name
            tensors[name] = t.detach().contiguous().cpu()
    return tensors, aliases


def layer_method(entry: dict) -> str:
    """Replacement method of a manifest layer entry (older manifests: bias)."""
    return entry.get("replacement_method", "bias")


def replaced_expert_prefixes(manifest: dict) -> list[str]:
    return [
        prefix
        for layer in manifest["layers"].values()
        for prefix in layer["expert_state_prefixes"].values()
    ]


def expected_bias_keys(manifest: dict) -> list[str]:
    """State-dict keys that must exist: the biases of bias-replaced experts."""
    return [
        prefix + "bias"
        for layer in manifest["layers"].values()
        if layer_method(layer) == "bias"
        for prefix in layer["expert_state_prefixes"].values()
    ]


def save_pruned_model(
    model: nn.Module,
    tokenizer,
    output_dir: Path,
    manifest: dict,
    experiment_config: dict,
    readme: str | None = None,
) -> Path:
    """Write the checkpoint and verify that no replaced expert weights are stored."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors, aliases = deduplicate_state_dict(model.state_dict())

    prefixes = replaced_expert_prefixes(manifest)
    for name in tensors:
        for prefix in prefixes:
            if name.startswith(prefix) and name != prefix + "bias":
                raise CheckpointError(f"Refusing to save original weights of a replaced expert: {name}")
    missing_biases = [k for k in expected_bias_keys(manifest) if k not in tensors]
    if missing_biases:
        raise CheckpointError(f"Bias parameters missing from the model: {missing_biases}")

    manifest = {
        **manifest,
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "tied_aliases": aliases,
        "num_stored_tensors": len(tensors),
        "num_stored_parameters": int(sum(t.numel() for t in tensors.values())),
    }
    save_file(tensors, str(output_dir / WEIGHTS_NAME), metadata={"format": "pt", "pruning_format": FORMAT_NAME})
    model.config.save_pretrained(output_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    write_json(output_dir / MANIFEST_NAME, manifest)
    write_json(output_dir / EXPERIMENT_CONFIG_NAME, experiment_config)
    (output_dir / "README.md").write_text(readme or _default_readme(manifest), encoding="utf-8")
    logger.info(
        "Saved pruned checkpoint to %s (%d tensors, %.1fM parameters)",
        output_dir, len(tensors), manifest["num_stored_parameters"] / 1e6,
    )
    return output_dir


def _default_readme(manifest: dict) -> str:
    lines = [
        "# Pruned MoE checkpoint",
        "",
        f"Base model: `{manifest.get('model_source')}`",
        "",
        "Selected experts were removed. Replacement per layer: `bias` = gate-scaled constant vector "
        "(`BiasExpert`), `drop` = zero output, `reroute` = tokens sent to their best remaining expert.",
        "The original weights of removed experts are **not** stored in this checkpoint.",
        "",
        "Load with the project's custom loader (plain `from_pretrained` is not supported):",
        "",
        "```python",
        "from moe_pruning.serialization.checkpoint import load_pruned_model",
        "model, tokenizer, manifest = load_pruned_model('<this directory>', device='cpu')",
        "```",
        "",
        "## Replaced experts",
        "",
    ]
    for layer_id, layer in manifest["layers"].items():
        lines.append(f"- `{layer_id}`: experts {layer['expert_ids']} ({layer_method(layer)})")
    return "\n".join(lines) + "\n"


def load_pruned_model(model_path: Path | str, device: torch.device | str = "cpu"):
    """Rebuild a pruned model from a checkpoint written by :func:`save_pruned_model`.

    Returns:
        (model in eval mode on ``device``, tokenizer, manifest)
    """
    model_path = Path(model_path)
    device = torch.device(device)
    manifest_path = model_path / MANIFEST_NAME
    if not manifest_path.exists():
        raise CheckpointError(f"{model_path} has no {MANIFEST_NAME}; not a pruned checkpoint")
    manifest = read_json(manifest_path)
    if manifest.get("format") != FORMAT_NAME or manifest.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(f"Unsupported checkpoint format {manifest.get('format')} v{manifest.get('format_version')}")

    # 1. Architecture from the saved configuration, without allocating weights.
    config = AutoConfig.from_pretrained(model_path)
    auto_cls = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
    with torch.device("meta"):
        model = auto_cls.from_config(config)

    # 2. Install the replacement modules in the manifest's expert slots before loading.
    adapter = get_adapter(model)
    layers = {layer.layer_id: layer for layer in adapter.discover_layers(model)}
    for layer_id, entry in manifest["layers"].items():
        if layer_id not in layers:
            raise CheckpointError(f"Manifest layer {layer_id} not found in the rebuilt architecture")
        layer = layers[layer_id]
        method = layer_method(entry)
        for expert_id in entry["expert_ids"]:
            expected_prefix = entry["expert_state_prefixes"][str(expert_id)]
            if adapter.expert_state_prefix(layer, expert_id) != expected_prefix:
                raise CheckpointError(f"State prefix mismatch for {layer_id} expert {expert_id}")
            module = make_replacement_module(method, layer.expert_output_dim, device="meta")
            adapter.set_expert(model, layer, expert_id, module)

    # 3. Load tensors, restoring tied aliases.
    tensors = load_file(str(model_path / WEIGHTS_NAME), device="cpu")
    for alias, canonical in manifest.get("tied_aliases", {}).items():
        if canonical not in tensors:
            raise CheckpointError(f"Tied alias {alias} refers to missing tensor {canonical}")
        tensors[alias] = tensors[canonical]

    # 4. Validate keys. Original weights of replaced experts are absent from the
    #    expected set by construction; anything else missing/unexpected is an error.
    expected = set(model.state_dict().keys())
    missing = sorted(expected - tensors.keys())
    unexpected = sorted(tensors.keys() - expected)
    if missing or unexpected:
        raise CheckpointError(
            f"State dict mismatch.\n  missing ({len(missing)}): {missing[:10]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:10]}"
        )
    model.load_state_dict(tensors, strict=True, assign=True)
    model.tie_weights()

    leftover_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if leftover_meta:
        raise CheckpointError(f"Tensors were not materialised: {leftover_meta[:10]}")

    model.to(device)
    model.eval()
    for layer_id, entry in manifest["layers"].items():
        if layer_method(entry) == "reroute":
            adapter.set_routing_mask(model, layers[layer_id], entry["expert_ids"])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info(
        "Loaded pruned checkpoint %s (%d replaced experts) on %s",
        model_path, sum(len(v["expert_ids"]) for v in manifest["layers"].values()), device,
    )
    return model, tokenizer, manifest
