"""Load the configured original model and tokenizer."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from moe_pruning.config import ExperimentConfig
from moe_pruning.utils.device import check_dtype_supported, log_device, resolve_device, resolve_dtype

logger = logging.getLogger(__name__)

# switch-base-8 ships only pytorch_model.bin. transformers would otherwise start a
# background request to the public "safetensors-convert" Space on every load.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    dtype: torch.dtype
    source: str
    revision: str | None
    plugin: object  # moe_pruning.models.ModelPlugin


def _source(cfg: ExperimentConfig) -> tuple[str, str | None]:
    # load_config already verified that a non-empty MODEL_PATH is a valid directory.
    if cfg.model_path:
        return cfg.model_path, None
    return cfg.hf_model_id, cfg.hf_revision


def load_tokenizer(cfg: ExperimentConfig) -> PreTrainedTokenizerBase:
    source, revision = _source(cfg)
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    if tokenizer.pad_token_id is None:
        # Right padding is always masked out; any id works, EOS is the usual choice.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model_plugin(cfg: ExperimentConfig):
    """Model plugin of the configured model (reads only its config.json)."""
    from moe_pruning.models import plugin_for_source

    source, revision = _source(cfg)
    return plugin_for_source(source, revision)


def load_original_model(cfg: ExperimentConfig, device: torch.device | None = None) -> LoadedModel:
    """Load the unmodified model in evaluation mode on the configured device."""
    source, revision = _source(cfg)
    device = device or resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.model_dtype)
    check_dtype_supported(device, dtype)
    log_device(device, dtype)

    origin = "local directory" if cfg.model_path else "Hugging Face Hub"
    logger.info("Loading model %s from %s (revision=%s)", source, origin, revision or "default")
    plugin = get_model_plugin(cfg)
    logger.info("Model plugin: %s (%s)", plugin.name, plugin.description)
    model = plugin.auto_model_class.from_pretrained(source, revision=revision, dtype=dtype)
    model.to(device)
    model.eval()
    tokenizer = load_tokenizer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Loaded %s with %.1fM parameters", type(model).__name__, n_params / 1e6)
    return LoadedModel(model, tokenizer, device, dtype, source, revision, plugin)


class CheckpointReader:
    """Lazy, memory-mapped access to the tensors of the ORIGINAL checkpoint on disk.

    Reads safetensors shards (``*.safetensors``) or PyTorch ``pytorch_model*.bin``
    files. Tensor names are those stored on disk, which may differ from the
    in-memory module names (e.g. transformers 5 fuses OLMoE experts on load);
    adapters translate in ``MoEAdapter.load_original_expert``.
    """

    def __init__(self, model_dir: Path):
        from safetensors import safe_open

        self.model_dir = Path(model_dir)
        self._handles = []
        self._where: dict[str, object] = {}
        files = sorted(self.model_dir.glob("*.safetensors"))
        if files:
            for path in files:
                handle = safe_open(str(path), framework="pt")
                self._handles.append(handle)
                for name in handle.keys():
                    self._where[name] = handle
        else:
            for path in sorted(self.model_dir.glob("pytorch_model*.bin")):
                state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
                self._handles.append(state)
                for name in state:
                    self._where[name] = state
        if not self._where:
            raise FileNotFoundError(f"No checkpoint files (*.safetensors / pytorch_model*.bin) in {self.model_dir}")

    def keys(self) -> list[str]:
        return list(self._where)

    def has(self, name: str) -> bool:
        return name in self._where

    def get(self, name: str) -> torch.Tensor:
        source = self._where[name]
        if isinstance(source, dict):
            return source[name].clone()
        return source.get_tensor(name)


def open_original_checkpoint(cfg: ExperimentConfig) -> CheckpointReader:
    """CheckpointReader for the configured original model (local dir or HF cache)."""
    from huggingface_hub import snapshot_download

    source, revision = _source(cfg)
    if cfg.model_path:
        return CheckpointReader(Path(cfg.model_path))
    model_dir = snapshot_download(source, revision=revision, allow_patterns=["*.safetensors", "*.bin", "*.json"])
    return CheckpointReader(Path(model_dir))


class StateDictReader:
    """CheckpointReader over an in-memory state dict (tests, already-loaded weights)."""

    def __init__(self, state: dict[str, torch.Tensor]):
        self._state = state

    def keys(self) -> list[str]:
        return list(self._state)

    def has(self, name: str) -> bool:
        return name in self._state

    def get(self, name: str) -> torch.Tensor:
        return self._state[name].clone()
