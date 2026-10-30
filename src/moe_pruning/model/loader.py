"""Load the configured original model and tokenizer."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

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


def _source(cfg: ExperimentConfig) -> tuple[str, str | None]:
    # load_config already verified that a non-empty MODEL_PATH is a valid directory.
    if cfg.model_path:
        return cfg.model_path, None
    return cfg.hf_model_id, cfg.hf_revision


def load_tokenizer(cfg: ExperimentConfig) -> PreTrainedTokenizerBase:
    source, revision = _source(cfg)
    return AutoTokenizer.from_pretrained(source, revision=revision)


def load_original_model(cfg: ExperimentConfig, device: torch.device | None = None) -> LoadedModel:
    """Load the unmodified model in evaluation mode on the configured device."""
    source, revision = _source(cfg)
    device = device or resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.model_dtype)
    check_dtype_supported(device, dtype)
    log_device(device, dtype)

    origin = "local directory" if cfg.model_path else "Hugging Face Hub"
    logger.info("Loading model %s from %s (revision=%s)", source, origin, revision or "default")
    hf_config = AutoConfig.from_pretrained(source, revision=revision)
    auto_cls = AutoModelForSeq2SeqLM if hf_config.is_encoder_decoder else AutoModelForCausalLM
    model = auto_cls.from_pretrained(source, revision=revision, dtype=dtype)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Loaded %s with %.1fM parameters", type(model).__name__, n_params / 1e6)
    return LoadedModel(model, tokenizer, device, dtype, source, revision)


def load_original_tensors(cfg: ExperimentConfig, prefixes: list[str]) -> dict[str, torch.Tensor]:
    """Read only the tensors whose names start with one of ``prefixes`` from the
    original checkpoint (memory-mapped; the full model is never materialised).

    Used to measure the reconstruction error of replaced experts without
    keeping a second copy of the model in memory.
    """
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    source, revision = _source(cfg)
    if cfg.model_path:
        model_dir = Path(cfg.model_path)
    else:
        model_dir = Path(
            snapshot_download(source, revision=revision, allow_patterns=["*.safetensors", "*.bin", "*.json"])
        )
    wanted = lambda name: any(name.startswith(p) for p in prefixes)  # noqa: E731
    out: dict[str, torch.Tensor] = {}
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if safetensor_files:
        for path in safetensor_files:
            with safe_open(str(path), framework="pt") as fh:
                for name in fh.keys():
                    if wanted(name):
                        out[name] = fh.get_tensor(name)
    else:
        for path in sorted(model_dir.glob("pytorch_model*.bin")):
            state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            out.update({k: v.clone() for k, v in state.items() if wanted(k)})
            del state
    if not out:
        raise FileNotFoundError(f"No tensors with prefixes {prefixes} found in {model_dir}")
    return out
