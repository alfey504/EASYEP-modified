"""Where model weights live: one device, or sharded over several GPUs.

``DEVICE_MAP`` empty  -> the whole model on ``DEVICE`` (default; switch-base-8).
``DEVICE_MAP=auto``   -> layers distributed over all visible GPUs with
                         accelerate (e.g. OLMoE-1B-7B on Kaggle 2x T4).
Other values: ``balanced``, ``balanced_low_0``, ``sequential``, or a JSON
object ``{"module.name": "cuda:0", ...}``. ``MAX_MEMORY`` caps the memory per
device (``0:13GiB,1:13GiB``) to leave room for activations.

The pipeline never assumes a single device: inputs go to the input
embedding's device, and per-layer work (bias training, reference experts,
replacement modules) runs on the device of that layer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import chain

import torch
from torch import nn

logger = logging.getLogger(__name__)

NAMED_DEVICE_MAPS = ("auto", "balanced", "balanced_low_0", "sequential")


@dataclass(frozen=True)
class ModelPlacement:
    device: torch.device  # single-device placement / primary device
    device_map: str | dict | None = None
    max_memory: dict | None = None

    @property
    def sharded(self) -> bool:
        return self.device_map is not None

    def describe(self) -> str:
        if not self.sharded:
            return str(self.device)
        return f"device_map={self.device_map!r} max_memory={self.max_memory}"


def parse_device_map(value: str) -> str | dict | None:
    value = (value or "").strip()
    if not value:
        return None
    if value.startswith("{"):
        try:
            mapping = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"DEVICE_MAP is not valid JSON: {exc}") from exc
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("DEVICE_MAP JSON must be a non-empty object")
        return mapping
    if value not in NAMED_DEVICE_MAPS:
        raise ValueError(f"DEVICE_MAP must be empty, one of {NAMED_DEVICE_MAPS} or a JSON object (got {value!r})")
    return value


def parse_max_memory(value: str) -> dict | None:
    """``"0:13GiB,1:13GiB,cpu:20GiB"`` -> ``{0: "13GiB", 1: "13GiB", "cpu": "20GiB"}``."""
    value = (value or "").strip()
    if not value:
        return None
    out: dict = {}
    for item in value.split(","):
        if ":" not in item:
            raise ValueError(f"MAX_MEMORY entries must look like 0:13GiB (got {item!r})")
        key, amount = (part.strip() for part in item.split(":", 1))
        out[int(key) if key.isdigit() else key] = amount
    return out


def placement_from_config(cfg) -> ModelPlacement:
    from moe_pruning.utils.device import resolve_device

    return ModelPlacement(
        device=resolve_device(cfg.device),
        device_map=parse_device_map(cfg.device_map),
        max_memory=parse_max_memory(cfg.max_memory),
    )


def from_pretrained_kwargs(placement: ModelPlacement) -> dict:
    """Keyword arguments for ``from_pretrained`` (weights go straight to their devices)."""
    if not placement.sharded:
        return {}
    kwargs = {"device_map": placement.device_map}
    if placement.max_memory:
        kwargs["max_memory"] = placement.max_memory
    return kwargs


def place_model(model: nn.Module, placement: ModelPlacement) -> nn.Module:
    """Move an in-memory model to its device(s)."""
    if not placement.sharded:
        return model.to(placement.device)
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_balanced_memory

    device_map = placement.device_map
    if isinstance(device_map, str):
        no_split = getattr(model, "_no_split_modules", None) or []
        dtype = next(model.parameters()).dtype
        max_memory = placement.max_memory
        if device_map != "sequential":
            max_memory = get_balanced_memory(
                model, max_memory=max_memory, no_split_module_classes=no_split, dtype=dtype,
                low_zero=device_map == "balanced_low_0",
            )
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split, dtype=dtype)
    logger.info("Dispatching model over devices: %s", sorted({str(v) for v in device_map.values()}))
    return dispatch_model(model, device_map=device_map)


def module_device(module: nn.Module) -> torch.device:
    """Device of a module's first parameter (or buffer)."""
    for tensor in chain(module.parameters(), module.buffers()):
        if not tensor.is_meta:
            return tensor.device
    raise ValueError(f"{type(module).__name__} has no materialised parameters")


def input_device(model: nn.Module) -> torch.device:
    """Device that model inputs must be placed on (the input embedding's)."""
    return module_device(model.get_input_embeddings())


def model_devices(model: nn.Module) -> list[torch.device]:
    return sorted({p.device for p in model.parameters()}, key=str)
