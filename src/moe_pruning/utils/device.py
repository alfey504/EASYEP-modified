"""Centralised device and dtype selection."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class DeviceUnavailableError(RuntimeError):
    """Raised when an explicitly requested device is not available."""


def is_available(kind: str) -> bool:
    if kind == "cpu":
        return True
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        return torch.backends.mps.is_available()
    raise ValueError(f"Unknown device type {kind!r}")


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a device request.

    ``auto`` picks CUDA, then MPS, then CPU. An explicit request for an
    unavailable device raises instead of silently falling back.
    """
    requested = requested.lower()
    if requested == "auto":
        for kind in ("cuda", "mps"):
            if is_available(kind):
                return torch.device(kind)
        return torch.device("cpu")
    if requested not in ("cuda", "mps", "cpu"):
        raise ValueError(f"DEVICE must be auto, cuda, mps or cpu (got {requested!r})")
    if not is_available(requested):
        raise DeviceUnavailableError(
            f"DEVICE={requested} was requested but is not available on this machine. "
            "Set DEVICE=auto or choose an available device."
        )
    return torch.device(requested)


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[name]
    except KeyError as exc:
        raise ValueError(f"MODEL_DTYPE must be one of {list(_DTYPES)} (got {name!r})") from exc


def check_dtype_supported(device: torch.device, dtype: torch.dtype) -> None:
    if device.type == "cpu" and dtype == torch.float16:
        logger.warning("float16 on CPU is slow and poorly supported; float32 is recommended.")
    if device.type == "mps" and dtype == torch.bfloat16:
        logger.warning("bfloat16 support on MPS depends on the macOS/PyTorch version.")


def log_device(device: torch.device, dtype: torch.dtype) -> None:
    logger.info("Selected device: %s | model dtype: %s", device, str(dtype).replace("torch.", ""))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
