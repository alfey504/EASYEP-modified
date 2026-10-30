"""Seeding, environment capture and resource measurements."""

from __future__ import annotations

import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import psutil
import torch

# Known sources of non-determinism, recorded in experiment manifests.
DETERMINISM_NOTES = {
    "cpu": "Deterministic for the operations used here (fixed seeds, no dropout in eval mode).",
    "mps": (
        "MPS does not guarantee bit-identical results between runs for all kernels "
        "(e.g. reductions and index_add_). Differences are at float32 rounding level."
    ),
    "cuda": (
        "CUDA index_add_/scatter use atomics and are not bit-deterministic; "
        "torch.use_deterministic_algorithms is not enabled to avoid unsupported-op errors."
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_info(device: torch.device) -> dict:
    import transformers

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(device),
        "determinism_note": DETERMINISM_NOTES.get(device.type, ""),
    }


def memory_info(device: torch.device) -> dict:
    info = {"process_rss_gb": round(psutil.Process(os.getpid()).memory_info().rss / 1e9, 3)}
    if device.type == "mps":
        info["mps_allocated_gb"] = round(torch.mps.current_allocated_memory() / 1e9, 3)
        info["mps_driver_allocated_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 3)
    elif device.type == "cuda":
        info["cuda_max_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 3)
    return info


@contextmanager
def timed() -> Iterator[dict]:
    """Context manager that records elapsed wall-clock seconds in ``result['seconds']``."""
    result: dict = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = round(time.perf_counter() - start, 3)
