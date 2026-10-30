"""Logging setup shared by all scripts."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(level=level, format=_FORMAT, datefmt="%H:%M:%S", handlers=handlers, force=True)
    # Third-party libraries are noisy at INFO.
    for name in ("urllib3", "httpx", "httpx2", "huggingface_hub", "datasets", "filelock", "fsspec"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")
