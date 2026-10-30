"""Artifact locations and small JSON helpers.

Every stage reads and writes through these helpers so that file names are
defined in exactly one place.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from moe_pruning.config import ExperimentConfig


class MissingArtifactError(FileNotFoundError):
    """Raised when a stage depends on an artifact that has not been produced yet."""


class ArtifactPaths:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    # Model inspection
    @property
    def architecture_report(self) -> Path:
        return self.cfg.run_dir / "architecture_report.json"

    # Statistics and scoring
    @property
    def expert_stats(self) -> Path:
        return self.cfg.run_stats_dir / "expert_stats.pt"

    @property
    def expert_stats_manifest(self) -> Path:
        return self.cfg.run_stats_dir / "expert_stats_manifest.json"

    @property
    def expert_scores_json(self) -> Path:
        return self.cfg.run_dir / "expert_scores.json"

    @property
    def expert_scores_csv(self) -> Path:
        return self.cfg.run_dir / "expert_scores.csv"

    @property
    def expert_selection(self) -> Path:
        return self.cfg.run_dir / "expert_selection.json"

    # Models
    @property
    def replaced_model_dir(self) -> Path:
        return self.cfg.run_dir / "replaced-model"

    @property
    def trained_model_dir(self) -> Path:
        return self.cfg.run_dir / "trained-model"

    @property
    def bias_initialization_report(self) -> Path:
        return self.cfg.run_dir / "bias_initialization.json"

    # Training
    @property
    def layer_cache_dir(self) -> Path:
        return self.cfg.run_cache_dir / "layer_reference"

    @property
    def training_report(self) -> Path:
        return self.cfg.run_dir / "training_report.json"

    # Evaluation
    @property
    def eval_dir(self) -> Path:
        return self.cfg.run_dir / "evaluation"

    def eval_metrics(self, stage: str) -> Path:
        return self.eval_dir / f"{stage}_metrics.json"

    def eval_per_example(self, stage: str) -> Path:
        return self.eval_dir / f"{stage}_per_example.pt"

    @property
    def eval_comparison_json(self) -> Path:
        return self.eval_dir / "comparison.json"

    @property
    def eval_comparison_md(self) -> Path:
        return self.eval_dir / "comparison.md"

    @property
    def eval_examples_cache(self) -> Path:
        return self.cfg.run_cache_dir / "eval_examples.pt"

    # Logs
    @property
    def log_file(self) -> Path:
        return self.cfg.run_dir / "experiment.log"

    @property
    def config_snapshot_dir(self) -> Path:
        return self.cfg.run_dir / "config"

    def require(self, path: Path, produced_by: str) -> Path:
        if not path.exists():
            raise MissingArtifactError(f"Required artifact {path} not found. Run `{produced_by}` first.")
        return path


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(data: Any) -> str:
    """Stable short hash of a JSON-serialisable object."""
    payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
