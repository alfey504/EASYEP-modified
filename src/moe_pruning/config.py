"""Centralised, validated experiment configuration.

The ``.env`` file in the project root is the primary configuration source.
Scripts may override individual values with ``--set KEY=VALUE`` (the key is the
environment-variable name), which is applied on top of the environment.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

VALID_DTYPES = ("float32", "float16", "bfloat16")
VALID_DEVICES = ("auto", "cuda", "mps", "cpu")
VALID_PRUNING_METHODS = ("easyep",)
VALID_BIAS_INITIALIZATIONS = ("gate_token_weighted_mean",)
VALID_TRAINING_OBJECTIVES = ("layer_mse",)
VALID_UNOBSERVED_POLICIES = ("exclude", "allow")
VALID_REPLACEMENT_METHODS = ("bias", "drop", "reroute")


class ConfigError(ValueError):
    """Raised when the configuration is missing or invalid."""


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class ExperimentConfig:
    # Model
    model_path: str
    hf_model_id: str
    hf_revision: str | None
    model_dtype: str
    # Device
    device: str
    # Datasets
    calibration_dataset_path: Path
    evaluation_dataset_path: Path
    calibration_max_samples: int
    calibration_batch_size: int
    max_source_length: int
    max_target_length: int
    # C4
    c4_dataset_id: str
    c4_config: str
    c4_split: str
    c4_domain: str
    c4_max_samples: int
    c4_max_scanned: int
    c4_min_text_length: int
    c4_max_text_length: int
    c4_min_keyword_matches: int
    c4_support_keywords: tuple[str, ...]
    c4_url_keywords: tuple[str, ...]
    c4_allowed_domains: tuple[str, ...]
    # Pruning
    pruning_method: str
    prune_experts_per_layer: int
    prune_layers: str
    prune_unobserved_policy: str
    replacement_method: str
    # Bias
    bias_initialization: str
    bias_training_objective: str
    bias_learning_rate: float
    bias_training_steps: int
    bias_batch_size: int
    bias_validation_fraction: float
    bias_eval_interval: int
    # Evaluation
    eval_max_samples: int
    eval_batch_size: int
    eval_noise_density: float
    eval_mean_span_length: float
    eval_seed: int
    eval_qualitative_samples: int
    # Reproducibility
    seed: int
    # Outputs
    experiment_name: str
    pruned_model_path: Path
    experiment_dir: Path
    stats_dir: Path
    cache_dir: Path
    log_level: str = "INFO"
    # Raw values, kept for experiment manifests.
    raw: dict[str, str] = field(default_factory=dict, compare=False, repr=False)

    # ------------------------------------------------------------------
    # Derived locations
    # ------------------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        """Directory holding every artifact of the current experiment."""
        return self.experiment_dir / self.experiment_name

    @property
    def run_stats_dir(self) -> Path:
        return self.stats_dir / self.experiment_name

    @property
    def run_cache_dir(self) -> Path:
        return self.cache_dir / self.experiment_name

    @property
    def model_source(self) -> str:
        """Local path if configured, otherwise the Hugging Face model id."""
        return self.model_path or self.hf_model_id

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name == "raw":
                continue
            value = getattr(self, f.name)
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, tuple):
                value = list(value)
            out[f.name] = value
        return out


# ----------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------
class _Reader:
    def __init__(self, values: dict[str, str]):
        self.values = values
        self.errors: list[str] = []

    def _get(self, key: str, default: str | None) -> str | None:
        value = self.values.get(key)
        if value is None or value.strip() == "":
            return default
        return value.strip()

    def str(self, key: str, default: str | None = None, required: bool = True) -> str:
        value = self._get(key, default)
        if value is None:
            if required:
                self.errors.append(f"{key} is required")
            return ""
        return value

    def choice(self, key: str, choices: tuple[str, ...], default: str | None = None) -> str:
        value = self.str(key, default).lower()
        if value and value not in choices:
            self.errors.append(f"{key}={value!r} is invalid; expected one of {choices}")
        return value

    def int(self, key: str, default: int | None = None, minimum: int | None = None) -> int:
        raw = self._get(key, None if default is None else str(default))
        if raw is None:
            self.errors.append(f"{key} is required")
            return 0
        try:
            value = int(raw)
        except ValueError:
            self.errors.append(f"{key}={raw!r} is not an integer")
            return 0
        if minimum is not None and value < minimum:
            self.errors.append(f"{key}={value} must be >= {minimum}")
        return value

    def float(
        self,
        key: str,
        default: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        exclusive_min: bool = False,
    ) -> float:
        raw = self._get(key, None if default is None else str(default))
        if raw is None:
            self.errors.append(f"{key} is required")
            return 0.0
        try:
            value = float(raw)
        except ValueError:
            self.errors.append(f"{key}={raw!r} is not a number")
            return 0.0
        if minimum is not None and (value <= minimum if exclusive_min else value < minimum):
            op = ">" if exclusive_min else ">="
            self.errors.append(f"{key}={value} must be {op} {minimum}")
        if maximum is not None and value > maximum:
            self.errors.append(f"{key}={value} must be <= {maximum}")
        return value

    def path(self, key: str, default: str | None = None) -> Path:
        value = self.str(key, default)
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path


def _collect_values(env_file: Path | None, overrides: dict[str, str] | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if env_file is not None and env_file.exists():
        values.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    # Real environment variables take precedence over the .env file, as with
    # python-dotenv's default ``override=False`` behaviour.
    for key in list(values) + [k for k in os.environ if k.isupper()]:
        if key in os.environ:
            values[key] = os.environ[key]
    if overrides:
        values.update(overrides)
    return values


def load_config(
    env_file: Path | str | None = DEFAULT_ENV_FILE,
    overrides: dict[str, str] | None = None,
) -> ExperimentConfig:
    """Load and validate the configuration.

    Raises:
        ConfigError: if any value is missing or invalid. All problems are
            reported at once.
    """
    env_path = Path(env_file) if env_file is not None else None
    values = _collect_values(env_path, overrides)
    r = _Reader(values)

    model_path = r.str("MODEL_PATH", "", required=False)
    revision = r.str("HF_REVISION", "", required=False) or None

    cfg = ExperimentConfig(
        model_path=model_path,
        hf_model_id=r.str("HF_MODEL_ID", "google/switch-base-8"),
        hf_revision=revision,
        model_dtype=r.choice("MODEL_DTYPE", VALID_DTYPES, "float32"),
        device=r.choice("DEVICE", VALID_DEVICES, "auto"),
        calibration_dataset_path=r.path("CALIBRATION_DATASET_PATH", "./data/calibration.jsonl"),
        evaluation_dataset_path=r.path(
            "EVALUATION_DATASET_PATH", "./data/evaluation/c4_support_eval.jsonl"
        ),
        calibration_max_samples=r.int("CALIBRATION_MAX_SAMPLES", 128, minimum=1),
        calibration_batch_size=r.int("CALIBRATION_BATCH_SIZE", 1, minimum=1),
        max_source_length=r.int("MAX_SOURCE_LENGTH", 256, minimum=8),
        max_target_length=r.int("MAX_TARGET_LENGTH", 128, minimum=4),
        c4_dataset_id=r.str("C4_DATASET_ID", "allenai/c4"),
        c4_config=r.str("C4_CONFIG", "en"),
        c4_split=r.str("C4_SPLIT", "validation"),
        c4_domain=r.str("C4_DOMAIN", "support"),
        c4_max_samples=r.int("C4_MAX_SAMPLES", 500, minimum=1),
        c4_max_scanned=r.int("C4_MAX_SCANNED", 100000, minimum=1),
        c4_min_text_length=r.int("C4_MIN_TEXT_LENGTH", 200, minimum=0),
        c4_max_text_length=r.int("C4_MAX_TEXT_LENGTH", 8000, minimum=1),
        c4_min_keyword_matches=r.int("C4_MIN_KEYWORD_MATCHES", 2, minimum=1),
        c4_support_keywords=_split_csv(r.str("C4_SUPPORT_KEYWORDS")),
        c4_url_keywords=_split_csv(r.str("C4_URL_KEYWORDS", "", required=False)),
        c4_allowed_domains=_split_csv(r.str("C4_ALLOWED_DOMAINS", "", required=False)),
        pruning_method=r.choice("PRUNING_METHOD", VALID_PRUNING_METHODS, "easyep"),
        prune_experts_per_layer=r.int("PRUNE_EXPERTS_PER_LAYER", 2, minimum=1),
        prune_layers=r.str("PRUNE_LAYERS", "first"),
        prune_unobserved_policy=r.choice(
            "PRUNE_UNOBSERVED_POLICY", VALID_UNOBSERVED_POLICIES, "exclude"
        ),
        replacement_method=r.choice("REPLACEMENT_METHOD", VALID_REPLACEMENT_METHODS, "bias"),
        bias_initialization=r.choice(
            "BIAS_INITIALIZATION", VALID_BIAS_INITIALIZATIONS, "gate_token_weighted_mean"
        ),
        bias_training_objective=r.choice(
            "BIAS_TRAINING_OBJECTIVE", VALID_TRAINING_OBJECTIVES, "layer_mse"
        ),
        bias_learning_rate=r.float("BIAS_LEARNING_RATE", 1e-3, minimum=0.0, exclusive_min=True),
        bias_training_steps=r.int("BIAS_TRAINING_STEPS", 100, minimum=0),
        bias_batch_size=r.int("BIAS_BATCH_SIZE", 1, minimum=1),
        bias_validation_fraction=r.float(
            "BIAS_VALIDATION_FRACTION", 0.1, minimum=0.0, maximum=0.9, exclusive_min=True
        ),
        bias_eval_interval=r.int("BIAS_EVAL_INTERVAL", 10, minimum=1),
        eval_max_samples=r.int("EVAL_MAX_SAMPLES", 500, minimum=1),
        eval_batch_size=r.int("EVAL_BATCH_SIZE", 1, minimum=1),
        eval_noise_density=r.float(
            "EVAL_NOISE_DENSITY", 0.15, minimum=0.0, maximum=0.9, exclusive_min=True
        ),
        eval_mean_span_length=r.float("EVAL_MEAN_SPAN_LENGTH", 3.0, minimum=1.0),
        eval_seed=r.int("EVAL_SEED", 42),
        eval_qualitative_samples=r.int("EVAL_QUALITATIVE_SAMPLES", 3, minimum=0),
        seed=r.int("SEED", 42),
        experiment_name=r.str("EXPERIMENT_NAME", "default"),
        pruned_model_path=r.path("PRUNED_MODEL_PATH", "./outputs/pruned-model"),
        experiment_dir=r.path("EXPERIMENT_DIR", "./outputs/experiments"),
        stats_dir=r.path("STATS_DIR", "./outputs/stats"),
        cache_dir=r.path("CACHE_DIR", "./outputs/cache"),
        log_level=r.str("LOG_LEVEL", "INFO").upper(),
        raw=dict(values),
    )

    if cfg.c4_min_text_length >= cfg.c4_max_text_length:
        r.errors.append("C4_MIN_TEXT_LENGTH must be smaller than C4_MAX_TEXT_LENGTH")
    if not cfg.c4_support_keywords:
        r.errors.append("C4_SUPPORT_KEYWORDS must contain at least one keyword")
    if "/" in cfg.experiment_name or cfg.experiment_name in (".", ".."):
        r.errors.append("EXPERIMENT_NAME must be a plain directory name")
    if cfg.model_path:
        local = Path(cfg.model_path).expanduser()
        if not local.is_dir():
            # Never silently fall back to downloading another model.
            r.errors.append(
                f"MODEL_PATH={cfg.model_path!r} is not an existing directory. "
                "Fix the path or leave MODEL_PATH empty to download HF_MODEL_ID."
            )
        elif not (local / "config.json").exists():
            r.errors.append(f"MODEL_PATH={cfg.model_path!r} does not contain a config.json")

    if r.errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(r.errors))
    return cfg


# ----------------------------------------------------------------------
# Command-line integration
# ----------------------------------------------------------------------
def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Path to the .env file (default: project root .env).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a configuration value, e.g. --set PRUNE_EXPERTS_PER_LAYER=3. Repeatable.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    overrides: dict[str, str] = {}
    for item in args.overrides:
        if "=" not in item:
            raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip().upper()] = value
    return load_config(args.env_file, overrides)
