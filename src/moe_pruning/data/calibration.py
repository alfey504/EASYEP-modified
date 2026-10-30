"""Calibration dataset: JSONL records with ``human`` (encoder) and ``response`` (decoder target)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("human", "response")


class CalibrationDataError(ValueError):
    pass


@dataclass(frozen=True)
class CalibrationRecord:
    human: str
    response: str
    line_number: int


@dataclass
class CalibrationLoadReport:
    path: str
    lines_read: int = 0
    loaded: int = 0
    skipped_blank_lines: int = 0
    skipped_malformed_json: int = 0
    skipped_missing_fields: int = 0
    skipped_empty_fields: int = 0
    truncated_at_max_samples: bool = False
    problems: list[str] = field(default_factory=list)

    MAX_PROBLEMS = 20

    def problem(self, message: str) -> None:
        if len(self.problems) < self.MAX_PROBLEMS:
            self.problems.append(message)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def load_calibration_records(
    path: Path, max_samples: int | None = None
) -> tuple[list[CalibrationRecord], CalibrationLoadReport]:
    """Read and validate a calibration JSONL file line by line.

    Malformed or incomplete records are skipped and reported; the first few
    problems are kept with their line numbers.
    """
    if not path.exists():
        raise CalibrationDataError(
            f"Calibration dataset not found at {path}. Set CALIBRATION_DATASET_PATH in .env."
        )
    report = CalibrationLoadReport(path=str(path))
    records: list[CalibrationRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            report.lines_read += 1
            if not line.strip():
                report.skipped_blank_lines += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                report.skipped_malformed_json += 1
                report.problem(f"line {line_number}: invalid JSON ({exc.msg} at column {exc.colno})")
                continue
            if not isinstance(obj, dict):
                report.skipped_malformed_json += 1
                report.problem(f"line {line_number}: expected a JSON object, got {type(obj).__name__}")
                continue
            missing = [f for f in REQUIRED_FIELDS if f not in obj]
            if missing:
                report.skipped_missing_fields += 1
                report.problem(f"line {line_number}: missing field(s) {missing}")
                continue
            values = {f: obj[f] for f in REQUIRED_FIELDS}
            if not all(isinstance(v, str) and v.strip() for v in values.values()):
                report.skipped_empty_fields += 1
                report.problem(f"line {line_number}: 'human' and 'response' must be non-empty strings")
                continue
            records.append(CalibrationRecord(values["human"].strip(), values["response"].strip(), line_number))
            if max_samples is not None and len(records) >= max_samples:
                report.truncated_at_max_samples = True
                break
    report.loaded = len(records)
    skipped = (
        report.skipped_malformed_json + report.skipped_missing_fields + report.skipped_empty_fields
    )
    if skipped:
        logger.warning("Skipped %d invalid calibration record(s) in %s", skipped, path)
        for msg in report.problems:
            logger.warning("  %s", msg)
    if not records:
        raise CalibrationDataError(f"No valid calibration records in {path}")
    logger.info("Loaded %d calibration records from %s", len(records), path)
    return records, report


@dataclass(frozen=True)
class TokenizedPair:
    """One teacher-forced example: encoder ids and decoder labels (no padding)."""

    example_id: int
    input_ids: list[int]
    labels: list[int]


def tokenize_calibration(
    records: list[CalibrationRecord],
    tokenizer: PreTrainedTokenizerBase,
    max_source_length: int,
    max_target_length: int,
) -> list[TokenizedPair]:
    """Tokenize ``human`` as encoder input and ``response`` as decoder labels.

    The tokenizer appends EOS to both. The model builds the decoder input by
    shifting the labels right (teacher forcing).
    """
    out: list[TokenizedPair] = []
    truncated_src = truncated_tgt = 0
    for i, rec in enumerate(records):
        src = tokenizer(rec.human, truncation=False).input_ids
        tgt = tokenizer(text_target=rec.response, truncation=False).input_ids
        if len(src) > max_source_length:
            truncated_src += 1
            src = src[: max_source_length - 1] + [tokenizer.eos_token_id]
        if len(tgt) > max_target_length:
            truncated_tgt += 1
            tgt = tgt[: max_target_length - 1] + [tokenizer.eos_token_id]
        out.append(TokenizedPair(i, src, tgt))
    if truncated_src or truncated_tgt:
        logger.info(
            "Truncated %d source(s) to %d tokens and %d target(s) to %d tokens",
            truncated_src, max_source_length, truncated_tgt, max_target_length,
        )
    return out


def split_train_validation(
    items: list, validation_fraction: float, seed: int
) -> tuple[list, list]:
    """Deterministic shuffle-split; at least one item in each part when possible."""
    import random

    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    n_val = max(1, round(len(items) * validation_fraction)) if len(items) > 1 else 0
    val_idx = set(order[:n_val])
    train = [item for i, item in enumerate(items) if i not in val_idx]
    val = [item for i, item in enumerate(items) if i in val_idx]
    return train, val


@dataclass
class PreparedCalibration:
    """Tokenized calibration data with its deterministic train/validation split."""

    items: list[TokenizedPair]
    train_ids: set[int]
    validation_ids: set[int]
    load_report: CalibrationLoadReport
    metadata: dict

    @property
    def train(self) -> list[TokenizedPair]:
        return [it for it in self.items if it.example_id in self.train_ids]

    @property
    def validation(self) -> list[TokenizedPair]:
        return [it for it in self.items if it.example_id in self.validation_ids]


def prepare_calibration(cfg, tokenizer: PreTrainedTokenizerBase, task) -> PreparedCalibration:
    """Load, tokenize (with the model's task objective) and split the calibration set.

    The split is used consistently by every stage: EASY-EP importance uses all
    examples, bias initialisation and training use only the training split, and
    the validation split is held out for monitoring.
    """
    from moe_pruning.utils.paths import file_sha256, fingerprint

    records, report = load_calibration_records(cfg.calibration_dataset_path, cfg.calibration_max_samples)
    items = task.tokenize_calibration(records, tokenizer, cfg.max_source_length, cfg.max_target_length)
    train, val = split_train_validation(items, cfg.bias_validation_fraction, cfg.seed)
    metadata = {
        "path": str(cfg.calibration_dataset_path),
        "sha256": file_sha256(cfg.calibration_dataset_path),
        "max_samples": cfg.calibration_max_samples,
        "num_examples": len(items),
        "num_train": len(train),
        "num_validation": len(val),
        "validation_fraction": cfg.bias_validation_fraction,
        "split_seed": cfg.seed,
        "max_source_length": cfg.max_source_length,
        "max_target_length": cfg.max_target_length,
        "task": task.name,
        "input_tokens": sum(len(it.input_ids) for it in items),
        "target_tokens": sum(sum(1 for t in it.labels if t != -100) for it in items),
    }
    metadata["fingerprint"] = fingerprint({k: v for k, v in metadata.items() if k != "path"})
    return PreparedCalibration(
        items, {it.example_id for it in train}, {it.example_id for it in val}, report, metadata
    )
