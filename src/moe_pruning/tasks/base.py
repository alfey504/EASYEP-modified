"""Training/evaluation objective of a model family.

A task decides how calibration records and evaluation documents become model
inputs, which token positions are real (per stack), how the per-token loss is
computed and how qualitative predictions are produced. Model plugins pick a
task; the pipeline only talks to this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Sequence

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from moe_pruning.data.calibration import CalibrationRecord, TokenizedPair
from moe_pruning.data.collators import ModelBatch


class TaskObjective(ABC):
    #: short identifier recorded in manifests and evaluation fingerprints
    name: str = ""
    #: stacks whose MoE layers see real tokens ("encoder", "decoder")
    stacks: tuple[str, ...] = ()
    #: human-readable definition of the evaluation metric
    metric_definition: str = ""
    #: whether exp(mean token loss) is a meaningful perplexity for this objective
    reports_perplexity: bool = False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    @abstractmethod
    def tokenize_calibration(
        self,
        records: list[CalibrationRecord],
        tokenizer: PreTrainedTokenizerBase,
        max_source_length: int,
        max_target_length: int,
    ) -> list[TokenizedPair]:
        """``human`` is the input/prompt, ``response`` the teacher-forced target."""

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------
    @abstractmethod
    def collate(self, items: Sequence, pad_token_id: int) -> ModelBatch:
        """Right-padded batch with per-stack real-token masks."""

    def iter_batches(self, items: Sequence, batch_size: int, pad_token_id: int) -> Iterator[ModelBatch]:
        for start in range(0, len(items), batch_size):
            yield self.collate(items[start : start + batch_size], pad_token_id)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @abstractmethod
    def eval_settings(self, cfg) -> dict:
        """Task-specific settings that determine the evaluation examples."""

    @abstractmethod
    def build_eval_example(self, example_id: int, text: str, tokenizer: PreTrainedTokenizerBase, cfg):
        """One evaluation example (must expose ``example_id``, ``input_ids``, ``labels``)."""

    @abstractmethod
    def token_nll(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-example (sum of token NLL, number of scored tokens)."""

    @abstractmethod
    def predictions(self, logits: torch.Tensor, labels: torch.Tensor) -> list[list[int]]:
        """Per-example argmax token at every scored position (same order as ``token_nll``)."""

    @abstractmethod
    def qualitative(
        self, model: nn.Module, tokenizer: PreTrainedTokenizerBase, examples: list, device: torch.device, cfg
    ) -> list[dict]:
        """``[{"example_id", "inputs": {label: text}, "prediction"}]`` from greedy generation."""


def clean_decode(tokenizer: PreTrainedTokenizerBase, ids: list[int]) -> str:
    """Decode keeping special tokens such as T5 sentinels, but without pad/EOS noise."""
    text = tokenizer.decode(ids, skip_special_tokens=False)
    for tok in (tokenizer.pad_token, tokenizer.eos_token):
        if tok:
            text = text.replace(tok, "")
    return " ".join(text.split())
