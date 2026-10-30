"""Batching.

Sequences are always right-padded. This matters for Switch Transformers: the
router assigns expert capacity in sequence order, so right padding guarantees
that padding tokens never take capacity away from real tokens. Causal models
also see real tokens first, so attention masking needs no position shifts.

Each batch carries per-stack real-token masks (``encoder`` / ``decoder``),
aligned with that stack's hidden states, so the pipeline never needs to know
how a model family lays out its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import torch

LABEL_PAD = -100


@dataclass
class ModelBatch:
    example_ids: list[int]
    input_ids: torch.Tensor  # [B, S]
    attention_mask: torch.Tensor  # [B, S], 1 = real token
    labels: torch.Tensor  # [B, S_labels], -100 = not scored
    masks: dict[str, torch.Tensor] = field(default_factory=dict)  # stack -> [B, S_stack] bool

    def model_inputs(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids.to(device),
            "attention_mask": self.attention_mask.to(device),
            "labels": self.labels.to(device),
        }

    def token_masks(self, device: torch.device | None = None) -> dict[str, torch.Tensor]:
        if device is None:
            return dict(self.masks)
        return {k: v.to(device) for k, v in self.masks.items()}


# Backwards-compatible name.
Seq2SeqBatch = ModelBatch


def _pad(items: Sequence, attr: str, value: int) -> torch.Tensor:
    length = max(len(getattr(it, attr)) for it in items)
    out = torch.full((len(items), length), value, dtype=torch.long)
    for i, it in enumerate(items):
        seq = getattr(it, attr)
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def collate_seq2seq(items: Sequence, pad_token_id: int) -> ModelBatch:
    """Encoder-decoder batch.

    Decoder position j receives label j-1 (or the start token) as input, so the
    positions with ``labels != -100`` are exactly the real decoder positions.
    """
    input_ids = _pad(items, "input_ids", pad_token_id)
    attention_mask = torch.zeros_like(input_ids)
    for i, it in enumerate(items):
        attention_mask[i, : len(it.input_ids)] = 1
    labels = _pad(items, "labels", LABEL_PAD)
    masks = {"encoder": attention_mask.bool(), "decoder": labels != LABEL_PAD}
    return ModelBatch([it.example_id for it in items], input_ids, attention_mask, labels, masks)


def collate_causal(items: Sequence, pad_token_id: int) -> ModelBatch:
    """Decoder-only batch: ``labels`` has the same length as ``input_ids``.

    All real tokens (prompt and response) are processed by the MoE layers, so
    the ``decoder`` mask is the attention mask; the labels decide what is scored.
    """
    if any(len(it.labels) != len(it.input_ids) for it in items):
        raise ValueError("Causal examples need labels aligned with input_ids")
    input_ids = _pad(items, "input_ids", pad_token_id)
    attention_mask = torch.zeros_like(input_ids)
    for i, it in enumerate(items):
        attention_mask[i, : len(it.input_ids)] = 1
    labels = _pad(items, "labels", LABEL_PAD)
    return ModelBatch([it.example_id for it in items], input_ids, attention_mask, labels,
                      {"decoder": attention_mask.bool()})


def iter_batches(items: Sequence, batch_size: int, pad_token_id: int) -> Iterator[ModelBatch]:
    """Encoder-decoder batches (kept for compatibility; prefer ``task.iter_batches``)."""
    for start in range(0, len(items), batch_size):
        yield collate_seq2seq(items[start : start + batch_size], pad_token_id)
