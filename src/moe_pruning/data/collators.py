"""Batching for encoder-decoder examples.

Sequences are right-padded. This matters for Switch Transformers: the router
assigns expert capacity in sequence order, so right padding guarantees that
padding tokens can never take capacity away from real tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch

LABEL_PAD = -100


@dataclass
class Seq2SeqBatch:
    example_ids: list[int]
    input_ids: torch.Tensor  # [B, S_enc]
    attention_mask: torch.Tensor  # [B, S_enc], 1 = real token
    labels: torch.Tensor  # [B, S_dec], -100 = padding

    def model_inputs(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids.to(device),
            "attention_mask": self.attention_mask.to(device),
            "labels": self.labels.to(device),
        }

    def token_masks(self, device: torch.device | None = None) -> dict[str, torch.Tensor]:
        """Real-token masks per stack, aligned with each stack's hidden states.

        Decoder position j receives label j-1 (or the start token) as input, so
        positions with ``labels != -100`` are exactly the non-padding decoder
        positions.
        """
        masks = {"encoder": self.attention_mask.bool(), "decoder": self.labels != LABEL_PAD}
        if device is not None:
            masks = {k: v.to(device) for k, v in masks.items()}
        return masks


def collate_seq2seq(
    items: Sequence, pad_token_id: int
) -> Seq2SeqBatch:
    """Collate objects with ``example_id``, ``input_ids`` and ``labels`` lists."""
    enc_len = max(len(it.input_ids) for it in items)
    dec_len = max(len(it.labels) for it in items)
    input_ids = torch.full((len(items), enc_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(items), enc_len), dtype=torch.long)
    labels = torch.full((len(items), dec_len), LABEL_PAD, dtype=torch.long)
    for i, it in enumerate(items):
        input_ids[i, : len(it.input_ids)] = torch.tensor(it.input_ids)
        attention_mask[i, : len(it.input_ids)] = 1
        labels[i, : len(it.labels)] = torch.tensor(it.labels)
    return Seq2SeqBatch([it.example_id for it in items], input_ids, attention_mask, labels)


def iter_batches(items: Sequence, batch_size: int, pad_token_id: int) -> Iterator[Seq2SeqBatch]:
    for start in range(0, len(items), batch_size):
        yield collate_seq2seq(items[start : start + batch_size], pad_token_id)
