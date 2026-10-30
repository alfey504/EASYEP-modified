"""Deterministic T5 span corruption for denoising evaluation.

``random_spans_noise_mask`` is a NumPy port of the reference implementation in
google-research/text-to-text-transfer-transformer (``t5.data.preprocessors``),
the objective switch-base-8 was pre-trained with. Each example gets its own
RNG seeded with ``(seed, example_index)``, so the mask of an example does not
depend on which other examples are evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from transformers import PreTrainedTokenizerBase


def random_spans_noise_mask(
    length: int, noise_density: float, mean_noise_span_length: float, rng: np.random.Generator
) -> np.ndarray:
    """Boolean mask of ``length`` with ~noise_density noise in spans of mean length."""
    orig_length = length
    length = max(length, 2)
    num_noise_tokens = int(round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = max(int(round(num_noise_tokens / mean_noise_span_length)), 1)
    num_nonnoise_tokens = length - num_noise_tokens

    def _random_segmentation(num_items: int, num_segments: int) -> np.ndarray:
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        rng.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
    interleaved = np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1).reshape(-1)
    span_starts = np.cumsum(interleaved)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = 1
    span_num = np.cumsum(span_start_indicator)
    is_noise = span_num % 2 == 1
    return is_noise[:orig_length]


def sentinel_ids(tokenizer: PreTrainedTokenizerBase, count: int = 100) -> list[int]:
    ids = tokenizer.convert_tokens_to_ids([f"<extra_id_{i}>" for i in range(count)])
    if any(i is None or i == tokenizer.unk_token_id for i in ids):
        raise ValueError("Tokenizer has no T5 <extra_id_N> sentinel tokens")
    return ids


def replace_spans_with_sentinels(tokens: np.ndarray, mask: np.ndarray, sentinels: list[int]) -> list[int]:
    """Replace each masked span by one sentinel (in order) and drop the rest of the span."""
    out: list[int] = []
    n_span = 0
    prev = False
    for tok, m in zip(tokens.tolist(), mask.tolist()):
        if m:
            if not prev:
                if n_span >= len(sentinels):
                    raise ValueError("More noise spans than available sentinel tokens")
                out.append(sentinels[n_span])
                n_span += 1
        else:
            out.append(tok)
        prev = m
    return out


@dataclass(frozen=True)
class CorruptedExample:
    example_id: int
    source_ids: list[int]  # uncorrupted (truncated) document tokens, no EOS
    input_ids: list[int]  # corrupted encoder input, ends with EOS
    labels: list[int]  # sentinel-delimited removed spans, ends with EOS
    noise_mask: list[bool]


def corrupt_example(
    example_id: int,
    token_ids: list[int],
    tokenizer: PreTrainedTokenizerBase,
    noise_density: float,
    mean_noise_span_length: float,
    seed: int,
    max_source_length: int,
    max_target_length: int,
) -> CorruptedExample:
    sentinels = sentinel_ids(tokenizer)
    eos = tokenizer.eos_token_id
    length = min(len(token_ids), max_source_length - 1)
    while True:
        tokens = np.asarray(token_ids[:length], dtype=np.int64)
        rng = np.random.default_rng([seed, example_id])
        mask = random_spans_noise_mask(length, noise_density, mean_noise_span_length, rng)
        inputs = replace_spans_with_sentinels(tokens, mask, sentinels) + [eos]
        targets = replace_spans_with_sentinels(tokens, ~mask, sentinels) + [eos]
        if len(targets) <= max_target_length or length <= 8:
            break
        length -= 8  # shorten the document until the target fits
    return CorruptedExample(example_id, tokens.tolist(), inputs, targets[:max_target_length], mask.tolist())
