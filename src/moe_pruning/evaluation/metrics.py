"""Loss and comparison metrics.

The reported loss is the mean per-token cross-entropy of the T5 span-corruption
(denoising) targets under teacher forcing. It is NOT autoregressive language
modelling perplexity and must not be reported as such.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def per_example_token_nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of token NLL and number of target tokens per example (padding = -100 ignored)."""
    nll = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="none"
    ).view(labels.shape)
    mask = labels != -100
    return (nll * mask).sum(dim=-1), mask.sum(dim=-1)


def summarize_losses(loss_sums: list[float], token_counts: list[int]) -> dict:
    total_tokens = sum(token_counts)
    per_example = [s / n for s, n in zip(loss_sums, token_counts) if n > 0]
    return {
        "mean_token_denoising_loss": sum(loss_sums) / max(total_tokens, 1),
        "mean_example_denoising_loss": sum(per_example) / max(len(per_example), 1),
        "num_examples": len(token_counts),
        "num_target_tokens": total_tokens,
    }


def compare_to_baseline(stage: dict, baseline: dict) -> dict:
    loss = stage["mean_token_denoising_loss"]
    base = baseline["mean_token_denoising_loss"]
    return {
        "loss_difference": loss - base,
        "relative_loss_increase_percent": 100.0 * (loss - base) / base if base else math.nan,
    }


def per_example_comparison(stage_losses: torch.Tensor, base_losses: torch.Tensor) -> dict:
    delta = stage_losses - base_losses
    return {
        "per_example_loss_delta_mean": float(delta.mean()),
        "per_example_loss_delta_median": float(delta.median()),
        "per_example_loss_delta_max": float(delta.max()),
        "fraction_examples_worse": float((delta > 1e-6).float().mean()),
    }


def prediction_agreement(stage_preds: list[list[int]], base_preds: list[list[int]]) -> float:
    """Fraction of target positions whose teacher-forced argmax matches the baseline."""
    same = total = 0
    for a, b in zip(stage_preds, base_preds):
        same += sum(int(x == y) for x, y in zip(a, b))
        total += len(b)
    return same / max(total, 1)
