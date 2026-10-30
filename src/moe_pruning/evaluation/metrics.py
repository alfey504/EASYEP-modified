"""Loss and comparison metrics.

The meaning of the loss is defined by the model's task objective: for
encoder-decoder models it is the T5 span-corruption denoising loss (not
perplexity); for causal LMs it is next-token cross-entropy, for which
perplexity = exp(loss) is also reported.
"""

from __future__ import annotations

import math

import torch


def summarize_losses(loss_sums: list[float], token_counts: list[int], perplexity: bool = False) -> dict:
    total_tokens = sum(token_counts)
    per_example = [s / n for s, n in zip(loss_sums, token_counts) if n > 0]
    mean = sum(loss_sums) / max(total_tokens, 1)
    out = {
        "mean_token_loss": mean,
        "mean_example_loss": sum(per_example) / max(len(per_example), 1),
        "num_examples": len(token_counts),
        "num_target_tokens": total_tokens,
    }
    if perplexity:
        out["perplexity"] = math.exp(mean)
    return out


def compare_to_baseline(stage: dict, baseline: dict) -> dict:
    loss = stage["mean_token_loss"]
    base = baseline["mean_token_loss"]
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
