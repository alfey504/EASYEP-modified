"""Architecture-independent helpers built on top of ``MoEAdapter.instrument``."""

from __future__ import annotations

import torch

from moe_pruning.model.architecture import MoEForwardRecord


def reconstruct_moe_output(record: MoEForwardRecord) -> torch.Tensor:
    """Recompute sum_k gate_k * E_k(x) from the recorded routing and expert outputs."""
    return (record.gate.unsqueeze(-1).to(record.expert_output.dtype) * record.expert_output).sum(dim=2)


def gate_consistency_error(record: MoEForwardRecord) -> dict[str, float]:
    """Compare the layer's real output with sum_k g_k * E_k(x).

    A near-zero error confirms (a) that recorded expert outputs are un-gated,
    (b) that the parent applies the gate exactly once and (c) that every
    recorded output was associated with the right token.
    """
    expected = reconstruct_moe_output(record).float()
    actual = record.moe_output.float()
    abs_err = (expected - actual).abs()
    scale = actual.abs().max().clamp_min(1e-12)
    return {
        "max_abs_error": float(abs_err.max()),
        "max_rel_error": float(abs_err.max() / scale),
    }


class RecordCollector:
    """Callback that keeps records in memory (for small probes and tests only)."""

    def __init__(self, to_cpu: bool = True):
        self.records: list[MoEForwardRecord] = []
        self.to_cpu = to_cpu

    def __call__(self, record: MoEForwardRecord) -> None:
        if self.to_cpu:
            record = MoEForwardRecord(
                layer_id=record.layer_id,
                residual=record.residual.cpu(),
                moe_input=record.moe_input.cpu(),
                moe_output=record.moe_output.cpu(),
                expert_index=record.expert_index.cpu(),
                gate=record.gate.cpu(),
                expert_output=record.expert_output.cpu(),
            )
        self.records.append(record)

    def by_layer(self) -> dict[str, list[MoEForwardRecord]]:
        out: dict[str, list[MoEForwardRecord]] = {}
        for record in self.records:
            out.setdefault(record.layer_id, []).append(record)
        return out


def gate_consistency_tolerance(dtype: torch.dtype) -> float:
    """Relative tolerance for ``gate_consistency_error`` at the model's precision.

    Recomputed expert outputs are summed in a different order than the model's
    kernel, so the two agree only to the dtype's rounding level (bfloat16 has
    ~3 significant digits).
    """
    if dtype in (torch.float16, torch.bfloat16):
        return 3e-2
    return 1e-4
