r"""Gate-conditioned, token-weighted bias initialisation.

For the tokens :math:`T_e` routed to expert :math:`e` (training split, no
padding), with EASY-EP token contribution :math:`s_t` and the gate
:math:`g_{e,t}` that scales the expert output:

.. math::

    b_e = \frac{\sum_{t\in T_e} s_t g_{e,t}^2 E_e(h_t)}{\sum_{t\in T_e} s_t g_{e,t}^2}

Derivation. The replacement changes the layer output for token :math:`t` by
:math:`g_{e,t}(E_e(h_t) - b_e)`. Weighting the squared error by the token's
contribution gives

.. math::

    J(b) = \sum_t s_t \lVert g_{e,t}E_e(h_t) - g_{e,t} b\rVert^2
         = \sum_t s_t g_{e,t}^2 \lVert E_e(h_t) - b\rVert^2 .

:math:`J` is a convex quadratic; :math:`\nabla_b J = -2\sum_t w_t (E_e(h_t) - b) = 0`
with :math:`w_t = s_t g_{e,t}^2` gives the weighted mean above (unique when
:math:`\sum_t w_t > 0`).

Residual diagnostics are computed from running sums without storing tokens:
:math:`\sum_t w_t\lVert E_t - b\rVert^2 = \sum_t w_t\lVert E_t\rVert^2 - 2b^\top\sum_t w_tE_t + \lVert b\rVert^2\sum_t w_t`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import torch

from moe_pruning.scoring.statistics import LayerStatistics

logger = logging.getLogger(__name__)

EPS = 1e-12


class UnobservedExpertError(ValueError):
    pass


def weighted_mean(weighted_sum: torch.Tensor, weight_total: float | torch.Tensor) -> torch.Tensor | None:
    """sum_t w_t x_t / sum_t w_t, or None when the denominator is (numerically) zero."""
    total = float(weight_total)
    if not total > EPS:
        return None
    return weighted_sum / total


def weighted_residual(sq_sum: float, weighted_sum: torch.Tensor, weight_total: float, b: torch.Tensor) -> float:
    """sum_t w_t ||x_t - b||^2 from running sums (clamped at 0 for rounding)."""
    b = b.double()
    value = sq_sum - 2.0 * float(b @ weighted_sum.double()) + float(b @ b) * weight_total
    return max(value, 0.0)


@dataclass
class BiasInitResult:
    layer_id: str
    expert_id: int
    method: str  # method actually used
    fallback: bool
    token_count: int  # training tokens routed to the expert
    weight_sum: float  # sum s g^2
    bias_norm: float
    # Relative residuals (0 = perfect constant approximation, 1 = no better than zero vector)
    weighted_relative_residual: float | None  # sum w||E-b||^2 / sum w||E||^2
    gate_weighted_relative_residual: float | None  # sum g^2||E-b||^2 / sum g^2||E||^2
    # Absolute layer-output squared error introduced on training tokens, per token of the layer
    layer_output_sq_error_per_token: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def initialize_bias(
    stats: LayerStatistics,
    expert_id: int,
    allow_unobserved: bool = False,
) -> tuple[torch.Tensor, BiasInitResult]:
    """Compute b_e (float64) for one expert from accumulated statistics."""
    exp = stats.expert
    e = expert_id
    count = int(exp["count"][e])
    bias_count = int(exp["bias_count"][e])

    method = "gate_token_weighted_mean"
    fallback = False
    bias = weighted_mean(exp["w_output_sum"][e], exp["w_sum"][e])
    if bias is None:
        fallback = True
        bias = weighted_mean(exp["g2_output_sum"][e], exp["g2_sum"][e])
        method = "gate_weighted_mean"
        if bias is None and count > 0:
            bias = exp["output_sum"][e] / count
            method = "unweighted_mean_all_calibration_tokens"
        if bias is None:
            if not allow_unobserved:
                raise UnobservedExpertError(
                    f"{stats.layer_id} expert {e} was never activated on the calibration data; "
                    "refusing to invent a bias (PRUNE_UNOBSERVED_POLICY=exclude)."
                )
            bias = torch.zeros(stats.dim, dtype=torch.float64)
            method = "zero_vector_unobserved_expert"
        logger.warning(
            "%s expert %d: sum(s*g^2)=%.3g is zero; falling back to %s",
            stats.layer_id, e, float(exp["w_sum"][e]), method,
        )

    w_sum = float(exp["w_sum"][e])
    g2_sum = float(exp["g2_sum"][e])
    w_rel = g2_rel = per_token = None
    if w_sum > EPS and float(exp["w_output_sq_sum"][e]) > 0:
        w_rel = weighted_residual(float(exp["w_output_sq_sum"][e]), exp["w_output_sum"][e], w_sum, bias) / float(
            exp["w_output_sq_sum"][e]
        )
    if g2_sum > EPS and float(exp["g2_output_sq_sum"][e]) > 0:
        g2_res = weighted_residual(float(exp["g2_output_sq_sum"][e]), exp["g2_output_sum"][e], g2_sum, bias)
        g2_rel = g2_res / float(exp["g2_output_sq_sum"][e])
        per_token = g2_res / max(stats.layer["tokens"], 1.0)

    result = BiasInitResult(
        layer_id=stats.layer_id,
        expert_id=e,
        method=method,
        fallback=fallback,
        token_count=bias_count,
        weight_sum=w_sum,
        bias_norm=float(bias.norm()),
        weighted_relative_residual=w_rel,
        gate_weighted_relative_residual=g2_rel,
        layer_output_sq_error_per_token=per_token,
    )
    return bias.double(), result
