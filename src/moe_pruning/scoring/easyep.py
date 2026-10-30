r"""EASY-EP expert importance (Dong et al., 2025, arXiv:2504.06792).

Verified against the official implementation (github.com/RUCAIBox/EASYEP,
``pruning/model_new.py`` and ``pruning/expert_selection.py``):

* Output-aware expert importance (``MoE.forward``): for every token and every
  selected expert, the gate weight actually used to scale the expert output
  and the L2 norm of the un-gated expert output,

  .. math:: c_{e,t} = g_{e,t}\,\lVert E_e(x_t)\rVert_2 .

* Expert-level token contribution (``Block.forward``): one minus the cosine
  similarity between the residual stream before the routed experts and the
  residual stream after adding the routed-expert output, clipped at zero
  (``simibr = max(1 - cos_sim_0_1, 0)``),

  .. math:: s_t = \max\bigl(1 - \cos(h_t,\; h_t + \textstyle\sum_e g_{e,t}E_e(x_t)),\; 0\bigr).

* Expert importance (``expert_selection.py``), summed over all calibration
  tokens of all samples:

  .. math:: I_e = \sum_t c_{e,t}\, s_t = \sum_{t \in T_e} g_{e,t}\, s_t\, \lVert E_e(x_t)\rVert_2 .

  EASY-EP keeps the top-``k`` experts by :math:`I_e`; here the experts with the
  lowest :math:`I_e` are selected for replacement.

Switch Transformer adaptation
-----------------------------
* Top-1 routing: :math:`T_e` is the set of tokens dispatched to ``e``;
  :math:`g_{e,t}` is the top-1 softmax probability that the HF implementation
  multiplies into the expert output (DeepSeek uses normalised top-k sigmoid
  weights times ``route_scale``; in both cases it is the weight that actually
  scales the output).
* DeepSeek's shared experts are excluded from :math:`s_t` in EASY-EP
  (``x_after_rmoe`` contains only routed experts). Switch has no shared
  experts, so the full MoE output is the routed output.
* :math:`h_t` is the residual stream entering ``SwitchTransformersLayerFF``
  (before its pre-norm), matching ``x`` before ``ffn_norm`` in DeepSeek.
* Tokens dropped by Switch's expert capacity have no expert, zero MoE output
  and therefore :math:`s_t = 0`; they contribute nothing (DeepSeek has no
  capacity limit, so this case does not occur there).
* EASY-EP's selection script uses the 25 longest calibration samples; here all
  configured calibration samples are used (CALIBRATION_MAX_SAMPLES).
* Padding tokens are excluded (EASY-EP runs unpadded single sequences).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_contribution(residual: torch.Tensor, moe_output: torch.Tensor) -> torch.Tensor:
    """s_t = max(1 - cos(h_t, h_t + y_t), 0), computed in float32. Shapes [..., D] -> [...]."""
    before = residual.float()
    after = before + moe_output.float()
    return (1.0 - F.cosine_similarity(before, after, dim=-1)).clamp_min(0.0)


def output_aware_importance(gate: torch.Tensor, expert_output: torch.Tensor) -> torch.Tensor:
    """c_{e,t} = g_{e,t} * ||E_e(x_t)||_2. Shapes gate [...], expert_output [..., D] -> [...]."""
    return gate.float() * expert_output.float().norm(dim=-1)


def rank_experts(importance: torch.Tensor, observed: torch.Tensor) -> list[int]:
    """Expert ids ordered from least to most important.

    Deterministic: ties are broken by expert id. Unobserved experts are placed
    last (they are never "least important" by default).
    """
    keys = [
        (0 if bool(observed[e]) else 1, float(importance[e]), e) for e in range(importance.numel())
    ]
    return [e for _, _, e in sorted(keys)]


def select_lowest(
    importance: torch.Tensor,
    observed: torch.Tensor,
    num_to_select: int,
    allow_unobserved: bool = False,
) -> list[int]:
    """Pick the ``num_to_select`` least important experts of one layer."""
    order = rank_experts(importance, observed)
    candidates = [e for e in order if allow_unobserved or bool(observed[e])]
    if allow_unobserved:
        # Unobserved experts have zero importance, so they rank first when allowed.
        candidates = sorted(candidates, key=lambda e: (float(importance[e]) if observed[e] else -1.0, e))
    if num_to_select > len(candidates):
        raise ValueError(
            f"Cannot select {num_to_select} experts: only {len(candidates)} eligible "
            f"(observed={int(observed.sum())}, allow_unobserved={allow_unobserved})"
        )
    if num_to_select >= importance.numel():
        raise ValueError("Refusing to replace every expert of a layer")
    return sorted(candidates[:num_to_select])
