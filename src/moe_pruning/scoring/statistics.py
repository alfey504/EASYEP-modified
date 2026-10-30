r"""Streaming per-layer, per-expert statistics.

Nothing per-token is kept: every quantity needed for EASY-EP scoring and for
the bias initialisation is a running sum (float64, on CPU), so memory does not
grow with the calibration set.

For each expert e (over real, non-padding tokens t routed to e):

=====================  =========================================================
``count``              :math:`|T_e|`
``gate_sum``           :math:`\sum g`
``norm_sum``           :math:`\sum \lVert E\rVert`
``contribution_sum``   :math:`\sum s`
``importance``         :math:`\sum g\,s\,\lVert E\rVert`  (EASY-EP :math:`I_e`)
``output_sum``         :math:`\sum E`                      (unweighted mean, comparison)
``output_sq_sum``      :math:`\sum \lVert E\rVert^2`
``w_sum``              :math:`\sum s g^2`                  (bias-init denominator)
``w_output_sum``       :math:`\sum s g^2 E`                (bias-init numerator)
``w_output_sq_sum``    :math:`\sum s g^2 \lVert E\rVert^2`
``g2_sum``             :math:`\sum g^2`
``g2_output_sum``      :math:`\sum g^2 E`
``g2_output_sq_sum``   :math:`\sum g^2 \lVert E\rVert^2`
=====================  =========================================================

The ``w_*`` and ``g2_*`` sums are accumulated only over tokens flagged for bias
initialisation (the calibration training split), so the validation split stays
unseen by the initialisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from moe_pruning.model.architecture import MoEForwardRecord, MoELayerInfo
from moe_pruning.scoring.easyep import output_aware_importance, token_contribution

EXPERT_SCALARS = (
    "count",
    "gate_sum",
    "norm_sum",
    "contribution_sum",
    "importance",
    "output_sq_sum",
    "bias_count",
    "w_sum",
    "w_output_sq_sum",
    "g2_sum",
    "g2_output_sq_sum",
)
EXPERT_VECTORS = ("output_sum", "w_output_sum", "g2_output_sum")
LAYER_SCALARS = ("tokens", "dropped_tokens", "contribution_sum_all")


@dataclass
class LayerStatistics:
    layer_id: str
    num_experts: int
    dim: int
    expert: dict[str, torch.Tensor] = field(default_factory=dict)
    layer: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.expert:
            for name in EXPERT_SCALARS:
                self.expert[name] = torch.zeros(self.num_experts, dtype=torch.float64)
            for name in EXPERT_VECTORS:
                self.expert[name] = torch.zeros(self.num_experts, self.dim, dtype=torch.float64)
        if not self.layer:
            self.layer = {name: 0.0 for name in LAYER_SCALARS}

    @property
    def observed(self) -> torch.Tensor:
        return self.expert["count"] > 0

    def to_state(self) -> dict:
        return {
            "layer_id": self.layer_id,
            "num_experts": self.num_experts,
            "dim": self.dim,
            "expert": self.expert,
            "layer": self.layer,
        }

    @classmethod
    def from_state(cls, state: dict) -> "LayerStatistics":
        return cls(state["layer_id"], state["num_experts"], state["dim"], state["expert"], state["layer"])


class ExpertStatisticsAccumulator:
    def __init__(self, layers: list[MoELayerInfo]):
        self.layers = {layer.layer_id: layer for layer in layers}
        self.stats = {
            layer.layer_id: LayerStatistics(layer.layer_id, layer.num_experts, layer.expert_output_dim)
            for layer in layers
        }

    def update(
        self,
        record: MoEForwardRecord,
        token_mask: torch.Tensor,
        bias_mask: torch.Tensor | None = None,
    ) -> None:
        """Add one forward record.

        Args:
            token_mask: [B, S] bool, True for real (non-padding) tokens.
            bias_mask: [B, S] bool, tokens used for the bias initialisation sums
                (defaults to ``token_mask``). Must be a subset of ``token_mask``.
        """
        stats = self.stats[record.layer_id]
        if bias_mask is None:
            bias_mask = token_mask
        valid = token_mask.reshape(-1).bool()
        for_bias = (bias_mask.reshape(-1).bool() & valid)[valid]

        dim = record.moe_output.shape[-1]
        k = record.expert_index.shape[-1]
        valid = valid.to(record.moe_output.device)
        # Per-token quantities are computed on the model device in float32, then
        # moved to CPU for float64 accumulation (MPS has no float64).
        s = token_contribution(
            record.residual.reshape(-1, dim)[valid], record.moe_output.reshape(-1, dim)[valid]
        ).cpu()
        index = record.expert_index.reshape(-1, k)[valid].cpu()
        gate = record.gate.reshape(-1, k)[valid].float().cpu()
        outputs = record.expert_output.reshape(-1, k, dim)[valid].float().cpu()
        for_bias = for_bias.cpu()

        stats.layer["tokens"] += float(valid.sum())
        stats.layer["dropped_tokens"] += float((index < 0).all(dim=-1).sum())
        stats.layer["contribution_sum_all"] += float(s.double().sum())

        exp = stats.expert
        for slot in range(k):
            idx = index[:, slot]
            routed = idx >= 0
            if not routed.any():
                continue
            e = idx[routed]
            g = gate[routed, slot].double()
            out = outputs[routed, slot].double()
            s_t = s[routed].double()
            fb = for_bias[routed]
            norm = out.norm(dim=-1)
            norm_sq = norm.square()
            c = output_aware_importance(g, out).double()

            exp["count"].index_add_(0, e, torch.ones_like(g))
            exp["gate_sum"].index_add_(0, e, g)
            exp["norm_sum"].index_add_(0, e, norm)
            exp["contribution_sum"].index_add_(0, e, s_t)
            exp["importance"].index_add_(0, e, c * s_t)
            exp["output_sum"].index_add_(0, e, out)
            exp["output_sq_sum"].index_add_(0, e, norm_sq)

            if fb.any():
                eb, gb, ob, sb, nb = e[fb], g[fb], out[fb], s_t[fb], norm_sq[fb]
                g2 = gb.square()
                w = sb * g2
                exp["bias_count"].index_add_(0, eb, torch.ones_like(gb))
                exp["w_sum"].index_add_(0, eb, w)
                exp["w_output_sum"].index_add_(0, eb, w[:, None] * ob)
                exp["w_output_sq_sum"].index_add_(0, eb, w * nb)
                exp["g2_sum"].index_add_(0, eb, g2)
                exp["g2_output_sum"].index_add_(0, eb, g2[:, None] * ob)
                exp["g2_output_sq_sum"].index_add_(0, eb, g2 * nb)

    def state(self) -> dict:
        return {layer_id: s.to_state() for layer_id, s in self.stats.items()}


def load_statistics(state: dict) -> dict[str, LayerStatistics]:
    return {layer_id: LayerStatistics.from_state(s) for layer_id, s in state.items()}
