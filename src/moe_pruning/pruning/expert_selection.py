"""Per-layer EASY-EP ranking and selection of experts to replace."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from moe_pruning.model.architecture import MoELayerInfo
from moe_pruning.scoring.easyep import rank_experts, select_lowest
from moe_pruning.scoring.statistics import LayerStatistics


@dataclass
class ExpertScore:
    layer_id: str
    stack: str
    expert_id: int
    importance: float  # EASY-EP I_e = sum_t g * s * ||E||
    importance_share: float  # I_e / sum_e I_e within the layer
    rank: int  # 0 = least important (unobserved experts ranked last)
    token_count: int
    routing_frequency: float  # share of the layer's real tokens routed to e
    avg_gate: float | None
    avg_output_norm: float | None
    avg_token_contribution: float | None
    observed: bool
    layer_selected_for_pruning: bool
    selected_for_replacement: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_div(num: float, den: float) -> float | None:
    return num / den if den > 0 else None


def score_and_select(
    stats: dict[str, LayerStatistics],
    layers: list[MoELayerInfo],
    prune_layers: list[MoELayerInfo],
    experts_per_layer: int,
    allow_unobserved: bool,
) -> tuple[list[ExpertScore], dict[str, list[int]]]:
    """Rank experts independently in every layer; select in ``prune_layers`` only."""
    prune_ids = {layer.layer_id for layer in prune_layers}
    selection: dict[str, list[int]] = {}
    rows: list[ExpertScore] = []
    for layer in layers:
        st = stats[layer.layer_id]
        exp = st.expert
        importance = exp["importance"]
        observed = st.observed
        order = rank_experts(importance, observed)
        rank_of = {e: r for r, e in enumerate(order)}
        if layer.layer_id in prune_ids:
            selection[layer.layer_id] = select_lowest(
                importance, observed, experts_per_layer, allow_unobserved=allow_unobserved
            )
        chosen = set(selection.get(layer.layer_id, []))
        total_importance = float(importance.sum())
        tokens = st.layer["tokens"]
        for e in range(layer.num_experts):
            count = float(exp["count"][e])
            rows.append(
                ExpertScore(
                    layer_id=layer.layer_id,
                    stack=layer.stack,
                    expert_id=e,
                    importance=float(importance[e]),
                    importance_share=_safe_div(float(importance[e]), total_importance) or 0.0,
                    rank=rank_of[e],
                    token_count=int(count),
                    routing_frequency=_safe_div(count, tokens) or 0.0,
                    avg_gate=_safe_div(float(exp["gate_sum"][e]), count),
                    avg_output_norm=_safe_div(float(exp["norm_sum"][e]), count),
                    avg_token_contribution=_safe_div(float(exp["contribution_sum"][e]), count),
                    observed=bool(observed[e]),
                    layer_selected_for_pruning=layer.layer_id in prune_ids,
                    selected_for_replacement=e in chosen,
                )
            )
    return rows, selection
