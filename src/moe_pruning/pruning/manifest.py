"""Pruning manifest: the single description of what was replaced and how."""

from __future__ import annotations

from datetime import datetime, timezone

from moe_pruning.config import ExperimentConfig
from moe_pruning.model.architecture import MoEAdapter, MoELayerInfo

REPLACEMENT_DESCRIPTIONS = {
    "bias": "constant vector b_e; the parent MoE layer multiplies it by the router gate exactly once (g*b)",
    "drop": "expert removed; tokens routed to it receive zero MoE output (router unchanged)",
    "reroute": (
        "expert removed; a routing mask sends its tokens to their best remaining expert, scaled by that "
        "expert's original router probability (router weights unchanged; other tokens unaffected)"
    ),
}


def build_pruning_manifest(
    cfg: ExperimentConfig,
    adapter: MoEAdapter,
    model_type: str,
    layers: dict[str, MoELayerInfo],
    selection: dict[str, list[int]],
    bias_init: dict[str, dict[int, dict]],
    replacement_report: dict,
    calibration_fingerprint: str,
    stage: str,
) -> dict:
    return {
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_source": cfg.model_source,
        "hf_model_id": cfg.hf_model_id,
        "hf_revision": cfg.hf_revision,
        "model_type": model_type,
        "dtype": cfg.model_dtype,
        "pruning_method": cfg.pruning_method,
        "experts_per_layer": cfg.prune_experts_per_layer,
        "prune_layers_spec": cfg.prune_layers,
        "unobserved_policy": cfg.prune_unobserved_policy,
        "replacement_method": cfg.replacement_method,
        "replacement_description": REPLACEMENT_DESCRIPTIONS[cfg.replacement_method],
        "bias_initialization": cfg.bias_initialization if cfg.replacement_method == "bias" else None,
        "bias_formula": (
            "b_e = sum_t s_t g_{e,t}^2 E_e(h_t) / sum_t s_t g_{e,t}^2 (training split)"
            if cfg.replacement_method == "bias" else None
        ),
        "calibration_fingerprint": calibration_fingerprint,
        "replacement": replacement_report,
        "layers": {
            layer_id: {
                "module_path": layers[layer_id].module_path,
                "stack": layers[layer_id].stack,
                "replacement_method": cfg.replacement_method,
                "num_experts": layers[layer_id].num_experts,
                "expert_ids": sorted(expert_ids),
                "expert_state_prefixes": {
                    str(e): adapter.expert_state_prefix(layers[layer_id], e) for e in sorted(expert_ids)
                },
                "bias_initialization": (
                    {str(e): bias_init.get(layer_id, {}).get(e) for e in sorted(expert_ids)}
                    if cfg.replacement_method == "bias" else None
                ),
            }
            for layer_id, expert_ids in selection.items()
        },
    }
