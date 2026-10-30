"""Remove the selected experts and install their replacement (REPLACEMENT_METHOD).

* bias:    b_e = sum s g^2 E / sum s g^2 from the collected statistics, installed
           as BiasExpert (the parent layer applies the gate once: g*b).
* drop:    parameter-free ZeroExpert; routed tokens get zero MoE output.
* reroute: ZeroExpert + routing mask; tokens go to their best remaining expert.

The original expert weights are dropped in every case. The behaviour is
verified on a few calibration examples and the model is saved in the custom
checkpoint format.
"""

from __future__ import annotations

import logging

import torch

from moe_pruning.data.calibration import prepare_calibration
from moe_pruning.data.collators import collate_seq2seq
from moe_pruning.model.architecture import get_adapter
from moe_pruning.model.instrumentation import RecordCollector, gate_consistency_error
from moe_pruning.model.loader import load_original_model
from moe_pruning.pruning.bias_initialization import initialize_bias
from moe_pruning.pruning.manifest import build_pruning_manifest
from moe_pruning.pruning.replacement import replace_experts
from moe_pruning.scoring.statistics import load_statistics
from moe_pruning.serialization.checkpoint import save_pruned_model
from moe_pruning.utils.cli import bootstrap
from moe_pruning.utils.paths import read_json, write_json
from moe_pruning.utils.reproducibility import memory_info

logger = logging.getLogger("replace_experts")


def load_and_validate_selection(cfg, paths) -> tuple[dict, dict]:
    stats_manifest = read_json(paths.require(paths.expert_stats_manifest, "uv run python scripts/collect_stats.py"))
    selection = read_json(paths.require(paths.expert_selection, "uv run python scripts/score_experts.py"))
    problems = []
    if selection["model_source"] != cfg.model_source:
        problems.append(f"selection model {selection['model_source']} != configured {cfg.model_source}")
    if selection["experts_per_layer"] != cfg.prune_experts_per_layer:
        problems.append(
            f"selection used PRUNE_EXPERTS_PER_LAYER={selection['experts_per_layer']}, "
            f"config has {cfg.prune_experts_per_layer}"
        )
    if selection["prune_layers_spec"] != cfg.prune_layers:
        problems.append(f"selection used PRUNE_LAYERS={selection['prune_layers_spec']}, config has {cfg.prune_layers}")
    if selection["calibration_fingerprint"] != stats_manifest["calibration"]["fingerprint"]:
        problems.append("expert_selection.json is older than the collected statistics")
    if problems:
        raise SystemExit("Stale or inconsistent artifacts - re-run score_experts.py:\n  " + "\n  ".join(problems))
    return selection, stats_manifest


def main() -> None:
    _, cfg, paths = bootstrap(__doc__, "replace_experts")
    selection_doc, stats_manifest = load_and_validate_selection(cfg, paths)
    stats = load_statistics(torch.load(paths.expert_stats, weights_only=True)["layers"])

    loaded = load_original_model(cfg)
    model, device = loaded.model, loaded.device
    adapter = get_adapter(model)
    layers = {layer.layer_id: layer for layer in adapter.discover_layers(model)}

    # Verify the selection against the actual model.
    selection: dict[str, list[int]] = {}
    for layer_id, entry in selection_doc["layers"].items():
        if layer_id not in layers:
            raise SystemExit(f"Selected layer {layer_id} does not exist in the model")
        experts = entry["selected_experts"]
        if len(set(experts)) != len(experts) or not all(0 <= e < layers[layer_id].num_experts for e in experts):
            raise SystemExit(f"Invalid expert ids {experts} for {layer_id}")
        if entry["layer"] != layers[layer_id].to_dict():
            raise SystemExit(f"Layer description of {layer_id} changed since collect_stats.py")
        selection[layer_id] = experts

    method = cfg.replacement_method
    logger.info("Replacement method: %s", method)

    # Bias initialisation from the streamed statistics (bias method only).
    allow_unobserved = cfg.prune_unobserved_policy == "allow"
    biases: dict[str, dict[int, torch.Tensor]] = {}
    init_report: dict[str, dict[int, dict]] = {}
    if method == "bias":
        for layer_id, experts in selection.items():
            for e in experts:
                bias, result = initialize_bias(stats[layer_id], e, allow_unobserved)
                biases.setdefault(layer_id, {})[e] = bias.float()
                init_report.setdefault(layer_id, {})[e] = result.to_dict()
                logger.info(
                    "%s expert %d: %s over %d tokens | ||b||=%.2f | weighted rel. residual=%s | "
                    "gate-weighted rel. residual=%s",
                    layer_id, e, result.method, result.token_count, result.bias_norm,
                    f"{result.weighted_relative_residual:.3f}" if result.weighted_relative_residual is not None else "n/a",
                    f"{result.gate_weighted_relative_residual:.3f}"
                    if result.gate_weighted_relative_residual is not None else "n/a",
                )

    report = replace_experts(model, adapter, layers, selection, method, biases if method == "bias" else None)

    # Probe the replaced layers on a few calibration examples.
    calib = prepare_calibration(cfg, loaded.tokenizer)
    probe = collate_seq2seq(calib.items[:4], loaded.tokenizer.pad_token_id)
    collector = RecordCollector()
    pruned_layers = [layers[layer_id] for layer_id in selection]
    with torch.no_grad(), adapter.instrument(model, pruned_layers, collector):
        model(**probe.model_inputs(device))
    for rec in collector.records:
        err = gate_consistency_error(rec)["max_rel_error"]
        if err > 1e-4:
            raise RuntimeError(f"{rec.layer_id}: gate consistency failed after replacement ({err:.2e})")
        idx = rec.expert_index[..., 0]
        removed = selection[rec.layer_id]
        if method == "bias":
            for e in removed:
                routed = idx == e
                if routed.any():
                    outs = rec.expert_output[..., 0, :][routed]
                    expected = biases[rec.layer_id][e].to(outs.device, outs.dtype).expand_as(outs)
                    torch.testing.assert_close(outs, expected)
        elif method == "drop":
            for e in removed:
                routed = idx == e
                if routed.any() and rec.moe_output[routed].abs().max() != 0:
                    raise RuntimeError(f"{rec.layer_id}: dropped expert {e} produced a non-zero output")
        else:  # reroute
            if any(bool((idx == e).any()) for e in removed):
                raise RuntimeError(f"{rec.layer_id}: tokens were routed to a removed expert")
            with torch.no_grad():
                orig_idx, orig_gate = adapter.route(
                    model, layers[rec.layer_id], rec.moe_input.to(device), apply_mask=False
                )
            orig_idx, orig_gate = orig_idx[..., 0].cpu(), orig_gate[..., 0].cpu()
            # Tokens whose original expert is kept: same expert and gate (unless now capacity-dropped).
            kept = (orig_idx >= 0) & ~torch.isin(orig_idx, torch.tensor(removed, device=orig_idx.device)) & (idx >= 0)
            if not torch.equal(idx[kept], orig_idx[kept]) or not torch.equal(rec.gate[..., 0][kept], orig_gate[kept]):
                raise RuntimeError(f"{rec.layer_id}: rerouting changed tokens whose expert was not removed")
    logger.info("Verified replacement behaviour of %d layer(s) (method=%s)", len(collector.records), method)

    write_json(paths.bias_initialization_report, init_report)
    manifest = build_pruning_manifest(
        cfg, adapter, model.config.model_type, layers, selection, init_report,
        report.to_dict(), stats_manifest["calibration"]["fingerprint"], stage="replaced_before_training",
    )
    save_pruned_model(
        model, loaded.tokenizer, paths.replaced_model_dir, manifest,
        {"stage": "replace_experts", "config": cfg.to_dict(), "memory": memory_info(device)},
    )
    logger.info(
        "Removed %d parameters (%.2f%% of %d). Replaced model saved to %s",
        report.removed_parameters - report.added_parameters,
        100 * (report.parameters_before - report.parameters_after) / report.parameters_before,
        report.parameters_before, paths.replaced_model_dir,
    )


if __name__ == "__main__":
    main()
