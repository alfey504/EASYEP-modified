"""Inspect the configured MoE model and write a machine-readable architecture report.

Besides the static structure, a probe forward pass verifies empirically that
the layer output equals sum_k g_k * E_k(x), i.e. where and how often the gate
is applied, and that recorded expert outputs map to the right tokens.
"""

from __future__ import annotations

import logging

import torch

from moe_pruning.data.calibration import CalibrationRecord
from moe_pruning.model.architecture import get_adapter
from moe_pruning.model.instrumentation import RecordCollector, gate_consistency_error, gate_consistency_tolerance
from moe_pruning.model.loader import load_original_model
from moe_pruning.utils.cli import bootstrap
from moe_pruning.utils.paths import write_json
from moe_pruning.utils.reproducibility import environment_info, memory_info

logger = logging.getLogger("inspect_model")

PROBE_SOURCE = "How do I reset my password if I no longer have access to my email account?"
PROBE_TARGET = "Contact customer support and verify your identity to recover the account."


def main() -> None:
    _, cfg, paths = bootstrap(__doc__, "inspect_model")
    loaded = load_original_model(cfg)
    model, tok, device = loaded.model, loaded.tokenizer, loaded.device
    adapter = get_adapter(model)

    report = adapter.describe(model)
    layers = adapter.discover_layers(model)

    # --- Probe forward pass (inputs built by the plugin's task) -------------
    task = loaded.plugin.task()
    pair = task.tokenize_calibration([CalibrationRecord(PROBE_SOURCE, PROBE_TARGET, 0)], tok, 256, 128)
    batch = task.collate(pair, tok.pad_token_id)
    collector = RecordCollector()
    with torch.no_grad(), adapter.instrument(model, layers, collector):
        model(**batch.model_inputs(device))

    probe = {}
    for layer_id, records in collector.by_layer().items():
        rec = records[0]
        idx = rec.expert_index
        counts = torch.bincount(idx[idx >= 0].flatten(), minlength=layers[0].num_experts)
        probe[layer_id] = {
            "tokens": int(idx[..., 0].numel()),
            "dropped_by_capacity": int((idx < 0).all(dim=-1).sum()),
            "tokens_per_expert": counts.tolist(),
            "mean_gate": float(rec.gate[idx >= 0].mean()) if (idx >= 0).any() else None,
            "expert_output_shape": list(rec.expert_output.shape),
            **gate_consistency_error(rec),
        }
    max_err = max(p["max_rel_error"] for p in probe.values())
    gate_ok = max_err < gate_consistency_tolerance(loaded.dtype)
    report["probe"] = {
        "source": PROBE_SOURCE,
        "target": PROBE_TARGET,
        "layers": probe,
        "gate_applied_once_verified": gate_ok,
        "max_rel_error": max_err,
    }
    report["model_source"] = loaded.source
    report["model_plugin"] = loaded.plugin.name
    report["task"] = task.name
    report["revision"] = loaded.revision
    report["dtype"] = str(loaded.dtype).replace("torch.", "")
    report["environment"] = environment_info(device)
    report["memory"] = memory_info(device)
    write_json(paths.architecture_report, report)

    # --- Human-readable summary ---------------------------------------------
    first = layers[0]
    logger.info(
        "%s | plugin %s | %s | encoder-decoder=%s | hidden=%d expert=%d->%d->%d | experts=%d top-%d capacity=%s",
        report["model_class"], loaded.plugin.name, report["adapter"], report["is_encoder_decoder"],
        first.hidden_dim, first.expert_input_dim, first.expert_intermediate_dim, first.expert_output_dim,
        first.num_experts, first.top_k, first.expert_capacity,
    )
    logger.info(
        "Sparse MoE layers: %d (encoder %d, decoder %d)",
        report["num_sparse_layers"], report["num_encoder_sparse_layers"], report["num_decoder_sparse_layers"],
    )
    for layer in layers:
        p = probe[layer.layer_id]
        logger.info(
            "  [%2d] %-16s %-40s experts=%d in=%d mid=%d out=%d | probe tokens/expert=%s rel.err=%.1e",
            layer.position, layer.layer_id, layer.module_path, layer.num_experts,
            layer.expert_input_dim, layer.expert_intermediate_dim, layer.expert_output_dim,
            p["tokens_per_expert"], p["max_rel_error"],
        )
    logger.info("Gate application: %s", report["routing"]["gate_application"])
    if gate_ok:
        logger.info("Verified: layer output == sum_k g_k * E_k(x) (max rel. error %.2e)", max_err)
    else:
        logger.error("Gate consistency check FAILED (max rel. error %.2e) - do not prune", max_err)
    logger.info("Architecture report written to %s", paths.architecture_report)
    if not gate_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
