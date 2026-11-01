"""Export the trained pruned model to PRUNED_MODEL_PATH and verify the export.

The exported checkpoint contains retained experts, all non-MoE weights, the
original routers and (bias method) the trained bias vectors - but no weights of
replaced experts. For reroute, the routing mask is restored from the manifest. It can only be loaded with
``moe_pruning.serialization.checkpoint.load_pruned_model``.

Verification (sequential, never two models in memory):
  1. the safetensors file holds no tensor of a replaced expert except its bias;
  2. logits of the in-memory model are computed on fixed calibration inputs;
  3. the in-memory model is released, the export is reloaded with the custom
     loader, and its logits must match within tolerance.
"""

from __future__ import annotations

import gc
import logging
from datetime import datetime, timezone

import torch

from moe_pruning.data.calibration import prepare_calibration
from moe_pruning.model.architecture import get_adapter
from moe_pruning.model.loader import get_model_plugin, load_pruned_for_config
from moe_pruning.serialization.checkpoint import (
    CHECKPOINT_FORMATS,
    DELTA_WEIGHTS_NAME,
    WEIGHTS_NAME,
    expected_bias_keys,
    removed_weight_violations,
    save_pruned_model,
    stored_tensor_shapes,
)
from moe_pruning.utils.cli import bootstrap
from moe_pruning.utils.device import empty_cache
from moe_pruning.utils.paths import read_json, write_json

logger = logging.getLogger("export_model")

# Reloading reproduces the same weights and ops; half precision gets a small margin.
ATOL = {torch.float32: 1e-4, torch.float16: 5e-2, torch.bfloat16: 5e-2}


def _add_args(parser):
    parser.add_argument(
        "--source", choices=("trained", "replaced"), default=None,
        help="Model to export. Default: trained for REPLACEMENT_METHOD=bias, replaced otherwise "
        "(drop/reroute have nothing to train).",
    )
    parser.add_argument(
        "--format", choices=CHECKPOINT_FORMATS, default=None,
        help="Checkpoint format of the export (default: CHECKPOINT_FORMAT). 'full' is self-contained; "
        "'delta' stores only the biases and needs the base model to load.",
    )


def _optional(path):
    return read_json(path) if path.exists() else None


def main() -> None:
    args, cfg, paths = bootstrap(__doc__, "export_model", _add_args)
    source = args.source or ("trained" if cfg.replacement_method == "bias" else "replaced")
    if source == "trained":
        source_dir = paths.require(paths.trained_model_dir, "uv run python scripts/train_biases.py")
    else:
        source_dir = paths.require(paths.replaced_model_dir, "uv run python scripts/replace_experts.py")

    model, tokenizer, manifest, device = load_pruned_for_config(cfg, source_dir)
    model_dtype = next(model.parameters()).dtype
    checkpoint_format = args.format or cfg.checkpoint_format
    task = get_model_plugin(cfg).task()
    calib = prepare_calibration(cfg, tokenizer, task)
    probe = task.collate(calib.items[:4], tokenizer.pad_token_id)
    with torch.no_grad():
        reference_logits = model(**probe.model_inputs(device)).logits.float().cpu()

    experiment_metadata = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "exported_from": str(source_dir),
        "config": cfg.to_dict(),
        "statistics_manifest": _optional(paths.expert_stats_manifest),
        "expert_selection": _optional(paths.expert_selection),
        "bias_initialization": _optional(paths.bias_initialization_report),
        "training_report": _optional(paths.training_report),
        "evaluation_comparison": _optional(paths.eval_comparison_json),
    }
    out_dir = cfg.pruned_model_path
    save_pruned_model(
        model, tokenizer, out_dir, {**manifest, "exported": True}, experiment_metadata,
        checkpoint_format=checkpoint_format,
    )
    del model
    gc.collect()
    empty_cache(device)

    # 1. Reload with the custom loader (this also validates keys and shapes).
    reloaded, _, _, device = load_pruned_for_config(cfg, out_dir)

    # 2. No original weights of replaced experts in the file (adapter-specific rules,
    #    e.g. per-expert keys for Switch, compacted fused tensors for OLMoE). A delta
    #    checkpoint may contain nothing but the bias vectors.
    bias_keys = expected_bias_keys(manifest)
    if checkpoint_format == "delta":
        shapes = stored_tensor_shapes(out_dir / DELTA_WEIGHTS_NAME)
        leaked = sorted(set(shapes) - set(bias_keys))
    else:
        shapes = stored_tensor_shapes(out_dir / WEIGHTS_NAME)
        leaked = removed_weight_violations(get_adapter(reloaded), reloaded, shapes, manifest)
    biases_present = all(k in shapes for k in bias_keys)
    if leaked or not biases_present:
        raise RuntimeError(f"Export check failed: leaked={leaked} biases_present={biases_present}")

    # 3. Same outputs as the in-memory model.
    with torch.no_grad():
        reloaded_logits = reloaded(**probe.model_inputs(device)).logits.float().cpu()
    max_diff = float((reloaded_logits - reference_logits).abs().max())
    tolerance = ATOL.get(model_dtype, 1e-4)
    ok = max_diff <= tolerance
    n_removed = sum(len(entry["expert_ids"]) for entry in manifest["layers"].values())
    verification = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "replaced_expert_weights_absent": not leaked,
        "bias_vectors_present": biases_present,
        "replacement_method": manifest.get("replacement_method", "bias"),
        "reloaded_with": "moe_pruning.serialization.checkpoint.load_pruned_model",
        "reload_max_abs_logit_difference": max_diff,
        "tolerance": tolerance,
        "checkpoint_format": checkpoint_format,
        "passed": ok,
        "device": str(device),
    }
    write_json(out_dir / "export_verification.json", verification)
    if not ok:
        raise RuntimeError(f"Reloaded model differs from the in-memory model (max |diff| {max_diff:.2e})")
    logger.info(
        "Export verified: %d replaced experts absent from the checkpoint, reload max |Δlogit| = %.2e",
        n_removed, max_diff,
    )
    logger.info("Exported model: %s", out_dir)


if __name__ == "__main__":
    main()
