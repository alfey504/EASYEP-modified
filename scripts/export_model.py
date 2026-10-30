"""Export the trained pruned model to PRUNED_MODEL_PATH and verify the export.

The exported checkpoint contains retained experts, all non-MoE weights, the
original routers and the trained bias vectors - but no weights of replaced
experts. It can only be loaded with
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
from safetensors import safe_open

from moe_pruning.data.calibration import prepare_calibration
from moe_pruning.data.collators import collate_seq2seq
from moe_pruning.serialization.checkpoint import (
    WEIGHTS_NAME,
    load_pruned_model,
    replaced_expert_prefixes,
    save_pruned_model,
)
from moe_pruning.utils.cli import bootstrap
from moe_pruning.utils.device import empty_cache, resolve_device
from moe_pruning.utils.paths import read_json, write_json

logger = logging.getLogger("export_model")

ATOL = 1e-4


def _add_args(parser):
    parser.add_argument(
        "--source", choices=("trained", "replaced"), default="trained",
        help="Export the trained model (default) or the model before bias training.",
    )


def _optional(path):
    return read_json(path) if path.exists() else None


def main() -> None:
    args, cfg, paths = bootstrap(__doc__, "export_model", _add_args)
    device = resolve_device(cfg.device)
    if args.source == "trained":
        source_dir = paths.require(paths.trained_model_dir, "uv run python scripts/train_biases.py")
    else:
        source_dir = paths.require(paths.replaced_model_dir, "uv run python scripts/replace_experts.py")

    model, tokenizer, manifest = load_pruned_model(source_dir, device)
    calib = prepare_calibration(cfg, tokenizer)
    probe = collate_seq2seq(calib.items[:4], tokenizer.pad_token_id)
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
    save_pruned_model(model, tokenizer, out_dir, {**manifest, "exported": True}, experiment_metadata)
    del model
    gc.collect()
    empty_cache(device)

    # 1. No original weights of replaced experts in the file.
    prefixes = replaced_expert_prefixes(manifest)
    with safe_open(str(out_dir / WEIGHTS_NAME), framework="pt") as fh:
        stored = list(fh.keys())
    leaked = [k for k in stored for p in prefixes if k.startswith(p) and k != p + "bias"]
    biases_present = all(p + "bias" in stored for p in prefixes)
    if leaked or not biases_present:
        raise RuntimeError(f"Export check failed: leaked={leaked} biases_present={biases_present}")

    # 2./3. Reload and compare.
    reloaded, _, _ = load_pruned_model(out_dir, device)
    with torch.no_grad():
        reloaded_logits = reloaded(**probe.model_inputs(device)).logits.float().cpu()
    max_diff = float((reloaded_logits - reference_logits).abs().max())
    ok = max_diff <= ATOL
    verification = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "replaced_expert_weights_absent": not leaked,
        "bias_vectors_present": biases_present,
        "reloaded_with": "moe_pruning.serialization.checkpoint.load_pruned_model",
        "reload_max_abs_logit_difference": max_diff,
        "tolerance": ATOL,
        "passed": ok,
        "device": str(device),
    }
    write_json(out_dir / "export_verification.json", verification)
    if not ok:
        raise RuntimeError(f"Reloaded model differs from the in-memory model (max |diff| {max_diff:.2e})")
    logger.info(
        "Export verified: %d replaced experts absent from the checkpoint, reload max |Δlogit| = %.2e",
        len(prefixes), max_diff,
    )
    logger.info("Exported model: %s", out_dir)


if __name__ == "__main__":
    main()
