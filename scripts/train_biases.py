"""Fine-tune only the replacement bias vectors with layer-level reconstruction MSE.

1. Run the calibration set through the ORIGINAL model and cache, for every
   pruned layer, its input hidden states and reference output (real tokens
   only). The cache is reused while model, calibration data and layers are
   unchanged.
2. Release the original model, load the replaced model (initialised biases).
3. Freeze every parameter except BiasExpert.bias; train each layer's biases
   jointly on the calibration training split; monitor the validation split.
   The C4 evaluation data is never used here.
4. Save the trained model (custom checkpoint) and a training report.
"""

from __future__ import annotations

import gc
import logging
from datetime import datetime, timezone

import torch

from moe_pruning.data.calibration import prepare_calibration
from moe_pruning.model.architecture import get_adapter
from moe_pruning.model.loader import get_model_plugin, load_original_model, load_pruned_for_config, load_tokenizer
from moe_pruning.pruning.bias_expert import BiasExpert
from moe_pruning.pruning.replacement import freeze_all_but_biases
from moe_pruning.serialization.checkpoint import save_pruned_model
from moe_pruning.training.bias_trainer import (
    cache_is_valid,
    capture_layer_references,
    load_layer_reference,
    save_layer_reference,
    train_layer_biases,
)
from moe_pruning.utils.cli import bootstrap
from moe_pruning.utils.device import empty_cache
from moe_pruning.utils.paths import write_json
from moe_pruning.utils.reproducibility import environment_info, memory_info, set_seed, timed

logger = logging.getLogger("train_biases")


def main() -> None:
    _, cfg, paths = bootstrap(__doc__, "train_biases")
    if cfg.replacement_method != "bias":
        raise SystemExit(
            f"REPLACEMENT_METHOD={cfg.replacement_method} has no trainable parameters; nothing to train. "
            "Evaluate with `scripts/evaluate.py --stage before_training`."
        )
    replaced_dir = paths.require(paths.replaced_model_dir, "uv run python scripts/replace_experts.py")

    tokenizer = load_tokenizer(cfg)
    task = get_model_plugin(cfg).task()
    calib = prepare_calibration(cfg, tokenizer, task)
    from moe_pruning.utils.paths import read_json

    replaced_manifest = read_json(replaced_dir / "pruning_manifest.json")
    if replaced_manifest["calibration_fingerprint"] != calib.metadata["fingerprint"]:
        raise SystemExit(
            "The replaced model was built with different calibration data/settings. "
            "Re-run collect_stats.py, score_experts.py and replace_experts.py."
        )
    if replaced_manifest.get("replacement_method", "bias") != "bias":
        raise SystemExit("The replaced model was not built with REPLACEMENT_METHOD=bias. Re-run replace_experts.py.")
    if replaced_manifest["model_source"] != cfg.model_source:
        raise SystemExit("The replaced model was built from a different base model.")
    selection = {layer_id: entry["expert_ids"] for layer_id, entry in replaced_manifest["layers"].items()}
    layer_ids = list(selection)

    # 1. Reference activations from the original model (cached on disk).
    cache_meta = {
        "model_source": cfg.model_source,
        "dtype": cfg.model_dtype,
        "calibration_fingerprint": calib.metadata["fingerprint"],
    }
    cache_dir = paths.layer_cache_dir
    if cache_is_valid(cache_dir, layer_ids, cache_meta):
        logger.info("Reusing cached layer references from %s", cache_dir)
    else:
        loaded = load_original_model(cfg)
        device = loaded.device
        adapter = get_adapter(loaded.model)
        layers = {layer.layer_id: layer for layer in adapter.discover_layers(loaded.model)}
        batches = list(task.iter_batches(calib.items, cfg.calibration_batch_size, tokenizer.pad_token_id))
        with timed() as t:
            refs = capture_layer_references(
                loaded.model, adapter, [layers[i] for i in layer_ids], batches, device
            )
        for ref in refs.values():
            save_layer_reference(ref, cache_dir, cache_meta)
        logger.info("Cached reference inputs/outputs of %d layer(s) in %.1fs", len(refs), t["seconds"])
        del loaded, adapter, refs
        gc.collect()
        empty_cache(device)

    # 2. Replaced model with initialised biases.
    model, _, manifest, device = load_pruned_for_config(cfg, replaced_dir)
    adapter = get_adapter(model)
    layers = {layer.layer_id: layer for layer in adapter.discover_layers(model)}

    # 3. Freeze everything but the replacement biases and verify.
    trainable = freeze_all_but_biases(model)
    bias_ids = {id(m.bias) for m in model.modules() if isinstance(m, BiasExpert)}
    if {id(p) for p in trainable.values()} != bias_ids:
        raise RuntimeError("Trainable parameters are not exactly the BiasExpert biases")
    for layer_id in layer_ids:
        router = adapter.get_router(model, layers[layer_id])
        assert not any(p.requires_grad for p in router.parameters()), "router must be frozen"
    n_trainable = sum(p.numel() for p in trainable.values())
    logger.info("Trainable parameters: %d in %d BiasExpert biases: %s", n_trainable, len(trainable), sorted(trainable))
    frozen_reference = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if any(name.startswith(layers[layer_id].module_path) for layer_id in layer_ids) and not p.requires_grad
    }

    set_seed(cfg.seed)
    results = []
    with timed() as total:
        for layer_id in layer_ids:
            ref, _ = load_layer_reference(cache_dir, layer_id)
            row_of = {eid: i for i, eid in enumerate(ref.example_ids)}
            train_rows = [row_of[e] for e in sorted(calib.train_ids)]
            val_rows = [row_of[e] for e in sorted(calib.validation_ids)]
            results.append(
                train_layer_biases(
                    model, adapter, layers[layer_id], selection[layer_id], ref, train_rows, val_rows, device,
                    cfg.bias_learning_rate, cfg.bias_training_steps, cfg.bias_batch_size, cfg.bias_eval_interval,
                    cfg.seed,
                )
            )
            del ref

    # Frozen parameters of the trained layers must be bit-identical.
    for name, before in frozen_reference.items():
        if not torch.equal(dict(model.named_parameters())[name].detach(), before):
            raise RuntimeError(f"Frozen parameter {name} changed during training")
    logger.info("Verified: routers and retained experts of the trained layers are unchanged")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": cfg.bias_training_objective,
        "objective_formula": "(1/N) sum_i ||F_original(H_i) - F_replaced(H_i)||_2^2 over real tokens",
        "note": (
            "Layer-level reconstruction is a local objective; with several pruned layers, upstream changes "
            "alter downstream hidden states and routing, which this objective does not capture."
        ),
        "optimizer": "Adam",
        "learning_rate": cfg.bias_learning_rate,
        "steps": cfg.bias_training_steps,
        "batch_size": cfg.bias_batch_size,
        "trainable_parameters": n_trainable,
        "trainable_parameter_names": sorted(trainable),
        "calibration": calib.metadata,
        "device": str(device),
        "training_seconds": total["seconds"],
        "memory": memory_info(device),
        "environment": environment_info(device),
        "layers": [r.to_dict() for r in results],
    }
    write_json(paths.training_report, report)

    for p in model.parameters():
        p.requires_grad_(False)
    manifest = {
        **manifest,
        "stage": "trained",
        "training": {k: v for k, v in report.items() if k not in ("layers", "environment")}
        | {"layers": [{k: v for k, v in r.to_dict().items() if k != "history"} for r in results]},
    }
    save_pruned_model(
        model, tokenizer, paths.trained_model_dir, manifest,
        {"stage": "train_biases", "config": cfg.to_dict()},
        checkpoint_format=cfg.checkpoint_format,
    )
    for r in results:
        logger.info(
            "%s: train %.4f -> %.4f | val %.4f -> %.4f | closed-form val %s",
            r.layer_id, r.initial_train_loss, r.final_train_loss, r.initial_validation_loss,
            r.final_validation_loss,
            f"{r.closed_form_validation_loss:.4f}" if r.closed_form_validation_loss is not None else "n/a",
        )
    logger.info("Trained model saved to %s (training %.1fs on %s)", paths.trained_model_dir, total["seconds"], device)


if __name__ == "__main__":
    main()
