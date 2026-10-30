"""Collect EASY-EP statistics for every sparse MoE layer of the original model.

Encoder layers see the ``human`` text; decoder layers see the teacher-forced
``response``. Padding tokens are excluded. Statistics are streamed into float64
accumulators (no per-token activations are stored).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import torch
from tqdm import tqdm

from moe_pruning.data.calibration import prepare_calibration
from moe_pruning.data.collators import iter_batches
from moe_pruning.model.architecture import get_adapter
from moe_pruning.model.instrumentation import gate_consistency_error
from moe_pruning.model.loader import load_original_model
from moe_pruning.scoring.statistics import ExpertStatisticsAccumulator
from moe_pruning.utils.cli import bootstrap
from moe_pruning.utils.paths import write_json
from moe_pruning.utils.reproducibility import environment_info, memory_info, timed

logger = logging.getLogger("collect_stats")


def main() -> None:
    _, cfg, paths = bootstrap(__doc__, "collect_stats")
    loaded = load_original_model(cfg)
    model, device = loaded.model, loaded.device
    adapter = get_adapter(model)
    layers = adapter.discover_layers(model)
    stack_of = {layer.layer_id: layer.stack for layer in layers}

    calib = prepare_calibration(cfg, loaded.tokenizer)
    logger.info(
        "Calibration: %d examples (%d train / %d validation), %d encoder + %d decoder tokens",
        calib.metadata["num_examples"], calib.metadata["num_train"], calib.metadata["num_validation"],
        calib.metadata["encoder_tokens"], calib.metadata["decoder_tokens"],
    )

    acc = ExpertStatisticsAccumulator(layers)
    current: dict = {}
    max_gate_err = 0.0

    def on_record(record):
        nonlocal max_gate_err
        stack = stack_of[record.layer_id]
        token_mask = current["masks"][stack]
        bias_mask = token_mask & current["train_rows"][:, None]
        acc.update(record, token_mask, bias_mask)
        max_gate_err = max(max_gate_err, gate_consistency_error(record)["max_rel_error"])

    total_loss = 0.0
    total_tokens = 0
    batches = list(iter_batches(calib.items, cfg.calibration_batch_size, loaded.tokenizer.pad_token_id))
    with timed() as t, torch.no_grad(), adapter.instrument(model, layers, on_record):
        for batch in tqdm(batches, desc="collecting statistics", unit="batch"):
            current["masks"] = batch.token_masks(device)
            current["train_rows"] = torch.tensor(
                [eid in calib.train_ids for eid in batch.example_ids], device=device
            )
            out = model(**batch.model_inputs(device))
            n = int(current["masks"]["decoder"].sum())
            total_loss += float(out.loss) * n
            total_tokens += n

    if max_gate_err > 1e-4:
        raise RuntimeError(f"Gate consistency check failed during collection (rel. error {max_gate_err:.2e})")

    paths.expert_stats.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"layers": acc.state()}, paths.expert_stats)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_source": loaded.source,
        "revision": loaded.revision,
        "dtype": cfg.model_dtype,
        "layers": [layer.to_dict() for layer in layers],
        "calibration": calib.metadata,
        "calibration_load_report": calib.load_report.to_dict(),
        "batch_size": cfg.calibration_batch_size,
        "calibration_teacher_forced_loss": total_loss / max(total_tokens, 1),
        "gate_consistency_max_rel_error": max_gate_err,
        "runtime_seconds": t["seconds"],
        "memory": memory_info(device),
        "environment": environment_info(device),
    }
    write_json(paths.expert_stats_manifest, manifest)

    for layer in layers:
        st = acc.stats[layer.layer_id]
        logger.info(
            "%-16s tokens=%6d dropped=%4d tokens/expert=%s",
            layer.layer_id, int(st.layer["tokens"]), int(st.layer["dropped_tokens"]),
            st.expert["count"].long().tolist(),
        )
    logger.info("Statistics saved to %s (%.1fs)", paths.expert_stats, t["seconds"])


if __name__ == "__main__":
    main()
