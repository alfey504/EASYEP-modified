"""Evaluate a model stage on the deterministic C4 support evaluation set.

Stages:
  original         the unmodified model (baseline)
  before_training  experts replaced (replace_experts.py)
  after_training   biases fine-tuned (train_biases.py; bias method only)

The objective comes from the model plugin's task: T5 span-corruption denoising
loss for encoder-decoder models (not perplexity), next-token loss (and
perplexity) for causal LMs. All stages use exactly the same examples (cached
once). Models are evaluated one at a time: if the baseline is missing it is
evaluated first and released before the pruned model is loaded.
"""

from __future__ import annotations

import gc
import logging
from datetime import datetime, timezone

import torch

from moe_pruning.evaluation.evaluator import build_reference_probe, evaluate_model, prepare_eval_examples
from moe_pruning.evaluation.metrics import compare_to_baseline, per_example_comparison, prediction_agreement
from moe_pruning.model.architecture import get_adapter
from moe_pruning.model.loader import get_model_plugin, load_original_model, load_tokenizer, open_original_checkpoint
from moe_pruning.serialization.checkpoint import load_pruned_model
from moe_pruning.utils.cli import bootstrap
from moe_pruning.utils.device import empty_cache, resolve_device
from moe_pruning.utils.paths import read_json, write_json
from moe_pruning.utils.reproducibility import environment_info, memory_info, timed

logger = logging.getLogger("evaluate")

STAGES = ("original", "before_training", "after_training")
STAGE_TITLES = {
    "original": "Original model",
    "before_training": "Pruned (before training)",
    "after_training": "Pruned (after bias training)",
}


def _title(stage: str, metrics: dict) -> str:
    method = metrics.get("replacement_method")
    return STAGE_TITLES[stage] if not method else f"{STAGE_TITLES[stage]} [{method}]"


def _fmt(value, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _add_args(parser):
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--force-baseline", action="store_true", help="Re-evaluate the original model.")


def _model_dir(paths, stage: str):
    if stage == "after_training" and paths.cfg.replacement_method != "bias":
        raise SystemExit(
            f"REPLACEMENT_METHOD={paths.cfg.replacement_method} has no training stage; "
            "use --stage before_training."
        )
    if stage == "before_training":
        return paths.require(paths.replaced_model_dir, "uv run python scripts/replace_experts.py")
    return paths.require(paths.trained_model_dir, "uv run python scripts/train_biases.py")


def run_stage(stage: str, cfg, paths, examples, settings, tokenizer, task) -> dict:
    device = resolve_device(cfg.device)
    probe = None
    manifest = None
    if stage == "original":
        loaded = load_original_model(cfg, device)
        model = loaded.model
    else:
        model, _, manifest = load_pruned_model(_model_dir(paths, stage), device)
        if manifest.get("replacement_method", "bias") != cfg.replacement_method:
            raise SystemExit(
                f"Checkpoint uses REPLACEMENT_METHOD={manifest.get('replacement_method', 'bias')}, "
                f"config has {cfg.replacement_method}. Re-run replace_experts.py."
            )
        if manifest["model_source"] != cfg.model_source:
            raise SystemExit(f"Checkpoint was built from {manifest['model_source']}, config uses {cfg.model_source}")
        adapter = get_adapter(model)
        layers = {layer.layer_id: layer for layer in adapter.discover_layers(model)}
        replaced = {layer_id: entry["expert_ids"] for layer_id, entry in manifest["layers"].items()}
        reader = open_original_checkpoint(cfg)
        probe = build_reference_probe(model, adapter, layers, replaced, reader, device)
        del reader

    with timed() as t:
        result = evaluate_model(
            model, examples, device, cfg.eval_batch_size, tokenizer.pad_token_id, task,
            adapter=get_adapter(model) if probe else None, probe=probe, desc=f"eval[{stage}]",
        )
    qualitative = task.qualitative(model, tokenizer, examples[: cfg.eval_qualitative_samples], device, cfg)
    metrics = {
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task.name,
        "metric_definition": task.metric_definition,
        **result.metrics,
        "layer_reconstruction": result.layer_probe,
        "eval_settings": settings,
        "model": {"source": cfg.model_source, "checkpoint_stage": manifest["stage"] if manifest else "original"},
        "replaced_experts": {k: v["expert_ids"] for k, v in manifest["layers"].items()} if manifest else {},
        "replacement_method": manifest.get("replacement_method", "bias") if manifest else None,
        "runtime_seconds": t["seconds"],
        "memory": memory_info(device),
        "environment": environment_info(device),
        "qualitative": qualitative,
    }
    write_json(paths.eval_metrics(stage), metrics)
    torch.save(
        {
            "example_ids": result.example_ids,
            "loss_sums": torch.tensor(result.loss_sums, dtype=torch.float64),
            "token_counts": torch.tensor(result.token_counts),
            "predictions": result.predictions,
            "settings_fingerprint": settings["fingerprint"],
        },
        paths.eval_per_example(stage),
    )
    logger.info(
        "[%s] mean token loss %.4f%s over %d examples / %d scored tokens (%.1fs)",
        stage, metrics["mean_token_loss"],
        f" (perplexity {metrics['perplexity']:.3f})" if "perplexity" in metrics else "",
        metrics["num_examples"], metrics["num_target_tokens"], t["seconds"],
    )
    del model
    gc.collect()
    empty_cache(device)
    return metrics


def _baseline_valid(paths, settings) -> bool:
    path = paths.eval_metrics("original")
    return path.exists() and read_json(path)["eval_settings"]["fingerprint"] == settings["fingerprint"]


def write_comparison(paths, settings) -> None:
    stages = {}
    for stage in STAGES:
        path = paths.eval_metrics(stage)
        if path.exists():
            m = read_json(path)
            if m["eval_settings"]["fingerprint"] == settings["fingerprint"]:
                stages[stage] = m
            else:
                logger.warning("Ignoring %s: evaluated with different settings", path.name)
    if "original" not in stages:
        return
    base = stages["original"]
    base_pe = torch.load(paths.eval_per_example("original"), weights_only=False)
    base_losses = base_pe["loss_sums"] / base_pe["token_counts"]
    summary = {}
    for stage, m in stages.items():
        entry = {
            "mean_token_loss": m["mean_token_loss"],
            "mean_example_loss": m["mean_example_loss"],
            "perplexity": m.get("perplexity"),
            **compare_to_baseline(m, base),
            "layer_reconstruction": m.get("layer_reconstruction"),
            "title": _title(stage, m),
        }
        if stage != "original":
            pe = torch.load(paths.eval_per_example(stage), weights_only=False)
            if pe["example_ids"] != base_pe["example_ids"]:
                raise RuntimeError(f"{stage} was evaluated on different examples than the baseline")
            entry.update(per_example_comparison(pe["loss_sums"] / pe["token_counts"], base_losses))
            entry["teacher_forced_top1_agreement_with_original"] = prediction_agreement(
                pe["predictions"], base_pe["predictions"]
            )
        summary[stage] = entry
    qualitative = []
    for i, base_q in enumerate(base["qualitative"]):
        row = {"example_id": base_q["example_id"], "inputs": base_q["inputs"]}
        for stage, m in stages.items():
            if i < len(m["qualitative"]):
                row[f"prediction_{stage}"] = m["qualitative"][i]["prediction"]
        qualitative.append(row)
    write_json(paths.eval_comparison_json, {"eval_settings": settings, "stages": summary, "qualitative": qualitative})

    has_ppl = base.get("perplexity") is not None
    task_settings = {k: v for k, v in settings.items()
                     if k not in ("dataset_path", "dataset_sha256", "tokenizer_source", "fingerprint",
                                  "num_examples", "num_target_tokens", "max_samples")}
    lines = [
        "# Evaluation comparison",
        "",
        f"Dataset: `{settings['dataset_path']}` ({settings['num_examples']} documents, "
        f"{settings['num_target_tokens']} scored tokens). Settings: "
        + ", ".join(f"{k}={v}" for k, v in task_settings.items()) + ".",
        "",
        f"Metric: {base.get('metric_definition', 'mean per-token loss')}.",
        "",
        "| Stage | Token loss |" + (" Perplexity |" if has_ppl else "")
        + " Δ vs original | Rel. increase | Top-1 agreement | Examples worse |",
        "|---|---|" + ("---|" if has_ppl else "") + "---|---|---|---|",
    ]
    for stage, e in summary.items():
        agree = e.get("teacher_forced_top1_agreement_with_original")
        worse = e.get("fraction_examples_worse")
        ppl = f" {e['perplexity']:.3f} |" if has_ppl else ""
        lines.append(
            f"| {e['title']} | {e['mean_token_loss']:.4f} |{ppl} {e['loss_difference']:+.4f} | "
            f"{e['relative_loss_increase_percent']:+.2f}% | {'-' if agree is None else f'{agree:.4f}'} | "
            f"{'-' if worse is None else f'{100 * worse:.1f}%'} |"
        )
    probe_rows = [(s, lid, v) for s, e in summary.items() for lid, v in (e["layer_reconstruction"] or {}).items()]
    if probe_rows:
        lines += [
            "",
            "## Layer output reconstruction on the evaluation set",
            "",
            "Error of each pruned layer's output vs. the original layer (original routing, all original "
            "experts) on the same (actual) layer input.",
            "",
            "| Stage | Layer | Tokens originally → removed experts | MSE / token | Rel. error (all tokens) | Rel. error (those tokens) |",
            "|---|---|---|---|---|---|",
        ]
        for stage, layer_id, v in probe_rows:
            lines.append(
                f"| {summary[stage]['title']} | {layer_id} | {v['tokens_routed_to_replaced_experts']}/{v['tokens']} | "
                f"{v['mse_per_token']:.3f} | {_fmt(v['relative_error_all_tokens'])} | "
                f"{_fmt(v['relative_error_replaced_tokens'])} |"
            )
    lines += ["", "## Qualitative examples", "",
              "_Greedy predictions of pre-trained (not instruction-tuned) models; they illustrate model "
              "behaviour and are not customer-support answers._", ""]
    for row in qualitative:
        lines += [f"### Example {row['example_id']}", ""]
        lines += [f"- **{label}:** {text[:500]}" for label, text in row["inputs"].items()]
        for stage in stages:
            if f"prediction_{stage}" in row:
                lines.append(f"- **{summary[stage]['title']}:** {row[f'prediction_{stage}']}")
        lines.append("")
    paths.eval_comparison_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Comparison written to %s", paths.eval_comparison_md)
    for stage, e in summary.items():
        logger.info(
            "  %-16s loss %.4f  (%+.4f, %+.2f%%)", stage, e["mean_token_loss"],
            e["loss_difference"], e["relative_loss_increase_percent"],
        )


def main() -> None:
    args, cfg, paths = bootstrap(__doc__, "evaluate", _add_args)
    tokenizer = load_tokenizer(cfg)
    task = get_model_plugin(cfg).task()
    examples, settings = prepare_eval_examples(cfg, tokenizer, paths, task)
    logger.info(
        "Evaluation set: %d examples, task %s (fingerprint %s)", len(examples), task.name, settings["fingerprint"]
    )

    baseline_ok = _baseline_valid(paths, settings) and not args.force_baseline
    if args.stage == "original":
        if baseline_ok:
            logger.info("Baseline already evaluated with identical settings (use --force-baseline to redo)")
        else:
            run_stage("original", cfg, paths, examples, settings, tokenizer, task)
    else:
        if not baseline_ok:
            logger.info("Baseline metrics missing or stale - evaluating the original model first")
            run_stage("original", cfg, paths, examples, settings, tokenizer, task)
        run_stage(args.stage, cfg, paths, examples, settings, tokenizer, task)
    write_comparison(paths, settings)


if __name__ == "__main__":
    main()
