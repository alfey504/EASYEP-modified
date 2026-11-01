"""Deterministic evaluation (task-defined objective) and layer-reconstruction probing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import nn
from tqdm import tqdm

from moe_pruning.config import ExperimentConfig
from moe_pruning.data.c4_support import load_eval_documents
from moe_pruning.evaluation.metrics import summarize_losses
from moe_pruning.model.architecture import MoEAdapter, MoEForwardRecord, MoELayerInfo
from moe_pruning.model.instrumentation import assert_finite
from moe_pruning.model.placement import module_device
from moe_pruning.tasks.base import TaskObjective
from moe_pruning.utils.paths import ArtifactPaths, file_sha256, fingerprint

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Examples
# ----------------------------------------------------------------------
def eval_settings(cfg: ExperimentConfig, task: TaskObjective) -> dict:
    return {
        "dataset_path": str(cfg.evaluation_dataset_path),
        "dataset_sha256": file_sha256(cfg.evaluation_dataset_path),
        "max_samples": cfg.eval_max_samples,
        "tokenizer_source": cfg.model_source,
        **task.eval_settings(cfg),
    }


def prepare_eval_examples(
    cfg: ExperimentConfig, tokenizer, paths: ArtifactPaths, task: TaskObjective
) -> tuple[list, dict]:
    """Build the evaluation examples once and cache them; every stage reuses the cache."""
    settings = eval_settings(cfg, task)
    settings["fingerprint"] = fingerprint(settings)
    cache = paths.eval_examples_cache
    if cache.exists():
        cached = torch.load(cache, weights_only=False)
        if cached["settings"]["fingerprint"] == settings["fingerprint"]:
            return cached["examples"], cached["settings"]
        logger.info("Evaluation settings changed; rebuilding evaluation examples")
    docs = load_eval_documents(cfg.evaluation_dataset_path, cfg.eval_max_samples)
    examples = [task.build_eval_example(doc.example_id, doc.text, tokenizer, cfg) for doc in docs]
    examples = [e for e in examples if len(e.input_ids) >= 2]
    settings["num_examples"] = len(examples)
    settings["num_target_tokens"] = sum(sum(1 for t in e.labels if t != -100) for e in examples)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"settings": settings, "examples": examples}, cache)
    return examples, settings


# ----------------------------------------------------------------------
# Layer reconstruction probe
# ----------------------------------------------------------------------
@dataclass
class _ProbeAccumulator:
    tokens: float = 0.0
    affected_tokens: float = 0.0
    sq_error: float = 0.0
    affected_sq_error: float = 0.0
    original_sq_norm: float = 0.0
    affected_original_sq_norm: float = 0.0


@dataclass
class ReferenceExpertProbe:
    """Measures ||y_original - y_pruned||^2 of pruned layers on the evaluation data.

    For every token, the original layer output is recomputed exactly: original
    routing (any routing mask ignored), original gate, and the original expert.
    Retained experts are taken from the pruned model (they are unchanged); the
    removed experts are rebuilt from their original weights, which are the only
    original weights kept in memory (a few MB per expert).

    This works for every replacement method (bias, drop, reroute) and top-k
    routing. "Affected" tokens are those with a removed expert among their
    original top-k. The inputs are the
    pruned model's actual hidden states, so this is the *local* error of each
    layer given its real (possibly already perturbed) input.
    """

    model: nn.Module
    adapter: MoEAdapter
    layers: dict[str, MoELayerInfo]
    reference_experts: dict[str, dict[int, nn.Module]]
    acc: dict[str, _ProbeAccumulator] = field(default_factory=dict)

    @torch.no_grad()
    def original_layer_output(self, layer: MoELayerInfo, moe_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(y_original [B,S,D] float32, original expert indices [B,S,K])."""
        index, gate = self.adapter.route(self.model, layer, moe_input, apply_mask=False)
        gate = gate.float()
        y = torch.zeros(moe_input.shape, dtype=torch.float32, device=moe_input.device)
        reference = self.reference_experts[layer.layer_id]
        dtype = next(self.model.parameters()).dtype
        for e in torch.unique(index[index >= 0]).tolist():
            sel = index == e  # [..., K]
            tokens = sel.any(dim=-1)
            g = (gate * sel).sum(dim=-1)  # an expert appears at most once per token
            expert = reference[e] if e in reference else self.adapter.get_expert(self.model, layer, e)
            y[tokens] += g[tokens, None] * expert(moe_input[tokens].to(dtype)).float()
        return y, index

    def __call__(self, record: MoEForwardRecord, token_mask: torch.Tensor) -> None:
        layer = self.layers[record.layer_id]
        acc = self.acc.setdefault(record.layer_id, _ProbeAccumulator())
        y_orig, orig_idx = self.original_layer_output(layer, record.moe_input)
        valid = token_mask.to(record.moe_output.device).bool()
        removed = torch.tensor(sorted(self.reference_experts[layer.layer_id]), device=orig_idx.device)
        affected = torch.isin(orig_idx, removed).any(dim=-1) & valid
        sq = (y_orig - record.moe_output.float()).square().sum(dim=-1)
        norm = y_orig.square().sum(dim=-1)
        acc.tokens += float(valid.sum())
        acc.affected_tokens += float(affected.sum())
        acc.sq_error += float(sq[valid].sum())
        acc.affected_sq_error += float(sq[affected].sum())
        acc.original_sq_norm += float(norm[valid].sum())
        acc.affected_original_sq_norm += float(norm[affected].sum())

    def summary(self) -> dict:
        out = {}
        for layer_id, a in self.acc.items():
            out[layer_id] = {
                "tokens": int(a.tokens),
                "tokens_routed_to_replaced_experts": int(a.affected_tokens),
                "mse_per_token": a.sq_error / max(a.tokens, 1),
                "relative_error_all_tokens": a.sq_error / a.original_sq_norm if a.original_sq_norm else None,
                # Error on the tokens originally routed to removed experts. Under reroute,
                # capacity effects can also change a few other tokens; those count only above.
                "relative_error_replaced_tokens": (
                    a.affected_sq_error / a.affected_original_sq_norm if a.affected_original_sq_norm else None
                ),
            }
        return out


def build_reference_probe(
    model: nn.Module,
    adapter: MoEAdapter,
    layers: dict[str, MoELayerInfo],
    replaced: dict[str, list[int]],
    reader,
    device: torch.device,
) -> ReferenceExpertProbe:
    """Probe with the original weights of the removed experts (read lazily via ``reader``)."""
    reference: dict[str, dict[int, nn.Module]] = {}
    dtype = next(model.parameters()).dtype
    for layer_id, expert_ids in replaced.items():
        layer_device = module_device(adapter.get_router(model, layers[layer_id]))  # per-layer GPU when sharded
        for e in expert_ids:
            module = adapter.load_original_expert(reader, model, layers[layer_id], e)
            reference.setdefault(layer_id, {})[e] = module.to(device=layer_device, dtype=dtype).eval()
    return ReferenceExpertProbe(model, adapter, layers, reference)


# ----------------------------------------------------------------------
# Evaluation loop
# ----------------------------------------------------------------------
@dataclass
class EvalOutput:
    metrics: dict
    example_ids: list[int]
    loss_sums: list[float]
    token_counts: list[int]
    predictions: list[list[int]]  # teacher-forced argmax at every scored position
    layer_probe: dict | None = None


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    examples: list,
    device: torch.device,
    batch_size: int,
    pad_token_id: int,
    task: TaskObjective,
    adapter: MoEAdapter | None = None,
    probe: ReferenceExpertProbe | None = None,
    desc: str = "evaluating",
) -> EvalOutput:
    model.eval()
    ids, sums, counts, preds = [], [], [], []
    current: dict = {}
    hook_ctx = None
    if probe is not None:
        probe_layers = [probe.layers[layer_id] for layer_id in probe.reference_experts]
        hook_ctx = adapter.instrument(
            model, probe_layers, lambda rec: probe(rec, current["masks"][probe.layers[rec.layer_id].stack])
        )
        hook_ctx.__enter__()
    try:
        for batch in tqdm(list(task.iter_batches(examples, batch_size, pad_token_id)), desc=desc, unit="batch"):
            current["masks"] = batch.token_masks(device)
            inputs = batch.model_inputs(device)
            logits = model(**inputs).logits
            assert_finite(logits, "logits", next(model.parameters()).dtype)
            loss_sum, n_tok = task.token_nll(logits, inputs["labels"])
            batch_preds = task.predictions(logits, inputs["labels"])
            for row, eid in enumerate(batch.example_ids):
                ids.append(eid)
                sums.append(float(loss_sum[row]))
                counts.append(int(n_tok[row]))
                preds.append(batch_preds[row])
    finally:
        if hook_ctx is not None:
            hook_ctx.__exit__(None, None, None)
    metrics = summarize_losses(sums, counts, perplexity=task.reports_perplexity)
    return EvalOutput(metrics, ids, sums, counts, preds, probe.summary() if probe is not None else None)
