"""Deterministic span-corruption evaluation and layer-reconstruction probing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import nn
from tqdm import tqdm

from moe_pruning.config import ExperimentConfig
from moe_pruning.data.c4_support import load_eval_documents
from moe_pruning.data.collators import iter_batches
from moe_pruning.data.span_corruption import CorruptedExample, corrupt_example
from moe_pruning.evaluation.metrics import per_example_token_nll, summarize_losses
from moe_pruning.model.architecture import MoEAdapter, MoEForwardRecord, MoELayerInfo
from moe_pruning.utils.paths import ArtifactPaths, file_sha256, fingerprint

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Examples
# ----------------------------------------------------------------------
def eval_settings(cfg: ExperimentConfig) -> dict:
    return {
        "dataset_path": str(cfg.evaluation_dataset_path),
        "dataset_sha256": file_sha256(cfg.evaluation_dataset_path),
        "max_samples": cfg.eval_max_samples,
        "noise_density": cfg.eval_noise_density,
        "mean_noise_span_length": cfg.eval_mean_span_length,
        "seed": cfg.eval_seed,
        "max_source_length": cfg.max_source_length,
        "max_target_length": cfg.max_target_length,
        "tokenizer_source": cfg.model_source,
    }


def prepare_eval_examples(cfg: ExperimentConfig, tokenizer, paths: ArtifactPaths) -> tuple[list[CorruptedExample], dict]:
    """Corrupt the evaluation documents once and cache them; every stage reuses the cache."""
    settings = eval_settings(cfg)
    settings["fingerprint"] = fingerprint(settings)
    cache = paths.eval_examples_cache
    if cache.exists():
        cached = torch.load(cache, weights_only=False)
        if cached["settings"]["fingerprint"] == settings["fingerprint"]:
            return cached["examples"], cached["settings"]
        logger.info("Evaluation settings changed; rebuilding corrupted examples")
    docs = load_eval_documents(cfg.evaluation_dataset_path, cfg.eval_max_samples)
    examples = []
    for doc in docs:
        ids = tokenizer(doc.text, add_special_tokens=False).input_ids
        examples.append(
            corrupt_example(
                doc.example_id, ids, tokenizer, cfg.eval_noise_density, cfg.eval_mean_span_length,
                cfg.eval_seed, cfg.max_source_length, cfg.max_target_length,
            )
        )
    settings["num_examples"] = len(examples)
    settings["num_target_tokens"] = sum(len(e.labels) for e in examples)
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

    This works for every replacement method (bias, drop, reroute). "Affected"
    tokens are those originally routed to a removed expert. The inputs are the
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
        """(y_original [B,S,D] float32, original top-1 expert index [B,S])."""
        index, gate = self.adapter.route(self.model, layer, moe_input, apply_mask=False)
        idx, g = index[..., 0], gate[..., 0].float()
        y = torch.zeros(moe_input.shape, dtype=torch.float32, device=moe_input.device)
        reference = self.reference_experts[layer.layer_id]
        for e in range(layer.num_experts):
            sel = idx == e
            if not sel.any():
                continue
            expert = reference[e] if e in reference else self.adapter.get_expert(self.model, layer, e)
            x = moe_input[sel].to(next(expert.parameters()).dtype)
            y[sel] = g[sel, None] * expert(x).float()
        return y, idx

    def __call__(self, record: MoEForwardRecord, token_mask: torch.Tensor) -> None:
        layer = self.layers[record.layer_id]
        acc = self.acc.setdefault(record.layer_id, _ProbeAccumulator())
        y_orig, orig_idx = self.original_layer_output(layer, record.moe_input)
        valid = token_mask.to(record.moe_output.device).bool()
        removed = torch.tensor(sorted(self.reference_experts[layer.layer_id]), device=orig_idx.device)
        affected = torch.isin(orig_idx, removed) & valid
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
    original_tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> ReferenceExpertProbe:
    reference: dict[str, dict[int, nn.Module]] = {}
    dtype = next(model.parameters()).dtype
    for layer_id, expert_ids in replaced.items():
        layer = layers[layer_id]
        for e in expert_ids:
            prefix = adapter.expert_state_prefix(layer, e)
            module = adapter.build_original_expert(model, layer)
            state = {k[len(prefix):]: v for k, v in original_tensors.items() if k.startswith(prefix)}
            module.load_state_dict(state, strict=True)
            reference.setdefault(layer_id, {})[e] = module.to(device=device, dtype=dtype).eval()
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
    predictions: list[list[int]]  # teacher-forced argmax per target position
    layer_probe: dict | None = None


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    examples: list[CorruptedExample],
    device: torch.device,
    batch_size: int,
    pad_token_id: int,
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
        for batch in tqdm(list(iter_batches(examples, batch_size, pad_token_id)), desc=desc, unit="batch"):
            current["masks"] = batch.token_masks(device)
            inputs = batch.model_inputs(device)
            logits = model(**inputs).logits
            loss_sum, n_tok = per_example_token_nll(logits, inputs["labels"])
            argmax = logits.argmax(dim=-1).cpu()
            for row, eid in enumerate(batch.example_ids):
                n = int(n_tok[row])
                ids.append(eid)
                sums.append(float(loss_sum[row]))
                counts.append(n)
                preds.append(argmax[row, :n].tolist())
    finally:
        if hook_ctx is not None:
            hook_ctx.__exit__(None, None, None)
    metrics = summarize_losses(sums, counts)
    return EvalOutput(metrics, ids, sums, counts, preds, probe.summary() if probe is not None else None)


def _clean_decode(tokenizer, ids: list[int]) -> str:
    text = tokenizer.decode(ids, skip_special_tokens=False)
    for tok in (tokenizer.pad_token, tokenizer.eos_token):
        text = text.replace(tok, "")
    return " ".join(text.split())


@torch.no_grad()
def qualitative_predictions(
    model: nn.Module, tokenizer, examples: list[CorruptedExample], device: torch.device, max_new_tokens: int
) -> list[dict]:
    out = []
    for ex in examples:
        input_ids = torch.tensor([ex.input_ids], device=device)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        out.append(
            {
                "example_id": ex.example_id,
                "encoder_input_original": _clean_decode(tokenizer, ex.source_ids),
                "corrupted_input": _clean_decode(tokenizer, ex.input_ids),
                "target": _clean_decode(tokenizer, ex.labels),
                "prediction": _clean_decode(tokenizer, generated[0].tolist()),
            }
        )
    return out
