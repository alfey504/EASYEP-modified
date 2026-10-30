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
    replaced_tokens: float = 0.0
    sq_error: float = 0.0
    original_sq_norm: float = 0.0
    replaced_original_sq_norm: float = 0.0


@dataclass
class ReferenceExpertProbe:
    """Measures ||y_original - y_replaced||^2 for pruned layers on the evaluation data.

    The original weights of only the replaced experts are kept (a few MB). For
    a token routed to a replaced expert e, the original contribution would have
    been g * E_e(x); the model produces g * b_e. All other tokens are identical
    because the router and retained experts are unchanged. The inputs x are the
    pruned model's actual hidden states, so this is the *local* error of each
    layer given its real (possibly already perturbed) input.
    """

    adapter: MoEAdapter
    layers: dict[str, MoELayerInfo]
    reference_experts: dict[str, dict[int, nn.Module]]
    acc: dict[str, _ProbeAccumulator] = field(default_factory=dict)

    def __call__(self, record: MoEForwardRecord, token_mask: torch.Tensor) -> None:
        acc = self.acc.setdefault(record.layer_id, _ProbeAccumulator())
        dim = record.moe_output.shape[-1]
        valid = token_mask.reshape(-1).to(record.moe_output.device)
        x = record.moe_input.reshape(-1, dim)[valid].float()
        y = record.moe_output.reshape(-1, dim)[valid].float()
        idx = record.expert_index[..., 0].reshape(-1)[valid]
        g = record.gate[..., 0].reshape(-1)[valid].float()
        b_out = record.expert_output[..., 0, :].reshape(-1, dim)[valid].float()
        delta = torch.zeros_like(y)
        replaced_mask = torch.zeros_like(idx, dtype=torch.bool)
        with torch.no_grad():
            for e, expert in self.reference_experts[record.layer_id].items():
                sel = idx == e
                if sel.any():
                    original = expert(x[sel].to(next(expert.parameters()).dtype)).float()
                    delta[sel] = g[sel, None] * (original - b_out[sel])
                    replaced_mask |= sel
        y_orig = y + delta
        acc.tokens += float(valid.sum())
        acc.replaced_tokens += float(replaced_mask.sum())
        acc.sq_error += float(delta.square().sum())
        acc.original_sq_norm += float(y_orig.square().sum())
        acc.replaced_original_sq_norm += float(y_orig[replaced_mask].square().sum())

    def summary(self) -> dict:
        out = {}
        for layer_id, a in self.acc.items():
            out[layer_id] = {
                "tokens": int(a.tokens),
                "tokens_routed_to_replaced_experts": int(a.replaced_tokens),
                "mse_per_token": a.sq_error / max(a.tokens, 1),
                "relative_error_all_tokens": a.sq_error / a.original_sq_norm if a.original_sq_norm else None,
                "relative_error_replaced_tokens": (
                    a.sq_error / a.replaced_original_sq_norm if a.replaced_original_sq_norm else None
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
    return ReferenceExpertProbe(adapter, layers, reference)


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
