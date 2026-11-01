"""Remove selected experts and install their replacement (bias vector, drop or reroute)."""

from __future__ import annotations

import gc
import hashlib
import logging
from dataclasses import dataclass, field

import torch
from torch import nn

from moe_pruning.model.architecture import MoEAdapter, MoELayerInfo
from moe_pruning.model.placement import module_device
from moe_pruning.pruning.bias_expert import BiasExpert
from moe_pruning.pruning.zero_expert import ZeroExpert

logger = logging.getLogger(__name__)


def _module_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class ReplacementReport:
    replaced: dict[str, list[int]] = field(default_factory=dict)
    removed_parameters: int = 0
    added_parameters: int = 0
    parameters_before: int = 0
    parameters_after: int = 0

    def to_dict(self) -> dict:
        return {
            "replaced": self.replaced,
            "removed_parameters": self.removed_parameters,
            "added_parameters": self.added_parameters,
            "parameters_before": self.parameters_before,
            "parameters_after": self.parameters_after,
            "net_reduction": self.parameters_before - self.parameters_after,
            "net_reduction_fraction": (self.parameters_before - self.parameters_after)
            / max(self.parameters_before, 1),
        }


REPLACEMENT_METHODS = ("bias", "drop", "reroute")


def make_replacement_module(
    method: str,
    output_dim: int,
    init: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Module installed in a removed expert's slot for the given method."""
    if method == "bias":
        return BiasExpert(output_dim, init=init, dtype=dtype, device=device)
    if method == "drop":
        return ZeroExpert(output_dim)
    if method == "reroute":
        return ZeroExpert(output_dim, must_be_unused=True)
    raise ValueError(f"Unknown replacement method {method!r}; expected one of {REPLACEMENT_METHODS}")


def replace_experts(
    model: nn.Module,
    adapter: MoEAdapter,
    layers: dict[str, MoELayerInfo],
    selection: dict[str, list[int]],
    method: str,
    biases: dict[str, dict[int, torch.Tensor | None]] | None = None,
) -> ReplacementReport:
    """Remove the selected experts in place and install their replacement.

    * ``bias``: BiasExpert initialised from ``biases[layer][expert]`` (None = zeros).
    * ``drop``: parameter-free ZeroExpert; routed tokens get zero MoE output.
    * ``reroute``: ZeroExpert plus a routing mask sending the tokens to their best
      remaining expert.

    The original expert modules are dropped so their parameters are freed.
    Router weights are verified to be bit-identical afterwards.
    """
    if method not in REPLACEMENT_METHODS:
        raise ValueError(f"Unknown replacement method {method!r}")
    if method == "bias" and biases is None:
        raise ValueError("The bias method needs initial bias vectors")
    dtype = next(model.parameters()).dtype
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32  # keep trainable bias vectors in full precision
    report = ReplacementReport(parameters_before=sum(p.numel() for p in model.parameters()))
    router_digests = {
        layer_id: _module_digest(adapter.get_router(model, layers[layer_id])) for layer_id in selection
    }
    for layer_id, expert_ids in selection.items():
        layer = layers[layer_id]
        device = module_device(adapter.get_moe_module(model, layer))  # the layer's GPU when sharded
        modules: dict[int, nn.Module] = {}
        for expert_id in sorted(expert_ids):
            current = adapter.get_expert(model, layer, expert_id)
            if isinstance(current, (BiasExpert, ZeroExpert)):
                raise ValueError(f"{layer_id} expert {expert_id} is already replaced")
            init = biases[layer_id].get(expert_id) if method == "bias" else None
            modules[expert_id] = make_replacement_module(method, layer.expert_output_dim, init, dtype, device)
        report.removed_parameters += adapter.replace_experts_in_layer(model, layer, modules)
        report.added_parameters += sum(p.numel() for m in modules.values() for p in m.parameters())
        report.replaced[layer_id] = sorted(expert_ids)
        if method == "reroute":
            adapter.set_routing_mask(model, layer, sorted(expert_ids))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    for layer_id, digest in router_digests.items():
        if _module_digest(adapter.get_router(model, layers[layer_id])) != digest:
            raise RuntimeError(f"Router of {layer_id} changed during replacement")
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    for layer_id in report.replaced:
        layer = layers[layer_id]
        # All experts removed from the layer so far (this call and earlier ones).
        all_removed = [
            e for e in range(layer.num_experts)
            if isinstance(adapter.get_expert(model, layer, e), (BiasExpert, ZeroExpert))
        ]
        leftover = adapter.removed_weight_violations(shapes, layer, all_removed)
        if leftover:
            raise RuntimeError(f"Original parameters still present after replacement: {leftover}")
    report.parameters_after = sum(p.numel() for p in model.parameters())
    logger.info(
        "Replaced %d expert(s) [%s]: removed %d parameters, added %d (net %.2f%% of the model)",
        sum(len(v) for v in report.replaced.values()),
        method,
        report.removed_parameters,
        report.added_parameters,
        100.0 * (report.parameters_before - report.parameters_after) / report.parameters_before,
    )
    return report


def replace_experts_with_biases(
    model: nn.Module,
    adapter: MoEAdapter,
    layers: dict[str, MoELayerInfo],
    biases: dict[str, dict[int, torch.Tensor | None]],
) -> ReplacementReport:
    """Shorthand for ``replace_experts(..., method="bias")``."""
    selection = {layer_id: list(per_expert) for layer_id, per_expert in biases.items()}
    return replace_experts(model, adapter, layers, selection, "bias", biases)


def bias_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    """Named parameters belonging to BiasExpert modules."""
    out = {}
    for module_name, module in model.named_modules():
        if isinstance(module, BiasExpert):
            out[f"{module_name}.bias"] = module.bias
    return out


def freeze_all_but_biases(model: nn.Module) -> dict[str, nn.Parameter]:
    """Freeze every parameter except BiasExpert.bias and return the trainable ones."""
    trainable = bias_parameters(model)
    trainable_ids = {id(p) for p in trainable.values()}
    for p in model.parameters():
        p.requires_grad_(id(p) in trainable_ids)
    actually_trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if sorted(actually_trainable) != sorted(trainable):
        raise RuntimeError(f"Unexpected trainable parameters: {actually_trainable}")
    return trainable
