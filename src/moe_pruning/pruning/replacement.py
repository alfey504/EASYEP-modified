"""Install BiasExpert modules into expert slots, removing the original experts."""

from __future__ import annotations

import gc
import hashlib
import logging
from dataclasses import dataclass, field

import torch
from torch import nn

from moe_pruning.model.architecture import MoEAdapter, MoELayerInfo
from moe_pruning.pruning.bias_expert import BiasExpert

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


def replace_experts_with_biases(
    model: nn.Module,
    adapter: MoEAdapter,
    layers: dict[str, MoELayerInfo],
    biases: dict[str, dict[int, torch.Tensor | None]],
) -> ReplacementReport:
    """Replace experts in place.

    Args:
        biases: ``{layer_id: {expert_id: initial bias or None (zeros)}}``.

    The original expert modules are dropped from the model so their parameters
    are freed. Routers are verified to be bit-identical afterwards.
    """
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    report = ReplacementReport(parameters_before=sum(p.numel() for p in model.parameters()))
    router_digests = {
        layer_id: _module_digest(adapter.get_router(model, layers[layer_id])) for layer_id in biases
    }
    for layer_id, per_expert in biases.items():
        layer = layers[layer_id]
        for expert_id, init in sorted(per_expert.items()):
            current = adapter.get_expert(model, layer, expert_id)
            if isinstance(current, BiasExpert):
                raise ValueError(f"{layer_id} expert {expert_id} is already replaced")
            replacement = BiasExpert(layer.expert_output_dim, init=init, dtype=dtype, device=device)
            old = adapter.set_expert(model, layer, expert_id, replacement)
            report.removed_parameters += sum(p.numel() for p in old.parameters())
            report.added_parameters += replacement.bias.numel()
            del old
            report.replaced.setdefault(layer_id, []).append(expert_id)

    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    for layer_id, digest in router_digests.items():
        if _module_digest(adapter.get_router(model, layers[layer_id])) != digest:
            raise RuntimeError(f"Router of {layer_id} changed during replacement")
    state_keys = model.state_dict().keys()
    for layer_id, expert_ids in report.replaced.items():
        for expert_id in expert_ids:
            prefix = adapter.expert_state_prefix(layers[layer_id], expert_id)
            leftover = [k for k in state_keys if k.startswith(prefix) and k != prefix + "bias"]
            if leftover:
                raise RuntimeError(f"Original parameters still present after replacement: {leftover}")
    report.parameters_after = sum(p.numel() for p in model.parameters())
    logger.info(
        "Replaced %d expert(s): removed %d parameters, added %d (net %.2f%% of the model)",
        sum(len(v) for v in report.replaced.values()),
        report.removed_parameters,
        report.added_parameters,
        100.0 * (report.parameters_before - report.parameters_after) / report.parameters_before,
    )
    return report


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
