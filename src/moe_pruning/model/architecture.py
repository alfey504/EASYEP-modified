"""Architecture-independent interfaces for sparse MoE models.

The pruning, scoring, initialisation and training code only talks to an
:class:`MoEAdapter`. Supporting a new architecture (e.g. OLMoE) means writing
a new adapter, not touching the algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Callable, Iterator

import torch
from torch import nn


@dataclass(frozen=True)
class MoELayerInfo:
    """Static description of one sparse MoE layer."""

    layer_id: str  # stable identifier, e.g. "encoder.block.1"
    stack: str  # "encoder" or "decoder" (decoder-only models use "decoder")
    position: int  # 0-based order of discovery
    module_path: str  # dotted path of the sparse MoE module
    router_path: str
    experts_path: str
    num_experts: int
    top_k: int
    hidden_dim: int  # residual-stream width
    expert_input_dim: int
    expert_intermediate_dim: int
    expert_output_dim: int
    expert_capacity: int | None  # tokens per expert per sequence, None = unlimited

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MoEForwardRecord:
    """Everything observed in one forward call of one sparse MoE layer.

    Shapes use B = batch, S = sequence length, K = experts per token, D = width.
    A token that was not processed by any expert (e.g. dropped by expert
    capacity) has ``expert_index == -1`` and zero gate / expert output.
    """

    layer_id: str
    residual: torch.Tensor  # [B, S, D] residual stream entering the MoE sub-layer
    moe_input: torch.Tensor  # [B, S, D] input of the router and experts
    moe_output: torch.Tensor  # [B, S, D] gate-weighted sum of routed expert outputs
    expert_index: torch.Tensor  # [B, S, K] long
    gate: torch.Tensor  # [B, S, K] weight that multiplies each expert output
    expert_output: torch.Tensor  # [B, S, K, D] un-gated expert outputs E_e(x_t)


RecordCallback = Callable[[MoEForwardRecord], None]


class MoEAdapter(ABC):
    """Model-specific access to sparse MoE layers.

    Adapters are provided by model plugins (``moe_pruning.models``). Expert
    ids always refer to the original router indices, also after pruning.
    """

    @abstractmethod
    def discover_layers(self, model: nn.Module) -> list[MoELayerInfo]:
        """Find all sparse MoE layers by inspecting the loaded model."""

    @abstractmethod
    def get_moe_module(self, model: nn.Module, layer: MoELayerInfo) -> nn.Module: ...

    @abstractmethod
    def get_router(self, model: nn.Module, layer: MoELayerInfo) -> nn.Module: ...

    @abstractmethod
    def get_expert(self, model: nn.Module, layer: MoELayerInfo, expert_id: int) -> nn.Module: ...

    @abstractmethod
    def set_expert(
        self, model: nn.Module, layer: MoELayerInfo, expert_id: int, module: nn.Module
    ) -> nn.Module:
        """Install ``module`` in the expert slot and return the previous module."""

    @abstractmethod
    def build_original_expert(self, model: nn.Module, layer: MoELayerInfo) -> nn.Module:
        """New (uninitialised) expert module of the model's original expert class."""

    @abstractmethod
    def expert_state_prefix(self, layer: MoELayerInfo, expert_id: int) -> str:
        """State-dict key prefix of an expert slot (ends with a dot)."""

    @abstractmethod
    def route(
        self, model: nn.Module, layer: MoELayerInfo, moe_input: torch.Tensor, apply_mask: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (expert_index [B,S,K], gate [B,S,K]) exactly as used by the layer.

        With ``apply_mask=False`` any routing mask is ignored, giving the
        original model's routing decisions.
        """

    @abstractmethod
    def set_routing_mask(self, model: nn.Module, layer: MoELayerInfo, expert_ids: list[int]) -> None:
        """Make ``expert_ids`` unreachable: their tokens go to the best remaining expert.

        Router weights are not modified. An empty list removes the mask.
        """

    @abstractmethod
    def get_routing_mask(self, model: nn.Module, layer: MoELayerInfo) -> list[int]: ...

    @abstractmethod
    def run_moe_layer(self, model: nn.Module, layer: MoELayerInfo, moe_input: torch.Tensor) -> torch.Tensor:
        """Execute only the sparse MoE module on ``moe_input`` ([B,S,D])."""

    @abstractmethod
    @contextmanager
    def instrument(
        self, model: nn.Module, layers: list[MoELayerInfo], callback: RecordCallback
    ) -> Iterator[None]:
        """Call ``callback`` with a :class:`MoEForwardRecord` per layer forward."""

    @abstractmethod
    def describe(self, model: nn.Module) -> dict:
        """Machine-readable architecture report."""

    # ------------------------------------------------------------------
    # Overridable defaults (per-expert modules addressed by a state prefix)
    # ------------------------------------------------------------------
    def replace_experts_in_layer(
        self, model: nn.Module, layer: MoELayerInfo, modules: dict[int, nn.Module]
    ) -> int:
        """Install ``modules`` in their expert slots; return the number of removed parameters.

        Default: one ``set_expert`` per expert. Adapters whose experts share
        fused weight tensors override this to compact the tensors once.
        """
        removed = 0
        for expert_id, module in sorted(modules.items()):
            old = self.set_expert(model, layer, expert_id, module)
            removed += sum(p.numel() for p in old.parameters())
            del old
        return removed

    def removed_weight_violations(
        self, state_shapes: dict[str, tuple[int, ...]], layer: MoELayerInfo, removed: list[int]
    ) -> list[str]:
        """Names of stored tensors that still hold weights of removed experts.

        Default: any tensor under a removed expert's state prefix other than a
        replacement ``bias``.
        """
        bad = []
        for e in removed:
            prefix = self.expert_state_prefix(layer, e)
            bad += [k for k in state_shapes if k.startswith(prefix) and k != prefix + "bias"]
        return bad

    def load_original_expert(self, reader, model: nn.Module, layer: MoELayerInfo, expert_id: int) -> nn.Module:
        """Rebuild one original expert from the original checkpoint (``reader``: CheckpointReader).

        Default: tensors stored under the expert's state prefix.
        """
        prefix = self.expert_state_prefix(layer, expert_id)
        module = self.build_original_expert(model, layer)
        state = {k[len(prefix):]: reader.get(k) for k in reader.keys() if k.startswith(prefix)}
        if not state:
            raise KeyError(f"No tensors with prefix {prefix!r} in the original checkpoint")
        module.load_state_dict(state, strict=True)
        return module

    def validate_loaded(self, model: nn.Module, layers: list[MoELayerInfo]) -> None:
        """Called by the checkpoint loader after weights are loaded; raise on inconsistency."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def get_experts(self, model: nn.Module, layer: MoELayerInfo) -> dict[int, nn.Module]:
        return {e: self.get_expert(model, layer, e) for e in range(layer.num_experts)}

    def hidden_dims(self, model: nn.Module) -> dict[str, dict[str, int]]:
        return {
            layer.layer_id: {
                "hidden_dim": layer.hidden_dim,
                "expert_input_dim": layer.expert_input_dim,
                "expert_intermediate_dim": layer.expert_intermediate_dim,
                "expert_output_dim": layer.expert_output_dim,
            }
            for layer in self.discover_layers(model)
        }


def select_layers(layers: list[MoELayerInfo], spec: str) -> list[MoELayerInfo]:
    """Resolve PRUNE_LAYERS: ``first``, ``all`` or comma-separated layer ids."""
    spec = spec.strip()
    if not layers:
        raise ValueError("The model has no sparse MoE layers")
    if spec.lower() == "first":
        return [layers[0]]
    if spec.lower() == "all":
        return list(layers)
    by_id = {layer.layer_id: layer for layer in layers}
    wanted = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [w for w in wanted if w not in by_id]
    if unknown:
        raise ValueError(f"Unknown MoE layer id(s) {unknown}. Available: {sorted(by_id)}")
    # Keep model order, drop duplicates.
    return [layer for layer in layers if layer.layer_id in set(wanted)]


def get_adapter(model: nn.Module) -> MoEAdapter:
    """Adapter of the model's plugin (looked up by ``config.model_type``)."""
    from moe_pruning.models import plugin_for_model

    return plugin_for_model(model).adapter()
