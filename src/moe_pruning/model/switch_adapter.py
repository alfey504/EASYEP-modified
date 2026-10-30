"""Adapter for Hugging Face Switch Transformers (e.g. google/switch-base-8).

Relevant facts about the HF implementation (transformers 5.x), verified by
reading ``modeling_switch_transformers.py`` and by the gate-consistency probe in
``scripts/inspect_model.py``:

* Every sparse block's feed-forward sub-layer is ``SwitchTransformersLayerFF``:
  ``out = h + dropout(SparseMLP(LayerNorm(h)))``.
* ``SwitchTransformersSparseMLP`` holds ``router`` (``SwitchTransformersTop1Router``)
  and ``experts`` (``SwitchTransformersExperts``, a ``ModuleDict`` with keys
  ``expert_{i}``).
* The router computes ``softmax(classifier(x))`` in float32, takes the argmax
  (top-1), and masks tokens beyond ``expert_capacity`` *per sequence*
  (cumulative count along the sequence). It returns the one-hot dispatch mask
  and the max probability ``g_t`` (shape [B, S, 1]).
* ``SwitchTransformersExperts.forward`` evaluates each expert on its tokens and
  multiplies the result by ``g_t``: ``expert(x) * routing_weights``. Hence the
  gate is applied by the parent, **outside** the expert module. A replacement
  expert must therefore return the un-gated vector only.
* Tokens dropped by the capacity limit receive zero MoE output; only the
  residual connection carries them forward.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn

from moe_pruning.model.architecture import (
    MoEAdapter,
    MoEForwardRecord,
    MoELayerInfo,
    RecordCallback,
    register_adapter,
)


def _get_submodule(model: nn.Module, path: str) -> nn.Module:
    return model.get_submodule(path)


def _is_sparse_mlp(module: nn.Module) -> bool:
    # Structural check (router + experts container) with the HF class as a hint,
    # so that subclasses or renamed modules are still found.
    return (
        hasattr(module, "router")
        and hasattr(module, "experts")
        and isinstance(module.experts, nn.ModuleDict)
        and hasattr(module.router, "classifier")
    )


@register_adapter
class SwitchTransformersAdapter(MoEAdapter):
    model_types = ("switch_transformers",)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover_layers(self, model: nn.Module) -> list[MoELayerInfo]:
        layers: list[MoELayerInfo] = []
        for path, module in model.named_modules():
            if not _is_sparse_mlp(module):
                continue
            parts = path.split(".")
            # Expected path: [prefix.]{encoder|decoder}.block.{i}.layer.{j}.mlp
            if "block" not in parts:
                raise RuntimeError(f"Unexpected sparse MoE location {path!r}")
            b = parts.index("block")
            stack = parts[b - 1]
            if stack not in ("encoder", "decoder"):
                raise RuntimeError(f"Cannot determine encoder/decoder for {path!r}")
            layer_id = ".".join(parts[b - 1 : b + 2])  # e.g. encoder.block.1

            router = module.router
            experts = module.experts
            keys = list(experts.keys())
            expected = [f"expert_{i}" for i in range(len(keys))]
            if sorted(keys, key=lambda k: int(k.split("_")[-1])) != expected:
                raise RuntimeError(f"Unexpected expert keys in {path}: {keys}")
            # Dimensions come from any original (not replaced) expert.
            original = [m for m in experts.values() if hasattr(m, "wi") and hasattr(m, "wo")]
            if not original:
                raise RuntimeError(f"{path}: no original expert left to read dimensions from")
            wi, wo = original[0].wi.weight, original[0].wo.weight  # nn.Linear stores [out, in]
            num_experts = int(router.num_experts)
            if num_experts != len(keys) or router.classifier.out_features != num_experts:
                raise RuntimeError(f"Router/expert count mismatch in {path}")
            layers.append(
                MoELayerInfo(
                    layer_id=layer_id,
                    stack=stack,
                    position=len(layers),
                    module_path=path,
                    router_path=f"{path}.router",
                    experts_path=f"{path}.experts",
                    num_experts=num_experts,
                    top_k=1,
                    hidden_dim=int(router.classifier.in_features),
                    expert_input_dim=int(wi.shape[1]),
                    expert_intermediate_dim=int(wi.shape[0]),
                    expert_output_dim=int(wo.shape[0]),
                    expert_capacity=int(router.expert_capacity),
                )
            )
        return layers

    # ------------------------------------------------------------------
    # Module access
    # ------------------------------------------------------------------
    def get_moe_module(self, model: nn.Module, layer: MoELayerInfo) -> nn.Module:
        return _get_submodule(model, layer.module_path)

    def get_router(self, model: nn.Module, layer: MoELayerInfo) -> nn.Module:
        return _get_submodule(model, layer.router_path)

    def _experts(self, model: nn.Module, layer: MoELayerInfo) -> nn.ModuleDict:
        return _get_submodule(model, layer.experts_path)

    def get_expert(self, model: nn.Module, layer: MoELayerInfo, expert_id: int) -> nn.Module:
        self._check_expert_id(layer, expert_id)
        return self._experts(model, layer)[f"expert_{expert_id}"]

    def set_expert(
        self, model: nn.Module, layer: MoELayerInfo, expert_id: int, module: nn.Module
    ) -> nn.Module:
        self._check_expert_id(layer, expert_id)
        experts = self._experts(model, layer)
        key = f"expert_{expert_id}"
        previous = experts[key]
        experts[key] = module  # same slot, router indexing unchanged
        return previous

    def build_original_expert(self, model: nn.Module, layer: MoELayerInfo) -> nn.Module:
        from transformers.models.switch_transformers.modeling_switch_transformers import (
            SwitchTransformersDenseActDense,
        )

        return SwitchTransformersDenseActDense(model.config)

    def expert_state_prefix(self, layer: MoELayerInfo, expert_id: int) -> str:
        return f"{layer.experts_path}.expert_{expert_id}."

    def _layer_ff(self, model: nn.Module, layer: MoELayerInfo) -> nn.Module:
        parent_path = layer.module_path.rsplit(".", 1)[0]
        return _get_submodule(model, parent_path)

    @staticmethod
    def _check_expert_id(layer: MoELayerInfo, expert_id: int) -> None:
        if not 0 <= expert_id < layer.num_experts:
            raise IndexError(f"Expert {expert_id} out of range for {layer.layer_id}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    @staticmethod
    def _dispatch_to_index(dispatch: torch.Tensor, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert the one-hot capacity-masked dispatch mask into index/gate tensors."""
        routed = dispatch.sum(dim=-1) > 0  # [B, S]
        index = torch.where(routed, dispatch.argmax(dim=-1), torch.full_like(routed, -1, dtype=torch.long))
        gate = probs.squeeze(-1) * routed.to(probs.dtype)
        return index.unsqueeze(-1), gate.unsqueeze(-1)

    def route(
        self, model: nn.Module, layer: MoELayerInfo, moe_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dispatch, probs, _ = self.get_router(model, layer)(moe_input)
        return self._dispatch_to_index(dispatch, probs)

    def run_moe_layer(self, model: nn.Module, layer: MoELayerInfo, moe_input: torch.Tensor) -> torch.Tensor:
        return self.get_moe_module(model, layer)(moe_input)

    # ------------------------------------------------------------------
    # Instrumentation
    # ------------------------------------------------------------------
    @contextmanager
    def instrument(
        self, model: nn.Module, layers: list[MoELayerInfo], callback: RecordCallback
    ) -> Iterator[None]:
        handles = []
        try:
            for layer in layers:
                handles.extend(self._register_layer_hooks(model, layer, callback))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _register_layer_hooks(self, model: nn.Module, layer: MoELayerInfo, callback: RecordCallback) -> list:
        state: dict = {}
        layer_ff = self._layer_ff(model, layer)
        moe = self.get_moe_module(model, layer)
        router = self.get_router(model, layer)
        experts = self._experts(model, layer)

        def ff_pre(_module, args):
            state.clear()
            state["residual"] = args[0].detach()
            state["expert_out"] = {}

        def moe_pre(_module, args):
            state["moe_input"] = args[0].detach()

        def router_hook(_module, _args, output):
            dispatch, probs, _logits = output
            state["dispatch"] = dispatch.detach()
            state["probs"] = probs.detach()

        def make_expert_hook(expert_id: int):
            def expert_hook(_module, _args, output):
                # Un-gated output; rows follow ascending flat token order.
                state["expert_out"][expert_id] = output.detach()

            return expert_hook

        def moe_hook(_module, _args, output):
            x = state["moe_input"]
            bsz, seq, dim = x.shape
            dispatch = state["dispatch"].reshape(bsz * seq, -1)
            index, gate = self._dispatch_to_index(state["dispatch"], state["probs"])
            expert_output = torch.zeros(bsz * seq, dim, dtype=x.dtype, device=x.device)
            for expert_id, rows in state["expert_out"].items():
                tokens = torch.nonzero(dispatch[:, expert_id], as_tuple=False).squeeze(-1)
                if tokens.numel() != rows.shape[0]:
                    raise RuntimeError(
                        f"{layer.layer_id}: expert {expert_id} returned {rows.shape[0]} rows "
                        f"for {tokens.numel()} dispatched tokens"
                    )
                expert_output[tokens] = rows.to(x.dtype)
            callback(
                MoEForwardRecord(
                    layer_id=layer.layer_id,
                    residual=state["residual"],
                    moe_input=x,
                    moe_output=output.detach(),
                    expert_index=index,
                    gate=gate,
                    expert_output=expert_output.view(bsz, seq, 1, dim),
                )
            )

        handles = [
            layer_ff.register_forward_pre_hook(ff_pre),
            moe.register_forward_pre_hook(moe_pre),
            router.register_forward_hook(router_hook),
            moe.register_forward_hook(moe_hook),
        ]
        for key, expert in experts.items():
            handles.append(expert.register_forward_hook(make_expert_hook(int(key.split("_")[-1]))))
        return handles

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def describe(self, model: nn.Module) -> dict:
        layers = self.discover_layers(model)
        cfg = model.config
        report_layers = []
        for layer in layers:
            moe = self.get_moe_module(model, layer)
            layer_ff = self._layer_ff(model, layer)
            experts = self.get_experts(model, layer)
            report_layers.append(
                {
                    **layer.to_dict(),
                    "moe_class": type(moe).__name__,
                    "router_class": type(moe.router).__name__,
                    "router_classifier": {
                        "in_features": moe.router.classifier.in_features,
                        "out_features": moe.router.classifier.out_features,
                        "bias": moe.router.classifier.bias is not None,
                    },
                    "router_dtype": str(moe.router.dtype).replace("torch.", ""),
                    "parent_class": type(layer_ff).__name__,
                    "expert_classes": sorted({type(m).__name__ for m in experts.values()}),
                    "expert_parameters": {
                        str(e): sum(p.numel() for p in m.parameters()) for e, m in experts.items()
                    },
                }
            )
        return {
            "model_type": cfg.model_type,
            "model_class": type(model).__name__,
            "adapter": type(self).__name__,
            "is_encoder_decoder": bool(cfg.is_encoder_decoder),
            "d_model": cfg.d_model,
            "d_ff": cfg.d_ff,
            "num_encoder_layers": cfg.num_layers,
            "num_decoder_layers": cfg.num_decoder_layers,
            "encoder_sparse_step": cfg.encoder_sparse_step,
            "decoder_sparse_step": cfg.decoder_sparse_step,
            "num_experts": cfg.num_experts,
            "expert_capacity": cfg.expert_capacity,
            "router_jitter_noise": cfg.router_jitter_noise,
            "router_ignore_padding_tokens": cfg.router_ignore_padding_tokens,
            "num_sparse_layers": len(layers),
            "num_encoder_sparse_layers": sum(layer.stack == "encoder" for layer in layers),
            "num_decoder_sparse_layers": sum(layer.stack == "decoder" for layer in layers),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "expert_parameters_total": sum(
                sum(p.numel() for p in self.get_expert(model, layer, e).parameters())
                for layer in layers
                for e in range(layer.num_experts)
            ),
            "routing": {
                "type": "top-1 (Switch)",
                "gate_value": "max softmax probability of the router logits (float32)",
                "capacity": (
                    "per sequence: tokens are assigned in sequence order and tokens beyond "
                    "expert_capacity are dropped (zero MoE output, residual only)"
                ),
                "gate_application": (
                    "SwitchTransformersExperts.forward: expert(x) * routing_weights, i.e. the parent "
                    "multiplies the un-gated expert output by the gate exactly once"
                ),
                "padding": (
                    "router_ignore_padding_tokens is not used by the HF forward; padding tokens are "
                    "routed and consume capacity. Right padding keeps them after all real tokens, so "
                    "they never take capacity from real tokens."
                ),
            },
            "layers": report_layers,
        }
