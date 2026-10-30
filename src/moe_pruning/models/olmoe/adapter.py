"""Adapter for OLMoE (allenai/OLMoE-1B-7B-0125 and relatives) in transformers 5.

Verified against ``transformers/models/olmoe/modeling_olmoe.py`` (5.22):

* Decoder-only. Every layer is sparse: ``model.layers.{i}.mlp`` is an
  ``OlmoeSparseMoeBlock`` with ``gate`` (``OlmoeTopKRouter``) and ``experts``
  (``OlmoeExperts``, fused 3D weights).
* ``OlmoeDecoderLayer``: ``h = h_attn; out = h + mlp(post_attention_layernorm(h))``.
* Router: ``p = softmax(W x)`` in float32, top-k (8 of 64), optional
  renormalisation (``norm_topk_prob``; False for OLMoE-1B-7B), cast to the
  logits dtype. No capacity limit, no dropped tokens.
* Gate application: ``OlmoeExperts.forward`` multiplies each expert output by
  its top-k weight exactly once (``... * top_k_weights[token_idx, top_k_pos]``).
  ``PrunedFusedExperts`` keeps that contract, so BiasExpert returns b only.
* Hub checkpoints store experts per expert (``experts.{e}.gate_proj`` etc.);
  transformers fuses them on load. ``load_original_expert`` reads either layout.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn
from transformers.activations import ACT2FN

from moe_pruning.model.architecture import MoEAdapter, MoEForwardRecord, MoELayerInfo, RecordCallback
from moe_pruning.models.olmoe.experts import FusedExpertView, OlmoeReferenceExpert, PrunedFusedExperts


def _is_moe_block(module: nn.Module) -> bool:
    gate = getattr(module, "gate", None)
    experts = getattr(module, "experts", None)
    return (
        gate is not None
        and experts is not None
        and hasattr(gate, "top_k")
        and hasattr(gate, "weight")
        and hasattr(experts, "gate_up_proj")
        and hasattr(experts, "down_proj")
    )


class OlmoeAdapter(MoEAdapter):
    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover_layers(self, model: nn.Module) -> list[MoELayerInfo]:
        layers: list[MoELayerInfo] = []
        for path, module in model.named_modules():
            if not _is_moe_block(module):
                continue
            parts = path.split(".")
            if "layers" not in parts:
                raise RuntimeError(f"Unexpected MoE block location {path!r}")
            i = parts.index("layers")
            layer_id = ".".join(parts[i : i + 2])  # e.g. layers.3
            router, experts = module.gate, module.experts
            num_experts = int(router.weight.shape[0])
            hidden = int(router.weight.shape[1])
            intermediate = int(experts.down_proj.shape[2])
            layers.append(
                MoELayerInfo(
                    layer_id=layer_id,
                    stack="decoder",
                    position=len(layers),
                    module_path=path,
                    router_path=f"{path}.gate",
                    experts_path=f"{path}.experts",
                    num_experts=num_experts,
                    top_k=int(router.top_k),
                    hidden_dim=hidden,
                    expert_input_dim=hidden,
                    expert_intermediate_dim=intermediate,
                    expert_output_dim=int(experts.down_proj.shape[1]),
                    expert_capacity=None,
                )
            )
        return layers

    # ------------------------------------------------------------------
    # Module access
    # ------------------------------------------------------------------
    def get_moe_module(self, model, layer):
        return model.get_submodule(layer.module_path)

    def get_router(self, model, layer):
        return model.get_submodule(layer.router_path)

    def _experts(self, model, layer) -> nn.Module:
        return model.get_submodule(layer.experts_path)

    def get_expert(self, model, layer, expert_id):
        if not 0 <= expert_id < layer.num_experts:
            raise IndexError(f"Expert {expert_id} out of range for {layer.layer_id}")
        experts = self._experts(model, layer)
        if isinstance(experts, PrunedFusedExperts):
            return experts.expert(expert_id)
        return FusedExpertView(experts, expert_id)

    def build_original_expert(self, model, layer):
        return OlmoeReferenceExpert(
            layer.hidden_dim, layer.expert_intermediate_dim, ACT2FN[model.config.hidden_act]
        )

    def expert_state_prefix(self, layer, expert_id):
        return f"{layer.experts_path}.replaced.{expert_id}."

    def replace_experts_in_layer(self, model, layer, modules):
        experts = self._experts(model, layer)
        already = set(experts.removed_ids()) if isinstance(experts, PrunedFusedExperts) else set()
        overlap = already & set(modules)
        if overlap:
            raise ValueError(f"{layer.layer_id}: experts {sorted(overlap)} are already replaced")
        if len(already | set(modules)) >= layer.num_experts:
            raise ValueError("Refusing to replace every expert of a layer")
        per_expert = experts.gate_up_proj[0].numel() + experts.down_proj[0].numel()
        compact = PrunedFusedExperts.compact(experts, sorted(modules))
        for expert_id, module in modules.items():
            compact.replaced[str(expert_id)] = module
        self.get_moe_module(model, layer).experts = compact
        return per_expert * len(modules)

    def set_expert(self, model, layer, expert_id, module):
        previous = self.build_original_expert(model, layer)
        view = self.get_expert(model, layer, expert_id)
        if isinstance(view, FusedExpertView):
            owner = view._owner[0]
            with torch.no_grad():
                previous.gate_up_proj = nn.Parameter(owner.gate_up_proj[view.row].detach().clone())
                previous.down_proj = nn.Parameter(owner.down_proj[view.row].detach().clone())
        self.replace_experts_in_layer(model, layer, {expert_id: module})
        return previous

    def removed_weight_violations(self, state_shapes, layer, removed):
        bad = super().removed_weight_violations(state_shapes, layer, removed)
        rows = state_shapes.get(f"{layer.experts_path}.gate_up_proj")
        if rows is None or rows[0] != layer.num_experts - len(removed):
            bad.append(
                f"{layer.experts_path}.gate_up_proj has {None if rows is None else rows[0]} rows, "
                f"expected {layer.num_experts - len(removed)} (removed experts must not be stored)"
            )
        return bad

    def validate_loaded(self, model, layers):
        for layer in layers:
            experts = self._experts(model, layer)
            if isinstance(experts, PrunedFusedExperts):
                stored = [int(e) for e in experts.retained_expert_ids.tolist()]
                if stored != experts.retained_ids():
                    raise RuntimeError(
                        f"{layer.layer_id}: checkpoint rows belong to experts {stored}, "
                        f"manifest implies {experts.retained_ids()}"
                    )

    def load_original_expert(self, reader, model, layer, expert_id):
        """Read one expert from the hub layout (per expert) or the fused layout."""
        module = self.build_original_expert(model, layer)
        per_expert = f"{layer.module_path}.experts.{expert_id}."
        fused = f"{layer.experts_path}."
        if reader.has(per_expert + "gate_proj.weight"):
            gate = reader.get(per_expert + "gate_proj.weight")
            up = reader.get(per_expert + "up_proj.weight")
            gate_up = torch.cat([gate, up], dim=0)
            down = reader.get(per_expert + "down_proj.weight")
        elif reader.has(fused + "gate_up_proj"):
            gate_up = reader.get(fused + "gate_up_proj")[expert_id]
            down = reader.get(fused + "down_proj")[expert_id]
        else:
            raise KeyError(f"Original weights of {layer.layer_id} expert {expert_id} not found")
        module.load_state_dict({"gate_up_proj": gate_up, "down_proj": down}, strict=True)
        return module

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def route(self, model, layer, moe_input, apply_mask=True):
        router = self.get_router(model, layer)
        previous = getattr(router, "_moe_pruning_mask_disabled", False)
        router._moe_pruning_mask_disabled = not apply_mask
        try:
            _, weights, index = router(moe_input.reshape(-1, layer.hidden_dim))
        finally:
            router._moe_pruning_mask_disabled = previous
        shape = (*moe_input.shape[:-1], layer.top_k)
        return index.reshape(shape), weights.reshape(shape)

    @staticmethod
    def _masked_routing_hook(router, _args, output):
        """Top-k over the non-masked experts; gates are the original probabilities.

        Mirrors OlmoeTopKRouter.forward (float32 softmax, top-k, optional
        renormalisation over the selected experts, cast to the logits dtype).
        """
        masked = getattr(router, "_moe_pruning_masked_experts", ())
        if not masked or getattr(router, "_moe_pruning_mask_disabled", False):
            return None
        logits, _scores, _index = output
        probs = torch.softmax(logits, dtype=torch.float, dim=-1)
        selectable = probs.clone()
        selectable[..., list(masked)] = -1.0
        index = torch.topk(selectable, router.top_k, dim=-1).indices
        weights = probs.gather(-1, index)
        if router.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights.to(logits.dtype), index

    def set_routing_mask(self, model, layer, expert_ids):
        if len(set(expert_ids)) > layer.num_experts - layer.top_k:
            raise ValueError(f"At least top_k={layer.top_k} experts must remain reachable")
        router = self.get_router(model, layer)
        router._moe_pruning_masked_experts = tuple(sorted(set(expert_ids)))
        handle = getattr(router, "_moe_pruning_mask_handle", None)
        if expert_ids and handle is None:
            router._moe_pruning_mask_handle = router.register_forward_hook(self._masked_routing_hook, prepend=True)
        elif not expert_ids and handle is not None:
            handle.remove()
            router._moe_pruning_mask_handle = None

    def get_routing_mask(self, model, layer):
        return list(getattr(self.get_router(model, layer), "_moe_pruning_masked_experts", ()))

    def run_moe_layer(self, model, layer, moe_input):
        return self.get_moe_module(model, layer)(moe_input)

    # ------------------------------------------------------------------
    # Instrumentation
    # ------------------------------------------------------------------
    @contextmanager
    def instrument(self, model, layers, callback: RecordCallback) -> Iterator[None]:
        handles = []
        try:
            for layer in layers:
                handles += self._register(model, layer, callback)
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _register(self, model, layer, callback):
        state: dict = {}
        decoder_layer = model.get_submodule(layer.module_path.rsplit(".", 1)[0])
        moe = self.get_moe_module(model, layer)
        router = self.get_router(model, layer)

        def norm_pre(_m, args):
            state.clear()
            state["residual"] = args[0].detach()

        def moe_pre(_m, args):
            state["moe_input"] = args[0].detach()

        def router_hook(_m, _args, output):
            _logits, weights, index = output
            state["weights"], state["index"] = weights.detach(), index.detach()

        def moe_hook(_m, _args, output):
            x = state["moe_input"]
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            index, weights = state["index"], state["weights"]
            k = index.shape[-1]
            expert_output = torch.zeros(flat.shape[0], k, layer.expert_output_dim, dtype=x.dtype, device=x.device)
            # Expert outputs are not exposed by the fused kernel; recompute them
            # (un-gated) with the same weights for the routed (token, slot) pairs.
            with torch.no_grad():
                for expert_id in torch.unique(index).tolist():
                    token_idx, slot = torch.where(index == expert_id)
                    out = self.get_expert(model, layer, expert_id)(flat[token_idx])
                    expert_output[token_idx, slot] = out.to(x.dtype)
            callback(
                MoEForwardRecord(
                    layer_id=layer.layer_id,
                    residual=state["residual"],
                    moe_input=x,
                    moe_output=output.detach(),
                    expert_index=index.reshape(*shape[:-1], k),
                    gate=weights.reshape(*shape[:-1], k),
                    expert_output=expert_output.reshape(*shape[:-1], k, -1),
                )
            )

        return [
            decoder_layer.post_attention_layernorm.register_forward_pre_hook(norm_pre),
            moe.register_forward_pre_hook(moe_pre),
            router.register_forward_hook(router_hook),
            moe.register_forward_hook(moe_hook),
        ]

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def describe(self, model):
        layers = self.discover_layers(model)
        cfg = model.config
        return {
            "model_type": cfg.model_type,
            "model_class": type(model).__name__,
            "adapter": type(self).__name__,
            "is_encoder_decoder": False,
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_experts": cfg.num_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "norm_topk_prob": cfg.norm_topk_prob,
            "num_sparse_layers": len(layers),
            "num_encoder_sparse_layers": 0,
            "num_decoder_sparse_layers": len(layers),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "routing": {
                "type": f"top-{cfg.num_experts_per_tok} of {cfg.num_experts} (softmax, no capacity limit)",
                "gate_value": "softmax probability of each selected expert"
                + (" renormalised over the top-k" if cfg.norm_topk_prob else " (not renormalised)"),
                "gate_application": "OlmoeExperts.forward: expert(x) * top_k_weights, applied exactly once",
                "padding": "padding tokens are routed but excluded from every statistic and loss",
            },
            "layers": [
                {**layer.to_dict(), "moe_class": type(self.get_moe_module(model, layer)).__name__,
                 "experts_class": type(self._experts(model, layer)).__name__}
                for layer in layers
            ],
        }
