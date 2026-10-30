"""Expert containers for OLMoE in transformers 5.

transformers 5 stores all experts of a layer as two 3D tensors
(``OlmoeExperts.gate_up_proj`` [E, 2I, D] and ``down_proj`` [E, D, I]), so a
single expert cannot be swapped as a module. ``PrunedFusedExperts`` replaces
the whole container: it keeps only the rows of retained experts (so removed
weights really leave memory and the checkpoint) and holds the replacement
modules (BiasExpert / ZeroExpert) of removed experts, keyed by their ORIGINAL
expert id, so the router's indexing is unchanged.

Expert e (SwiGLU): ``down_proj[e] @ (act(gate) * up)`` with
``gate, up = (gate_up_proj[e] @ x).chunk(2)`` - the layout transformers uses
when it merges the hub's per-expert ``gate_proj``/``up_proj`` weights
(``Concatenate(dim=1)`` of [gate, up]).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _swiglu_expert(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, act_fn) -> torch.Tensor:
    gate, up = F.linear(x, gate_up).chunk(2, dim=-1)
    return F.linear(act_fn(gate) * up, down)


class FusedExpertView(nn.Module):
    """Callable view of one row of a fused expert container (owns no parameters)."""

    def __init__(self, owner: nn.Module, row: int):
        super().__init__()
        self._owner = [owner]  # a list keeps the owner out of this module's registry
        self.row = row

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        owner = self._owner[0]
        return _swiglu_expert(x, owner.gate_up_proj[self.row], owner.down_proj[self.row], owner.act_fn)


class OlmoeReferenceExpert(nn.Module):
    """A single standalone original expert (used to measure reconstruction errors)."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, act_fn, device=None, dtype=None):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(2 * intermediate_dim, hidden_dim, device=device, dtype=dtype))
        self.down_proj = nn.Parameter(torch.empty(hidden_dim, intermediate_dim, device=device, dtype=dtype))
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _swiglu_expert(x, self.gate_up_proj, self.down_proj, self.act_fn)


class PrunedFusedExperts(nn.Module):
    """Drop-in replacement for ``OlmoeExperts`` with some experts removed.

    Same forward signature: ``(hidden_states [T,D], top_k_index [T,K], top_k_weights [T,K])``.
    Each routed (token, slot) is computed by its expert and scaled by its
    weight exactly once, as in ``OlmoeExperts``; removed experts are served by
    their replacement module.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn,
        retained_ids: list[int],
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn
        r = len(retained_ids)
        self.gate_up_proj = nn.Parameter(torch.empty(r, 2 * intermediate_dim, hidden_dim, device=device, dtype=dtype))
        self.down_proj = nn.Parameter(torch.empty(r, hidden_dim, intermediate_dim, device=device, dtype=dtype))
        # Persistent: the checkpoint records which original experts the rows belong to.
        self.register_buffer("retained_expert_ids", torch.tensor(retained_ids, dtype=torch.long, device=device))
        self.replaced = nn.ModuleDict()
        self._row_of = {e: i for i, e in enumerate(retained_ids)}

    # ------------------------------------------------------------------
    @classmethod
    def compact(cls, experts: nn.Module, remove: list[int]) -> "PrunedFusedExperts":
        """New container without the experts in ``remove``.

        ``experts`` is an original ``OlmoeExperts`` or an existing
        ``PrunedFusedExperts`` (whose replacement modules are carried over).
        """
        existing = experts if isinstance(experts, PrunedFusedExperts) else None
        retained_before = (
            existing.retained_ids() if existing is not None else list(range(experts.gate_up_proj.shape[0]))
        )
        retained = [e for e in retained_before if e not in set(remove)]
        rows = torch.tensor([retained_before.index(e) for e in retained], dtype=torch.long)
        num_experts = existing.num_experts if existing is not None else experts.gate_up_proj.shape[0]
        module = cls(
            num_experts,
            experts.gate_up_proj.shape[2],
            experts.down_proj.shape[2],
            experts.act_fn,
            retained,
            device=experts.gate_up_proj.device,
            dtype=experts.gate_up_proj.dtype,
        )
        with torch.no_grad():
            index = rows.to(experts.gate_up_proj.device)
            module.gate_up_proj = nn.Parameter(experts.gate_up_proj.index_select(0, index).clone())
            module.down_proj = nn.Parameter(experts.down_proj.index_select(0, index).clone())
        if existing is not None:
            for key, replacement in existing.replaced.items():
                module.replaced[key] = replacement
        return module

    def retained_ids(self) -> list[int]:
        """Original ids of the retained experts, in row order."""
        return sorted(self._row_of, key=self._row_of.get)

    def removed_ids(self) -> list[int]:
        return sorted(int(k) for k in self.replaced.keys())

    def expert(self, expert_id: int) -> nn.Module:
        key = str(expert_id)
        if key in self.replaced:
            return self.replaced[key]
        if expert_id not in self._row_of:
            raise KeyError(f"Expert {expert_id} is neither retained nor replaced")
        return FusedExpertView(self, self._row_of[expert_id])

    # ------------------------------------------------------------------
    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            hit = torch.unique(top_k_index).tolist()
        for expert_id in hit:
            if not 0 <= expert_id < self.num_experts:
                continue  # transformers uses num_experts as a "no expert" sentinel
            token_idx, slot = torch.where(top_k_index == expert_id)
            out = self.expert(expert_id)(hidden_states[token_idx])
            out = out * top_k_weights[token_idx, slot, None]
            final.index_add_(0, token_idx, out.to(final.dtype))
        return final

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, retained={len(self._row_of)}, "
            f"replaced={self.removed_ids()}"
        )
