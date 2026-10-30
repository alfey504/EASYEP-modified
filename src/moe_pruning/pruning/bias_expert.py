r"""Constant replacement for a pruned expert.

:math:`\hat E_e(h_t) = b_e` for every token routed to the slot. The module does
*not* apply the router gate: the parent MoE layer multiplies whatever the
expert returns by :math:`g_{e,t}` (verified for Switch Transformers), giving
:math:`\hat y_{e,t} = g_{e,t} b_e` with the gate applied exactly once.

The class is architecture-agnostic: it only needs the expert output width.
"""

from __future__ import annotations

import torch
from torch import nn


class BiasExpert(nn.Module):
    def __init__(
        self,
        output_dim: int,
        init: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        if init is not None:
            if init.shape != (output_dim,):
                raise ValueError(f"init has shape {tuple(init.shape)}, expected ({output_dim},)")
            value = init.detach().to(dtype=dtype or init.dtype, device=device or init.device).clone()
        else:
            value = torch.zeros(output_dim, dtype=dtype, device=device)
        self.output_dim = output_dim
        self.bias = nn.Parameter(value)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return b broadcast to ``hidden_states.shape[:-1] + (output_dim,)`` (un-gated)."""
        shape = (*hidden_states.shape[:-1], self.output_dim)
        return self.bias.expand(shape).clone()

    def extra_repr(self) -> str:
        return f"output_dim={self.output_dim}"
