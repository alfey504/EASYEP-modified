"""Parameter-free placeholder for a removed expert.

Used by the ``drop`` and ``reroute`` replacement methods. The slot is kept so
that the router's expert indexing is unchanged; the original weights are gone.

* ``drop``: tokens routed to the slot get zero expert output, so the parent's
  gate multiplication also gives zero (the token keeps only its residual).
* ``reroute``: the routing mask guarantees that no token reaches the slot;
  ``must_be_unused=True`` turns any token that does arrive into an error
  instead of a silent zero.
"""

from __future__ import annotations

import torch
from torch import nn


class ZeroExpert(nn.Module):
    def __init__(self, output_dim: int, must_be_unused: bool = False):
        super().__init__()
        self.output_dim = output_dim
        self.must_be_unused = must_be_unused

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.must_be_unused and hidden_states.shape[:-1].numel() > 0:
            raise RuntimeError(
                f"{hidden_states.shape[:-1].numel()} token(s) reached a removed expert that should be "
                "unreachable (routing mask not applied?)"
            )
        return hidden_states.new_zeros((*hidden_states.shape[:-1], self.output_dim))

    def extra_repr(self) -> str:
        return f"output_dim={self.output_dim}, must_be_unused={self.must_be_unused}"
