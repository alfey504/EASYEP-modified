r"""Layer-level reconstruction training of replacement biases.

Objective for one sparse MoE layer, over the N real tokens of the cached
calibration inputs :math:`H_i`:

.. math::

    \mathcal L_{MSE} = \frac1N \sum_{i=1}^{N}\lVert F_{orig}(H_i) - F_{repl}(H_i)\rVert_2^2

:math:`F_{orig}(H_i)` is cached from the unmodified model; :math:`F_{repl}` is the
same layer with BiasExpert slots (router and retained experts identical, so
routing of the cached input is unchanged). Only the BiasExpert biases receive
gradients. This is a *local* objective: when several layers are pruned,
upstream changes alter downstream hidden states and routing, which the
per-layer objective does not see.

Because routing is fixed for the cached inputs, the objective is quadratic in
the biases. With top-1 routing it separates per expert, with the closed-form
minimiser :math:`b^*_e = \sum g^2E / \sum g^2` over the training tokens routed
to :math:`e` (no token-contribution weight), reported as a reference for the
optimiser. With top-k routing (k > 1) the biases are coupled and no closed form
is reported.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from moe_pruning.model.architecture import MoEAdapter, MoELayerInfo
from moe_pruning.pruning.bias_expert import BiasExpert
from moe_pruning.utils.paths import read_json, write_json

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Reference cache (original layer inputs/outputs, real tokens only)
# ----------------------------------------------------------------------
@dataclass
class LayerReference:
    layer_id: str
    example_ids: list[int]
    offsets: list[int]  # example i occupies rows offsets[i]:offsets[i+1]
    inputs: torch.Tensor  # [T, D] MoE input (after the pre-norm)
    outputs: torch.Tensor  # [T, D] original MoE output

    def sequence(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.inputs[a:b], self.outputs[a:b]


def _cache_file(cache_dir: Path, layer_id: str) -> Path:
    return cache_dir / f"{layer_id}.safetensors"


def save_layer_reference(ref: LayerReference, cache_dir: Path, meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "inputs": ref.inputs.contiguous(),
            "outputs": ref.outputs.contiguous(),
            "offsets": torch.tensor(ref.offsets, dtype=torch.long),
            "example_ids": torch.tensor(ref.example_ids, dtype=torch.long),
        },
        str(_cache_file(cache_dir, ref.layer_id)),
    )
    write_json(cache_dir / f"{ref.layer_id}.json", meta)


def load_layer_reference(cache_dir: Path, layer_id: str) -> tuple[LayerReference, dict]:
    t = load_file(str(_cache_file(cache_dir, layer_id)))
    meta = read_json(cache_dir / f"{layer_id}.json")
    ref = LayerReference(layer_id, t["example_ids"].tolist(), t["offsets"].tolist(), t["inputs"], t["outputs"])
    return ref, meta


def cache_is_valid(cache_dir: Path, layer_ids: list[str], expected_meta: dict) -> bool:
    for layer_id in layer_ids:
        meta_path = cache_dir / f"{layer_id}.json"
        if not meta_path.exists() or not _cache_file(cache_dir, layer_id).exists():
            return False
        if read_json(meta_path) != expected_meta:
            return False
    return True


@torch.no_grad()
def capture_layer_references(
    model: nn.Module,
    adapter: MoEAdapter,
    layers: list[MoELayerInfo],
    batches,
    device: torch.device,
) -> dict[str, LayerReference]:
    """Run the original model and keep (input, output) of each layer for real tokens."""
    store: dict[str, dict[str, list]] = {layer.layer_id: {"x": [], "y": [], "ids": []} for layer in layers}
    stack = {layer.layer_id: layer.stack for layer in layers}
    current: dict = {}

    def on_record(rec):
        mask = current["masks"][stack[rec.layer_id]]
        for row, eid in enumerate(current["ids"]):
            n = int(mask[row].sum())
            # Right padding: real tokens are the prefix of the row.
            if not bool(mask[row, :n].all()):
                raise RuntimeError("Expected right-padded sequences")
            store[rec.layer_id]["x"].append(rec.moe_input[row, :n].float().cpu())
            store[rec.layer_id]["y"].append(rec.moe_output[row, :n].float().cpu())
            store[rec.layer_id]["ids"].append(eid)

    with adapter.instrument(model, layers, on_record):
        for batch in batches:
            current["masks"] = batch.token_masks(device)
            current["ids"] = batch.example_ids
            model(**batch.model_inputs(device))

    refs = {}
    for layer_id, s in store.items():
        lengths = [x.shape[0] for x in s["x"]]
        offsets = [0]
        for n in lengths:
            offsets.append(offsets[-1] + n)
        refs[layer_id] = LayerReference(layer_id, s["ids"], offsets, torch.cat(s["x"]), torch.cat(s["y"]))
    return refs


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------
def masked_sequence_batch(
    ref: LayerReference, rows: list[int], device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad sequences with zeros. Returns x [B,S,D], y [B,S,D], mask [B,S]."""
    seqs = [ref.sequence(i) for i in rows]
    max_len = max(x.shape[0] for x, _ in seqs)
    dim = ref.inputs.shape[1]
    x = torch.zeros(len(rows), max_len, dim, dtype=dtype)
    y = torch.zeros(len(rows), max_len, dim, dtype=torch.float32)
    mask = torch.zeros(len(rows), max_len, dtype=torch.bool)
    for i, (xi, yi) in enumerate(seqs):
        n = xi.shape[0]
        x[i, :n] = xi.to(dtype)
        y[i, :n] = yi
        mask[i, :n] = True
    return x.to(device), y.to(device), mask.to(device)


def reconstruction_loss(y_ref: torch.Tensor, y_hat: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """(1/N) * sum over real tokens of ||y_ref - y_hat||_2^2, and N."""
    sq = (y_ref - y_hat.float()).square().sum(dim=-1)
    n = int(mask.sum())
    return (sq * mask).sum() / max(n, 1), n


@torch.no_grad()
def evaluate_reconstruction(
    model: nn.Module, adapter: MoEAdapter, layer: MoELayerInfo, ref: LayerReference, rows: list[int],
    device: torch.device, batch_size: int,
) -> float:
    dtype = next(model.parameters()).dtype
    total, count = 0.0, 0
    for start in range(0, len(rows), batch_size):
        x, y, mask = masked_sequence_batch(ref, rows[start : start + batch_size], device, dtype)
        loss, n = reconstruction_loss(y, adapter.run_moe_layer(model, layer, x), mask)
        total += float(loss) * n
        count += n
    return total / max(count, 1)


@torch.no_grad()
def closed_form_biases(
    model: nn.Module, adapter: MoEAdapter, layer: MoELayerInfo, ref: LayerReference, rows: list[int],
    expert_ids: list[int], device: torch.device, batch_size: int,
) -> dict[int, torch.Tensor | None]:
    """Minimiser of the layer MSE on ``rows``: b*_e = sum g^2 E / sum g^2.

    With top-1 routing the original output of a routed token is y = g * E_e(x),
    so E_e(x) = y / g is recovered exactly from the cached reference.
    """
    dtype = next(model.parameters()).dtype
    num = {e: torch.zeros(ref.inputs.shape[1], dtype=torch.float64) for e in expert_ids}
    den = {e: 0.0 for e in expert_ids}
    if layer.top_k != 1:
        raise NotImplementedError("closed-form reference implemented for top-1 routing")
    for start in range(0, len(rows), batch_size):
        x, y, mask = masked_sequence_batch(ref, rows[start : start + batch_size], device, dtype)
        index, gate = adapter.route(model, layer, x)
        idx, g = index[..., 0], gate[..., 0].float()
        for e in expert_ids:
            sel = (idx == e) & mask
            if sel.any():
                ge = g[sel].cpu().double()
                expert_out = y[sel].cpu().double() / ge[:, None]
                num[e] += (ge.square()[:, None] * expert_out).sum(0)
                den[e] += float(ge.square().sum())
    return {e: (num[e] / den[e] if den[e] > 0 else None) for e in expert_ids}


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
@dataclass
class LayerTrainingResult:
    layer_id: str
    expert_ids: list[int]
    trainable_parameters: int
    steps: int
    skipped_steps_without_routed_tokens: int
    initial_train_loss: float
    initial_validation_loss: float
    final_train_loss: float
    final_validation_loss: float
    closed_form_train_loss: float | None
    closed_form_validation_loss: float | None
    bias_change_norm: dict[str, float]
    distance_to_closed_form: dict[str, float | None]
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _layer_biases(model: nn.Module, adapter: MoEAdapter, layer: MoELayerInfo, expert_ids: list[int]) -> dict[int, BiasExpert]:
    out = {}
    for e in expert_ids:
        module = adapter.get_expert(model, layer, e)
        if not isinstance(module, BiasExpert):
            raise TypeError(f"{layer.layer_id} expert {e} is not a BiasExpert")
        out[e] = module
    return out


def train_layer_biases(
    model: nn.Module,
    adapter: MoEAdapter,
    layer: MoELayerInfo,
    expert_ids: list[int],
    ref: LayerReference,
    train_rows: list[int],
    val_rows: list[int],
    device: torch.device,
    learning_rate: float,
    steps: int,
    batch_size: int,
    eval_interval: int,
    seed: int,
) -> LayerTrainingResult:
    """Optimise all BiasExpert biases of one layer jointly with Adam."""
    start_time = time.perf_counter()
    model.eval()  # no dropout: the reference was produced in eval mode
    experts = _layer_biases(model, adapter, layer, expert_ids)
    params = [m.bias for m in experts.values()]
    if not all(p.requires_grad for p in params):
        raise RuntimeError("Bias parameters must be trainable (call freeze_all_but_biases first)")
    initial = {e: m.bias.detach().clone() for e, m in experts.items()}
    dtype = next(model.parameters()).dtype
    eval_bs = max(batch_size, 8)

    def evaluate(rows):
        return evaluate_reconstruction(model, adapter, layer, ref, rows, device, eval_bs) if rows else float("nan")

    # Closed-form reference (does not modify the model). Only for top-1 routing: with
    # top-k a token can mix several replaced experts, which couples their biases.
    if layer.top_k == 1:
        closed = closed_form_biases(model, adapter, layer, ref, train_rows, expert_ids, device, eval_bs)
    else:
        closed = {e: None for e in expert_ids}
    closed_train = closed_val = None
    if all(v is not None for v in closed.values()):
        with torch.no_grad():
            for e, m in experts.items():
                m.bias.copy_(closed[e].to(m.bias.dtype))
            closed_train, closed_val = evaluate(train_rows), evaluate(val_rows)
            for e, m in experts.items():
                m.bias.copy_(initial[e])

    history = [{"step": 0, "train_loss": evaluate(train_rows), "validation_loss": evaluate(val_rows)}]
    logger.info(
        "%s: initial train %.4f | val %.4f | closed-form optimum train %s val %s",
        layer.layer_id, history[0]["train_loss"], history[0]["validation_loss"],
        f"{closed_train:.4f}" if closed_train is not None else "n/a",
        f"{closed_val:.4f}" if closed_val is not None else "n/a",
    )

    optimizer = torch.optim.Adam(params, lr=learning_rate)
    rng = random.Random(seed)
    order: list[int] = []
    skipped_steps = 0
    for step in range(1, steps + 1):
        if len(order) < batch_size:
            refill = list(train_rows)
            rng.shuffle(refill)
            order.extend(refill)
        rows, order = order[:batch_size], order[batch_size:]
        x, y, mask = masked_sequence_batch(ref, rows, device, dtype)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = reconstruction_loss(y, adapter.run_moe_layer(model, layer, x), mask)
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
        else:
            # No token of this batch was routed to a replaced expert: the loss does
            # not depend on the biases. Skip the update rather than letting Adam
            # momentum move them on a zero gradient.
            skipped_steps += 1
        if step % eval_interval == 0 or step == steps:
            history.append(
                {"step": step, "batch_loss": float(loss.detach()), "train_loss": evaluate(train_rows),
                 "validation_loss": evaluate(val_rows)}
            )
            logger.info(
                "%s step %4d | batch %.4f | train %.4f | val %.4f",
                layer.layer_id, step, float(loss.detach()), history[-1]["train_loss"], history[-1]["validation_loss"],
            )

    return LayerTrainingResult(
        layer_id=layer.layer_id,
        expert_ids=expert_ids,
        trainable_parameters=sum(p.numel() for p in params),
        steps=steps,
        skipped_steps_without_routed_tokens=skipped_steps,
        initial_train_loss=history[0]["train_loss"],
        initial_validation_loss=history[0]["validation_loss"],
        final_train_loss=history[-1]["train_loss"],
        final_validation_loss=history[-1]["validation_loss"],
        closed_form_train_loss=closed_train,
        closed_form_validation_loss=closed_val,
        bias_change_norm={str(e): float((m.bias.detach() - initial[e]).norm()) for e, m in experts.items()},
        distance_to_closed_form={
            str(e): (float((m.bias.detach().cpu().double() - closed[e]).norm()) if closed[e] is not None else None)
            for e, m in experts.items()
        },
        history=history,
        seconds=round(time.perf_counter() - start_time, 3),
    )
