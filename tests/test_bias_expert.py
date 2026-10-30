import pytest
import torch

from moe_pruning.model.architecture import get_adapter
from moe_pruning.pruning.bias_expert import BiasExpert
from tests.conftest import make_tiny_model


def test_shape_dtype_device_and_gradients():
    init = torch.arange(4, dtype=torch.float32)
    expert = BiasExpert(4, init=init, dtype=torch.float64)
    out = expert(torch.randn(3, 5, 7))  # input width may differ from output width
    assert out.shape == (3, 5, 4)
    assert out.dtype == torch.float64 and out.device == expert.bias.device
    torch.testing.assert_close(out[2, 4], init.double())
    out.sum().backward()
    torch.testing.assert_close(expert.bias.grad, torch.full((4,), 15.0, dtype=torch.float64))
    assert isinstance(expert.bias, torch.nn.Parameter)


def test_output_is_a_copy():
    expert = BiasExpert(3, init=torch.ones(3))
    out = expert(torch.zeros(2, 3))
    out.add_(5)
    torch.testing.assert_close(expert.bias.data, torch.ones(3))


def test_rejects_wrong_init_shape():
    with pytest.raises(ValueError):
        BiasExpert(4, init=torch.zeros(5))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gate_applied_exactly_once_in_switch_layer(seed):
    """y_t must equal g_t * b for every token routed to the replaced slot."""
    model = make_tiny_model(seed=seed)
    adapter = get_adapter(model)
    layer = adapter.discover_layers(model)[0]
    # Spread router probabilities so that tokens get clearly different gates.
    with torch.no_grad():
        adapter.get_router(model, layer).classifier.weight.mul_(5.0)
    b = torch.randn(16)
    for e in range(layer.num_experts):
        adapter.set_expert(model, layer, e, BiasExpert(16, init=b * (e + 1)))
    x = torch.randn(2, 30, 16)
    with torch.no_grad():
        index, gate = adapter.route(model, layer, x)
        y = adapter.run_moe_layer(model, layer, x)
    gates = gate[..., 0]
    assert gates[index[..., 0] >= 0].std() > 0.01  # genuinely different probabilities
    for bi in range(2):
        for t in range(30):
            e = int(index[bi, t, 0])
            expected = torch.zeros(16) if e < 0 else gates[bi, t] * b * (e + 1)
            torch.testing.assert_close(y[bi, t], expected, rtol=1e-5, atol=1e-6)
    # Scaling: doubling-free check that output norm is proportional to g.
    e0 = index[..., 0] == 0
    if e0.sum() >= 2:
        ratio = y[e0].norm(dim=-1) / gates[e0]
        torch.testing.assert_close(ratio, torch.full_like(ratio, float(b.norm())), rtol=1e-4, atol=1e-4)
