import pytest
import torch

from moe_pruning.model.architecture import MoEForwardRecord, MoELayerInfo
from moe_pruning.pruning.bias_initialization import UnobservedExpertError, initialize_bias, weighted_mean
from moe_pruning.scoring.easyep import token_contribution
from moe_pruning.scoring.statistics import ExpertStatisticsAccumulator, LayerStatistics


def stats_from_tokens(E, g, s, dim=None, num_experts=1, expert=0) -> LayerStatistics:
    """Build statistics for one expert directly from per-token values."""
    E, g, s = E.double(), g.double(), s.double()
    st = LayerStatistics("layer", num_experts, dim or E.shape[1])
    w = s * g**2
    ex = st.expert
    ex["count"][expert] = ex["bias_count"][expert] = len(g)
    ex["output_sum"][expert] = E.sum(0)
    ex["w_sum"][expert] = w.sum()
    ex["w_output_sum"][expert] = (w[:, None] * E).sum(0)
    ex["w_output_sq_sum"][expert] = (w * E.square().sum(1)).sum()
    ex["g2_sum"][expert] = (g**2).sum()
    ex["g2_output_sum"][expert] = ((g**2)[:, None] * E).sum(0)
    ex["g2_output_sq_sum"][expert] = ((g**2) * E.square().sum(1)).sum()
    st.layer["tokens"] = float(len(g))
    return st


def objective(b, E, g, s):
    return float((s * g**2 * (E - b).square().sum(-1)).sum())


def test_matches_analytical_formula():
    torch.manual_seed(0)
    E, g, s = torch.randn(50, 8).double(), torch.rand(50).double(), torch.rand(50).double()
    bias, result = initialize_bias(stats_from_tokens(E, g, s), 0)
    expected = (s[:, None] * g[:, None] ** 2 * E).sum(0) / (s * g**2).sum()
    torch.testing.assert_close(bias, expected)
    assert result.method == "gate_token_weighted_mean" and not result.fallback
    # Diagnostics from running sums equal the brute-force residual.
    brute = objective(bias, E, g, s) / float((s * g**2 * E.square().sum(-1)).sum())
    assert result.weighted_relative_residual == pytest.approx(brute, rel=1e-9)


def test_bias_minimises_weighted_objective():
    torch.manual_seed(1)
    E, g, s = torch.randn(40, 6).double(), torch.rand(40).double(), torch.rand(40).double()
    bias, _ = initialize_bias(stats_from_tokens(E, g, s), 0)
    best = objective(bias, E, g, s)
    for _ in range(50):
        assert objective(bias + 0.05 * torch.randn(6).double(), E, g, s) > best
    assert objective(E.mean(0), E, g, s) >= best  # better than the unweighted mean
    # Gradient of the objective vanishes at b.
    b = bias.clone().requires_grad_(True)
    (s * g**2 * (E - b).square().sum(-1)).sum().backward()
    assert b.grad.abs().max() < 1e-9


def test_higher_gate_has_more_influence():
    E = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bias, _ = initialize_bias(stats_from_tokens(E, torch.tensor([0.9, 0.3]), torch.tensor([1.0, 1.0])), 0)
    assert bias[0] > bias[1]
    # Squared gate: weights 0.81 vs 0.09
    torch.testing.assert_close(bias, torch.tensor([0.9, 0.1], dtype=torch.float64))


def test_higher_token_contribution_has_more_influence():
    E = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bias, _ = initialize_bias(stats_from_tokens(E, torch.tensor([0.5, 0.5]), torch.tensor([0.2, 0.6])), 0)
    assert bias[1] > bias[0]
    torch.testing.assert_close(bias, torch.tensor([0.25, 0.75], dtype=torch.float64))


def test_zero_denominator_falls_back_and_is_flagged(caplog):
    E = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    st = stats_from_tokens(E, torch.tensor([0.5, 0.5]), torch.tensor([0.0, 0.0]))
    with caplog.at_level("WARNING"):
        bias, result = initialize_bias(st, 0)
    assert result.fallback and result.method == "gate_weighted_mean"
    torch.testing.assert_close(bias, torch.tensor([1.0, 2.0], dtype=torch.float64))
    assert "falling back" in caplog.text
    assert weighted_mean(torch.ones(2), 0.0) is None


def test_unobserved_expert_is_not_invented():
    st = LayerStatistics("layer", 2, 3)
    with pytest.raises(UnobservedExpertError):
        initialize_bias(st, 1)
    bias, result = initialize_bias(st, 1, allow_unobserved=True)
    assert result.method == "zero_vector_unobserved_expert" and result.fallback
    assert torch.all(bias == 0)


def test_accumulator_end_to_end_matches_brute_force():
    """Feed a synthetic forward record through the real accumulator."""
    torch.manual_seed(3)
    B, S, D = 2, 7, 5
    layer = MoELayerInfo("layer", "encoder", 0, "m", "m.r", "m.e", 3, 1, D, D, 8, D, None)
    index = torch.randint(-1, 3, (B, S, 1))
    gate = torch.rand(B, S, 1) * (index >= 0)
    expert_output = torch.randn(B, S, 1, D) * (index >= 0).unsqueeze(-1)
    moe_output = (gate.unsqueeze(-1) * expert_output).sum(2)
    residual = torch.randn(B, S, D)
    record = MoEForwardRecord("layer", residual, torch.randn(B, S, D), moe_output, index, gate, expert_output)
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[1, 5:] = False
    acc = ExpertStatisticsAccumulator([layer])
    acc.update(record, mask)

    s = token_contribution(residual, moe_output)
    for e in range(3):
        sel = (index[..., 0] == e) & mask
        if not sel.any():
            continue
        E, g, s_e = expert_output[..., 0, :][sel].double(), gate[..., 0][sel].double(), s[sel].double()
        expected = (s_e[:, None] * g[:, None] ** 2 * E).sum(0) / (s_e * g**2).sum()
        bias, _ = initialize_bias(acc.stats["layer"], e)
        torch.testing.assert_close(bias, expected, rtol=1e-5, atol=1e-6)
