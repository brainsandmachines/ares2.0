import pytest
import torch

from ares.utils.gradnorm import DBP, combine_gradnorm_objective


def _expected_dbp(gradients, eps, std, penalty_norm):
    per_sample = gradients.reshape(gradients.shape[0], -1)
    if penalty_norm == "l1":
        penalty = per_sample.abs().sum(dim=1)
    elif penalty_norm == "l2":
        penalty = per_sample.pow(2).sum(dim=1)
    else:
        raise ValueError(f"Unsupported penalty norm in test helper: {penalty_norm}")
    return (eps / std) * gradients.shape[0] * penalty.mean()


@pytest.mark.parametrize("grad_shape", [(8, 10), (4, 3, 8, 8)])
def test_dbp_l1_matches_existing_formula(grad_shape):
    gradients = torch.randn(*grad_shape)
    dbp = DBP(eps=4.0 / 255.0, std=0.225, penalty_norm="l1")

    loss = dbp(gradients, inputs=None)
    expected = _expected_dbp(gradients, 4.0 / 255.0, 0.225, "l1")

    assert torch.isclose(loss, expected, atol=1e-6)


@pytest.mark.parametrize("grad_shape", [(8, 10), (4, 3, 8, 8)])
def test_dbp_l2_matches_squared_l2_formula(grad_shape):
    gradients = torch.randn(*grad_shape)
    dbp = DBP(eps=4.0 / 255.0, std=0.225, penalty_norm="l2")

    loss = dbp(gradients, inputs=None)
    expected = _expected_dbp(gradients, 4.0 / 255.0, 0.225, "l2")

    assert torch.isclose(loss, expected, atol=1e-6)


def test_dbp_rejects_invalid_penalty_norm():
    with pytest.raises(ValueError, match="Unsupported gradnorm penalty norm"):
        DBP(penalty_norm="linf")


def test_current_gradnorm_objective_matches_legacy_uncapped_formula():
    ce_loss = torch.tensor(2.0)
    raw_reg = torch.tensor(0.5)

    loss, loss_reg, scale = combine_gradnorm_objective(
        ce_loss,
        raw_reg,
        objective="current",
        max_reg_to_ce_ratio=0,
    )

    assert torch.isclose(loss, torch.tensor(2.5))
    assert torch.isclose(loss_reg, torch.tensor(0.5))
    assert torch.isclose(scale, torch.tensor(1.0))


def test_weighted_gradnorm_objective_uses_ce_and_reg_weights():
    ce_loss = torch.tensor(2.0)
    raw_reg = torch.tensor(0.5)

    loss, loss_reg, scale = combine_gradnorm_objective(
        ce_loss,
        raw_reg,
        objective="weighted",
        ce_weight=0.8,
        gradnorm_weight=1.2,
        max_reg_to_ce_ratio=0,
    )

    assert torch.isclose(loss, torch.tensor(2.2))
    assert torch.isclose(loss_reg, torch.tensor(0.6))
    assert torch.isclose(scale, torch.tensor(1.0))


def test_gradnorm_cap_scales_weighted_regularizer():
    ce_loss = torch.tensor(2.0)
    raw_reg = torch.tensor(10.0)

    loss, loss_reg, scale = combine_gradnorm_objective(
        ce_loss,
        raw_reg,
        objective="weighted",
        ce_weight=0.8,
        gradnorm_weight=1.2,
        max_reg_to_ce_ratio=1.0,
    )

    assert torch.isclose(loss_reg, torch.tensor(2.0))
    assert torch.isclose(loss, torch.tensor(3.6))
    assert torch.isclose(scale, torch.tensor(1.0 / 6.0))


def test_gradnorm_objective_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported gradnorm objective"):
        combine_gradnorm_objective(torch.tensor(1.0), torch.tensor(1.0), objective="bad")
