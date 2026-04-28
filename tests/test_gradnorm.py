import pytest
import torch

from ares.utils.gradnorm import DBP


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
