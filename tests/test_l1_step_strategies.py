import torch

from data_analysis.l1_step_methods import (
    attack_success_from_preds,
    fw_onehot_update,
    gather_best_per_sample,
    l1_project,
    normalized_l1_direction,
    normalized_l2_direction,
    per_sample_l1,
    random_l1_start,
    raw_direction,
    select_best_run_indices,
    topk_sign_direction,
)


def test_projection_step_variants_respect_l1_constraint():
    torch.manual_seed(0)
    b, c, h, w = 4, 3, 16, 16
    eps = 5.0
    alpha = 1.0

    orig = torch.rand(b, c, h, w)
    grad = torch.randn_like(orig)

    cand_l2 = orig + alpha * normalized_l2_direction(grad)
    proj_l2 = l1_project(orig, cand_l2, eps)

    cand_l1 = orig + alpha * normalized_l1_direction(grad)
    proj_l1 = l1_project(orig, cand_l1, eps)

    cand_raw = orig + alpha * raw_direction(grad)
    proj_raw = l1_project(orig, cand_raw, eps)

    cand_topk = orig + alpha * topk_sign_direction(grad, rho=0.05)
    proj_topk = l1_project(orig, cand_topk, eps)

    for adv in (proj_l2, proj_l1, proj_raw, proj_topk):
        l1 = per_sample_l1(adv - orig)
        assert torch.all(l1 <= eps + 1e-5), f"Found L1 norm beyond eps: {l1}"


def test_fw_onehot_stays_within_l1_ball_without_projection():
    torch.manual_seed(1)
    b, c, h, w = 2, 3, 8, 8
    eps = 3.0
    delta = torch.zeros(b, c, h, w)

    for t in range(1, 11):
        grad = torch.randn_like(delta)
        delta = fw_onehot_update(delta, grad, eps=eps, t=t)

    l1 = per_sample_l1(delta)
    assert torch.all(l1 <= eps + 1e-5), f"FW update escaped L1 ball: {l1}"


def test_random_start_reproducible_with_seed():
    b, c, h, w = 2, 3, 8, 8
    eps = 4.0
    orig = torch.rand(b, c, h, w)

    torch.manual_seed(123)
    a = random_l1_start(orig, eps)
    torch.manual_seed(123)
    b_out = random_l1_start(orig, eps)

    assert torch.allclose(a, b_out), "Random starts with same seed should match"


def test_best_over_runs_selection():
    losses = torch.tensor(
        [
            [0.2, 0.9, 0.3],
            [0.5, 0.1, 0.7],
        ]
    )
    best_idx = select_best_run_indices(losses)
    assert best_idx.tolist() == [1, 0, 1]

    vals = torch.tensor(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
        ]
    )
    best_vals = gather_best_per_sample(vals, best_idx)
    assert best_vals.tolist() == [20.0, 11.0, 22.0]


def test_attack_success_matches_prediction_flip_definition():
    clean = torch.tensor([1, 2, 3, 4])
    adv = torch.tensor([1, 5, 3, 0])
    success = attack_success_from_preds(clean, adv)
    assert success.tolist() == [0.0, 1.0, 0.0, 1.0]
