import math
import pytest
import torch

import numpy as np

import socialforce


def test_rab():
    r = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    V = socialforce.potentials.PedPedPotential(0.4)
    assert V.r_ab(r).tolist() == [[
        [0.0, 0.0],
        [-1.0, 0.0],
    ], [
        [1.0, 0.0],
        [0.0, 0.0],
    ]]


def test_f_ab():
    s = socialforce.Simulator()
    initial_state = s.normalize_state([
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ])
    force_at_unit_distance = 0.25  # confirmed below
    assert s.f_ab(initial_state).detach().numpy() == pytest.approx(np.array([[
        [0.0, 0.0],
        [-force_at_unit_distance, 0.0],
    ], [
        [force_at_unit_distance, 0.0],
        [0.0, 0.0],
    ]]), abs=0.05)


def test_b_zero_vel():
    r_ab = torch.tensor([[
        [0.0, 0.0],
        [-1.0, 0.0],
    ], [
        [1.0, 0.0],
        [0.0, 0.0],
    ]])
    speeds = torch.tensor([0.0, 0.0])
    desired_directions = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    V = socialforce.potentials.PedPedPotential()

    # due to clipping for stability, a zero is clipped to 1e-8 which
    # is returned as 0.5 * sqrt(1e-8):
    assert V.b(r_ab, speeds, desired_directions).numpy() == pytest.approx(np.array([
        [0.00005, 1.0],
        [1.0, 0.00005],
    ]), abs=0.0001)


def test_torch_potential_gradient():
    state = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
    ])
    v0 = torch.Tensor([2.1])
    sigma = torch.Tensor([0.3])

    r = state[:, 0:2].detach().requires_grad_()
    r_a = r.unsqueeze(1)
    r_b = r.unsqueeze(0).detach()  # !!! gradient of b will accumulate into a without detach()
    r_ab = r_a - r_b
    r_ab_norm = torch.norm(r_ab, dim=-1)
    print(r_ab_norm)

    pedped_potential = v0 * torch.exp(-r_ab_norm / sigma)
    torch.diagonal(pedped_potential)[:] = 0.0
    # pedped_potential = torch.sum(pedped_potential, dim=1)
    print('value', pedped_potential)
    gradients = torch.ones_like(pedped_potential)
    pedped_potential.backward(gradients)
    print(r.grad)

    analytic_abs_grad_value = 2.1 * math.exp(-1.0 / 0.3) * 1.0 / 0.3
    print(analytic_abs_grad_value)
    assert r.grad[0][0] == analytic_abs_grad_value
    assert r.grad[1][0] == -analytic_abs_grad_value
