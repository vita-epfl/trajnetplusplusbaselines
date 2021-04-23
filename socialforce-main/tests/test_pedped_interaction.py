import numpy as np
import pytest
import socialforce


def test_rab():
    state = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ])
    V = socialforce.PedPedPotential(0.4)
    assert V.r_ab(state).tolist() == [[
        [0.0, 0.0],
        [-1.0, 0.0],
    ], [
        [1.0, 0.0],
        [0.0, 0.0],
    ]]


def test_fab():
    initial_state = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ])
    s = socialforce.Simulator(initial_state)
    force_at_unit_distance = 0.25  # TODO confirm
    assert s.f_ab() == pytest.approx(np.array([[
        [0.0, 0.0],
        [-force_at_unit_distance, 0.0],
    ], [
        [force_at_unit_distance, 0.0],
        [0.0, 0.0],
    ]]), abs=0.05)


def test_b_zero_vel():
    r_ab = np.array([[
        [0.0, 0.0],
        [-1.0, 0.0],
    ], [
        [1.0, 0.0],
        [0.0, 0.0],
    ]])
    speeds = np.array([0.0, 0.0])
    desired_directions = ([[1.0, 0.0], [-1.0, 0.0]])
    V = socialforce.PedPedPotential(0.4)
    assert V.b(r_ab, speeds, desired_directions).tolist() == [
        [0.0, 1.0],
        [1.0, 0.0],
    ]
