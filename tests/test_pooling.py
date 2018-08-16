import numpy as np
import pytest
import torch
import trajnetbaselines

NAN = float('nan')


def test_simple_grid():
    pool = trajnetbaselines.lstm.Pooling(n=2, pool_size=4, blur_size=3)
    obs = torch.Tensor([
        [0.0, 0.0],
        [-1.0, -1.0],
    ])
    occupancies = pool.occupancies(obs).numpy().tolist()
    assert occupancies == [[
        1, 0,
        0, 0,
    ], [
        0, 0,
        0, 1,
    ]]


def test_simple_grid_directional():
    pool = trajnetbaselines.lstm.Pooling(n=2, pool_size=4, type_='directional')
    obs1 = torch.Tensor([
        [0.0, 0.0],
        [-1.0, -1.0],
    ])
    obs2 = torch.Tensor([
        [0.1, 0.1],
        [-1.1, -1.1],
    ])
    occupancies = pool.directional(obs1, obs2).numpy().tolist()
    assert occupancies == pytest.approx(np.array([[
        -0.1, 0, 0, 0,
        -0.1, 0, 0, 0,
    ], [
        0, 0, 0, 0.1,
        0, 0, 0, 0.1,
    ]]), abs=0.01)


def test_simple_grid_midpoint():
    """Testing a midpoint between grid cells.

    Using a large pool size as a every data point has to go into a grid
    cell first. Therefore, data can never be exactly between two cells.
    """
    pool = trajnetbaselines.lstm.Pooling(n=2, pool_size=100, blur_size=99)
    obs = torch.Tensor([
        [0.0, 0.0],
        [-1.0, 0.0],
    ])
    occupancies = pool.occupancies(obs).numpy()
    assert occupancies == pytest.approx(np.array([[
        0.5, 0.5,
        0.0, 0,
    ], [
        0, 0.0,
        0.5, 0.5,
    ]]), abs=0.01)


def test_nan():
    pool = trajnetbaselines.lstm.Pooling(n=2)
    obs = torch.Tensor([
        [0.0, 0.0],
        [NAN, NAN],
    ])
    occupancies = pool.occupancies(obs).numpy().tolist()
    assert occupancies == [[
        0, 0,
        0, 0,
    ], [
        0, 0,
        0, 0,
    ]]


def test_embedding_shape():
    pool = trajnetbaselines.lstm.Pooling(n=2, hidden_dim=128)
    obs = torch.Tensor([
        [0.0, 0.0],
        [-0.2, -0.2],
    ])
    embedding = pool(None, None, obs)
    assert embedding.size(0) == 2
    assert embedding.size(1) == 128


def test_hiddenstatemlp_rel_pos():
    positions = torch.Tensor([
        [0.0, 0.0],
        [1.0, 1.0],
    ])
    rel = trajnetbaselines.lstm.pooling.HiddenStateMLPPooling.rel_obs(positions)
    assert rel.numpy().tolist() == [[
        [0.0, 0.0],
        [1.0, 1.0],
    ], [
        [-1.0, -1.0],
        [0.0, 0.0],
    ]]


def test_hiddenstatemlp():
    positions = torch.Tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])
    hidden = torch.zeros(3, 128)
    pool = trajnetbaselines.lstm.pooling.HiddenStateMLPPooling()
    result = pool(hidden, None, positions)
