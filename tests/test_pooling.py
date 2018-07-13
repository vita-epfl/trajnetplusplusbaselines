import torch

import trajnetbaselines

NAN = float('nan')


def test_simple_grid():
    pool = trajnetbaselines.lstm.Pooling(n=2)
    obs = torch.Tensor([
        [0.0, 0.0],
        [-0.2, -0.2],
    ])
    occupancies = pool.occupancies(obs).numpy().tolist()
    assert occupancies == [[
        1, 0,
        0, 0,
    ], [
        0, 0,
        0, 1,
    ]]


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
