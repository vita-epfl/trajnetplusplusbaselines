import math
import numpy
import pytest
import torch
import trajnetbaselines.lstm
from trajnettools.data import Row

NAN = float('nan')


def test_simple():
    gaussian_parameters = torch.Tensor([
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
    ])
    coordinates = torch.Tensor([
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    criterion = trajnetbaselines.lstm.PredictionLoss(reduce=False)
    loss = criterion(gaussian_parameters, coordinates).numpy().tolist()
    gauss_denom = 1/math.sqrt(2*math.pi)**2
    assert loss == pytest.approx([-math.log(gauss_denom), -math.log(gauss_denom)], 1e-4)


def test_narrower_progression():
    gaussian_parameters = torch.Tensor([
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.1, 0.1, 0.0],
    ])
    coordinates = torch.Tensor([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    criterion = trajnetbaselines.lstm.PredictionLoss(reduce=False)
    loss = criterion(gaussian_parameters, coordinates).numpy().tolist()
    assert loss[0] > loss[1]
    assert loss[1] > loss[2]


def test_scene_to_xy():
    scene = [
        [Row(0, 1, 1.0, 1.0), Row(10, 1, 1.0, 1.0), Row(20, 1, 1.0, 1.0)],
        [Row(10, 2, 2.0, 2.0), Row(20, 2, 2.0, 2.0)],
        [Row(0, 3, 3.0, 3.0), Row(10, 3, 3.0, 3.0)],
    ]

    xy = trajnetbaselines.lstm.lstm.scene_to_xy(scene).numpy()
    assert xy == pytest.approx(numpy.array([
        [[1.0, 1.0], [NAN, NAN], [3.0, 3.0]],
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        [[1.0, 1.0], [2.0, 2.0], [NAN, NAN]],
    ]), nan_ok=True)
