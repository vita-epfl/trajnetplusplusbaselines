import math
import numpy
import pytest
import torch
import trajnetbaselines.lstm
import trajnettools
from trajnettools.data import TrackRow

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
    criterion = trajnetbaselines.lstm.PredictionLoss(reduce=False, background_rate=0.0)
    loss = criterion(gaussian_parameters, coordinates).numpy().tolist()
    gauss_denom = 1.0 / math.sqrt(2.0*math.pi)**2
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
    criterion = trajnetbaselines.lstm.PredictionLoss(reduce=False, background_rate=0.0)
    loss = criterion(gaussian_parameters, coordinates).numpy().tolist()
    assert loss[0] > loss[1]
    assert loss[1] > loss[2]


def test_drop_distant():
    paths = [
        [TrackRow(0, 1, 1.0, 1.0), TrackRow(10, 1, 1.0, 1.0), TrackRow(20, 1, 1.0, 1.0)],
        [TrackRow(10, 2, 2.0, 2.0), TrackRow(20, 2, 2.0, 2.0)],
        [TrackRow(0, 3, 3.0, 3.0), TrackRow(10, 3, 3.0, 3.0)],
        [TrackRow(0, 4, 40.0, 40.0), TrackRow(10, 4, 40.0, 40.0)],
    ]

    xy = trajnettools.Reader.paths_to_xy(paths)
    xy = trajnetbaselines.lstm.lstm.drop_distant(xy)
    assert xy == pytest.approx(numpy.array([
        [[1.0, 1.0], [NAN, NAN], [3.0, 3.0]],
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        [[1.0, 1.0], [2.0, 2.0], [NAN, NAN]],
    ]), nan_ok=True)
