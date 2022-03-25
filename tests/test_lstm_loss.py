import math
import numpy
import pytest
import torch
import trajnetbaselines.lstm
import trajnettools
from trajnettools.data import TrackRow

NAN = float('nan')


def test_simple():
    gaussian_parameters = torch.Tensor([[
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
    ]])
    coordinates = torch.Tensor([[
        [0.0, 0.0],
        [0.0, 0.0],
    ]])
    batch_split = torch.LongTensor([0, 1, 2])
    criterion = trajnetbaselines.lstm.PredictionLoss(keep_batch_dim=True, background_rate=0.0)
    loss = criterion(gaussian_parameters, coordinates, batch_split).numpy().tolist()
    gauss_denom = 1.0 / math.sqrt(2.0*math.pi)**2
    assert loss == pytest.approx([-math.log(0.01 + 0.99 * gauss_denom), -math.log(0.01 + 0.99 * gauss_denom)], 1e-4)


def test_narrower_progression():
    gaussian_parameters = torch.Tensor([[
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.1, 0.1, 0.0],
    ]])
    coordinates = torch.Tensor([[
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]])
    batch_split = torch.LongTensor([0, 1, 2, 3])
    criterion = trajnetbaselines.lstm.PredictionLoss(keep_batch_dim=True, background_rate=0.0)
    loss = criterion(gaussian_parameters, coordinates, batch_split).numpy().tolist()
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
    xy, _ = trajnetbaselines.lstm.lstm.drop_distant(xy)
    assert xy == pytest.approx(numpy.array([
        [[1.0, 1.0], [NAN, NAN], [3.0, 3.0]],
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        [[1.0, 1.0], [2.0, 2.0], [NAN, NAN]],
    ]), nan_ok=True)


def test_col_loss():
    predictions = numpy.array([
                               [[0, 0],  [1, 0],  [2, 0],  [3, 0]],
                               [[0, 4],  [1, 3],  [2, 2],  [3, 1]],
                               [[0, -3], [1, -2], [2, -1], [3, -1]],
                               [[0, -8], [1, -8], [2, -8], [3, -8]],
                              ])
    predictions = predictions.transpose(1, 0, 2)
    batch_split = numpy.array([0, 4], numpy.int32)

    predictions = torch.Tensor(predictions)
    batch_split = torch.LongTensor(batch_split)

    col_loss = trajnetbaselines.lstm.loss.L2Loss.col_loss(predictions, batch_split, col_wt=2.0, col_distance=2.0)
    assert col_loss == 3.0

    col_loss = trajnetbaselines.lstm.loss.L2Loss.col_loss(predictions, batch_split, col_wt=4.0, col_distance=2.0)
    assert col_loss == 6.0

    col_loss = trajnetbaselines.lstm.loss.L2Loss.col_loss(predictions, batch_split, col_wt=2.0, col_distance=4.0)
    assert col_loss == 7.5


test_simple()
test_narrower_progression()
test_drop_distant()
test_col_loss()
