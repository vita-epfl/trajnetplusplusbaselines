"""Test field-of-view computations."""

import numpy as np
from socialforce.fieldofview import FieldOfView


def test_w():
    assert FieldOfView()(
        np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
        ]),
        np.array([[
            [0.0, 0.0],
            [1.0, 1.0],
        ], [
            [-1.0, 1.0],
            [0.0, 0.0],
        ]])
    ).tolist() == [
        [0, 1],
        [1, 0],
    ]
