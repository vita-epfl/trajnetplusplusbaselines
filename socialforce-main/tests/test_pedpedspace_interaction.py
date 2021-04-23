import numpy as np
import socialforce


def test_r_aB():
    state = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ])
    space = [
        np.array([[0.0, 100.0], [0.0, 0.5]])
    ]
    r_aB = socialforce.PedSpacePotential(space).r_aB(state)
    assert r_aB.tolist() == [
        [[0.0, -0.5]],
        [[1.0, -0.5]],
    ]
