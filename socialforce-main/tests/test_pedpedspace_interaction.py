import torch
import socialforce


# pylint: disable=invalid-name
def test_r_aB():
    state = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ])
    space = [
        torch.tensor([[0.0, 100.0], [0.0, 0.5]])
    ]
    r_aB = socialforce.potentials.PedSpacePotential(space).r_aB(state)
    assert r_aB.tolist() == [
        [[0.0, -0.5]],
        [[1.0, -0.5]],
    ]
