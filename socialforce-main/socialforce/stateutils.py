"""Utility functions to process state."""

import torch


def desired_directions(state):
    """Given the current state and destination, compute desired direction."""
    destination_vectors = state[:, 6:8] - state[:, 0:2]

    # support for prediction without given destination:
    # "desired_direction" is in the direction of the current velocity
    invalid_destination = torch.isnan(destination_vectors[:, 0])
    destination_vectors[invalid_destination] = state[invalid_destination, 2:4]

    norm_factors = torch.linalg.norm(destination_vectors, ord=2, dim=-1)
    norm_factors[norm_factors == 0.0] = 1.0
    return destination_vectors / norm_factors.unsqueeze(-1)


def speeds(state):
    """Return the speeds corresponding to a given state."""
    return torch.linalg.norm(state[:, 2:4], ord=2, dim=-1)
