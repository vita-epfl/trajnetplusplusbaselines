from collections import defaultdict

import torch

from .modules import Hidden2Normal, InputEmbedding


def one_cold(i, n):
    """Inverse one-hot encoding."""
    x = torch.ones(n, dtype=torch.uint8)
    x[i] = 0
    return x


class Pooling(torch.nn.Module):
    def __init__(self, cell_side=1.0, n=6, hidden_dim=128, type_='occupancy'):
        super(Pooling, self).__init__()
        self.cell_side = cell_side
        self.n = n
        self.hidden_dim = hidden_dim
        self.type_ = type_

        self.pooling_dim = 1
        if self.type_ == 'directional':
            self.pooling_dim = 2
        if self.type_ == 'social':
            self.pooling_dim = hidden_dim
        self.embedding = torch.nn.Linear(n * n * self.pooling_dim, hidden_dim)

    def forward(self, hidden_state, obs1, obs2):
        if self.type_ == 'occupancy':
            grid = self.occupancies(obs2)
        elif self.type_ == 'directional':
            grid = self.directional(obs1, obs2)
        elif self.type_ == 'social':
            grid = self.social(hidden_state, obs2)
        return self.embedding(grid)

    def occupancies(self, obs):
        n = obs.size(0)
        return torch.stack([
            self.occupancy(obs[i], obs[one_cold(i, n)])
            for i in range(n)
        ], dim=0)

    def directional(self, obs1, obs2):
        n = obs2.size(0)
        return torch.stack([
            self.occupancy(obs2[i], obs2[one_cold(i, n)], (obs2 - obs1)[one_cold(i, n)])
            for i in range(n)
        ], dim=0)

    def social(self, hidden_state, obs):
        n = obs.size(0)
        return torch.stack([
            self.occupancy(obs[i], obs[one_cold(i, n)], hidden_state[one_cold(i, n)])
            for i in range(n)
        ], dim=0)

    def occupancy(self, xy, other_xy, other_values=None):
        """Returns the occupancy."""
        if xy[0] != xy[0] or \
           other_xy.size(0) == 0:
            return torch.zeros(self.n * self.n * self.pooling_dim, device=xy.device)

        if other_values is None:
            other_values = torch.ones(other_xy.size(0), 1)

        mask = torch.isnan(other_xy[:, 0]) == 0
        oxy = other_xy[mask]
        other_values = other_values[mask]
        if not oxy.size(0):
            return torch.zeros(self.n * self.n * self.pooling_dim, device=xy.device)

        oij = ((oxy - xy) / self.cell_side + self.n / 2)
        range_violations = torch.sum((oij < 0) + (oij >= self.n), dim=1)
        oij = oij[range_violations == 0, :].long()
        if oij.size(0) == 0:
            return torch.zeros(self.n * self.n * self.pooling_dim, device=xy.device)
        oi = oij[:, 0] * self.n + oij[:, 1]
        occ = torch.zeros(self.n * self.n, self.pooling_dim, device=xy.device)
        for oii, v in zip(oi, other_values):
            occ[oii, :] += v

        return occ.view(-1)
