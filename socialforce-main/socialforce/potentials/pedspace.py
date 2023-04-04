import torch


class PedSpacePotential(torch.nn.Module):
    """Pedestrian-space interaction potential.

    space is a list of numpy arrays containing points of boundaries.

    u0 is in m^2 / s^2.
    r is in m
    """

    def __init__(self, space, u0=10, r=0.2):
        super().__init__()
        self.space = space or []
        self.u0 = u0
        self.r = r

    def value_r_aB(self, r_aB):
        """Compute value parametrized with r_aB."""
        return self.u0 * torch.exp(-1.0 * torch.norm(r_aB, dim=-1) / self.r)

    def r_aB(self, state):
        """r_aB"""
        if not self.space:
            return torch.zeros((state.shape[0], 0, 2))

        r_a = state[:, 0:2].unsqueeze(1)
        closest_i = [
            torch.argmin(torch.norm(r_a - B.unsqueeze(0), dim=-1), dim=1)
            for B in self.space
        ]
        closest_points = torch.transpose(
            torch.stack([B[i] for B, i in zip(self.space, closest_i)]),
            0, 1)  # index order: pedestrian, boundary, coordinates
        return r_a - closest_points

    def forward(self, state):
        return self.value_r_aB(self.r_aB(state))

    def grad_r_aB(self, state, delta=1e-3):
        """Compute gradient wrt r_aB using finite difference differentiation."""
        r_aB = self.r_aB(state)

        dx = torch.tensor([[[delta, 0.0]]])
        dy = torch.tensor([[[0.0, delta]]])

        v = self.value_r_aB(r_aB)
        dvdx = (self.value_r_aB(r_aB + dx) - v) / delta
        dvdy = (self.value_r_aB(r_aB + dy) - v) / delta

        return torch.stack((dvdx, dvdy), dim=-1)
