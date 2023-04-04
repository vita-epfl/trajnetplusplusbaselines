"""Interaction potentials."""

import torch

from .. import stateutils


class PedPedPotential(torch.nn.Module):
    """Ped-ped interaction potential based on distance b.

    v0 is in m^2 / s^2.
    sigma is in m.
    """
    delta_t_step = 0.4

    def __init__(self, v0=2.1, sigma=0.3):
        super().__init__()
        self.v0 = v0
        self.sigma = sigma

    def b(self, r_ab, speeds, desired_directions):
        """Calculate b."""
        speeds_b = speeds.unsqueeze(0)
        speeds_b_abc = speeds_b.unsqueeze(2)  # abc = alpha, beta, coordinates
        e_b = desired_directions.unsqueeze(0)

        in_sqrt = (
            self.norm_r_ab(r_ab)
            + self.norm_r_ab(r_ab - self.delta_t_step * speeds_b_abc * e_b)
        )**2 - (self.delta_t_step * speeds_b)**2

        # torch.diagonal(in_sqrt)[:] = 0.0  # protect forward pass
        in_sqrt = torch.clamp(in_sqrt, min=1e-8)
        out = 0.5 * torch.sqrt(in_sqrt)
        # torch.diagonal(out)[:] = 0.0  # protect backward pass

        return out

    def value_b(self, b):
        """Value of potential parametrized with b."""
        return self.v0 * torch.exp(-b / self.sigma)

    def value_r_ab(self, r_ab, speeds, desired_directions):
        """Value of potential explicitely parametrized with r_ab."""
        b = self.b(r_ab, speeds, desired_directions)
        return self.value_b(b)

    @staticmethod
    def r_ab(state):
        """Construct r_ab using broadcasting."""
        r = state[:, 0:2]
        r_a0 = r.unsqueeze(1)
        r_0b = r.unsqueeze(0).detach()  # detach others
        r_ab = r_a0 - r_0b
        torch.diagonal(r_ab)[:] = 0.0  # detach diagonal gradients
        return r_ab

    def forward(self, state):
        speeds = stateutils.speeds(state).detach()
        desired_directions = stateutils.desired_directions(state).detach()
        return self.value_r_ab(self.r_ab(state), speeds, desired_directions)

    def grad_r_ab_finite_difference(self, state, delta=1e-3):
        """Compute gradient wrt r_ab using finite difference differentiation."""
        r_ab = self.r_ab(state[:, 0:2])
        speeds = stateutils.speeds(state)
        desired_directions = stateutils.desired_directions(state)

        dx = torch.tensor([[[delta, 0.0]]])
        dy = torch.tensor([[[0.0, delta]]])

        v = self.value_r_ab(r_ab, speeds, desired_directions)
        dvdx = (self.value_r_ab(r_ab + dx, speeds, desired_directions) - v) / delta
        dvdy = (self.value_r_ab(r_ab + dy, speeds, desired_directions) - v) / delta

        # remove gradients from self-intereactions
        torch.diagonal(dvdx)[:] = 0.0
        torch.diagonal(dvdy)[:] = 0.0

        return torch.stack((dvdx, dvdy), dim=-1)

    def grad_r_ab(self, state):
        """Compute gradient wrt r_ab using autograd."""
        speeds = stateutils.speeds(state).detach()
        desired_directions = stateutils.desired_directions(state).detach()
        r_ab = self.r_ab(state)
        return self.grad_r_ab_(r_ab, speeds, desired_directions)

    def grad_r_ab_(self, r_ab, speeds, desired_directions):
        """Compute gradient wrt r_ab using autograd."""
        def compute(r_ab):
            return self.value_r_ab(r_ab, speeds, desired_directions)

        with torch.enable_grad():
            vector = torch.ones(r_ab.shape[0:2], requires_grad=False)
            _, r_ab_grad = torch.autograd.functional.vjp(
                compute, r_ab, vector,
                create_graph=True, strict=True)

        return r_ab_grad

    @staticmethod
    def norm_r_ab(r_ab):
        """Norm of r_ab.

        Special treatment of diagonal terms for backpropagation.

        Without this treatment, backpropagating through a norm of a
        zero vector gives nan gradients.
        """
        out = torch.linalg.norm(r_ab, ord=2, dim=2, keepdim=False)
        torch.diagonal(out)[:] = 0.0
        return out


class PedPedPotentialWall(PedPedPotential):
    """Ped-ped interaction potential based on distance b.

    v0 is in m^2 / s^2.
    sigma is in m.
    """
    delta_t_step = 0.4

    def __init__(self, sigma=0.3, w=0.1):
        super().__init__()
        self.sigma = sigma
        self.w = w

    def value_b(self, b):
        """Value of potential parametrized with b."""
        return torch.exp(-(b - self.sigma) / self.w)


class PedPedPotentialMLP(PedPedPotential):
    """Ped-ped interaction potential."""

    def __init__(self, *, hidden_units=5, small_init=False, dropout_p=0.0):
        super().__init__()

        lin1 = torch.nn.Linear(1, hidden_units)
        lin2 = torch.nn.Linear(hidden_units, 1)

        # initialize
        if small_init:
            torch.nn.init.normal_(lin1.weight, std=0.03)
            torch.nn.init.normal_(lin1.bias, std=0.03)
            torch.nn.init.normal_(lin2.weight, std=0.03)
            torch.nn.init.normal_(lin2.bias, std=0.03)

        if dropout_p == 0.0:
            self.mlp = torch.nn.Sequential(
                lin1, torch.nn.Softplus(),
                lin2, torch.nn.Softplus(),
            )
        else:
            self.mlp = torch.nn.Sequential(
                lin1, torch.nn.Softplus(), torch.nn.Dropout(dropout_p),
                lin2, torch.nn.Softplus(),
            )

    def value_b(self, b):
        """Calculate value given b."""
        b = torch.clamp(b, max=100.0)
        return self.mlp(b.view(-1, 1)).view(b.shape)
