# coding=utf-8

"""Synthetic pedestrian behavior according to the Social Force model.

See Helbing and Molnár 1998.
"""

import torch

from .potentials import PedPedPotential
from .field_of_view import FieldOfView
from . import stateutils


class Simulator(torch.nn.Module):
    """Simulate social force model.

    Main interface is the state. Every pedestrian is an entry in the state and
    represented by a vector (x, y, v_x, v_y, [a_x, a_y], [d_x, d_y], [tau]),
    which are the coordinates for position, velocity and destination.
    destination and tau are optional in this vector.

    ped_space is an instance of PedSpacePotential.
    ped_ped is an instance of PedPedPotential.

    delta_t in seconds.
    tau in seconds: either float or numpy array of shape[n_ped].
    field_of_view: use -1 to remove
    """
    max_speed_multiplier = 1.3

    def __init__(self, *,
                 ped_space=None, ped_ped=None, field_of_view=None,
                 delta_t=0.4, tau=0.5,
                 oversampling=10, dtype=None, integrator=None):
        super().__init__()

        self.tau = tau
        self.dtype = dtype if dtype is not None else torch.float32

        self.delta_t = delta_t / oversampling
        self.oversampling = oversampling

        if integrator is None:
            integrator = LeapfrogIntegrator(self.delta_t)
        self.integrator = integrator

        # potentials
        self.V = ped_ped or PedPedPotential()
        self.U = ped_space

        # field of view
        self.w = field_of_view or FieldOfView()

    def normalize_state(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=self.dtype)
        assert len(state.shape) == 2

        if state.shape[1] == 4:
            # accelerations and destinations not given
            no_destinations = torch.full((state.shape[0], 2), float('nan'),
                                         dtype=state.dtype)
            state = torch.cat((state, no_destinations), dim=-1)
        if state.shape[1] == 5:
            # accelerations and destinations not given but tau is given
            no_dest_acc = torch.full((state.shape[0], 4), float('nan'),
                                     dtype=state.dtype)
            state = torch.cat((state[:, 0:4], no_dest_acc, state[:, 4:]), dim=-1)
        if state.shape[1] in (6, 7):
            # accelerations not given
            no_accelerations = torch.zeros((state.shape[0], 2), dtype=state.dtype)
            state = torch.cat((state[:, :4], no_accelerations, state[:, 4:]), dim=-1)
        if state.shape[1] == 8:
            # tau not given
            if hasattr(self.tau, 'shape'):
                tau = self.tau
            else:
                tau = self.tau * torch.ones(state.shape[0], dtype=state.dtype)
            state = torch.cat((state, tau.unsqueeze(-1)), dim=-1)
        if state.shape[1] == 9:
            # preferred speed not given
            preferred_speeds = stateutils.speeds(state)
            state = torch.cat((state, preferred_speeds.unsqueeze(-1)), dim=-1)

        assert state.shape[1] == 10, state.shape[1]
        return state

    def f_ab(self, state):
        """Compute f_ab."""
        return -1.0 * self.V.grad_r_ab(state)

    def f_aB(self, state):
        """Compute f_aB."""
        if self.U is None:
            return None
        return -1.0 * self.U.grad_r_aB(state)

    def cap_velocity(self, state):
        """Scale down a desired velocity to its capped speed."""
        desired_velocity = state[:, 2:4]
        desired_speeds = torch.linalg.norm(desired_velocity, ord=2, dim=-1, keepdims=True)
        max_speeds = state[:, 9:10] * self.max_speed_multiplier
        factor = torch.clamp(max_speeds / desired_speeds, max=1.0)

        state = torch.clone(state)
        state[:, 2:4] = desired_velocity * factor
        return state

    def forward(self, *args):
        """Do oversampling steps."""
        state = args[0]
        state = self.normalize_state(state.clone().detach())

        for _ in range(self.oversampling):
            state = self._step(state)

        return state

    def _step(self, state):
        """Do one step in the simulation and update the state in place."""

        # accelerate to preferred velocity
        e = stateutils.desired_directions(state).detach()
        vel = state[:, 2:4].detach()
        tau = state[:, 8:9].detach()
        preferred_speeds = state[:, 9:10].detach()
        F0 = 1.0 / tau * (preferred_speeds * e - vel)

        # repulsive terms between pedestrians
        f_ab = self.f_ab(state.detach())

        # field of view modulation
        if f_ab is not None and self.w is not None and self.w != -1:
            # w = self.w(e, -f_ab).unsqueeze(-1).detach()
            r_ab = PedPedPotential.r_ab(state)
            w = self.w(e, -r_ab).unsqueeze(-1).detach()
            F_ab = w * f_ab
        else:
            F_ab = f_ab

        # repulsive terms between pedestrians and boundaries
        F_aB = self.f_aB(state.detach())

        # social force
        F = F0
        if F_ab is not None:
            F += torch.sum(F_ab, dim=1)
        if F_aB is not None:
            F += torch.sum(F_aB, dim=1)

        state = self.integrator(state, F)
        state = self.cap_velocity(state)
        return state

    def run(self, state, n_steps):
        state = self.normalize_state(state.clone().detach())
        states = [state]
        for _ in range(n_steps):
            last_state = states[-1]
            states.append(self(last_state).clone())

        return torch.stack(states)


class EulerIntegrator:
    def __init__(self, delta_t):
        self.delta_t = delta_t

    def __call__(self, state, acceleration):
        # before applying updates to state, make a copy of it
        previous_state = state
        new_state = state.clone().detach()  # gradients will be connected below

        # update state
        new_state[:, 0:2] = previous_state[:, 0:2] + self.delta_t * previous_state[:, 2:4]
        new_state[:, 2:4] = previous_state[:, 2:4] + self.delta_t * previous_state[:, 4:6]
        new_state[:, 4:6] = acceleration

        return new_state


class LeapfrogIntegrator:
    def __init__(self, delta_t):
        self.delta_t = delta_t

    def __call__(self, state, acceleration):
        # before applying updates to state, make a copy of it
        previous_state = state
        new_state = state.clone().detach()  # gradients will be connected below

        # update position
        new_state[:, 0:2] = (previous_state[:, 0:2]
                             + previous_state[:, 2:4] * self.delta_t
                             + 0.5 * acceleration * self.delta_t**2)

        # update velocity
        v = previous_state[:, 2:4] + 0.5 * (previous_state[:, 4:6] + acceleration) * self.delta_t
        new_state[:, 2:4] = v

        # update acceleration
        new_state[:, 4:6] = acceleration

        return new_state


class PeriodicBoundary:
    def __init__(self, integrator, *, x_boundary=None, y_boundary=None):
        self.integrator = integrator
        self.x_boundary = x_boundary
        self.y_boundary = y_boundary

    def __call__(self, state, acceleration):
        old_state = self.integrator(state, acceleration)

        new_state = old_state.clone()
        if self.x_boundary is not None:
            width = self.x_boundary[1] - self.x_boundary[0]
            new_state[old_state[:, 0] < self.x_boundary[0], 0] += width
            new_state[old_state[:, 0] > self.x_boundary[1], 0] -= width
        if self.y_boundary is not None:
            height = self.y_boundary[1] - self.y_boundary[0]
            new_state[old_state[:, 1] < self.y_boundary[0], 1] += height
            new_state[old_state[:, 1] > self.y_boundary[1], 1] -= height

        return new_state
