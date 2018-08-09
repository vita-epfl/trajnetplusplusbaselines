"""Synthetic pedestrian behavior according to the Social Force model.

See Helbing and Molnár 1998.
"""

import numpy as np

MEAN_VELOCITY = 1.34  # m/s
SIGMA_VEL = 0.26  # std dev in m/s
MAX_SPEED_MULTIPLIER = 1.3  # with respect to initial speed

V0 = 2.1  # m^2 / s^2
SIGMA_V = 0.3  # m

U0 = 10  # m^2 / s^2
R_U = 0.2  # m

TWOPHI = 200  # degrees field of view
COSPHI = np.cos(TWOPHI / 2.0 / 180.0 * np.pi)  # for field of view computation
C = 0.5

DELTA_T = 1.0 / 2.5  # seconds

TAU = 0.5  # seconds



class SocialForceSim(object):
    """Simulate social force model.

    Main interface is the state. Every person is represented by a vector in
    state space (x, y, v_x, v_y).
    """
    def __init__(self, initial_state, destinations):
        self.state = initial_state
        self.destinations = destinations

        self.initial_speeds = self.speeds()
        self.max_speeds = MAX_SPEED_MULTIPLIER * self.initial_speeds

    def speeds(self):
        """Calculate the speeds of all pedestrians."""
        velocities = self.state[:, 2:4]
        return np.linalg.norm(velocities, axis=1)

    def desired_direction(self):
        """Given the current state and destination, compute desired direction."""
        destination_vectors = self.destinations - self.state[:, 0:2]
        norm_factors = np.linalg.norm(destination_vectors, axis=-1)
        return destination_vectors / np.expand_dims(norm_factors, -1)

    def rab(self):
        """r_ab"""
        r = self.state[:, 0:2]
        r_a = np.expand_dims(r, 1)
        r_b = np.expand_dims(r, 0)
        return r_a - r_b

    def b(self, rab):
        e_b = np.expand_dims(self.desired_direction(), axis=0)
        speeds_b = np.expand_dims(self.speeds(), axis=0)
        speeds_b_abc = np.expand_dims(speeds_b, axis=2)  # abc = alpha, beta, coordinates

        in_sqrt = (
            np.linalg.norm(rab, axis=-1) +
            np.linalg.norm(rab - DELTA_T * speeds_b_abc * e_b, axis=-1)
        )**2 - (DELTA_T * speeds_b)**2
        np.fill_diagonal(in_sqrt, 0.0)

        return 0.5 * np.sqrt(in_sqrt)

    def V(self, rab):
        return V0 * np.exp(-self.b(rab) / SIGMA_V)

    def fab(self, delta=1e-3):
        """Compute f_ab using finite difference differentiation."""
        rab = self.rab()
        dx = np.array([[[delta, 0.0]]])
        dy = np.array([[[0.0, delta]]])

        v = self.V(rab)
        dvdx = (self.V(rab + dx) - v) / delta
        dvdy = (self.V(rab + dy) - v) / delta

        # remove gradients from self-intereactions
        np.fill_diagonal(dvdx, 0.0)
        np.fill_diagonal(dvdy, 0.0)

        fab = -1.0 * np.stack((dvdx, dvdy), axis=-1)
        return fab

    @staticmethod
    def w(e, f):
        """Weighting factor for field of view.

        Assumes e is normalized."""
        in_sight = np.einsum('aj,abj->ab', e, f) > np.linalg.norm(f, axis=-1) * COSPHI
        out = C * np.ones_like(in_sight)
        out[in_sight] = 1.0
        np.fill_diagonal(out, 0.0)
        return out

    def capped_velocity(self, desired_velocity):
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, self.max_speeds / desired_speeds)
        return desired_velocity * np.expand_dims(factor, -1)

    def step(self):
        # accelerate to desired velocity
        e = self.desired_direction()
        vel = self.state[:, 2:4]
        F0 = 1.0 / TAU * (np.expand_dims(self.initial_speeds, -1) * e - vel)

        # repulsive terms between pedestrians
        fab = self.fab()
        w = np.expand_dims(self.w(e, -fab), -1)
        Fab = w * fab

        # social force
        F = F0 + np.sum(Fab, axis=1)
        # desired velocity
        w = self.state[:, 2:4] + DELTA_T * F
        # velocity
        v = self.capped_velocity(w)

        # update state
        self.state[:, 0:2] += v * DELTA_T
        self.state[:, 2:4] = v

        return self
