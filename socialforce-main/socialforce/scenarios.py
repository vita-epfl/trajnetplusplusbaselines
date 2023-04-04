import math

import numpy as np
import torch

from . import potentials
from .simulator import Simulator


class Circle:
    def __init__(self, ped_ped=None, **kwargs):
        self.ped_ped = ped_ped or potentials.PedPedPotential(2.1)
        self.simulator = Simulator(ped_ped=self.ped_ped, **kwargs)

    def generate(self, n, *, steps_per_scenario=21):
        # ped0 always left to right
        speed0 = 0.7 + 0.4 * torch.rand(1).item()
        ped0 = np.array([-5.0, 0.0, speed0, 0.0, 6.0, 0.0])

        generator_initial_states = []
        # Minimum 15degree angle (to avoid pedestrians starting on top of each other).
        # 15 degrees is about 0.25.
        for theta in 0.25 + torch.rand(n) * (2.0 * math.pi - 0.5):
            # ped1 at a random angle with +/-20% speed variation
            c, s = np.cos(theta), np.sin(theta)
            r = np.array([[c, -s], [s, c]])
            ped1 = np.concatenate((
                np.matmul(r, ped0[0:2]),
                np.matmul(r, ped0[2:4] * (0.7 + 0.4 * torch.rand(1).item())),
                np.matmul(r, ped0[4:6]),
            ))
            generator_initial_states.append(
                self.simulator.normalize_state(np.stack((ped0, ped1)))
            )

        with torch.no_grad():
            return [
                self.simulator.run(initial_state, steps_per_scenario)
                for initial_state in generator_initial_states
            ]


class ParallelOvertake:
    def __init__(self, ped_ped=None, *, b_center=0.0, **kwargs):
        self.ped_ped = ped_ped or potentials.PedPedPotential(2.1)
        self.b_center = b_center
        self.simulator = Simulator(ped_ped=self.ped_ped, **kwargs)

    def generate(self, n):
        # ped0 always left to right
        speed0 = 0.7 + 0.4 * torch.rand(1).item()
        ped0 = [-5.0, 0.0, speed0, 0.0, 6.0, 0.0]  # or tau=, 0.5

        generator_initial_states = []
        for b in 0.75 * torch.randn(n) + self.b_center:
            # with 20% speed variation
            speed = 1.2 + 0.2 * torch.rand(1).item()
            ped1 = [-7.0, b, speed, 0.0, 7.0, b]  # or tau=, 0.1

            state = np.array([ped0, ped1])
            state = self.simulator.normalize_state(state)
            generator_initial_states.append(state)

        with torch.no_grad():
            return [
                self.simulator.run(initial_state, 21)
                for initial_state in generator_initial_states
            ]


class ParallelAvoidance:
    def __init__(self, ped_ped=None, *, b_center=0.0, **kwargs):
        self.ped_ped = ped_ped or potentials.PedPedPotential(2.1)
        self.b_center = b_center
        self.simulator = Simulator(ped_ped=self.ped_ped, **kwargs)

    def generate(self, n):
        # ped0 always left to right
        speed0 = 0.7 + 0.4 * torch.rand(1).item()
        ped0 = [-5.0, 0.0, speed0, 0.0, 6.0, 0.0]  # or tau=, 0.5

        generator_initial_states = []
        for b in 0.75 * torch.randn(n) + self.b_center:
            # with 20% speed variation
            speed = 1.2 + 0.2 * torch.rand(1).item()
            ped1 = [7.0, b, -speed, 0.0, -7.0, b]  # or tau=, 0.1

            state = np.array([ped0, ped1])
            state = self.simulator.normalize_state(state)
            generator_initial_states.append(state)

        with torch.no_grad():
            return [
                self.simulator.run(initial_state, 21)
                for initial_state in generator_initial_states
            ]
