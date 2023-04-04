import logging
import random

import torch

from .potentials import PedPedPotential

LOG = logging.getLogger(__name__)


class Trainer:
    """Trainer

    train_experience: list of state tuples (each state is a torch.Tensor)
    """

    def __init__(self, simulator, optimizer, *,
                 batch_size=1, loss=None):
        self.simulator = simulator
        self.optimizer = optimizer

        self.batch_size = batch_size
        self.loss = loss
        if loss is None:
            self.loss = torch.nn.L1Loss()

    @staticmethod
    def scenes_to_experience(scenes, radius=3.0, acc_abs=0.1):
        experience = [
            (state1, state2)
            for scene in scenes
            for state1, state2 in zip(scene[:-1], scene[1:])
        ]
        n_total = len(experience)

        def keep(state1, state2):
            valid_state1 = torch.isfinite(state1[:, 0])
            valid_state2 = torch.isfinite(state2[:, 0])

            small_distance = (
                (PedPedPotential.norm_r_ab(PedPedPotential.r_ab(state1)) < radius)
                | (PedPedPotential.norm_r_ab(PedPedPotential.r_ab(state2)) < radius)
            )
            torch.diagonal(small_distance)[:] = False
            small_distance = torch.any(small_distance, dim=-1)

            acc = (torch.abs(state1[:, 4:6]) > acc_abs) | (torch.abs(state2[:, 4:6]) > acc_abs)
            acc = torch.any(acc, dim=-1)
            # keep 10% of samples without acc:
            acc[~acc] = (torch.rand(acc[~acc].shape) < 0.1)
            acc[:] = torch.any(acc)  # symmetrize

            return valid_state1 & valid_state2 & small_distance & acc

        keep_pedestrians = [keep(state1, state2) for state1, state2 in experience]
        experience = [
            (state1[k], state2[k])
            for k, (state1, state2) in zip(keep_pedestrians, experience)
            if torch.any(k)
        ]

        LOG.info('from %d scenes, extracted %d experiences, filtered to %d',
                 len(scenes), n_total, len(experience))
        return experience

    def epoch(self, train_experience, val_experience=None):
        assert val_experience is None, 'TODO'  # TODO
        random.shuffle(train_experience)

        n_batches = 0
        epoch_loss = 0.0

        for i in range(0, len(train_experience), self.batch_size):
            data = train_experience[i:i + self.batch_size]

            loss = 0.0
            for e in data:
                Y = e[1]
                X = self.simulator(e[0])
                loss = loss + self.loss(X[:, :2], Y[:, :2])

            epoch_loss += float(loss.item())
            n_batches += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        epoch_loss /= n_batches
        return epoch_loss

    def loop(self, n_epochs, train_experience, val_experience=None, *, log_interval=1):
        for i in range(n_epochs):
            loss = self.epoch(train_experience, val_experience)
            if (i + 1) % log_interval == 0:
                print(f'epoch {i + 1}: {loss}')
