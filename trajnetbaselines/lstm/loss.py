import math
import random

import torch

class PredictionLoss(torch.nn.Module):
    """2D Gaussian with a flat background.

    p(x) = 0.2 * N(x|mu, 3.0)  +  0.8 * N(x|mu, sigma)
    """
    def __init__(self, keep_batch_dim=False, background_rate=0.2,
                 col_wt=0.0, col_distance=0.2):
        super(PredictionLoss, self).__init__()
        self.keep_batch_dim = keep_batch_dim
        self.background_rate = background_rate
        self.loss_multiplier = 1

        self.col_wt = col_wt
        self.col_distance = col_distance
        if self.col_wt:
            print("Using Auxiliary collision loss")

    @staticmethod
    def gaussian_2d(mu1mu2s1s2rho, x1x2):
        """This supports backward().

        Inspired by
        https://github.com/naba89/RNN-Handwriting-Generation-Pytorch/blob/master/loss_functions.py
        """

        x1, x2 = x1x2[:, 0], x1x2[:, 1]
        mu1, mu2, s1, s2, rho = (
            mu1mu2s1s2rho[:, 0],
            mu1mu2s1s2rho[:, 1],
            mu1mu2s1s2rho[:, 2],
            mu1mu2s1s2rho[:, 3],
            mu1mu2s1s2rho[:, 4],
        )

        norm1 = x1 - mu1
        norm2 = x2 - mu2

        sigma1sigma2 = s1 * s2

        z = (norm1 / s1) ** 2 + (norm2 / s2) ** 2 - 2 * rho * norm1 * norm2 / sigma1sigma2

        numerator = torch.exp(-z / (2 * (1 - rho ** 2)))
        denominator = 2 * math.pi * sigma1sigma2 * torch.sqrt(1 - rho ** 2)

        return numerator / denominator

    def forward(self, inputs, targets, batch_split, positions=None):
        pred_length, batch_size = targets.size(0), batch_split[:-1].size(0)

        # Tranpose
        inputs = inputs.permute(1, 0, 2)
        targets = targets.permute(1, 0, 2)

        ## Loss calculation
        inputs = inputs.reshape(-1, 5)
        targets = targets.reshape(-1, 2)
        inputs_bg = inputs.clone()
        inputs_bg[:, 2] = 3.0  # sigma_x
        inputs_bg[:, 3] = 3.0  # sigma_y
        inputs_bg[:, 4] = 0.0  # rho

        values = -torch.log(
            0.01 +
            self.background_rate * self.gaussian_2d(inputs_bg, targets) +
            (0.99 - self.background_rate) * self.gaussian_2d(inputs, targets)
        )

        ## Used in variety loss (SGAN)
        if self.keep_batch_dim:
            values = values.reshape(pred_length, batch_size)
            return values.mean(dim=0) * self.loss_multiplier

        # Give equal weight to each sample (based to sample size)
        sample_sizes = batch_split[1:] - batch_split[:-1]
        sample_weights = torch.empty(batch_split[-1] * pred_length, device=targets.device)
        current_sample_start = 0
        for sample_size in sample_sizes:
            sample_weights[current_sample_start * pred_length: (current_sample_start + sample_size) * pred_length] = 1 / (pred_length * sample_size)
            current_sample_start += sample_size
        sample_weights = sample_weights / batch_size
        return (torch.sum(values * sample_weights) * self.loss_multiplier)
