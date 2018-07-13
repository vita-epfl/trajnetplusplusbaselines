import math
import torch


class PredictionLoss(torch.nn.Module):
    """Negative Log of a 2D Gaussian."""
    def __init__(self, size_average=True, reduce=True):
        super(PredictionLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    @staticmethod
    def log_gaussian_2d(mu1mu2s1s2rho, x1x2):
        """This supports backward().

        Insprired by
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

        log_numerator = -z / (2 * (1 - rho ** 2))
        denominator = 2 * math.pi * sigma1sigma2 * torch.sqrt(1 - rho ** 2)

        return log_numerator - torch.log(denominator)

    def forward(self, inputs, targets):
        values = -self.log_gaussian_2d(inputs, targets)
        if not self.reduce:
            return values
        if self.size_average:
            return torch.mean(values)
        return torch.sum(values)
