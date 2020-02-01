import math
import random

import torch

class PredictionLoss(torch.nn.Module):
    """2D Gaussian with a flat background.

    p(x) = 0.2 * N(x|mu, 3.0)  +  0.8 * N(x|mu, sigma)
    """
    def __init__(self, size_average=True, reduce=True, background_rate=0.2):
        super(PredictionLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.background_rate = background_rate

    @staticmethod
    def gaussian_2d(mu1mu2s1s2rho, x1x2):
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

        numerator = torch.exp(-z / (2 * (1 - rho ** 2)))
        denominator = 2 * math.pi * sigma1sigma2 * torch.sqrt(1 - rho ** 2)

        return numerator / denominator

    def forward(self, inputs, targets):
        inputs_bg = inputs.clone()
        inputs_bg[:, 2] = 3.0  # sigma_x
        inputs_bg[:, 3] = 3.0  # sigma_y
        inputs_bg[:, 4] = 0.0  # rho

        values = -torch.log(
            0.01 +
            self.background_rate * self.gaussian_2d(inputs_bg, targets) +
            (0.99 - self.background_rate) * self.gaussian_2d(inputs, targets)
        )
        if not self.reduce:
            return values
        if self.size_average:
            return torch.mean(values)
        return torch.sum(values)


class L2Loss(torch.nn.Module):
    """Pytorch L2 Loss between Mean of predicted gaussians and targets
    """
    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, targets):
        return self.loss(inputs[:, :2], targets)

def bce_loss(input_, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input_.abs()
    loss = input_.clamp(min=0) - input_ * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss
    """
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


def variety_loss(inputs, target, pred_length=12):
    """Variety Loss acc to SGAN
    """
    criterion = torch.nn.MSELoss()
    min_loss_value = 1e10
    for sample in inputs:
        tmp_loss = criterion(sample[-pred_length:, 0, :2], target) * 100
        if tmp_loss.detach() < min_loss_value:
            loss = tmp_loss
            min_loss_value = tmp_loss.detach()
    return loss
