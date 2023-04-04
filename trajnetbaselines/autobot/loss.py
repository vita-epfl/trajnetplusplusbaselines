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
        ## Extract primary pedestrians
        # [pred_length, num_tracks, 2] --> [pred_length, batch_size, 2]
        targets = targets.transpose(0, 1)
        targets = targets[batch_split[:-1]]
        targets = targets.transpose(0, 1)

        col_loss = 0
        if self.col_wt:
            assert positions is not None, "Prediction positions required to calculate collision loss"
            col_loss = CollisionLoss(positions, batch_split, self.col_wt, self.col_distance)

        # [pred_length, num_tracks, 5] --> [pred_length, batch_size, 5]
        inputs = inputs.transpose(0, 1)
        inputs = inputs[batch_split[:-1]]
        inputs = inputs.transpose(0, 1)

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
        
        if self.col_wt:
            return torch.mean(values) * self.loss_multiplier + col_loss * self.loss_multiplier
        return (torch.mean(values) * self.loss_multiplier)

class L2Loss(torch.nn.Module):
    """L2 Loss (deterministic version of PredictionLoss)

    This Loss penalizes only the primary trajectories
    """
    def __init__(self, keep_batch_dim=False,
                 col_wt=0.0, col_distance=0.2):
        super(L2Loss, self).__init__()
        self.loss = torch.nn.MSELoss(reduction='none')
        self.keep_batch_dim = keep_batch_dim
        self.loss_multiplier = 100

        self.col_wt = col_wt
        self.col_distance = col_distance
        if self.col_wt:
            print("Using Auxiliary collision loss")

    def forward(self, inputs, targets, batch_split, positions=None):
        ## Extract primary pedestrians
        # [pred_length, num_tracks, 2] --> [pred_length, batch_size, 2]
        targets = targets.transpose(0, 1)
        targets = targets[batch_split[:-1]]
        targets = targets.transpose(0, 1)

        col_loss = 0.0
        if self.col_wt:
            assert positions is not None, "Prediction positions required to calculate collision loss"
            col_loss = CollisionLoss(positions, batch_split, self.col_wt, self.col_distance)

        # [pred_length, num_tracks, 5] --> [pred_length, batch_size, 5]
        inputs = inputs.transpose(0, 1)
        inputs = inputs[batch_split[:-1]]
        inputs = inputs.transpose(0, 1)

        loss = self.loss(inputs[:, :, :2], targets)

        ## Used in variety loss (SGAN)
        if self.keep_batch_dim:
            return loss.mean(dim=0).mean(dim=1) * self.loss_multiplier
        
        if self.col_wt:
            return torch.mean(loss) * self.loss_multiplier + col_loss * self.loss_multiplier
        return (torch.mean(loss) * self.loss_multiplier)


def CollisionLoss(predictions, batch_split, col_wt=10.0, col_distance=0.2):
    """
    Penalizes model when primary pedestrian prediction comes close
    to the neighbour predictions
    primary: Tensor [pred_length, 1, 2]
    neighbours: Tensor [pred_length, num_neighbours, 2]
    col_wt: Weight of collision loss
    col_distance: distance threshold post which collision occurs
    """

    predictions[predictions != predictions] = -1000
    col_loss = 0.0
    for (start, end) in zip(batch_split[:-1], batch_split[1:]):
        primary = predictions[:, start:start+1, :2]
        if (start + 1) == end:  # No neighbours
            continue
        neighs = predictions[:, start+1:end, :2].detach()
        distance_to_neighs = torch.norm(primary - neighs, dim=-1).view(-1)
        colliding_neigh_mask = (distance_to_neighs <= col_distance).detach()
        if not colliding_neigh_mask.any():  # No collisions
            continue
        colliding_neighs_dist = distance_to_neighs[colliding_neigh_mask] / col_distance
        col_val = 1 - colliding_neighs_dist
        col_loss += col_wt * col_val.sum()
    return col_loss


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
