"""Loss used in the VAE model"""

import math
import torch

class KLDLoss(torch.nn.Module):
    """Kullback-Leibler divergence Loss
    """
    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, inputs, batch_split, targets=None):
        """
        Forward path

        Parameters:
        -----------
        inputs : Tensor [num_tracks, 2*latent_dim]
            Tensor containing multivariate distribution mean and logarithmic variance
        targets : Tensor [num_tracks, 2*latent_dim]
            Tensor containing target multivariate distribution mean and logarithmic variance
            Default: standard normal distribution (zero mean and unit variance)
        
        Output:
        -----------
        loss : Tensor [1]
            Tensor containing Kullback-Leibler divergence loss
    
        """

        inputs = inputs[batch_split[:-1]]
        if targets is not None:
            targets = targets[batch_split[:-1]]

        ## Adapted from https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
        if targets is None:
            # Default KLD Loss (with standard normal distribution)
            z_mu, z_log_var = torch.split(inputs, split_size_or_sections=inputs.size(1)//2, dim=1)
            latent_loss = -0.5 * torch.sum(1.0 + z_log_var - torch.square(z_mu) - torch.exp(z_log_var), dim=1)
        else:
            # KLD Loss between the distributions inputs and targets
            z_mu, z_log_var = torch.split(inputs, split_size_or_sections=inputs.size(1)//2, dim=1)
            z_mu_t, z_log_var_t = torch.split(targets, split_size_or_sections=targets.size(1)//2, dim=1)
            z_var = torch.exp(z_log_var)
            z_var_t = torch.exp(z_log_var_t)
            # latent_loss = 0.5 * (((1/z_var_t)*z_var).sum(dim=1) \
            #               + ((z_mu_t-z_mu)**2 * (1/z_var_t)).sum(dim=1) \
            #               + torch.sum(z_log_var_t, dim=1) / torch.sum(z_log_var, dim=1))
            ## Stable
            latent_loss = 0.5 * (((1/z_var_t)*z_var).sum(dim=1) \
                          + ((z_mu_t-z_mu)**2 * (1/z_var_t)).sum(dim=1))
        return torch.mean(latent_loss)
