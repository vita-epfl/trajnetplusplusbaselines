import numpy as np
import torch

def sample_multivariate_distribution(mean, var_log):
    """
    Draw random samples from a multivariate normal distribution 

    Parameters
    ----------
    mean : Tensor [num_tracks, dim]  
        Mean of the multivariate distribution  
    var_log : Tensor [num_tracks, dim]
        Logarithm of the diagonal coefficients of the covariance matrix

    Returns
    -------
    samples : Tensor [num_tracks, dim]  
        The drawn samples of size [num_tracks, dim]
    """
    samples = torch.zeros_like(mean)
    for track in range(mean.size(0)):
        cov_matrix = np.diag(torch.exp(var_log[track, :]).numpy())
        samples[track, :] = torch.Tensor(np.random.multivariate_normal(mean[track, :].numpy(), cov_matrix))
    return samples