from collections import defaultdict
import numpy as np

import torch

def one_cold(i, n):
    """Inverse one-hot encoding."""
    x = torch.ones(n, dtype=torch.bool)
    x[i] = 0
    return x

def rel_position_batch(obs):
    """ Provides relative position of neighbours wrt one another

    Parameters
    ----------
    obs :  Tensor [batch, num_tracks, 2]
        x-y positions of all agents

    Returns
    -------
    relative : Tensor [batch, num_tracks, num_tracks, 2]
    """
    unfolded = obs.unsqueeze(1).repeat(1, obs.size(1), 1, 1)
    relative = unfolded - obs.unsqueeze(2)
    return relative

def rel_directional_batch(obs1, obs2):
    """ Provides relative velocity of neighbours wrt one another

    Parameters
    ----------
    obs1 :  Tensor [batch, num_tracks, 2]
        x-y positions of all agents at previous time-step t-1
    obs2 :  Tensor [batch, num_tracks, 2]
        x-y positions of all agents at current time-step t

    Returns
    -------
    relative : Tensor [batch, num_tracks, num_tracks, 2]
    """

    vel = obs2 - obs1
    unfolded = vel.unsqueeze(1).repeat(1, vel.size(1), 1, 1)
    relative = unfolded - vel.unsqueeze(2)
    return relative

def remove_diagonal(relative):
    """ Remove relative position and velocity with self

    # [batch, num_tracks, num_tracks, 2] --> [batch, num_tracks, num_tracks - 1, 2]
    """

    num_human = relative.size(1)
    batch = relative.size(0)
    relative = relative.permute(1, 2, 0, 3)
    relative = relative[~torch.eye(num_human).bool()].reshape(num_human, num_human-1, batch, 2)
    relative = relative.permute(2, 0, 1, 3)    
    return relative

class NN_Pooling(torch.nn.Module):
    """ Interaction vector is obtained by concatenating the relative coordinates of
        top-n neighbours selected according to criterion (euclidean distance)
        
        Attributes
        ----------
        n : Scalar
            Number of neighbours per batch [REQUIRED: Constant across scenes!]
        out_dim: Scalar
            Dimension of resultant interaction vector
    """
    def __init__(self, n=4, out_dim=32):
        super(NN_Pooling, self).__init__()
        self.num_neighbor = n
        self.input_dim = 4
        self.out_dim = out_dim

        ## Fixed size embedding. Each neighbour gets equal-sized representation
        ## Currently, n must divide out_dim
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, int(out_dim/self.num_neighbor)),
            torch.nn.ReLU(),
        )


    def forward(self, _, obs1, obs2):
        """ Forward function. All agents must belong to the same scene

        Parameters
        ----------
        obs1 :  Tensor [batch, num_tracks, 2]
            x-y positions of all agents at previous time-step t-1
        obs2 :  Tensor [batch, num_tracks, 2]
            x-y positions of all agents at current time-step t

        Returns
        -------
        interaction_vector : Tensor [batch, num_tracks, self.out_dim]
            interaction vector of all agents in the scene
        """


        B = obs2.size(0)
        num_tracks = obs2.size(1)

        # Get relative positions with dim [B, num_tracks, num_tracks, 2]
        rel_position = rel_position_batch(obs2)
        # Removal diagonal (Ped wrt itself), with dim [B, num_tracks, num_tracks-1, 2]
        rel_position = remove_diagonal(rel_position)

        # Get relative velocities, with dim [B, num_tracks, num_tracks, 2]
        rel_direction = rel_directional_batch(obs1, obs2)
        # Removal diagonal (Ped wrt itself), with dim [B, num_tracks, num_tracks-1, 2]
        rel_direction = remove_diagonal(rel_direction)

        # Merge relative attributes of neighbours with dim [B, num_tracks, num_tracks-1, 4]
        neighbour_relative_attributes = torch.cat([rel_position, rel_direction], dim=3)
        neighbor_embed = self.embedding(neighbour_relative_attributes)

        return neighbor_embed.view(B, num_tracks, self.out_dim)

