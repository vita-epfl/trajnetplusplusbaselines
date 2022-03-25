from collections import defaultdict
import os

import numpy as np
import matplotlib.pyplot as plt

import torch

def one_cold(i, n):
    """Inverse one-hot encoding."""
    x = torch.ones(n, dtype=torch.bool)
    x[i] = 0
    return x

class GridBasedPooling(torch.nn.Module):
    def __init__(self, cell_side=2.0, n=4, hidden_dim=128, out_dim=None,
                 type_='occupancy', pool_size=1, blur_size=1, front=False,
                 embedding_arch='one_layer', pretrained_pool_encoder=None,
                 constant=0, norm=0, layer_dims=None, latent_dim=16):
        """
        Pools in a grid of size 'n * cell_side' centred at the ped location
        cell_side: Scalar
            size of each cell in real world
        n: Scalar
            number of cells along one dimension
        out_dim: Scalar
            dimension of resultant interaaction vector
        type_: ('occupancy', 'directional', 'social', 'dir_social')
            type of grid-based pooling
        front: Bool 
            if True, pools neighbours only in the front of pedestrian
        embedding_arch: ('one_layer', 'two_layer', 'three_layer', 'lstm_layer')
            architecture to encoder grid tensor
        pretrained_pool_encoder: None
            autoencoder to reduce dimensionality of grid
        constant: int
            background values of pooling grid
        norm: Scalar 
            normalization scheme of pool grid [Default: None]
        """
        super(GridBasedPooling, self).__init__()
        self.cell_side = cell_side
        self.n = n
        self.type_ = type_
        self.pool_size = pool_size
        self.blur_size = blur_size

        self.norm_pool = False
        self.front = front
        if self.front:
            self.norm_pool = True
        self.constant = constant
        self.norm = norm
        self.pool_scale = 1.0

        ## Type of pooling
        self.pooling_dim = 1
        if self.type_ == 'directional':
            self.pooling_dim = 2
        if self.type_ == 'social':
            ## Encode hidden-dim into latent-dim vector (faster computation)
            self.hidden_dim_encoding = torch.nn.Linear(hidden_dim, latent_dim)
            self.pooling_dim = latent_dim
        if self.type_ == 'dir_social':
            ## Encode hidden-dim into latent-dim vector (faster computation)
            self.hidden_dim_encoding = torch.nn.Linear(hidden_dim, latent_dim)
            self.pooling_dim = latent_dim + 2

        ## Final Representation Size
        if out_dim is None:
            out_dim = hidden_dim
        self.out_dim = out_dim

        ## Pretrained AE
        self.pretrained_model = pretrained_pool_encoder
        input_dim = None
        if self.pretrained_model is not None:
            input_dim = self.pretrained_model[-1].out_features
            if embedding_arch == 'None':
                self.out_dim = input_dim

        ## Embedding Grid / AE Representation
        self.embedding = None
        self.embedding_arch = embedding_arch
        if self.embedding_arch == 'one_layer':
            self.embedding = self.one_layer(input_dim)
        elif self.embedding_arch == 'two_layer':
            self.embedding = self.two_layer(input_dim, layer_dims)
        elif self.embedding_arch == 'three_layer':
            self.embedding = self.three_layer(input_dim, layer_dims)
        elif self.embedding_arch == 'lstm_layer':
            self.embedding = self.lstm_layer(hidden_dim)

    def forward(self, hidden_state, obs1, obs2):
        batch_size, num_tracks = obs1.size(0), obs1.size(1) 
        ## Make chosen grid
        if self.type_ == 'occupancy':
            grid = self.occupancies(obs1, obs2)
        elif self.type_ == 'directional':
            grid = self.directional(obs1, obs2)
        elif self.type_ == 'social':
            grid = self.social(hidden_state, obs1, obs2)
        elif self.type_ == 'dir_social':
            grid = self.dir_social(hidden_state, obs1, obs2)

        ## Embed grid
        grid = grid.reshape(batch_size * num_tracks, -1)
        if self.embedding:
            return self.embedding(grid)
        return grid

    def occupancies(self, obs1, obs2):
        ## Generate the Occupancy Map
        # obs1 : batch_size, num_tracks, 2
        # obs2 : batch_size, num_tracks, 2
        return self.occupancy(obs2, past_obs=obs1)

    def directional(self, obs1, obs2):
        ## Makes the Directional Grid
        # obs1 : batch_size, num_tracks, 2
        # obs2 : batch_size, num_tracks, 2

        num_tracks = obs2.size(1)
        batch_size = obs2.size(0)

        ## if only primary pedestrian present
        if num_tracks == 1:
            return self.occupancy(obs2, None, past_obs=obs1)

        ## Generate values to input in directional grid tensor (relative velocities in this case) 
        vel = obs2 - obs1
        unfolded = vel.unsqueeze(1).repeat(1, vel.size(1), 1, 1)
        ## [batch_size, num_tracks, 2] --> [batch_size, num_tracks, num_tracks, 2]
        relative = unfolded - vel.unsqueeze(2)
        ## Deleting Diagonal (Ped wrt itself)
        ## mask: [batch_size, num_tracks, num_tracks]
        mask = ~torch.eye(num_tracks).unsqueeze(0).repeat(batch_size, 1, 1).bool()
        ## [batch_size, num_tracks, num_tracks, 2] --> [batch_size, num_tracks, num_tracks-1, 2]
        relative = relative[mask].reshape(batch_size, num_tracks, num_tracks-1, 2)
        relative = torch.nan_to_num(relative)

        ## Generate Occupancy Map
        return self.occupancy(obs2, relative, past_obs=obs1)

    def social(self, hidden_state, obs1, obs2):
        ## Makes the Social Grid
        # hidden_state : batch_size, num_tracks, hidden_dim
        # obs1 : batch_size, num_tracks, 2
        # obs2 : batch_size, num_tracks, 2

        num_tracks = obs2.size(1)
        batch_size = obs2.size(0)

        ## if only primary pedestrian present
        if num_tracks == 1:
            return self.occupancy(obs2, None, past_obs=obs1)

        ## Generate values to input in hiddenstate grid tensor (compressed hidden-states in this case)
        ## [batch_size, num_tracks, hidden_dim] --> [batch_size, num_tracks, num_tracks, pooling_dim]
        hidden_state_grid = hidden_state.unsqueeze(1).repeat(1, num_tracks, 1, 1)
        ## Deleting Diagonal (Ped wrt itself)
        ## mask: [batch_size, num_tracks, num_tracks]
        mask = ~torch.eye(num_tracks).unsqueeze(0).repeat(batch_size, 1, 1).bool()
        ## [batch_size, num_tracks, num_tracks, 2] --> [batch_size, num_tracks, num_tracks-1, 2]
        hidden_state_grid = hidden_state_grid[mask].reshape(batch_size, num_tracks, num_tracks-1, -1)
        hidden_state_grid = torch.nan_to_num(hidden_state_grid)
        hidden_state_grid = self.hidden_dim_encoding(hidden_state_grid)
        
        ## Generate Occupancy Map
        return self.occupancy(obs2, hidden_state_grid, past_obs=obs1)

    def dir_social(self, hidden_state, obs1, obs2):
        ## Makes the Directional + Social Grid
        # hidden_state : batch_size, num_tracks, hidden_dim
        # obs1 : batch_size, num_tracks, 2
        # obs2 : batch_size, num_tracks, 2

        num_tracks = obs2.size(1)
        batch_size = obs2.size(0)

        ## if only primary pedestrian present
        if num_tracks == 1:
            return self.occupancy(obs2, None, past_obs=obs1)

        ## Generate values to input in directional grid tensor (relative velocities in this case) 
        vel = obs2 - obs1
        unfolded = vel.unsqueeze(1).repeat(1, vel.size(1), 1, 1)
        ## [batch_size, num_tracks, 2] --> [batch_size, num_tracks, num_tracks, 2]
        relative = unfolded - vel.unsqueeze(2)
        ## Deleting Diagonal (Ped wrt itself)
        ## mask: [batch_size, num_tracks, num_tracks]
        mask = ~torch.eye(num_tracks).unsqueeze(0).repeat(batch_size, 1, 1).bool()
        ## [batch_size, num_tracks, num_tracks, 2] --> [batch_size, num_tracks, num_tracks-1, 2]
        relative = relative[mask].reshape(batch_size, num_tracks, num_tracks-1, 2)
        relative = torch.nan_to_num(relative)

        ## Generate values to input in hiddenstate grid tensor (compressed hidden-states in this case)
        ## [batch_size, num_tracks, hidden_dim] --> [batch_size, num_tracks, num_tracks, pooling_dim]
        hidden_state_grid = hidden_state.unsqueeze(1).repeat(1, num_tracks, 1, 1)
        ## Deleting Diagonal (Ped wrt itself)
        ## mask: [batch_size, num_tracks, num_tracks]
        mask = ~torch.eye(num_tracks).unsqueeze(0).repeat(batch_size, 1, 1).bool()
        ## [batch_size, num_tracks, num_tracks, 2] --> [batch_size, num_tracks, num_tracks-1, 2]
        hidden_state_grid = hidden_state_grid[mask].reshape(batch_size, num_tracks, num_tracks-1, -1)
        hidden_state_grid = torch.nan_to_num(hidden_state_grid)
        hidden_state_grid = self.hidden_dim_encoding(hidden_state_grid)

        # Combine representations
        dir_social_rep = torch.cat([relative, hidden_state_grid], dim=2)

        ## Generate Occupancy Map
        return self.occupancy(obs2, dir_social_rep, past_obs=obs1)

    @staticmethod
    def normalize(relative, obs, past_obs):
        ## Normalize pooling grid along direction of pedestrian motion
        diff = torch.cat([obs[:, 1:] - past_obs[:, 1:], obs[:, 0:1] - past_obs[:, 0:1]], dim=1)
        velocity = np.arctan2(diff[:, 0].clone(), diff[:, 1].clone())
        theta = (np.pi / 2) - velocity
        ct = torch.cos(theta)
        st = torch.sin(theta)
        ## Cleaner?
        relative = torch.stack([torch.einsum('tc,ci->ti', pos_instance, torch.Tensor([[ct[i], st[i]], [-st[i], ct[i]]])) for
                                i, pos_instance in enumerate(relative)], dim=0)
        return relative

    def occupancy(self, obs, other_values=None, past_obs=None):
        """Returns the occupancy map filled with respective attributes.
        A different occupancy map with respect to each pedestrian
        Parameters
        ----------
        obs: Tensor [num_tracks, 2]
            Current x-y positions of all pedestrians, used to construct occupancy map.
        other_values: Tensor [num_tracks, num_tracks-1,  2]
            Attributes (self.pooling_dim) of the neighbours relative to pedestrians, to be filled in the occupancy map
            e.g. Relative velocities of pedestrians
        past_obs: Tensor [num_tracks, 2]
            Previous x-y positions of all pedestrians, used to construct occupancy map.
            Useful for normalizing the grid tensor.
        Returns
        -------
        grid: Tensor [num_tracks, self.pooling_dim, self.n, self.n]
        """
        batch_size = obs.size(0)
        num_tracks = obs.size(1)

        ##mask unseen
        mask = torch.isnan(obs).any(dim=-1)
        obs[mask] = -500.0

        ## if only primary pedestrian present
        if num_tracks == 1:
            return self.constant*torch.ones(1, self.pooling_dim, self.n, self.n, device=obs.device)

        ## Get relative position
        ## [batch_size, num_tracks, 2] --> [batch_size, num_tracks, num_tracks, 2]
        unfolded = obs.unsqueeze(1).repeat(1, obs.size(1), 1, 1)
        relative = unfolded - obs.unsqueeze(2)
        ## Deleting Diagonal (Ped wrt itself)
        ## mask: [batch_size, num_tracks, num_tracks]
        mask = ~torch.eye(num_tracks).unsqueeze(0).repeat(batch_size, 1, 1).bool()
        ## [batch_size, num_tracks, num_tracks, 2] --> [batch_size, num_tracks, num_tracks-1, 2]
        relative = relative[mask].reshape(batch_size, num_tracks, num_tracks-1, 2)

        ## In case of 'occupancy' pooling
        if other_values is None:
            other_values = torch.ones(batch_size, num_tracks, num_tracks-1, self.pooling_dim, device=obs.device)

        ## Normalize pooling grid along direction of pedestrian motion
        # if self.norm_pool:
        #     relative = self.normalize(relative, obs, past_obs)

        if self.front:
            oij = (relative / (self.cell_side / self.pool_size) + torch.Tensor([self.n * self.pool_size / 2, 0]))
        else:
            oij = (relative / (self.cell_side / self.pool_size) + self.n * self.pool_size / 2)

        range_violations = torch.sum((oij < 0) + (oij >= self.n * self.pool_size), dim=-1)
        range_mask = range_violations == 0

        oij[~range_mask] = 0
        other_values[~range_mask] = self.constant
        other_values = other_values.view(batch_size * num_tracks, num_tracks-1, -1)
        oij = oij.long()

        ## Flatten
        oi = oij[:, :, :, 0] * self.n * self.pool_size + oij[:, :, :, 1]
        oi = oi.view(batch_size * num_tracks, -1)
        # faster occupancy
        occ = self.constant*torch.ones(batch_size * num_tracks, self.n**2 * self.pool_size**2, self.pooling_dim, device=obs.device)

        ## Fill occupancy map with attributes
        occ[torch.arange(occ.size(0)).unsqueeze(1), oi] = other_values
        occ = torch.transpose(occ, 1, 2)
        occ_2d = occ.view(batch_size * num_tracks, -1, self.n * self.pool_size, self.n * self.pool_size)

        if self.blur_size == 1:
            occ_blurred = occ_2d
        else:
            occ_blurred = torch.nn.functional.avg_pool2d(
                occ_2d, self.blur_size, 1, int(self.blur_size / 2), count_include_pad=True)

        occ_summed = torch.nn.functional.lp_pool2d(occ_blurred, 1, self.pool_size)
        # occ_summed = torch.nn.functional.avg_pool2d(occ_blurred, self.pool_size)  # faster?
        return occ_summed

    ## Architectures of Encoding Grid
    def one_layer(self, input_dim=None):
        if input_dim is None:
            input_dim = self.n * self.n * self.pooling_dim
        return  torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.out_dim),
            torch.nn.ReLU(),)

    ## Default Layer Dims: 1024
    def two_layer(self, input_dim=None, layer_dims=None):
        if input_dim is None:
            input_dim = self.n * self.n * self.pooling_dim
        return  torch.nn.Sequential(
            torch.nn.Linear(input_dim, layer_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_dims[0], self.out_dim),
            torch.nn.ReLU(),)

    ## Default Layer Dims: 1024, 512
    def three_layer(self, input_dim=None, layer_dims=None):
        if input_dim is None:
            input_dim = self.n * self.n * self.pooling_dim
        return  torch.nn.Sequential(
            torch.nn.Linear(input_dim, layer_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_dims[0], layer_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_dims[1], self.out_dim),
            torch.nn.ReLU(),)

    def lstm_layer(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.pool_lstm = torch.nn.LSTMCell(self.out_dim, self.hidden_dim)
        self.hidden2pool = torch.nn.Linear(self.hidden_dim, self.out_dim)
        return torch.nn.Sequential(
                         torch.nn.Linear(self.n * self.n * self.pooling_dim, self.out_dim),
                         torch.nn.ReLU(),)

    def reset(self, num_tracks, max_num_neigh, device):
        self.track_mask = None
        if self.embedding_arch == 'lstm_layer':
            self.hidden_cell_state = (
                [torch.zeros(self.hidden_dim, device=device) for _ in range(num_tracks)],
                [torch.zeros(self.hidden_dim, device=device) for _ in range(num_tracks)],
            )

    def lstm_forward(self, grid):
        """ Forward process for LSTM-based grid encoding"""
        grid_embedding = self.embedding(grid)

        num_tracks = grid.size(0)
        ## If only primary pedestrian of the scene present
        if torch.sum(self.track_mask).item() == 1:
            return torch.zeros(num_tracks, self.out_dim, device=grid.device)

        hidden_cell_stacked = [
            torch.stack([h for m, h in zip(self.track_mask, self.hidden_cell_state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(self.track_mask, self.hidden_cell_state[1]) if m], dim=0),
        ]

        ## Update interaction-encoder LSTM
        hidden_cell_stacked = self.pool_lstm(grid_embedding, hidden_cell_stacked)
        interaction_vector = self.hidden2pool(hidden_cell_stacked[0])

        ## Save hidden-cell-states
        mask_index = [i for i, m in enumerate(self.track_mask) if m]
        for i, h, c in zip(mask_index,
                           hidden_cell_stacked[0],
                           hidden_cell_stacked[1]):
            self.hidden_cell_state[0][i] = h
            self.hidden_cell_state[1][i] = c

        return interaction_vector

    def make_grid(self, obs):
        """ Make the grids for all time-steps together 
            Only supports Occupancy and Directional pooling
        """
        if obs.ndim == 2:
            obs = obs.unsqueeze(0)
        timesteps = obs.size(0)

        grid = []
        for i in range(1, timesteps):
            obs1 = obs[i-1]
            obs2 = obs[i]
            ## Remove NANs
            track_mask = (torch.isnan(obs1[:, 0]) + torch.isnan(obs2[:, 0])) == 0
            obs1, obs2 = obs1[track_mask], obs2[track_mask]
            if self.type_ == 'occupancy':
                grid.append(self.occupancies(obs1, obs2))
            elif self.type_ == 'directional':
                grid.append(self.directional(obs1, obs2))
        return grid
