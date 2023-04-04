from collections import defaultdict
import numpy as np

import torch

def one_cold(i, n):
    """Inverse one-hot encoding."""
    x = torch.ones(n, dtype=torch.bool)
    x[i] = 0
    return x


def rel_obs(obs):
    """ Provides relative position of neighbours wrt one another
    obs :  Tensor [batch_size, num_tracks, 2]
        x-y positions of all agents
    relative : Tensor [batch_size, num_tracks, num_tracks, 2]
    """
    ## [batch_size, num_tracks, 2] --> [batch_size, num_tracks, num_tracks, 2]
    unfolded = obs.unsqueeze(1).repeat(1, obs.size(1), 1, 1)
    relative = unfolded - obs.unsqueeze(2)
    return relative


def rel_directional(obs1, obs2):
    """ Provides relative velocity of neighbours wrt one another
    obs1 :  Tensor [batch_size, num_tracks, 2]
        x-y positions of all agents at previous time-step t-1
    obs2 :  Tensor [batch_size, num_tracks, 2]
        x-y positions of all agents at current time-step t
    relative : Tensor [batch_size, num_tracks, num_tracks, 2]
    """
    ## Generate values to input in directional grid tensor (relative velocities in this case) 
    vel = obs2 - obs1
    unfolded = vel.unsqueeze(1).repeat(1, vel.size(1), 1, 1)
    ## [batch_size, num_tracks, 2] --> [batch_size, num_tracks, num_tracks, 2]
    relative = unfolded - vel.unsqueeze(2)
    return relative


def delete_diagonal(input, batch_size, num_tracks):
    """ Deletes the effects of each pedestrian on itself.
    input: Tensor [batch_size, num_tracks, num_tracks, 2]
    output : Tensor [batch_size, num_tracks, num_tracks-1, 2]
    """
    # mask: [batch_size, num_tracks, num_tracks]
    mask = ~torch.eye(num_tracks).unsqueeze(0).repeat(batch_size, 1, 1).bool()
    last_dim = input.size(-1)
    # [batch_size, num_tracks, num_tracks, last_dim] --> [batch_size, num_tracks, num_tracks-1, last_dim]
    return input[mask].reshape(batch_size, num_tracks, num_tracks-1, last_dim)


def embed_with_masking(embedding_module, input, out_dim, fill_value=-100):
    """ Embed the parts of the inputs that do not corresponding to NaNs.
    Fill the rest with 'fill_value'."""
    nan_mask = torch.isnan(input).any(dim=-1)
    # [batch_size, num_tracks, num_tracks, 2] --> [batch_size, num_tracks, num_tracks, self.mlp_dim_spatial]
    embedding_vis = embedding_module(input[~nan_mask])
    embedding = torch.empty(*input.size()[:-1], out_dim).fill_(fill_value).to(input.device)  # placeholder
    embedding[~nan_mask] = embedding_vis
    return embedding


class NearestNeighborMLP(torch.nn.Module):
    """ Interaction vector is obtained by concatenating the relative coordinates of
        top-n neighbours selected according to criterion (euclidean distance)
        
        Attributes
        ----------
        n : Scalar
            Number of neighbours to select
        no_vel: Bool
            If True, does not concatenate the relative velocity of neighbour 
            to the relative position
        out_dim: Scalar
            Dimension of resultant interaction vector
    """
    def __init__(self, n=4, out_dim=32, no_vel=False):
        super(NearestNeighborMLP, self).__init__()
        self.n = n
        self.out_dim = out_dim
        self.no_velocity = no_vel
        self.input_dim = 2 if self.no_velocity else 4

        # Fixed size embedding. Each neighbour gets equal-sized representation
        # Currently, n must divide out_dim !
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, int(out_dim/self.n)),
            torch.nn.ReLU(),
        )

    def reset(self, num_tracks, max_num_neigh, device):
        self.track_mask = None

    def forward(self, _, obs1, obs2):
        """ Forward function. All agents must belong to the same scene

        Parameters
        ----------
        obs1 :  Tensor [batch_size, num_tracks, 2]
            x-y positions of all agents at previous time-step t-1
        obs2 :  Tensor [batch_size, num_tracks, 2]
            x-y positions of all agents at current time-step t

        Returns
        -------
        interaction_vector : Tensor [batch_size, num_tracks, self.out_dim]
            interaction vector of all agents in the scene
        """

        num_tracks = obs2.size(1)
        batch_size = obs2.size(0)

        # Get relative position of all agents wrt one another 
        # [batch_size, num_tracks, 2] --> [batch_size, num_tracks, num_tracks, 2]
        rel_position = rel_obs(obs2)
        ## Deleting Diagonal (Ped wrt itself)
        ## [batch_size, num_tracks, num_tracks, 2] --> [batch_size, num_tracks, num_tracks-1, 2]
        rel_position = delete_diagonal(rel_position, batch_size, num_tracks)

        # Get relative velocities of all agents wrt one another 
        # [batch_size, num_tracks, 2] --> [batch_size, num_tracks, num_tracks, 2]
        rel_direction = rel_directional(obs1, obs2)
        ## Deleting Diagonal (Ped wrt itself)
        ## [batch_size, num_tracks, num_tracks, 2] --> [batch_size, num_tracks, num_tracks-1, 2]
        rel_direction = delete_diagonal(rel_direction, batch_size, num_tracks)

        # Combine [batch_size, num_tracks, num_tracks - 1, self.input_dim]
        overall_grid = torch.cat([rel_position, rel_direction], dim=-1) if not self.no_velocity else rel_position

        # Get nearest n neighours
        rel_distance = torch.norm(rel_position, dim=-1)
        rel_distance = torch.nan_to_num(rel_distance, nan=1000)  # High dummy distance
        if (num_tracks - 1) < self.n:
            nearest_grid = torch.zeros((batch_size, num_tracks, self.n, self.input_dim), device=obs2.device)
            _, dist_index = torch.topk(-rel_distance, num_tracks-1, dim=-1)
            nearest_grid[:, :, :(num_tracks-1)] = torch.gather(overall_grid, 2, dist_index.unsqueeze(-1).repeat(1, 1, 1, self.input_dim))
        else:
            _, dist_index = torch.topk(-rel_distance, self.n, dim=-1)
            nearest_grid = torch.gather(overall_grid, 2, dist_index.unsqueeze(-1).repeat(1, 1, 1, self.input_dim))

        # Remove NaNs
        nearest_grid = torch.nan_to_num(nearest_grid)

        ## Embed top-n relative neighbour attributes
        nearest_grid = self.embedding(nearest_grid)
        return nearest_grid.view(batch_size * num_tracks, -1)


class HiddenStateMLPPooling(torch.nn.Module):
    """ Interaction vector is obtained by max-pooling the embeddings of relative coordinates
        and hidden-state of all neighbours. Proposed in Social GAN
        
        Attributes
        ----------
        mlp_dim : Scalar
            Embedding dimension of each neighbour
        mlp_dim_spatial : Scalar
            Embedding dimension of relative spatial coordinates
        mlp_dim_vel: Scalar
            Embedding dimension of relative velocity coordinates
        out_dim: Scalar
            Dimension of resultant interaction vector
    """
    def __init__(self, hidden_dim=128, mlp_dim=128, mlp_dim_spatial=32, mlp_dim_vel=32, out_dim=None):
        super(HiddenStateMLPPooling, self).__init__()
        self.out_dim = out_dim or hidden_dim
        self.hidden_dim = hidden_dim

        self.mlp_dim = mlp_dim
        self.mlp_dim_spatial = mlp_dim_spatial
        self.mlp_dim_vel = mlp_dim_vel
        self.mlp_dim_hidden = mlp_dim - mlp_dim_spatial - mlp_dim_vel

        self.spatial_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, self.mlp_dim_spatial),
            torch.nn.ReLU(),
        )

        if self.mlp_dim_vel:
            self.vel_embedding = torch.nn.Sequential(
                torch.nn.Linear(2, self.mlp_dim_vel),
                torch.nn.ReLU(),
            )

        if self.mlp_dim_hidden:
            self.hidden_embedding = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim, self.mlp_dim_hidden),
                torch.nn.ReLU(),
            )
        self.out_projection = torch.nn.Linear(self.mlp_dim, self.out_dim)

    def reset(self, num_tracks, max_num_neigh, device):
        self.track_mask = None

    def forward(self, hidden_states, obs1, obs2):
        """ Forward function. All agents must belong to the same scene

        Parameters
        ----------
        obs1 :  Tensor [num_tracks, 2]
            x-y positions of all agents at previous time-step t-1
        obs2 :  Tensor [num_tracks, 2]
            x-y positions of all agents at current time-step t
        hidden_states :  Tensor [num_tracks, hidden_dim]
            LSTM hidden state of all agents at current time-step t

        Returns
        -------
        interaction_vector : Tensor [num_tracks, self.out_dim]
            interaction vector of all agents in the scene
        """

        num_tracks = obs2.size(1)
        batch_size = obs2.size(0)

        # Embed relative position with proper masking
        # [batch_size, num_tracks, 2] --> [batch_size, num_tracks, num_tracks, 2]
        relative_obs = rel_obs(obs2)
        embedded = embed_with_masking(self.spatial_embedding, relative_obs, self.mlp_dim_spatial)

        if self.mlp_dim_hidden:
            # Embed hidden states with proper masking
            # [batch_size, num_tracks, hidden_dim] --> [batch_size, num_tracks, mlp_dim_hidden]
            hidden = embed_with_masking(self.hidden_embedding, hidden_states, self.mlp_dim_hidden)
            # [batch_size, num_tracks, mlp_dim_hidden] --> [batch_size, num_tracks, num_tracks, mlp_dim_hidden]
            hidden_unfolded = hidden.unsqueeze(1).repeat(1, num_tracks, 1, 1)
            embedded = torch.cat([embedded, hidden_unfolded], dim=-1)

        if self.mlp_dim_vel:
            # Embed relative velocity with proper masking
            # [batch_size, num_tracks, 2] --> [batch_size, num_tracks, num_tracks, 2]
            rel_vel = rel_directional(obs1, obs2)
            directional = embed_with_masking(self.vel_embedding, rel_vel*4, self.mlp_dim_vel)
            embedded = torch.cat([embedded, directional], dim=-1)

        # Max Pool
        pooled, _ = torch.max(embedded, dim=2)
        return self.out_projection(pooled).view(batch_size * num_tracks, -1)


class AttentionMLPPooling(torch.nn.Module):
    """ Interaction vector is obtained by attention-weighting the embeddings of relative coordinates
        and hidden-state of all neighbours. Proposed in S-BiGAT
        
        Attributes
        ----------
        mlp_dim : Scalar
            Embedding dimension of each neighbour
        mlp_dim_spatial : Scalar
            Embedding dimension of relative spatial coordinates
        mlp_dim_vel: Scalar
            Embedding dimension of relative velocity coordinates
        out_dim: Scalar
            Dimension of resultant interaction vector
    """
    def __init__(self, hidden_dim=128, mlp_dim=128, mlp_dim_spatial=32, mlp_dim_vel=32, out_dim=None, fill_value=-10):
        super(AttentionMLPPooling, self).__init__()
        self.out_dim = out_dim or hidden_dim
        self.hidden_dim = hidden_dim
        self.fill_value = fill_value

        self.mlp_dim = mlp_dim
        self.mlp_dim_spatial = mlp_dim_spatial
        self.mlp_dim_vel = mlp_dim_vel
        self.mlp_dim_hidden = mlp_dim - mlp_dim_spatial - mlp_dim_vel

        self.spatial_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, self.mlp_dim_spatial),
            torch.nn.ReLU(),
        )

        if self.mlp_dim_vel:
            self.vel_embedding = torch.nn.Sequential(
                torch.nn.Linear(2, self.mlp_dim_vel),
                torch.nn.ReLU(),
            )

        if self.mlp_dim_hidden:
            self.hidden_embedding = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim, self.mlp_dim_hidden),
                torch.nn.ReLU(),
            )

        ## Attention Embeddings (Query, Key, Value)
        self.wq = torch.nn.Linear(self.mlp_dim, self.mlp_dim, bias=False)
        self.wk = torch.nn.Linear(self.mlp_dim, self.mlp_dim, bias=False)
        self.wv = torch.nn.Linear(self.mlp_dim, self.mlp_dim, bias=False)

        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=self.mlp_dim, num_heads=1)
        self.out_projection = torch.nn.Linear(self.mlp_dim, self.out_dim)

    def reset(self, num_tracks, max_num_neigh, device):
        self.track_mask = None

    def forward(self, hidden_states, obs1, obs2):
        """ Forward function. All agents must belong to the same scene

        Parameters
        ----------
        obs1 :  Tensor [batch_size, num_tracks, 2]
            x-y positions of all agents at previous time-step t-1
        obs2 :  Tensor [batch_size, num_tracks, 2]
            x-y positions of all agents at current time-step t
        hidden_states :  Tensor [batch_size, num_tracks, hidden_dim]
            LSTM hidden state of all agents at current time-step t

        Returns
        -------
        interaction_vector : Tensor [batch_size * num_tracks, self.out_dim]
            interaction vector of all agents in the scene
        """

        num_tracks = obs2.size(1)
        batch_size = obs2.size(0)

        # Embed relative position with proper masking
        # [batch_size, num_tracks, 2] --> [batch_size, num_tracks, num_tracks, 2]
        relative_obs = rel_obs(obs2)
        embedded = embed_with_masking(self.spatial_embedding, relative_obs, self.mlp_dim_spatial, self.fill_value)

        if self.mlp_dim_hidden:
            # Embed hidden states with proper masking
            # [batch_size, num_tracks, hidden_dim] --> [batch_size, num_tracks, mlp_dim_hidden]
            hidden = embed_with_masking(self.hidden_embedding, hidden_states, self.mlp_dim_hidden, fill_value=0)
            # [batch_size, num_tracks, mlp_dim_hidden] --> [batch_size, num_tracks, num_tracks, mlp_dim_hidden]
            hidden_unfolded = hidden.unsqueeze(1).repeat(1, num_tracks, 1, 1)
            embedded = torch.cat([embedded, hidden_unfolded], dim=-1)

        if self.mlp_dim_vel:
            # Embed relative velocity with proper masking
            # [batch_size, num_tracks, 2] --> [batch_size, num_tracks, num_tracks, 2]
            rel_vel = rel_directional(obs1, obs2)
            directional = embed_with_masking(self.vel_embedding, rel_vel*4, self.mlp_dim_vel, self.fill_value)
            embedded = torch.cat([embedded, directional], dim=-1)

        ## Attention
        # [batch_size, num_tracks, num_tracks, mlp_dim] --> [batch_size * num_tracks, num_tracks, mlp_dim]
        embedded = embedded.view(batch_size * num_tracks, num_tracks, -1)
        # [batch, seq, mlp_dim] --> [seq, batch, mlp_dim]
        embedded = embedded.transpose(0, 1)
        query = self.wq(embedded)
        key = self.wk(embedded)
        value = self.wv(embedded)
        attn_output, _ = self.multihead_attn(query, key, value)

        # We need to select entries along diagonal of each scene; along the first axis.
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(batch_size, num_tracks, num_tracks, -1)
        attn_output_ped = attn_output[torch.eye(num_tracks).bool().unsqueeze(0).repeat(batch_size, 1, 1)]
        return self.out_projection(attn_output_ped)


class NearestNeighborLSTM(torch.nn.Module):
    """ Interaction vector is obtained by concatenating the relative coordinates of
        top-n neighbours filtered according to criterion (euclidean distance).
        The concatenated vector is passed through an LSTM.
        
        Attributes
        ----------
        n : Scalar
            Number of neighbours to select
        track_mask : Bool [num_tracks,]
            Mask of tracks visible at the current time-instant
            as well as tracks belonging to the particular scene 
        hidden_dim : Scalar
            Hidden-state dimension of interaction-encoder LSTM
        out_dim: Scalar
            Dimension of resultant interaction vector
    """

    def __init__(self, n=4, hidden_dim=256, out_dim=32):
        super(NearestNeighborLSTM, self).__init__()
        self.n = n
        self.out_dim = out_dim
        self.input_dim = 4
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, int(out_dim/self.n)),
            torch.nn.ReLU(),
        )
        self.hidden_dim = hidden_dim
        self.pool_lstm = torch.nn.LSTMCell(out_dim, hidden_dim)
        self.hidden2pool = torch.nn.Linear(hidden_dim, out_dim)

    def reset(self, num_tracks, max_num_neigh, device):
        self.hidden_cell_state = [
            [torch.zeros(self.hidden_dim, device=device) for _ in range(num_tracks)],
            [torch.zeros(self.hidden_dim, device=device) for _ in range(num_tracks)],
        ]

    def forward(self, _, obs1, obs2):
        """ Forward function. All agents must belong to the same scene

        Parameters
        ----------
        obs1 :  Tensor [num_tracks, 2]
            x-y positions of all agents at previous time-step t-1
        obs2 :  Tensor [num_tracks, 2]
            x-y positions of all agents at current time-step t

        Returns
        -------
        interaction_vector : Tensor [num_tracks, self.out_dim]
            interaction vector of all agents in the scene
        """

        batch_size = obs2.size(0)
        num_tracks = obs2.size(1)

        hidden_cell_stacked = [
            torch.stack(self.hidden_cell_state[0], dim=0),
            torch.stack(self.hidden_cell_state[1], dim=0),
        ]

        # Get relative position of all agents wrt one another 
        # [batch_size, num_tracks, 2] --> [batch_size, num_tracks, num_tracks, 2]
        rel_position = rel_obs(obs2)
        ## Deleting Diagonal (Ped wrt itself)
        ## [batch_size, num_tracks, num_tracks, 2] --> [batch_size, num_tracks, num_tracks-1, 2]
        rel_position = delete_diagonal(rel_position, batch_size, num_tracks)

        # Get relative velocities of all agents wrt one another 
        # [batch_size, num_tracks, 2] --> [batch_size, num_tracks, num_tracks, 2]
        rel_direction = rel_directional(obs1, obs2)
        ## Deleting Diagonal (Ped wrt itself)
        ## [batch_size, num_tracks, num_tracks, 2] --> [batch_size, num_tracks, num_tracks-1, 2]
        rel_direction = delete_diagonal(rel_direction, batch_size, num_tracks)

        # Combine [batch_size, num_tracks, num_tracks - 1, 4]
        overall_grid = torch.cat([rel_position, rel_direction], dim=-1)

        # Get nearest n neighours
        rel_distance = torch.norm(rel_position, dim=-1)
        rel_distance = torch.nan_to_num(rel_distance, nan=1000)  # High dummy distance
        if (num_tracks - 1) < self.n:
            nearest_grid = torch.zeros((batch_size, num_tracks, self.n, self.input_dim), device=obs2.device)
            _, dist_index = torch.topk(-rel_distance, num_tracks-1, dim=-1)
            nearest_grid[:, :, :(num_tracks-1)] = torch.gather(overall_grid, 2, dist_index.unsqueeze(-1).repeat(1, 1, 1, self.input_dim))
        else:
            _, dist_index = torch.topk(-rel_distance, self.n, dim=-1)
            nearest_grid = torch.gather(overall_grid, 2, dist_index.unsqueeze(-1).repeat(1, 1, 1, self.input_dim))

        # Remove NaNs
        nearest_grid = torch.nan_to_num(nearest_grid)
        ## Embed top-n relative neighbour attributes
        nearest_grid = self.embedding(nearest_grid)
        nearest_grid = nearest_grid.view(batch_size * num_tracks, -1)

        ## Update interaction-encoder LSTM
        hidden_cell_stacked = self.pool_lstm(nearest_grid, hidden_cell_stacked)
        interaction_vector = self.hidden2pool(hidden_cell_stacked[0])

        self.hidden_cell_state[0] = list(hidden_cell_stacked[0])
        self.hidden_cell_state[1] = list(hidden_cell_stacked[1])
        return interaction_vector

class TrajectronPooling(torch.nn.Module):
    """ Interaction vector is obtained by sum-pooling the absolute coordinates and passed
        through Interaction encoder LSTM. Proposed in Trajectron
        
        Attributes
        ----------
        track_mask : Bool [num_tracks,]
            Mask of tracks visible at the current time-instant
            as well as tracks belonging to the particular scene 
        hidden_dim : Scalar
            Hidden-state dimension of interaction-encoder LSTM
        out_dim: Scalar
            Dimension of resultant interaction vector
    """
    def __init__(self, n=4, hidden_dim=256, out_dim=32, track_mask=None):
        super(TrajectronPooling, self).__init__()
        self.n = n
        self.out_dim = out_dim
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(8, out_dim),
            torch.nn.ReLU(),
        )
        self.hidden_dim = hidden_dim
        self.pool_lstm = torch.nn.LSTMCell(out_dim, hidden_dim)
        self.hidden2pool = torch.nn.Linear(hidden_dim, out_dim)
        self.track_mask = track_mask

    def reset(self, num_tracks, max_num_neigh, device):
        self.hidden_cell_state = [
            [torch.zeros(self.hidden_dim, device=device) for _ in range(num_tracks)],
            [torch.zeros(self.hidden_dim, device=device) for _ in range(num_tracks)],
        ]

    def forward(self, _, obs1, obs2):
        """ Forward function. All agents must belong to the same scene

        Parameters
        ----------
        obs1 :  Tensor [num_tracks, 2]
            x-y positions of all agents at previous time-step t-1
        obs2 :  Tensor [num_tracks, 2]
            x-y positions of all agents at current time-step t

        Returns
        -------
        interaction_vector : Tensor [num_tracks, self.out_dim]
            interaction vector of all agents in the scene
        """

        batch_size = obs2.size(0)
        num_tracks = obs2.size(1)

        hidden_cell_stacked = [
            torch.stack(self.hidden_cell_state[0], dim=0),
            torch.stack(self.hidden_cell_state[1], dim=0),
        ]

        ## Construct Neighbour grid using current position and velocity (We need "relative" !)
        curr_vel = obs2 - obs1
        curr_pos = obs2
        states = torch.cat([curr_pos, curr_vel], dim=-1)
        states = states.view(batch_size * num_tracks, -1)

        ## Only consider visible pedestrians
        neigh_grid = torch.zeros(batch_size * num_tracks, self.out_dim, device=obs1.device)
        nan_mask = torch.isnan(states).any(dim=-1)
        states_vis = states[~nan_mask]
        ## Get neighbour configuration embedding
        neigh_grid_vis = torch.stack([
            torch.cat([states_vis[i], torch.sum(states_vis[one_cold(i, len(states_vis))], dim=0)])
            for i in range(len(states_vis))], dim=0)
        neigh_grid_vis = self.embedding(neigh_grid_vis)
        neigh_grid[~nan_mask] = neigh_grid_vis

        ## Update interaction-encoder LSTM
        hidden_cell_stacked = self.pool_lstm(neigh_grid, hidden_cell_stacked)
        interaction_vector = self.hidden2pool(hidden_cell_stacked[0])

        self.hidden_cell_state[0] = list(hidden_cell_stacked[0])
        self.hidden_cell_state[1] = list(hidden_cell_stacked[1])

        return interaction_vector
