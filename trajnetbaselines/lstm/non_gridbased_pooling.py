from collections import defaultdict
import numpy as np

import torch

def one_cold(i, n):
    """Inverse one-hot encoding."""
    x = torch.ones(n, dtype=torch.bool)
    x[i] = 0
    return x

def front_ped(xy, other_xy, past_xy):
    """ Provides indices of neighbours in front of chosen pedestrian

    Parameters
    ----------
    xy :  Tensor [2,]
        x-y position of the chosen pedestrian at time t
    other_xy :  Tensor [num_tracks, 2]
        x-y position of all neighbours of the chosen pedestrian at current time-step t
    past_xy :  Tensor [2,]
        x-y position of the chosen pedestrian at time t-1

    Returns
    -------
    angle_index : Bool Tensor [num_tracks,]
        1 if the corresponding neighbour is present in front of current pedestrian
    """
    primary_direction = torch.atan2(xy[1] - past_xy[1], xy[0] - past_xy[0])
    relative_neigh = other_xy - xy
    neigh_direction = torch.atan2(relative_neigh[:, 1], relative_neigh[:, 0])
    angle_index = torch.abs((neigh_direction - primary_direction) * 180 / np.pi) < 90
    return angle_index

# 
def rel_obs(obs):
    """ Provides relative position of neighbours wrt one another

    Parameters
    ----------
    obs :  Tensor [num_tracks, 2]
        x-y positions of all agents

    Returns
    -------
    relative : Tensor [num_tracks, num_tracks, 2]
    """
    unfolded = obs.unsqueeze(0).repeat(obs.size(0), 1, 1)
    relative = unfolded - obs.unsqueeze(1)
    return relative

def rel_directional(obs1, obs2):
    """ Provides relative velocity of neighbours wrt one another

    Parameters
    ----------
    obs1 :  Tensor [num_tracks, 2]
        x-y positions of all agents at previous time-step t-1
    obs2 :  Tensor [num_tracks, 2]
        x-y positions of all agents at current time-step t

    Returns
    -------
    relative : Tensor [num_tracks, num_tracks, 2]
    """
    vel = obs2 - obs1
    unfolded = vel.unsqueeze(0).repeat(vel.size(0), 1, 1)
    relative = unfolded - vel.unsqueeze(1)
    return relative

class NN_Pooling(torch.nn.Module):
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
        super(NN_Pooling, self).__init__()
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

    def reset(self, _, device):
        self.track_mask = None

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

        num_tracks = obs2.size(0)

        # Get relative position of all agents wrt one another 
        # [num_tracks, 2] --> [num_tracks, num_tracks, 2]
        rel_position = rel_obs(obs2)
        # Deleting Diagonal (agents wrt themselves) 
        # [num_tracks, num_tracks, 2] --> [num_tracks, num_tracks - 1, 2]
        rel_position = rel_position[~torch.eye(num_tracks).bool()].reshape(num_tracks, num_tracks-1, 2)


        # Get relative velocities of all agents wrt one another 
        # [num_tracks, 2] --> [num_tracks, num_tracks, 2]
        rel_direction = rel_directional(obs1, obs2)
        # Deleting Diagonal (agents wrt themselves) 
        # [num_tracks, num_tracks, 2] --> [num_tracks, num_tracks - 1, 2]
        rel_direction = rel_direction[~torch.eye(num_tracks).bool()].reshape(num_tracks, num_tracks-1, 2)

        # Combine [num_tracks, num_tracks - 1, self.input_dim]
        overall_grid = torch.cat([rel_position, rel_direction], dim=2) if not self.no_velocity else rel_position

        # Get nearest n neighours
        if (num_tracks - 1) < self.n:
            nearest_grid = torch.zeros((num_tracks, self.n, self.input_dim), device=obs2.device)
            nearest_grid[:, :(num_tracks-1)] = overall_grid
        else:
            rel_distance = torch.norm(rel_position, dim=2)
            _, dist_index = torch.topk(-rel_distance, self.n, dim=1)
            nearest_grid = torch.gather(overall_grid, 1, dist_index.unsqueeze(2).repeat(1, 1, self.input_dim))

        ## Embed top-n relative neighbour attributes
        nearest_grid = self.embedding(nearest_grid)
        return nearest_grid.view(num_tracks, -1)

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
        self.spatial_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, mlp_dim_spatial),
            torch.nn.ReLU(),
        )

        self.vel_embedding = None
        if mlp_dim_vel:
            self.vel_embedding = torch.nn.Sequential(
                torch.nn.Linear(2, mlp_dim_vel),
                torch.nn.ReLU(),
            )

        #mlp_dim_hidden = mlp_dim - mlp_dim_spatial - mlp_dim_vel
        self.hidden_embedding = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, mlp_dim - mlp_dim_spatial - mlp_dim_vel),
            torch.nn.ReLU(),
        )
        self.out_projection = torch.nn.Linear(mlp_dim, self.out_dim)

    def reset(self, _, device):
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

        # Obtain and embed relative position
        # [num_tracks, 2] --> [num_tracks, num_tracks, 2]
        relative_obs = rel_obs(obs2)
        # [num_tracks, num_tracks, 2] --> [num_tracks, num_tracks, self.mlp_dim_spatial]
        spatial = self.spatial_embedding(relative_obs)

        # Embed hidden states
        # [num_tracks, hidden_dim] --> [num_tracks, mlp_dim_hidden]
        hidden = self.hidden_embedding(hidden_states)
        # [num_tracks, mlp_dim_hidden] --> [num_tracks, num_tracks, mlp_dim_hidden]
        hidden_unfolded = hidden.unsqueeze(0).repeat(hidden.size(0), 1, 1)

        # Obtain and embed relative position
        if self.vel_embedding is not None:
            # [num_tracks, 2] --> [num_tracks, num_tracks, 2]
            rel_vel = rel_directional(obs1, obs2)
            # [num_tracks, num_tracks, 2] --> [num_tracks, num_tracks, self.mlp_dim_vel]
            directional = self.vel_embedding(rel_vel*4)
            embedded = torch.cat([spatial, directional, hidden_unfolded], dim=2)
        else:
            embedded = torch.cat([spatial, hidden_unfolded], dim=2)

        # Max Pool
        pooled, _ = torch.max(embedded, dim=1)
        return self.out_projection(pooled)

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
    def __init__(self, hidden_dim=128, mlp_dim=128, mlp_dim_spatial=32, mlp_dim_vel=32, out_dim=None):
        super(AttentionMLPPooling, self).__init__()
        self.out_dim = out_dim or hidden_dim
        self.spatial_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, mlp_dim_spatial),
            torch.nn.ReLU(),
        )
        self.vel_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, mlp_dim_vel),
            torch.nn.ReLU(),
        )

        #mlp_dim_hidden = mlp_dim - mlp_dim_spatial - mlp_dim_vel
        self.hidden_embedding = None
        if mlp_dim_spatial + mlp_dim_vel < mlp_dim:
            self.hidden_embedding = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, mlp_dim - mlp_dim_spatial - mlp_dim_vel),
                torch.nn.ReLU(),
            )

        ## Attention Embeddings (Query, Key, Value)
        self.wq = torch.nn.Linear(mlp_dim, mlp_dim, bias=False)
        self.wk = torch.nn.Linear(mlp_dim, mlp_dim, bias=False)
        self.wv = torch.nn.Linear(mlp_dim, mlp_dim, bias=False)

        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=mlp_dim, num_heads=1)

        self.out_projection = torch.nn.Linear(mlp_dim, self.out_dim)

    def reset(self, _, device):
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

        # [num_tracks, 2] --> [num_tracks, num_tracks, 2]
        relative_obs = rel_obs(obs2)
        # [num_tracks, num_tracks, 2] --> [num_tracks, num_tracks, self.mlp_dim_spatial]
        spatial = self.spatial_embedding(relative_obs)

        # [num_tracks, 2] --> [num_tracks, num_tracks, 2]
        rel_vel = rel_directional(obs1, obs2)
        # [num_tracks, num_tracks, 2] --> [num_tracks, num_tracks, self.mlp_dim_vel]
        directional = self.vel_embedding(rel_vel*4)

        if self.hidden_embedding is not None:
            # [num_tracks, hidden_dim] --> [num_tracks, mlp_dim_hidden]
            hidden = self.hidden_embedding(hidden_states)
            # [num_tracks, mlp_dim_hidden] --> [num_tracks, num_tracks, mlp_dim_hidden]
            hidden_unfolded = hidden.unsqueeze(0).repeat(hidden.size(0), 1, 1)
            embedded = torch.cat([spatial, directional, hidden_unfolded], dim=2)
        else:
            embedded = torch.cat([spatial, directional], dim=2)

        ## Attention
        # [num_tracks, num_tracks, mlp_dim] --> [num_tracks, num_tracks, mlp_dim]
        # i.e. [batch, seq, mlp_dim] --> [seq, batch, mlp_dim]
        embedded = embedded.transpose(0, 1)
        query = self.wq(embedded)
        key = self.wk(embedded)
        value = self.wv(embedded)
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = attn_output.transpose(0, 1)

        return self.out_projection(attn_output[torch.eye(len(obs2)).bool()])

class DirectionalMLPPooling(torch.nn.Module):
    """ Interaction vector is obtained by max-pooling the embeddings of relative coordinates
        and relative velocity of all neighbours.
        
        Attributes
        ----------
        mlp_dim : Scalar
            Embedding dimension of each neighbour
        mlp_dim_spatial : Scalar
            Embedding dimension of relative spatial coordinates
        out_dim: Scalar
            Dimension of resultant interaction vector
    """
    def __init__(self, hidden_dim=128, mlp_dim=128, mlp_dim_spatial=64, out_dim=None):
        super(DirectionalMLPPooling, self).__init__()
        self.out_dim = out_dim or hidden_dim
        self.spatial_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, mlp_dim_spatial),
            torch.nn.ReLU(),
        )

        # mlp_dim_vel = mlp_dim - mlp_dim_spatial
        self.directional_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, mlp_dim - mlp_dim_spatial),
            torch.nn.ReLU(),
        )
        self.out_projection = torch.nn.Linear(mlp_dim, self.out_dim)

    def reset(self, _, device):
        self.track_mask = None

    def forward(self, _, obs1, obs2):
        # [num_tracks, 2] --> [num_tracks, num_tracks, 2]
        relative_obs = rel_obs(obs2)
        # [num_tracks, num_tracks, 2] --> [num_tracks, num_tracks, self.mlp_dim_spatial]
        spatial = self.spatial_embedding(relative_obs)

        # [num_tracks, 2] --> [num_tracks, num_tracks, 2]
        rel_vel = rel_directional(obs1, obs2)
        # [num_tracks, num_tracks, 2] --> [num_tracks, num_tracks, mlp_dim_vel]
        directional = self.directional_embedding(rel_vel*4)

        embedded = torch.cat([spatial, directional], dim=2)
        pooled, _ = torch.max(embedded, dim=1)
        return self.out_projection(pooled)

class NN_LSTM(torch.nn.Module):
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

    def __init__(self, n=4, hidden_dim=256, out_dim=32, track_mask=None):
        super(NN_LSTM, self).__init__()
        self.n = n
        self.out_dim = out_dim
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(4, int(out_dim/self.n)),
            torch.nn.ReLU(),
        )
        self.hidden_dim = hidden_dim
        self.pool_lstm = torch.nn.LSTMCell(out_dim, hidden_dim)
        self.hidden2pool = torch.nn.Linear(hidden_dim, out_dim)
        self.track_mask = track_mask

    def reset(self, num_tracks, device):
        self.hidden_cell_state = (
            [torch.zeros(self.hidden_dim, device=device) for _ in range(num_tracks)],
            [torch.zeros(self.hidden_dim, device=device) for _ in range(num_tracks)],
        )

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
        num_tracks = obs2.size(0)

        ## If only primary pedestrian of the scene present
        if torch.sum(self.track_mask).item() == 1:
            return torch.zeros(num_tracks, self.out_dim, device=obs1.device)

        hidden_cell_stacked = [
            torch.stack([h for m, h in zip(self.track_mask, self.hidden_cell_state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(self.track_mask, self.hidden_cell_state[1]) if m], dim=0),
        ]

        # Get relative position of all agents wrt one another 
        # [num_tracks, 2] --> [num_tracks, num_tracks, 2]
        rel_position = rel_obs(obs2)
        # Deleting Diagonal (agents wrt themselves) 
        # [num_tracks, num_tracks, 2] --> [num_tracks, num_tracks - 1, 2]
        rel_position = rel_position[~torch.eye(num_tracks).bool()].reshape(num_tracks, num_tracks-1, 2)


        # Get relative velocities of all agents wrt one another 
        # [num_tracks, 2] --> [num_tracks, num_tracks, 2]
        rel_direction = rel_directional(obs1, obs2)
        # Deleting Diagonal (agents wrt themselves) 
        # [num_tracks, num_tracks, 2] --> [num_tracks, num_tracks - 1, 2]
        rel_direction = rel_direction[~torch.eye(num_tracks).bool()].reshape(num_tracks, num_tracks-1, 2)

        # Combine [num_tracks, num_tracks - 1, 4]
        overall_grid = torch.cat([rel_position, rel_direction], dim=2)

        # Get nearest n neighours
        if (num_tracks - 1) < self.n:
            nearest_grid = torch.zeros((num_tracks, self.n, 4), device=obs2.device)
            nearest_grid[:, :(num_tracks-1)] = overall_grid
        else:
            rel_distance = torch.norm(rel_position, dim=2)
            _, dist_index = torch.topk(-rel_distance, self.n, dim=1)
            nearest_grid = torch.gather(overall_grid, 1, dist_index.unsqueeze(2).repeat(1, 1, 4))

        ## Embed top-n relative neighbour attributes
        nearest_grid = self.embedding(nearest_grid)
        nearest_grid = nearest_grid.view(num_tracks, -1)

        ## Update interaction-encoder LSTM
        hidden_cell_stacked = self.pool_lstm(nearest_grid, hidden_cell_stacked)
        interaction_vector = self.hidden2pool(hidden_cell_stacked[0])

        ## Save hidden-cell-states
        mask_index = [i for i, m in enumerate(self.track_mask) if m]
        for i, h, c in zip(mask_index,
                           hidden_cell_stacked[0],
                           hidden_cell_stacked[1]):
            self.hidden_cell_state[0][i] = h
            self.hidden_cell_state[1][i] = c

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

    def reset(self, num_tracks, device):
        self.hidden_cell_state = (
            [torch.zeros(self.hidden_dim, device=device) for _ in range(num_tracks)],
            [torch.zeros(self.hidden_dim, device=device) for _ in range(num_tracks)],
        )

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

        num_tracks = obs2.size(0)

        ## If only primary pedestrian of the scene present
        if torch.sum(self.track_mask).item() == 1:
            return torch.zeros(num_tracks, self.out_dim, device=obs1.device)

        hidden_cell_stacked = [
            torch.stack([h for m, h in zip(self.track_mask, self.hidden_cell_state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(self.track_mask, self.hidden_cell_state[1]) if m], dim=0),
        ]

        ## Construct neighbour grid using current position and velocity
        curr_vel = obs2 - obs1
        curr_pos = obs2
        states = torch.cat([curr_pos, curr_vel], dim=1)
        neigh_grid = torch.stack([
            torch.cat([states[i], torch.sum(states[one_cold(i, num_tracks)], dim=0)])
            for i in range(num_tracks)], dim=0)
        neigh_grid = self.embedding(neigh_grid).view(num_tracks, -1)

        ## Update interaction-encoder LSTM
        hidden_cell_stacked = self.pool_lstm(neigh_grid, hidden_cell_stacked)
        interaction_vector = self.hidden2pool(hidden_cell_stacked[0])

        ## Save hidden-cell-states
        mask_index = [i for i, m in enumerate(self.track_mask) if m]
        for i, h, c in zip(mask_index,
                           hidden_cell_stacked[0],
                           hidden_cell_stacked[1]):
            self.hidden_cell_state[0][i] = h
            self.hidden_cell_state[1][i] = c

        return interaction_vector

class SAttention_fast(torch.nn.Module):
    """ Interaction vector is obtained by attention-weighting the embeddings of relative coordinates obtained
        using Interaction Encoder LSTM. Proposed in S-Attention
        
        Attributes
        ----------
        track_mask : Bool [num_tracks,]
            Mask of tracks visible at the current time-instant
            as well as tracks belonging to the particular scene 
        spatial_dim : Scalar
            Embedding dimension of relative position of neighbour       
        hidden_dim : Scalar
            Hidden-state dimension of interaction-encoder LSTM
        out_dim: Scalar
            Dimension of resultant interaction vector
    """

    def __init__(self, n=4, spatial_dim=32, hidden_dim=256, out_dim=32, track_mask=None):
        super(SAttention_fast, self).__init__()
        self.n = n
        if out_dim is None:
            out_dim = hidden_dim
        self.out_dim = out_dim
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(2, spatial_dim),
            torch.nn.ReLU(),
        )
        self.spatial_dim = spatial_dim
        self.hiddentospat = torch.nn.Linear(hidden_dim, spatial_dim)
        self.hidden_dim = hidden_dim
        self.pool_lstm = torch.nn.LSTMCell(spatial_dim, spatial_dim)
        self.track_mask = track_mask

        ## Attention Embeddings (Query, Key, Value)
        self.wq = torch.nn.Linear(spatial_dim, spatial_dim, bias=False)
        self.wk = torch.nn.Linear(spatial_dim, spatial_dim, bias=False)
        self.wv = torch.nn.Linear(spatial_dim, spatial_dim, bias=False)
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=spatial_dim, num_heads=1)

        self.out_projection = torch.nn.Linear(spatial_dim, self.out_dim)

    def reset(self, num_tracks, device):
        self.hidden_cell_state = (
            torch.zeros((num_tracks, num_tracks, self.spatial_dim), device=device),
            torch.zeros((num_tracks, num_tracks, self.spatial_dim), device=device),
        )

    def forward(self, hidden_state, obs1, obs2):
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

        # Make Adjacency Matrix of visible pedestrians
        num_tracks_in_batch = len(self.track_mask)
        adj_vector = self.track_mask.unsqueeze(1).float()
        adj_matrix = torch.mm(adj_vector, adj_vector.transpose(0, 1)).bool()
        # Remove reference to self
        adj_matrix[torch.eye(num_tracks_in_batch).bool()] = False
        ## Filter hidden cell state
        hidden_cell_stacked = [self.hidden_cell_state[0][adj_matrix], self.hidden_cell_state[1][adj_matrix]]

        ## Current Pedestrians in Scene
        num_tracks = obs2.size(0)

        ## If only primary pedestrian of the scene present
        if torch.sum(self.track_mask).item() == 1:
            return torch.zeros(num_tracks, self.out_dim, device=obs1.device)


        # [num_tracks, 2] --> [num_tracks, num_tracks, 2]
        rel_position = rel_obs(obs2)
        # Deleting Diagonal (agents wrt themselves)
        # [num_tracks, num_tracks, 2] --> [num_tracks * (num_tracks - 1), 2]
        rel_position = rel_position[~torch.eye(num_tracks).bool()]
        rel_embed = self.embedding(rel_position)

        ## Update interaction-encoder LSTMs
        pool_hidden_states = self.pool_lstm(rel_embed, hidden_cell_stacked)
        
        ## Save hidden-cell-states
        self.hidden_cell_state[0][adj_matrix] = pool_hidden_states[0]
        self.hidden_cell_state[1][adj_matrix] = pool_hidden_states[1]

        ## Attention between hidden_states of motion encoder & hidden_states of interactions encoders ##
        
        # Embed Hidden-state of Motion LSTM: [num_tracks, hidden_dim] --> [num_tracks, self.spatial_dim]
        hidden_state_spat = self.hiddentospat(hidden_state)
        
        # Concat Hidden-state of Motion LSTM to Hidden-state of Interaction LSTMs
        # embedded.shape = [num_tracks, num_tracks, self.spatial_dim]
        embedded = torch.cat([hidden_state_spat.unsqueeze(1), pool_hidden_states[0].reshape(num_tracks, num_tracks-1, self.spatial_dim)], dim=1)
        
        ## Attention
        # [num_tracks, num_tracks, self.spatial_dim] --> [num_tracks, num_tracks, self.spatial_dim]
        # i.e. [batch, seq, self.spatial_dim] --> [seq, batch, self.spatial_dim]
        embedded = embedded.transpose(0, 1)
        query = self.wq(embedded)
        key = self.wk(embedded)
        value = self.wv(embedded)
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_vectors = attn_output[0]
        return self.out_projection(attn_vectors)
