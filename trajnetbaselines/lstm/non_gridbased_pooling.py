from collections import defaultdict
import numpy as np

import torch

def one_cold(i, n):
    """Inverse one-hot encoding."""
    x = torch.ones(n, dtype=torch.bool)
    x[i] = 0
    return x

# Provides indices of neighbours in front of current ped
def front_ped(xy, other_xy, past_xy):
    primary_direction = torch.atan2(xy[1] - past_xy[1], xy[0] - past_xy[0])
    relative_neigh = other_xy - xy
    neigh_direction = torch.atan2(relative_neigh[:, 1], relative_neigh[:, 0])
    angle_index = torch.abs((neigh_direction - primary_direction) * 180 / np.pi) < 90
    return angle_index

# Provides relative position of neighbours
def rel_obs(obs):
    unfolded = obs.unsqueeze(0).repeat(obs.size(0), 1, 1)
    relative = unfolded - obs.unsqueeze(1)
    return relative

# Provides relative velocity of neighbours
def rel_directional(obs1, obs2):
    vel = obs2 - obs1
    unfolded = vel.unsqueeze(0).repeat(vel.size(0), 1, 1)
    relative = unfolded - vel.unsqueeze(1)
    return relative

class NN_Pooling(torch.nn.Module):
    """ Interaction vector is obtained by concatenating the relative coordinates of
        top-n neighbours filtered according to criterion (here, euclidean distance)
        Parameters:
        n : number of neighbours to concatenate
        no_vel: bool to indicate whether to consider relative velocity
        out_dim: dimension of resultant interaction vector
    """
    def __init__(self, n=4, out_dim=32, no_vel=False):
        super(NN_Pooling, self).__init__()
        self.n = n
        self.out_dim = out_dim
        self.no_velocity = no_vel
        self.input_dim = 2 if self.no_velocity else 4

        ## Fixed size embedding. Each neighbour gets equal-sized representation
        ## Currently, n must divide out_dim
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, int(out_dim/self.n)),
            torch.nn.ReLU(),
        )

    def forward(self, _, obs1, obs2):
        obs = obs2
        n = obs.size(0)

        ## Get relative velocities
        rel_direction = rel_directional(obs1, obs2)
        ## Deleting Diagonal (Ped wrt itself)
        rel_direction = rel_direction[~torch.eye(n).bool()].reshape(n, n-1, 2)

        nearest_grid = torch.stack([
            self.nearest(obs[i], obs[one_cold(i, n)], obs1[i], rel_direction[i])
            for i in range(n)], dim=0)

        nearest_grid = nearest_grid.view(n, self.n, self.input_dim)
        nearest_grid = self.embedding(nearest_grid)
        return nearest_grid.view(n, -1)

    def nearest(self, xy, other_xy, past_xy, rel_direction):
        ## Calculates the top-n neighbours
        """
        xy: current position of current ped
        other_xy: current positions of neighbours of current ped
        past_xy: previous position of current ped
        rel_direction: relative velocity of neighbours w.r.t. current ped
        """

        ## Considers Pedestrians in Front Only
        # angle_index = front_ped(xy, other_xy, past_xy)
        # other_xy = other_xy[angle_index]
        # rel_direction = rel_direction[angle_index]


        ## Consider Nearest n based on Euclidean distance
        nearest = self.dist(xy, other_xy, rel_direction)
        return nearest

    def dist(self, xy, other_xy, rel_direction):
        rel_position = other_xy - xy

        ## If more than n, select nearest n
        if rel_position.shape[0] >= self.n:
            rel_distance = torch.norm(rel_position, dim=1)
            _, dist_index = torch.topk(-rel_distance, self.n)
            nearest_pos = rel_position[dist_index[:self.n]]
            nearest_vel = rel_direction[dist_index[:self.n]]
        else:
            nearest_pos = torch.zeros((self.n, 2), device=xy.device)
            nearest_vel = torch.zeros((self.n, 2), device=xy.device)
            for i, row in enumerate(rel_position):
                nearest_pos[i] = row
            for i, row in enumerate(rel_direction):
                nearest_vel[i] = row

        if self.no_velocity:
            nearest = nearest_pos
        else:
            nearest = torch.cat([nearest_pos, nearest_vel], dim=1)

        return nearest.view(-1)

    def make_grid(self, obs):
        ## Provides NN interaction vector for all time-steps at once
        timesteps = obs.size(0)
        overall_grid = []
        for i in range(1, timesteps):
            obs1 = obs[i-1]
            obs2 = obs[i]
            grid = torch.zeros((len(obs2), self.out_dim), device=obs.device)
            ## Remove NANs
            track_mask = (torch.isnan(obs1[:, 0]) + torch.isnan(obs2[:, 0])) == 0
            obs1, obs2 = obs1[track_mask], obs2[track_mask]

            ## Pooling maps of only visible pedestrians.
            visible_grid = self.forward(None, obs1, obs2)
            grid[track_mask] = visible_grid
            overall_grid.append(grid)

        return torch.stack(overall_grid, dim=0)

class HiddenStateMLPPooling(torch.nn.Module):
    """ Interaction vector is obtained by max-pooling the embeddings of relative coordinates
        and hidden-state of all neighbours. Proposed in Social GAN
        Parameters:
        mlp_dim_spatial: embedding size of relative spatial coordinates
        mlp_dim_vel: embedding size of relative velocity coordinates
        mlp_dim - (mlp_dim_spatial + mlp_dim_vel): embedding size of hidden-state
        out_dim: dimension of resultant interaction vector
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

        self.hidden_embedding = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, mlp_dim - mlp_dim_spatial - mlp_dim_vel),
            torch.nn.ReLU(),
        )
        self.out_projection = torch.nn.Linear(mlp_dim, self.out_dim)

    def forward(self, hidden_states, obs1, obs):
        relative_obs = rel_obs(obs)
        spatial = self.spatial_embedding(relative_obs)
        hidden = self.hidden_embedding(hidden_states)
        hidden_unfolded = hidden.unsqueeze(0).repeat(hidden.size(0), 1, 1)

        if self.vel_embedding is not None:
            rel_vel = rel_directional(obs1, obs)
            directional = self.vel_embedding(rel_vel*4)
            embedded = torch.cat([spatial, directional, hidden_unfolded], dim=2)
        else:
            embedded = torch.cat([spatial, hidden_unfolded], dim=2)

        pooled, _ = torch.max(embedded, dim=1)
        return self.out_projection(pooled)

class AttentionMLPPooling(torch.nn.Module):
    """ Interaction vector is obtained by attention-weighting the embeddings of relative coordinates
        and hidden-state of all neighbours. Proposed in S-BiGAT
        Parameters:
        mlp_dim_spatial: embedding size of relative spatial coordinates
        mlp_dim_vel: embedding size of relative velocity coordinates
        mlp_dim - (mlp_dim_spatial + mlp_dim_vel): embedding size of hidden-state
        out_dim: dimension of resultant interaction vector
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

        self.hidden_embedding = None
        if mlp_dim_spatial + mlp_dim_vel < mlp_dim:
            self.hidden_embedding = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, mlp_dim - mlp_dim_spatial - mlp_dim_vel),
                torch.nn.ReLU(),
            )
        self.wq = torch.nn.Linear(mlp_dim, mlp_dim, bias=False)
        self.wk = torch.nn.Linear(mlp_dim, mlp_dim, bias=False)
        self.wv = torch.nn.Linear(mlp_dim, mlp_dim, bias=False)

        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=mlp_dim, num_heads=1)

        self.out_projection = torch.nn.Linear(mlp_dim, self.out_dim)

    def forward(self, hidden_states, obs1, obs):
        relative_obs = rel_obs(obs)
        spatial = self.spatial_embedding(relative_obs)
        rel_vel = rel_directional(obs1, obs)
        directional = self.vel_embedding(rel_vel*4)

        if self.hidden_embedding is not None:
            hidden = self.hidden_embedding(hidden_states)
            hidden_unfolded = hidden.unsqueeze(0).repeat(hidden.size(0), 1, 1)
            embedded = torch.cat([spatial, directional, hidden_unfolded], dim=2)
        else:
            embedded = torch.cat([spatial, directional], dim=2)

        embedded = embedded.transpose(0, 1)
        query = self.wq(embedded)
        key = self.wk(embedded)
        value = self.wv(embedded)
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = attn_output.transpose(0, 1)

        return self.out_projection(attn_output[torch.eye(len(obs)).bool()])

class DirectionalMLPPooling(torch.nn.Module):
    """ Interaction vector is obtained by max-pooling the embeddings of relative coordinates
        of all neighbours
        Parameters:
        mlp_dim_spatial: embedding size of relative spatial coordinates
        mlp_dim - (mlp_dim_spatial): embedding size of relative velocity coordinates
        out_dim: dimension of resultant interaction vector
    """
    def __init__(self, hidden_dim=128, mlp_dim=128, mlp_dim_spatial=64, out_dim=None):
        super(DirectionalMLPPooling, self).__init__()
        self.out_dim = out_dim or hidden_dim
        self.spatial_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, mlp_dim_spatial),
            torch.nn.ReLU(),
        )
        self.directional_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, mlp_dim - mlp_dim_spatial),
            torch.nn.ReLU(),
        )
        self.out_projection = torch.nn.Linear(mlp_dim, self.out_dim)

    def forward(self, _, obs1, obs2):
        relative_obs = rel_obs(obs2)
        spatial = self.spatial_embedding(relative_obs)

        rel_vel = rel_directional(obs1, obs2)
        directional = self.directional_embedding(rel_vel*4)

        embedded = torch.cat([spatial, directional], dim=2)
        pooled, _ = torch.max(embedded, dim=1)
        return self.out_projection(pooled)

class NN_LSTM(torch.nn.Module):
    """ Interaction vector is obtained by concatenating the relative coordinates of
        top-n neighbours filtered according to criterion (here, euclidean distance).
        The concatenated vector is passed through an LSTM.
        Parameters:
        n : number of neighbours to concatenate
        track_mask: mask of pedestrians visible at a particular time-instant
        out_dim: dimension of resultant interaction vector
    """
    def __init__(self, n=4, hidden_dim=256, out_dim=32, track_mask=None):
        super(NN_LSTM, self).__init__()
        self.n = n
        if out_dim is None:
            out_dim = hidden_dim
        self.out_dim = out_dim
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(4, int(out_dim/self.n)),
            torch.nn.ReLU(),
        )
        self.hidden_dim = hidden_dim
        self.pool_lstm = torch.nn.LSTMCell(out_dim, hidden_dim)
        self.hidden2pool = torch.nn.Linear(hidden_dim, out_dim)
        self.track_mask = track_mask

    def reset(self, num, device):
        self.hidden_cell_state = (
            [torch.zeros(self.hidden_dim, device=device) for _ in range(num)],
            [torch.zeros(self.hidden_dim, device=device) for _ in range(num)],
        )

    def forward(self, _, obs1, obs2):
        hidden_cell_stacked = [
            torch.stack([h for m, h in zip(self.track_mask, self.hidden_cell_state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(self.track_mask, self.hidden_cell_state[1]) if m], dim=0),
        ]

        obs = obs2
        n = obs.size(0)
        ## If only primary present
        if not torch.any(self.track_mask[1:]):
            return torch.zeros(n, self.out_dim, device=obs1.device)

        rel_direction = rel_directional(obs1, obs2)
        ## Deleting Diagonal (Ped wrt itself)
        rel_direction = rel_direction[~torch.eye(n).bool()].reshape(n, n-1, 2)

        nearest_grid = torch.stack([
            self.nearest(obs[i], obs[one_cold(i, n)], obs1[i], rel_direction[i])
            for i in range(n)], dim=0)
        nearest_grid = nearest_grid.view(n, self.n, 4)
        nearest_grid = self.embedding(nearest_grid).view(n, -1)


        hidden_cell_stacked = self.pool_lstm(nearest_grid, hidden_cell_stacked)
        normal_masked = self.hidden2pool(hidden_cell_stacked[0])
        mask_index = [i for i, m in enumerate(self.track_mask) if m]
        for i, h, c in zip(mask_index,
                           hidden_cell_stacked[0],
                           hidden_cell_stacked[1]):
            self.hidden_cell_state[0][i] = h
            self.hidden_cell_state[1][i] = c

        return normal_masked

    def nearest(self, xy, other_xy, past_xy, rel_direction):

        ## Considers Pedestrians in Front Only
        # angle_index = front_ped(xy, other_xy, past_xy)
        # other_xy = other_xy[angle_index]
        # rel_direction = rel_direction[angle_index]

        ## Consider Nearest n
        nearest = self.dist(xy, other_xy, rel_direction)
        return nearest

    def dist(self, xy, other_xy, rel_direction):
        rel_position = other_xy - xy
        ## If more than n, select nearest n
        if rel_position.shape[0] >= self.n:
            rel_distance = torch.norm(rel_position, dim=1)
            _, dist_index = torch.topk(-rel_distance, self.n)
            nearest_pos = rel_position[dist_index[:self.n]]
            nearest_vel = rel_direction[dist_index[:self.n]]
        else:
            nearest_pos = torch.zeros((self.n, 2), device=xy.device)
            nearest_vel = torch.zeros((self.n, 2), device=xy.device)
            for i, row in enumerate(rel_position):
                nearest_pos[i] = row
            for i, row in enumerate(rel_direction):
                nearest_vel[i] = row

        nearest = torch.cat([nearest_pos, nearest_vel], dim=1)
        return nearest.view(-1)

class TrajectronPooling(torch.nn.Module):
    """ Interaction vector is obtained by sum-pooling the absolute coordinates and passed
        through LSTM-Encoder. Proposed in Trajectron
        Parameters:
        n : number of neighbours (to remove, depends on scene)
        track_mask: mask of pedestrians visible at a particular time-instant
        out_dim: dimension of resultant interaction vector
    """
    def __init__(self, n=4, hidden_dim=256, out_dim=32, track_mask=None):
        super(TrajectronPooling, self).__init__()
        self.n = n
        if out_dim is None:
            out_dim = hidden_dim
        self.out_dim = out_dim
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(8, out_dim),
            torch.nn.ReLU(),
        )
        self.hidden_dim = hidden_dim
        self.pool_lstm = torch.nn.LSTMCell(out_dim, hidden_dim)
        self.hidden2pool = torch.nn.Linear(hidden_dim, out_dim)
        self.track_mask = track_mask

    def reset(self, num, device):
        self.hidden_cell_state = (
            [torch.zeros(self.hidden_dim, device=device) for _ in range(num)],
            [torch.zeros(self.hidden_dim, device=device) for _ in range(num)],
        )

    def forward(self, _, obs1, obs2):
        hidden_cell_stacked = [
            torch.stack([h for m, h in zip(self.track_mask, self.hidden_cell_state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(self.track_mask, self.hidden_cell_state[1]) if m], dim=0),
        ]

        obs = obs2
        n = obs.size(0)
        ## If only primary present
        if not torch.any(self.track_mask[1:]):
            return torch.zeros(n, self.out_dim, device=obs2.device)

        vel = obs2 - obs1
        pos = obs2
        states = torch.cat([pos, vel], dim=1)
        nearest_grid = torch.stack([
            torch.cat([states[i], torch.sum(states[one_cold(i, n)], dim=0)])
            for i in range(n)], dim=0)

        nearest_grid = self.embedding(nearest_grid).view(n, -1)

        hidden_cell_stacked = self.pool_lstm(nearest_grid, hidden_cell_stacked)
        normal_masked = self.hidden2pool(hidden_cell_stacked[0])
        mask_index = [i for i, m in enumerate(self.track_mask) if m]
        for i, h, c in zip(mask_index,
                           hidden_cell_stacked[0],
                           hidden_cell_stacked[1]):
            self.hidden_cell_state[0][i] = h
            self.hidden_cell_state[1][i] = c

        return normal_masked

class SAttention(torch.nn.Module):
    """ Interaction vector is obtained by attention-weighting the embeddings of relative coordinates obtained
        using LSTM-Encoder. Proposed in S-Attention
        Parameters:
        n : number of neighbours (to remove, depends on scene)
        track_mask: mask of pedestrians visible at a particular time-instant
        out_dim: dimension of resultant interaction vector
    """
    def __init__(self, n=4, hidden_dim=256, out_dim=32, track_mask=None):
        super(SAttention, self).__init__()
        self.n = n
        if out_dim is None:
            out_dim = hidden_dim
        self.out_dim = out_dim
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(2, out_dim),
            torch.nn.ReLU(),
        )
        self.hidden_dim = hidden_dim
        self.pool_lstm = torch.nn.LSTMCell(out_dim, hidden_dim)
        self.track_mask = track_mask

        self.wq = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wk = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wv = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1)

        self.out_projection = torch.nn.Linear(hidden_dim, self.out_dim)

    def reset(self, num, device):
        self.hidden_cell_state = (
            torch.zeros((num, num, self.hidden_dim), device=device),
            torch.zeros((num, num, self.hidden_dim), device=device),
        )

    def forward(self, hidden_state, obs1, obs2):

        # Overall Adjacency Matrix
        n_tot = len(self.track_mask)
        ## Make track_mask wrt each pedestrian (make reference to self False)
        adj_vector = torch.zeros((n_tot, 1), device=obs1.device)
        adj_vector[self.track_mask] = 1
        adj_matrix = torch.mm(adj_vector, adj_vector.transpose(0, 1)).bool()
        adj_matrix[torch.eye(n_tot).bool()] = False
        ## Filter hidden cell state
        hidden_cell_stacked = [self.hidden_cell_state[0][adj_matrix], self.hidden_cell_state[1][adj_matrix]]

        ## Current Pedestrians in Scene
        obs = obs2
        n = obs.size(0)
        ## If only primary present
        if not torch.any(self.track_mask[1:]):
            return torch.zeros(n, self.out_dim, device=obs2.device)

        ## relative possitions
        rel_position = rel_obs(obs)
        ## Deleting Diagonal (Ped wrt itself)
        rel_position = rel_position[~torch.eye(n).bool()]
        rel_embed = self.embedding(rel_position)

        pool_hidden_states = self.pool_lstm(rel_embed, hidden_cell_stacked)
        ## Update Hidden State
        self.hidden_cell_state[0][adj_matrix] = pool_hidden_states[0]
        self.hidden_cell_state[1][adj_matrix] = pool_hidden_states[1]

        ## Attention between pool_hidden_states & lstm_hidden_state
        embedded = torch.cat([hidden_state.unsqueeze(1), pool_hidden_states[0].reshape(n, n-1, self.hidden_dim)], dim=1)
        ## EMBEDDED
        embedded = embedded.transpose(0, 1) ## Batch_size to second dimension
        query = self.wq(embedded)
        key = self.wk(embedded)
        value = self.wv(embedded)
        attn_output, _ = self.multihead_attn(query, key, value)
        nearest_grid = attn_output[0]
        return self.out_projection(nearest_grid)

class NN_Tag_Pooling(torch.nn.Module):
    def __init__(self, n=4, hidden_dim=128, out_dim=32, no_vel=False):
        super(NN_Tag_Pooling, self).__init__()
        self.n = n
        self.out_dim = out_dim
        self.no_velocity = no_vel
        self.input_dim = 3 if self.no_velocity else 6

        ## Fixed size embedding. Each neighbour gets equal-sized representation
        ## Currently, n must divide out_dim
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, int(out_dim/self.n)),
            torch.nn.ReLU(),
        )

    def forward(self, _, obs1, obs2):
        obs = obs2
        n = obs.size(0)

        ## Get relative velocities
        rel_direction = rel_directional(obs1, obs2)
        ## Deleting Diagonal (Ped wrt itself)
        rel_direction = rel_direction[~torch.eye(n).bool()].reshape(n, n-1, 2)

        nearest_grid = torch.stack([
            self.nearest(obs[i], obs[one_cold(i, n)], obs1[i], rel_direction[i])
            for i in range(n)], dim=0)

        nearest_grid = nearest_grid.view(n, self.n, self.input_dim)
        nearest_grid = self.embedding(nearest_grid)
        return nearest_grid.view(n, -1)

    def nearest(self, xy, other_xy, past_xy, rel_direction):
        ## Calculates the top-n neighbours
        """
        xy: current position of current ped
        other_xy: current positions of neighbours of current ped
        past_xy: previous position of current ped
        rel_direction: relative velocity of neighbours w.r.t. current ped
        """

        ## Considers Pedestrians in Front Only
        # angle_index = front_ped(xy, other_xy, past_xy)
        # other_xy = other_xy[angle_index]
        # rel_direction = rel_direction[angle_index]


        ## Consider Nearest n based on Euclidean distance
        nearest = self.dist(xy, other_xy, rel_direction)
        return nearest

    def dist(self, xy, other_xy, rel_direction):
        rel_position = other_xy - xy
        visible_tag = torch.ones((len(rel_position), 1), device=xy.device)
        rel_position = torch.cat([rel_position, visible_tag], dim=1)
        rel_direction = torch.cat([rel_direction, visible_tag], dim=1)

        ## If more than n, select nearest n
        if rel_position.shape[0] >= self.n:
            rel_distance = torch.norm(rel_position, dim=1)
            _, dist_index = torch.topk(-rel_distance, self.n)
            nearest_pos = rel_position[dist_index[:self.n]]
            nearest_vel = rel_direction[dist_index[:self.n]]
        else:
            nearest_pos = torch.zeros((self.n, 3), device=xy.device)
            nearest_vel = torch.zeros((self.n, 3), device=xy.device)
            for i, row in enumerate(rel_position):
                nearest_pos[i] = row
            for i, row in enumerate(rel_direction):
                nearest_vel[i] = row

        if self.no_velocity:
            nearest = nearest_pos
        else:
            nearest = torch.cat([nearest_pos, nearest_vel], dim=1)

        return nearest.view(-1)


class SAttention_fast(torch.nn.Module):
    """ Interaction vector is obtained by attention-weighting the embeddings of relative coordinates obtained
        using LSTM-Encoder. Proposed in S-Attention
        Parameters:
        n : number of neighbours (to remove, depends on scene)
        track_mask: mask of pedestrians visible at a particular time-instant
        out_dim: dimension of resultant interaction vector
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

        self.wq = torch.nn.Linear(spatial_dim, spatial_dim, bias=False)
        self.wk = torch.nn.Linear(spatial_dim, spatial_dim, bias=False)
        self.wv = torch.nn.Linear(spatial_dim, spatial_dim, bias=False)

        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=spatial_dim, num_heads=1)

        self.out_projection = torch.nn.Linear(spatial_dim, self.out_dim)

    def reset(self, num, device):
        self.hidden_cell_state = (
            torch.zeros((num, num, self.spatial_dim), device=device),
            torch.zeros((num, num, self.spatial_dim), device=device),
        )

    def forward(self, hidden_state, obs1, obs2):

        # Overall Adjacency Matrix
        n_tot = len(self.track_mask)
        ## Make track_mask wrt each pedestrian (make reference to self False)
        adj_vector = torch.zeros((n_tot, 1), device=obs1.device)
        adj_vector[self.track_mask] = 1
        adj_matrix = torch.mm(adj_vector, adj_vector.transpose(0, 1)).bool()
        adj_matrix[torch.eye(n_tot).bool()] = False
        ## Filter hidden cell state
        hidden_cell_stacked = [self.hidden_cell_state[0][adj_matrix], self.hidden_cell_state[1][adj_matrix]]

        ## Current Pedestrians in Scene
        obs = obs2
        n = obs.size(0)
        ## If only primary present
        if not torch.any(self.track_mask[1:]):
            return torch.zeros(n, self.out_dim, device=obs1.device)

        ## relative possitions
        rel_position = rel_obs(obs)
        ## Deleting Diagonal (Ped wrt itself)
        rel_position = rel_position[~torch.eye(n).bool()]
        rel_embed = self.embedding(rel_position)

        pool_hidden_states = self.pool_lstm(rel_embed, hidden_cell_stacked)
        ## Update Hidden State
        self.hidden_cell_state[0][adj_matrix] = pool_hidden_states[0]
        self.hidden_cell_state[1][adj_matrix] = pool_hidden_states[1]

        ## Attention between pool_hidden_states & lstm_hidden_state
        hidden_state_spat = self.hiddentospat(hidden_state)
        embedded = torch.cat([hidden_state_spat.unsqueeze(1), pool_hidden_states[0].reshape(n, n-1, self.spatial_dim)], dim=1)
        ## EMBEDDED
        embedded = embedded.transpose(0, 1) ## Batch_size to second dimension
        query = self.wq(embedded)
        key = self.wk(embedded)
        value = self.wv(embedded)
        attn_output, _ = self.multihead_attn(query, key, value)
        nearest_grid = attn_output[0]
        return self.out_projection(nearest_grid)
