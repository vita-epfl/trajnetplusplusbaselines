from collections import defaultdict

import torch


def rel_obs(obs):
    """ Provides relative position of neighbours wrt one another

    Returns
    -------
    relative : Tensor [num_tracks, num_tracks, 2]
    """
    unfolded = obs.unsqueeze(0).repeat(obs.size(0), 1, 1)
    relative = unfolded - obs.unsqueeze(1)
    return relative

def rel_directional(obs1, obs2):
    """ Provides relative velocity of neighbours wrt one another

    Returns
    -------
    relative : Tensor [num_tracks, num_tracks, 2]
    """
    vel = obs2 - obs1
    unfolded = vel.unsqueeze(0).repeat(vel.size(0), 1, 1)
    relative = unfolded - vel.unsqueeze(1)
    return relative


class NMMP(torch.nn.Module):
    """ Interaction vector is obtained by message passing between
        hidden-state of all neighbours. Proposed in NMMP, CVPR 2020
        Parameters:
        mlp_dim: embedding size of hidden-state
        k: number of iterations of message passing
        out_dim: dimension of resultant interaction vector
        
        Attributes
        ----------
        mlp_dim : Scalar
            Embedding dimension of hidden-state of LSTM
        k : Scalar
            Number of iterations of message passing
        out_dim: Scalar
            Dimension of resultant interaction vector
    """
    def __init__(self, hidden_dim=128, mlp_dim=32, k=5, out_dim=None):
        super(NMMP, self).__init__()
        self.out_dim = out_dim or hidden_dim

        self.hidden_embedding = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, mlp_dim),
            torch.nn.ReLU(),
        )

        self.mlp_dim = mlp_dim
        self.node_to_edge_embedding = torch.nn.Linear(2*mlp_dim, mlp_dim)
        self.edge_to_node_embedding = torch.nn.Linear(2*mlp_dim, mlp_dim)

        self.out_projection = torch.nn.Linear(mlp_dim, self.out_dim)
        self.k = k

    def message_pass(self, node_embeddings):
        # Perform a single iteration of message passing
        n = node_embeddings.size(0)
        arrange1 = node_embeddings.repeat(n, 1, 1) ## c
        arrange2 = arrange1.transpose(0, 1) ## d

        ## e_out
        e_out_all = torch.cat([arrange2, arrange1], dim=2)
        e_out_neighbours = e_out_all[~torch.eye(n).bool()].reshape(n, n-1, 2*self.mlp_dim)
        e_out_edges = self.node_to_edge_embedding(e_out_neighbours)
        e_out_sumpool = torch.mean(e_out_edges, dim=1)

        ## e_in
        e_in_all = torch.cat([arrange1, arrange2], dim=2)
        e_in_neighbours = e_in_all[~torch.eye(n).bool()].reshape(n, n-1, 2*self.mlp_dim)
        e_in_edges = self.node_to_edge_embedding(e_in_neighbours)
        e_in_sumpool = torch.mean(e_in_edges, dim=1)

        ## [e_in; e_out]
        concat_nodes = torch.cat([e_in_sumpool, e_out_sumpool], dim=1)

        ## refined node
        refined_embeddings = self.edge_to_node_embedding(concat_nodes)
        return refined_embeddings

    def reset(self, _, device):
        self.track_mask = None

    def forward(self, hidden_states, _, obs2):

        ## If only primary present
        num_tracks = obs2.size(0)
        if num_tracks == 1:
            return torch.zeros(1, self.out_dim, device=obs2.device)

        ## Embed hidden-state
        node_embeddings = self.hidden_embedding(hidden_states)
        ## Iterative Message Passing
        for _ in range(self.k):
            node_embeddings = self.message_pass(node_embeddings)

        return self.out_projection(node_embeddings)


class Directional_SAttention(torch.nn.Module):
    """ Interaction vector is obtained by attention-weighting the embeddings of relative coordinates and
        relative velocities obtained using Interaction Encoder LSTM.
        "S-Attention + Directional"
        
        Attributes
        ----------
        track_mask : Bool [num_tracks,]
            Mask of tracks visible at the current time-instant
            as well as tracks belonging to the particular scene 
        spatial_dim : Scalar
            Embedding dimension of relative position of neighbour   
        vel_dim : Scalar
            Embedding dimension of relative velocity of neighbour       
        hidden_dim : Scalar
            Hidden-state dimension of interaction-encoder LSTM
        out_dim: Scalar
            Dimension of resultant interaction vector
    """

    def __init__(self, spatial_dim=32, vel_dim=32, hidden_dim=256, out_dim=32, track_mask=None):
        super(Directional_SAttention, self).__init__()
        if out_dim is None:
            out_dim = hidden_dim
        self.out_dim = out_dim
        self.spatial_dim = spatial_dim
        self.spatial_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, self.spatial_dim),
            torch.nn.ReLU(),
        )
        self.vel_dim = vel_dim
        self.directional_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, self.vel_dim),
            torch.nn.ReLU(),
        )

        self.neigh_dim = self.spatial_dim + self.vel_dim
        self.hiddentospat = torch.nn.Linear(hidden_dim, self.neigh_dim)
        self.hidden_dim = hidden_dim
        self.pool_lstm = torch.nn.LSTMCell(self.neigh_dim, self.neigh_dim)
        self.track_mask = track_mask

        ## Attention Embeddings (Query, Key, Value)
        self.wq = torch.nn.Linear(self.neigh_dim, self.neigh_dim, bias=False)
        self.wk = torch.nn.Linear(self.neigh_dim, self.neigh_dim, bias=False)
        self.wv = torch.nn.Linear(self.neigh_dim, self.neigh_dim, bias=False)
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=self.neigh_dim, num_heads=1)

        self.out_projection = torch.nn.Linear(self.neigh_dim, self.out_dim)

    def reset(self, num_tracks, device):
        self.hidden_cell_state = (
            torch.zeros((num_tracks, num_tracks, self.neigh_dim), device=device),
            torch.zeros((num_tracks, num_tracks, self.neigh_dim), device=device),
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
        spat_embed = self.spatial_embedding(rel_position)

        # [num_tracks, 2] --> [num_tracks, num_tracks, 2]
        rel_vel = rel_directional(obs1, obs2)
        # Deleting Diagonal (agents wrt themselves)
        # [num_tracks, num_tracks, 2] --> [num_tracks * (num_tracks - 1), 2]
        rel_vel = rel_vel[~torch.eye(num_tracks).bool()]
        vel_embed = self.directional_embedding(rel_vel * 4)
        rel_embed = torch.cat([spat_embed, vel_embed], dim=1)
    
        ## Update interaction-encoder LSTMs
        pool_hidden_states = self.pool_lstm(rel_embed, hidden_cell_stacked)
        
        ## Save hidden-cell-states
        self.hidden_cell_state[0][adj_matrix] = pool_hidden_states[0]
        self.hidden_cell_state[1][adj_matrix] = pool_hidden_states[1]

        ## Attention between hidden_states of motion encoder & hidden_states of interactions encoders ##
        
        # Embed Hidden-state of Motion LSTM: [num_tracks, hidden_dim] --> [num_tracks, self.neigh_dim]
        hidden_state_spat = self.hiddentospat(hidden_state)
        
        # Concat Hidden-state of Motion LSTM to Hidden-state of Interaction LSTMs
        # embedded.shape = [num_tracks, num_tracks, self.neigh_dim]
        embedded = torch.cat([hidden_state_spat.unsqueeze(1), pool_hidden_states[0].reshape(num_tracks, num_tracks-1, self.neigh_dim)], dim=1)
        
        ## Attention
        # [num_tracks, num_tracks, self.neigh_dim] --> [num_tracks, num_tracks, self.neigh_dim]
        # i.e. [batch, seq, self.neigh_dim] --> [seq, batch, self.neigh_dim]
        embedded = embedded.transpose(0, 1)
        query = self.wq(embedded)
        key = self.wk(embedded)
        value = self.wv(embedded)
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_vectors = attn_output[0]
        return self.out_projection(attn_vectors)
