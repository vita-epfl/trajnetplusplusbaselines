from collections import defaultdict

import torch

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
