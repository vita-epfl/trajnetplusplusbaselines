import itertools

import numpy as np
import torch

import trajnetplusplustools

from .modules import Hidden2Normal, InputEmbedding

from .. import augmentation
from .utils import center_scene, visualize_scene, visualize_grid, visualize_lrp, animate_lrp
from .lrp_linear_layer import *

NAN = float('nan')
TIME_STEPS = 19

def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask], mask


class LSTM(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128, pool=None, pool_to_input=True, goal_dim=None, goal_flag=False):
        """ Initialize the LSTM forecasting model

        Attributes
        ----------
        embedding_dim : Embedding dimension of location coordinates
        hidden_dim : Dimension of hidden state of LSTM
        pool : interaction module
        pool_to_input : Bool
            if True, the interaction vector is concatenated to the input embedding of LSTM [preferred]
            if False, the interaction vector is added to the LSTM hidden-state
        goal_dim : Embedding dimension of the unit vector pointing towards the goal
        goal_flag: Bool
            if True, the embedded goal vector is concatenated to the input embedding of LSTM 
        """

        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.pool = pool
        self.pool_to_input = pool_to_input

        ## Location
        scale = 4.0
        self.input_embedding = InputEmbedding(2, self.embedding_dim, scale)

        ## Goal
        self.goal_flag = goal_flag
        self.goal_dim = goal_dim or embedding_dim
        self.goal_embedding = InputEmbedding(2, self.goal_dim, scale)
        goal_rep_dim = self.goal_dim if self.goal_flag else 0

        ## Pooling
        pooling_dim = 0
        if pool is not None and self.pool_to_input:
            pooling_dim = self.pool.out_dim 
        
        ## LSTMs
        self.encoder = torch.nn.LSTMCell(self.embedding_dim + goal_rep_dim + pooling_dim, self.hidden_dim)
        self.decoder = torch.nn.LSTMCell(self.embedding_dim + goal_rep_dim + pooling_dim, self.hidden_dim)

        # Predict the parameters of a multivariate normal:
        # mu_vel_x, mu_vel_y, sigma_vel_x, sigma_vel_y, rho
        self.hidden2normal = Hidden2Normal(self.hidden_dim)

    def init_lrp(self):
        """
        Load trained model weights for LRP.
        """

        d = self.hidden_dim

        self.Wxh_Left_E  = self.encoder.weight_ih  # shape 4d*e
        self.bxh_Left_E  = self.encoder.bias_ih # shape 4d
        self.Whh_Left_E  = self.encoder.weight_hh  # shape 4d*d
        self.bhh_Left_E  = self.encoder.bias_hh # shape 4d

        self.Wxh_Left_D  = self.decoder.weight_ih # shape 4d*e
        self.bxh_Left_D  = self.decoder.bias_ih  # shape 4d
        self.Whh_Left_D  = self.decoder.weight_hh   # shape 4d*d
        self.bhh_Left_D  = self.decoder.bias_hh   # shape 4d

        self.Why_Left  = self.hidden2normal.linear.weight  # shape C*d
        self.bhy_Left  = self.hidden2normal.linear.bias   # shape C

        self.W_pool = self.pool.embedding[0].weight 
        self.b_pool = self.pool.embedding[0].bias 
        self.b_pool_units = self.pool.n * self.pool.n * self.pool.pooling_dim

        if self.pool.embedding_arch != 'one_layer':
            self.W_pool_2 = self.pool.embedding[2].weight 
            self.b_pool_2 = self.pool.embedding[2].bias 
            self.b_pool_units_2 = len(self.pool.embedding[0].bias)

    def init_lrp_new_scene(self):
        """
        Initialize relevant attributes for LRP.
        """

        self.T = TIME_STEPS
    
        # initialize
        d = self.hidden_dim
        T = self.T

        E = self.encoder.weight_ih.shape[1]

        ## Input
        self.x              = torch.zeros((T, E), device=self.encoder.weight_ih.device) 

        ## Sequence LSTM
        self.h_Left         = torch.zeros((T+1, d), device=self.encoder.weight_ih.device) 
        self.c_Left         = torch.zeros((T+1, d), device=self.encoder.weight_ih.device) 

        self.gates_xh_Left  = torch.zeros((T, 4*d), device=self.encoder.weight_ih.device)  
        self.gates_hh_Left  = torch.zeros((T, 4*d), device=self.encoder.weight_ih.device)  
        self.gates_pre_Left = torch.zeros((T, 4*d), device=self.encoder.weight_ih.device)   # gates pre-activation
        self.gates_Left     = torch.zeros((T, 4*d), device=self.encoder.weight_ih.device)   # gates activation

        ## Interaction Encoder (Grid-Based)
        if self.pool.embedding_arch == 'one_layer':
            self.pre_pool       = torch.zeros((T, self.pool.n * self.pool.n * self.pool.pooling_dim), device=self.encoder.weight_ih.device)  
            self.post_pool      = torch.zeros((T, self.pool.out_dim), device=self.encoder.weight_ih.device)  
        else:
            self.pre_pool       = torch.zeros((T, self.pool.n * self.pool.n * self.pool.pooling_dim), device=self.encoder.weight_ih.device)  
            self.post_pool      = torch.zeros((T, self.b_pool_units_2), device=self.encoder.weight_ih.device)  

            self.relu = torch.nn.ReLU()

            self.pre_pool_2       = torch.zeros((T, self.b_pool_units_2), device=self.encoder.weight_ih.device)  
            self.post_pool_2      = torch.zeros((T, self.pool.out_dim), device=self.encoder.weight_ih.device)  

        ## Output
        self.s              = torch.zeros((T, 5), device=self.encoder.weight_ih.device)   # gates activation

        ## Neighbour mapping
        self.range_mask = {}
        self.track_mask = {}
        self.occupancy_mapping = {}

        self.time_step = 0

    def step(self, lstm, hidden_cell_state, obs1, obs2, goals, batch_split):
        """Do one step of prediction: two inputs to one normal prediction.
        
        Parameters
        ----------
        lstm: torch nn module [Encoder / Decoder]
            The module responsible for prediction
        hidden_cell_state : tuple (hidden_state, cell_state)
            Current hidden_cell_state of the pedestrians
        obs1 : Tensor [num_tracks, 2]
            Previous x-y positions of the pedestrians
        obs2 : Tensor [num_tracks, 2]
            Current x-y positions of the pedestrians
        goals : Tensor [num_tracks, 2]
            Goal coordinates of the pedestrians
        
        Returns
        -------
        hidden_cell_state : tuple (hidden_state, cell_state)
            Updated hidden_cell_state of the pedestrians
        normals : Tensor [num_tracks, 5]
            Parameters of a multivariate normal of the predicted position 
            with respect to the current position
        """
        num_tracks = len(obs2)
        # mask for pedestrians absent from scene (partial trajectories)
        # consider only the hidden states of pedestrains present in scene
        track_mask = (torch.isnan(obs1[:, 0]) + torch.isnan(obs2[:, 0])) == 0

        ## Masked Hidden Cell State
        hidden_cell_stacked = [
            torch.stack([h for m, h in zip(track_mask, hidden_cell_state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(track_mask, hidden_cell_state[1]) if m], dim=0),
        ]

        ## Mask current velocity & embed
        curr_velocity = obs2 - obs1
        curr_velocity = curr_velocity[track_mask]
        input_emb = self.input_embedding(curr_velocity)

        ## Mask Goal direction & embed
        if self.goal_flag:
            ## Get relative direction to goals (wrt current position)
            norm_factors = (torch.norm(obs2 - goals, dim=1))
            goal_direction = (obs2 - goals) / norm_factors.unsqueeze(1)
            goal_direction[norm_factors == 0] = torch.tensor([0., 0.], device=obs1.device)
            goal_direction = goal_direction[track_mask]
            goal_emb = self.goal_embedding(goal_direction)
            input_emb = torch.cat([input_emb, goal_emb], dim=1)

        ## Mask & Pool per scene
        if self.pool is not None:
            hidden_states_to_pool = torch.stack(hidden_cell_state[0]).clone() # detach?
            batch_pool = []
            ## Iterate over scenes
            for (start, end) in zip(batch_split[:-1], batch_split[1:]):
                ## Mask for the scene
                scene_track_mask = track_mask[start:end]
                ## Get observations and hidden-state for the scene
                prev_position = obs1[start:end][scene_track_mask]
                curr_position = obs2[start:end][scene_track_mask]
                curr_hidden_state = hidden_states_to_pool[start:end][scene_track_mask]

                ## Provide track_mask to the interaction encoders
                ## Everyone absent by default. Only those visible in current scene are present
                interaction_track_mask = torch.zeros(num_tracks, device=obs1.device).bool()
                interaction_track_mask[start:end] = track_mask[start:end]
                self.pool.track_mask = interaction_track_mask

                pool_sample = self.pool(curr_hidden_state, prev_position, curr_position)
                batch_pool.append(pool_sample)

            pooled = torch.cat(batch_pool)
            if self.pool_to_input:
                input_emb = torch.cat([input_emb, pooled], dim=1)
            else:
                hidden_cell_stacked[0] += pooled

        ## LRP LSTM STEP Forward Propagation ####################################################
        if self.time_step < self.T:
            t = self.time_step
            atol = 1e-8
            ## Note: Pytorch ORDER !!!
            # W_i|W_f|W_g|W_o
            # gate indices (assuming the gate ordering in the LSTM weights is i,g,f,o):
            d = self.hidden_dim
            idx  = np.hstack((np.arange(0,2*d), np.arange(3*d,4*d))).astype(int) # indices of gates i,f,o together
            idx_i, idx_g, idx_f, idx_o = np.arange(0,d), np.arange(2*d,3*d), np.arange(d,2*d), np.arange(3*d,4*d) # indices of gates i,g,f,o separately
            self.Wxh_Left = self.Wxh_Left_E if t < 8 else self.Wxh_Left_D
            self.Whh_Left = self.Whh_Left_E if t < 8 else self.Whh_Left_D
            self.bxh_Left = self.bxh_Left_E if t < 8 else self.bxh_Left_D
            self.bhh_Left = self.bhh_Left_E if t < 8 else self.bhh_Left_D

            ## Pooling
            self.track_mask[t] = track_mask
            rg_mk, ps = self.pool.occupancy_neigh_map(curr_position)
            self.range_mask[t] = rg_mk
            self.occupancy_mapping[t] = ps 
            if self.pool.type_ == 'directional':
                grid = self.pool.directional(prev_position, curr_position)
            elif self.pool.type_ == 'occupancy':
                grid = self.pool.occupancies(prev_position, curr_position)
            elif self.pool.type_ == 'social':
                grid = self.pool.social(curr_hidden_state, prev_position, curr_position)

            grid = grid.view(len(curr_position), -1)
            self.pre_pool[t] = grid[0]
            # pool_manual = self.pool.embedding(grid)
            self.post_pool[t] = torch.matmul(self.W_pool, self.pre_pool[t]) + self.b_pool
            # assert torch.all(torch.isclose(batch_pool[0][0] , relu(self.post_pool[t]), atol=atol))
            if self.pool.embedding_arch != 'one_layer':
                self.pre_pool_2[t] = self.relu(self.post_pool[t])
                self.post_pool_2[t] = torch.matmul(self.W_pool_2, self.pre_pool_2[t]) + self.b_pool_2
                # assert torch.all(torch.isclose(batch_pool[0][0] , self.relu(self.post_pool_2[t]), atol=atol))

            self.x[t] = input_emb[0]
            self.gates_xh_Left[t]     = torch.matmul(self.Wxh_Left, self.x[t]) 
            self.gates_hh_Left[t]     = torch.matmul(self.Whh_Left, self.h_Left[t-1]) 
            self.gates_pre_Left[t]    = self.gates_xh_Left[t] + self.gates_hh_Left[t] + self.bxh_Left + self.bhh_Left
            self.gates_Left[t,idx]    = 1.0/(1.0 + torch.exp(-self.gates_pre_Left[t,idx]))
            self.gates_Left[t,idx_g]  = torch.tanh(self.gates_pre_Left[t,idx_g]) 
            self.c_Left[t]            = self.gates_Left[t,idx_f]*self.c_Left[t-1] + self.gates_Left[t,idx_i]*self.gates_Left[t,idx_g]
            self.h_Left[t]            = self.gates_Left[t,idx_o]*torch.tanh(self.c_Left[t])
            
            self.s[t]  = torch.matmul(self.Why_Left,  self.h_Left[t]) + self.bhy_Left # self.y_Left 
            self.time_step += 1
        ##########################################################################################

        # LSTM step
        hidden_cell_stacked = lstm(input_emb, hidden_cell_stacked)
        normal_masked = self.hidden2normal(hidden_cell_stacked[0])
        # assert torch.all(torch.isclose(hidden_cell_stacked[0][0] , self.h_Left[t], atol=atol))
        # assert torch.all(torch.isclose(hidden_cell_stacked[1][0] , self.c_Left[t], atol=atol))
        # assert torch.all(torch.isclose(normal_masked[0][:2] , self.s[:2], atol=atol))


        # unmask [Update hidden-states and next velocities of pedestrians]
        normal = torch.full((track_mask.size(0), 5), NAN, device=obs1.device)
        mask_index = [i for i, m in enumerate(track_mask) if m]
        for i, h, c, n in zip(mask_index,
                              hidden_cell_stacked[0],
                              hidden_cell_stacked[1],
                              normal_masked):
            hidden_cell_state[0][i] = h
            hidden_cell_state[1][i] = c
            normal[i] = n

        return hidden_cell_state, normal

    def lrp(self, timestep, LRP_class=0, eps=0.001, bias_factor=0.0, debug=False):
        """ Backward Relevance propagation """

        T = timestep + 1
        d = self.hidden_dim

        # initialize Relevances
        Rx       = torch.zeros_like(self.x)
        Rp       = torch.zeros((T, self.pool.out_dim), device=self.encoder.weight_ih.device)

        if self.pool.embedding_arch != 'one_layer':
            Rp_mid = torch.zeros((T, self.b_pool_units_2), device=self.encoder.weight_ih.device)

        Rv       = torch.zeros((T, self.embedding_dim), device=self.encoder.weight_ih.device)
        Rgrid    = torch.zeros((T, self.pool.n * self.pool.n * self.pool.pooling_dim), device=self.encoder.weight_ih.device)
        
        Rh_Left  = torch.zeros((T+1, d), device=self.encoder.weight_ih.device)
        Rc_Left  = torch.zeros((T+1, d), device=self.encoder.weight_ih.device)
        Rg_Left  = torch.zeros((T, d), device=self.encoder.weight_ih.device) # gate g only

        Rout_mask            = torch.zeros((5))
        Rout_mask[LRP_class] = 1.0  

        # format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
        Rh_Left[T-1]  = lrp_linear(self.h_Left[T-1],  self.Why_Left.T , self.bhy_Left, self.s[timestep], self.s[timestep]*Rout_mask, 128, eps, bias_factor, debug=debug)

        d = self.hidden_dim
        e = self.encoder.weight_ih.shape[1]
        idx  = np.hstack((np.arange(0,2*d), np.arange(3*d,4*d))).astype(int) # indices of gates i,f,o together
        idx_i, idx_g, idx_f, idx_o = np.arange(0,d), np.arange(2*d,3*d), np.arange(d,2*d), np.arange(3*d,4*d) # indices of gates i,g,f,o separately
        
        ## Backward Pass
        for t in reversed(range(T)):
            self.Wxh_Left = self.Wxh_Left_E if t < 8 else self.Wxh_Left_D
            self.Whh_Left = self.Whh_Left_E if t < 8 else self.Whh_Left_D
            self.bxh_Left = self.bxh_Left_E if t < 8 else self.bxh_Left_D
            self.bhh_Left = self.bhh_Left_E if t < 8 else self.bhh_Left_D

            Rc_Left[t]   += Rh_Left[t]
            Rc_Left[t-1]  = lrp_linear(self.gates_Left[t,idx_f]*self.c_Left[t-1],         torch.eye(d), torch.zeros((d)), self.c_Left[t], Rc_Left[t], 2*d, eps, bias_factor, debug=debug)
            Rg_Left[t]    = lrp_linear(self.gates_Left[t,idx_i]*self.gates_Left[t,idx_g], torch.eye(d), torch.zeros((d)), self.c_Left[t], Rc_Left[t], 2*d, eps, bias_factor, debug=debug)
            Rx[t]         = lrp_linear(self.x[t],        self.Wxh_Left[idx_g].T, self.bxh_Left[idx_g]+self.bhh_Left[idx_g], self.gates_pre_Left[t,idx_g], Rg_Left[t], d+e, eps, bias_factor, debug=debug)
            Rh_Left[t-1]  = lrp_linear(self.h_Left[t-1], self.Whh_Left[idx_g].T, self.bxh_Left[idx_g]+self.bhh_Left[idx_g], self.gates_pre_Left[t,idx_g], Rg_Left[t], d+e, eps, bias_factor, debug=debug)
            
            Rv[t]         = Rx[t, :self.embedding_dim]
            Rp[t]         = Rx[t, self.embedding_dim:]

            # format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
            if self.pool.embedding_arch != 'one_layer':
                Rp_mid[t]  = lrp_linear(self.pre_pool_2[t], self.W_pool_2.T, self.b_pool_2, self.post_pool_2[t], Rp[t], self.b_pool_units_2, eps, bias_factor, debug=debug)
                Rgrid[t]  = lrp_linear(self.pre_pool[t],  self.W_pool.T, self.b_pool, self.post_pool[t], Rp_mid[t], self.b_pool_units, eps, bias_factor, debug=debug)
            else:
                Rgrid[t]  = lrp_linear(self.pre_pool[t],  self.W_pool.T, self.b_pool, self.post_pool[t], Rp[t], self.b_pool_units, eps, bias_factor, debug=debug)

        return Rx, Rh_Left[-1].sum()+Rc_Left[-1].sum(), Rgrid, Rv

    def forward(self, observed, goals, batch_split, prediction_truth=None, n_predict=None):
        """Forecast the entire sequence 
        
        Parameters
        ----------
        observed : Tensor [obs_length, num_tracks, 2]
            Observed sequences of x-y coordinates of the pedestrians
        goals : Tensor [num_tracks, 2]
            Goal coordinates of the pedestrians
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene        
        prediction_truth : Tensor [pred_length - 1, num_tracks, 2]
            Prediction sequences of x-y coordinates of the pedestrians
            Helps in teacher forcing wrt neighbours positions during training
        n_predict: Int
            Length of sequence to be predicted during test time

        Returns
        -------
        rel_pred_scene : Tensor [pred_length, num_tracks, 5]
            Predicted velocities of pedestrians as multivariate normal
            i.e. positions relative to previous positions
        pred_scene : Tensor [pred_length, num_tracks, 2]
            Predicted positions of pedestrians i.e. absolute positions
        """

        assert ((prediction_truth is None) + (n_predict is None)) == 1
        if n_predict is not None:
            # -1 because one prediction is done by the encoder already
            prediction_truth = [None for _ in range(n_predict - 1)]

        # initialize: Because of tracks with different lengths and the masked
        # update, the hidden state for every LSTM needs to be a separate object
        # in the backprop graph. Therefore: list of hidden states instead of
        # a single higher rank Tensor.
        num_tracks = observed.size(1)
        hidden_cell_state = (
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(num_tracks)],
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(num_tracks)],
        )

        ## Reset LSTMs of Interaction encoders
        self.pool.reset(num_tracks, device=observed.device)

        # list of predictions
        normals = []  # predicted normal parameters for both phases
        positions = []  # true (during obs phase) and predicted positions

        if len(observed) == 2:
            positions = [observed[-1]]

        positions.append(observed[0])  # no sampling, just mean
        positions.append(observed[1])  # no sampling, just mean

        # encoder
        for obs1, obs2 in zip(observed[:-1], observed[1:]):
            ##LSTM Step
            hidden_cell_state, normal = self.step(self.encoder, hidden_cell_state, obs1, obs2, goals, batch_split)

            # concat predictions
            normals.append(normal)

            positions.append(obs2 + normal[:, :2])  # no sampling, just mean

        # initialize predictions with last position to form velocity
        prediction_truth = list(itertools.chain.from_iterable(
            (observed[-1:], prediction_truth)
        ))

        # decoder, predictions
        for obs1, obs2 in zip(prediction_truth[:-1], prediction_truth[1:]):
            if obs1 is None:
                obs1 = positions[-2].detach()  # DETACH!!!
            else:
                for primary_id in batch_split[:-1]:
                    obs1[primary_id] = positions[-2][primary_id].detach()  # DETACH!!!
            if obs2 is None:
                obs2 = positions[-1].detach()
            else:
                for primary_id in batch_split[:-1]:
                    obs2[primary_id] = positions[-1][primary_id].detach()  # DETACH!!!
            hidden_cell_state, normal = self.step(self.decoder, hidden_cell_state, obs1, obs2, goals, batch_split)

            # concat predictions
            normals.append(normal)
            positions.append(obs2 + normal[:, :2])  # no sampling, just mean

        vel_weights, neigh_weights = self.calculate_lrp_scores_all()

        # Pred_scene: Tensor [seq_length, num_tracks, 2]
        #    Absolute positions of all pedestrians
        # Rel_pred_scene: Tensor [seq_length, num_tracks, 5]
        #    Velocities of all pedestrians
        rel_pred_scene = torch.stack(normals, dim=0)
        pred_scene = torch.stack(positions, dim=0)

        return rel_pred_scene, pred_scene, vel_weights, neigh_weights

    def calculate_lrp_scores_all(self):
        overall_vel_weights = []
        overall_neigh_weights = []

        for t in range(7, TIME_STEPS):
            ## x and y coordinates
            Rx_x, Rh_x, Rp_x, Rv_x = self.lrp(t, LRP_class=0, bias_factor=0.0, debug=False, eps=0.001)
            Rx_y, Rh_y, Rp_y, Rv_y = self.lrp(t, LRP_class=1, bias_factor=0.0, debug=False, eps=0.001)
            # R_tot = Rx_x.sum() + Rh_x.sum()      # sum of all "input" relevances
            # print("Sanity check passed? ", R_tot, self.s[timestep][0])

            ## Get relevance scores of inputs
            vel_weights, neigh_weights = self.get_scores(Rp_y + Rp_x, Rv_x + Rv_y, t)
            overall_vel_weights.append(vel_weights.copy())
            overall_neigh_weights.append(neigh_weights.copy())

        return overall_vel_weights, overall_neigh_weights

    def get_scores(self, Rp, Rv, timestep):

        ## Past velocity Weights
        vel_weights = [0.0]
        for t, vel_embed in enumerate(Rv):
            vel_weights.append(torch.mean(vel_embed).item())
        vel_weights.append(0.)
        vel_weights = np.array(vel_weights)
        vel_weights = (vel_weights - np.min(vel_weights)) / (np.max(vel_weights) - np.min(vel_weights)) + 0.3

        ## Neighbour Weights
        Rp_last_step = Rp[-1].reshape(-1, self.pool.n, self.pool.n)
        neigh_weights = []
        occupancy_map = self.occupancy_mapping[timestep][0]
        range_mask = self.range_mask[timestep][0]
        track_mask = self.track_mask[timestep]
        range_mask_iter = 0
        ## Mapping between neighbours and their respective grid positions
        for id_, visible in enumerate(track_mask[1:]):
            if not visible:
                neigh_weights.append(0.)
                continue
            if not range_mask[range_mask_iter]:
                neigh_weights.append(0.)
            else:
                assert visible and range_mask[range_mask_iter]
                x_pos = occupancy_map[range_mask_iter, 0]
                y_pos = occupancy_map[range_mask_iter, 1]
                neigh_value = Rp_last_step[:, x_pos, y_pos]
                neigh_weights.append(torch.sum(neigh_value).item())
            range_mask_iter += 1
        neigh_weights = np.abs(np.array(neigh_weights))
        neigh_weights = (neigh_weights) / (np.sum(neigh_weights) + 0.00001)

        return vel_weights, neigh_weights

class LSTMPredictor(object):
    def __init__(self, model):
        self.model = model

    def save(self, state, filename):
        with open(filename, 'wb') as f:
            torch.save(self, f)

        # # during development, good for compatibility across API changes:
        # # Save state for optimizer to continue training in future
        with open(filename + '.state', 'wb') as f:
            torch.save(state, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return torch.load(f)


    def __call__(self, paths, scene_goal, n_predict=12, modes=1, predict_all=True, obs_length=9, start_length=0, scene_id=0, args=None):
        self.model.eval()
        # self.model.train()
        with torch.no_grad():
            self.model.init_lrp()
            self.model.init_lrp_new_scene()
            xy = trajnetplusplustools.Reader.paths_to_xy(paths)
            batch_split = [0, xy.shape[1]]

            if args.normalize_scene:
                xy, rotation, center, scene_goal = center_scene(xy, obs_length, goals=scene_goal)
            
            xy = torch.Tensor(xy)  #.to(self.device)
            scene_goal = torch.Tensor(scene_goal) #.to(device)
            batch_split = torch.Tensor(batch_split).long()

            multimodal_outputs = {}
            for num_p in range(modes):
                # _, output_scenes = self.model(xy[start_length:obs_length], scene_goal, batch_split, xy[obs_length:-1].clone())
                _, output_scenes, vel_weights, neigh_weights = self.model(xy[start_length:obs_length], scene_goal, batch_split, n_predict=n_predict)
                output_scenes = output_scenes.numpy()

                # visualize_lrp(output_scenes, vel_weights, neigh_weights, TIME_STEPS)
                print("Scene ID: ", scene_id)
                animate_lrp(output_scenes, vel_weights, neigh_weights, TIME_STEPS, scene_id, self.model.pool.type_)
                if args.normalize_scene:
                    output_scenes = augmentation.inverse_scene(output_scenes, rotation, center)
                output_primary = output_scenes[-n_predict:, 0]
                output_neighs = output_scenes[-n_predict:, 1:]
                ## Dictionary of predictions. Each key corresponds to one mode
                multimodal_outputs[num_p] = [output_primary, output_neighs]

        ## Return Dictionary of predictions. Each key corresponds to one mode
        return multimodal_outputs
