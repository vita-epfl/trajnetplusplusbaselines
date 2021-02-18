import itertools
import copy

import numpy as np
import torch

import trajnetplusplustools

from .. import augmentation
from ..lstm.utils import center_scene
from ..lstm.modules import Hidden2Normal, InputEmbedding

from .utils import sample_multivariate_distribution

NAN = float('nan')

def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask], mask

class VAE(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128, pool=None, pool_to_input=True, goal_dim=None, goal_flag=False,
                 num_modes=1, desire_approach=False, latent_dim=128):
        """ Initialize the VAE forecasting model

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

        super(VAE, self).__init__()
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
        self.obs_encoder = torch.nn.LSTMCell(self.embedding_dim + goal_rep_dim + pooling_dim, self.hidden_dim)
        self.pred_encoder = torch.nn.LSTMCell(self.embedding_dim + goal_rep_dim + pooling_dim, self.hidden_dim)
        self.decoder = torch.nn.LSTMCell(self.embedding_dim + goal_rep_dim + pooling_dim, self.hidden_dim)

        # Predict the parameters of a multivariate normal:
        # mu_vel_x, mu_vel_y, sigma_vel_x, sigma_vel_y, rho
        self.hidden2normal = Hidden2Normal(self.hidden_dim)

        ## VAE Specific
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        self.desire = desire_approach

        self.vae_encoder_xy = VAEEncoder(2*self.hidden_dim, 2*self.latent_dim)
        self.vae_encoder_x = VAEEncoder(self.hidden_dim, 2*self.latent_dim)
        self.vae_decoder = VAEDecoder(self.latent_dim, self.hidden_dim)

    def concat(self, hidden_cell_state, hidden_cell_state_pred):
        return  (list(torch.cat([torch.stack(hidden_cell_state[0]), torch.stack(hidden_cell_state_pred[0])], dim=1)),
                 list(torch.cat([torch.stack(hidden_cell_state[1]), torch.stack(hidden_cell_state_pred[1])], dim=1))
                )

    def add_noise(self, hidden_cell_state, z_mu, z_var_log, z_mu_obs, z_var_log_obs):

        if self.training:
            ## Sampling using "reparametrization trick"
            # See Kingma & Wellig, Auto-Encoding Variational Bayes, 2014 (arXiv:1312.6114)
            epsilon = torch.empty(size=z_mu.size()).normal_(mean=0, std=1)
            z_val = z_mu + torch.exp(0.5*z_var_log) * epsilon

        else:
            # Draw a sample from the learned multivariate distribution (z_mu, z_var_log)
            z_val = sample_multivariate_distribution(z_mu_obs, z_var_log_obs)

        ## VAE decoder
        decoder_output = self.vae_decoder(z_val)

        ## Update Hidden-Cell-State
        hidden_state_new = [hidden_state * dec_output for dec_output, hidden_state in zip(decoder_output, hidden_cell_state[0])]
        cell_state_new = [cell_state for cell_state in hidden_cell_state[1]]

        return (hidden_state_new, cell_state_new)

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

                ## Pool
                pool_sample = self.pool(curr_hidden_state, prev_position, curr_position)
                batch_pool.append(pool_sample)

            pooled = torch.cat(batch_pool)
            if self.pool_to_input:
                input_emb = torch.cat([input_emb, pooled], dim=1)
            else:
                hidden_cell_stacked[0] += pooled

        # LSTM step
        hidden_cell_stacked = lstm(input_emb, hidden_cell_stacked)
        normal_masked = self.hidden2normal(hidden_cell_stacked[0])

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

        ## Reset LSTMs of Interaction Encoders.
        if self.pool is not None:
            self.pool.reset(num_tracks, device=observed.device)

        # list of predictions store a dictionary. Each key corresponds to one mode
        normals = {mode: [] for mode in range(self.num_modes)} # predicted normal parameters for both phases
        positions = {mode: [] for mode in range(self.num_modes)} # true (during obs phase) and predicted positions

        # encoder
        for obs1, obs2 in zip(observed[:-1], observed[1:]):
            ##LSTM Step
            hidden_cell_state, normal = self.step(self.obs_encoder, hidden_cell_state, obs1, obs2, goals, batch_split)

            # concat predictions
            for mode_n, mode_p in zip(normals.keys(), positions.keys()):
                normals[mode_n].append(normal)
                positions[mode_p].append(obs2 + normal[:, :2]) # no sampling, just mean
    
        # initialize predictions with last position to form velocity. DEEP COPY !!!
        prediction_truth = copy.deepcopy(list(itertools.chain.from_iterable(
            (observed[-1:], prediction_truth)
        )))

        ## Encode Prediction Truth (during training)
        if self.training:
            assert prediction_truth is not None
            # Initialize hidden cell state for prediction encoder
            hidden_cell_state_pred = (
                [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(num_tracks)],
                [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(num_tracks)],
            )

            ## Encode
            for obs1, obs2 in zip(prediction_truth[:-1], prediction_truth[1:]):
                # LSTM Step
                hidden_cell_state_pred, _ = self.step(self.pred_encoder, hidden_cell_state_pred, obs1, obs2, goals, batch_split)

        ## Get z_xy and z_x ###################################
        ## VAE encoder, latent distribution
        z_distr_xy = None
        z_mu, z_var_log = None, None
        if self.training:
            ## Concatenate Hidden State of Observation and Prediction
            hidden_cell_state_full = self.concat(hidden_cell_state, hidden_cell_state_pred)
            z_mu, z_var_log = self.vae_encoder_xy(hidden_cell_state_full[0])
            z_distr_xy = torch.cat((z_mu, z_var_log), dim=1)

        # Compute target latent distribution (depending only on observation)
        z_distr_x = None
        z_mu_obs = torch.zeros(num_tracks, self.latent_dim)
        z_var_log_obs = torch.ones(num_tracks, self.latent_dim)
        if not self.desire:
            z_mu_obs, z_var_log_obs = self.vae_encoder_x(hidden_cell_state[0])
            z_distr_x = torch.cat((z_mu_obs, z_var_log_obs), dim=1)
        ########################################################

        # Make num_modes predictions
        for k in range(self.num_modes):
            hidden_cell_state_dec = self.add_noise(hidden_cell_state, z_mu, z_var_log, z_mu_obs, z_var_log_obs)

            # decoder, predictions
            for obs1, obs2 in zip(prediction_truth[:-1], prediction_truth[1:]):
                if obs1 is None:
                    obs1 = positions[k][-2].detach()  # DETACH!!!
                else:
                    for primary_id in batch_split[:-1]:
                        obs1[primary_id] = positions[k][-2][primary_id].detach()  # DETACH!!!
                if obs2 is None:
                    obs2 = positions[k][-1].detach()
                else:
                    for primary_id in batch_split[:-1]:
                        obs2[primary_id] = positions[k][-1][primary_id].detach()  # DETACH!!!
                hidden_cell_state_dec, normal = self.step(self.decoder, hidden_cell_state_dec, obs1, obs2, goals, batch_split)
                # concat predictions
                normals[k].append(normal)
                positions[k].append(obs2 + normal[:, :2])  # no sampling, just mean

        # Pred_scene: Tensor [seq_length, num_tracks, 2]
        #    Absolute positions of all pedestrians
        # Rel_pred_scene: Tensor [seq_length, num_tracks, 5]
        #    Velocities of all pedestrians
        rel_pred_scene = [torch.stack(normals[mode_n], dim=0) for mode_n in normals.keys()]
        pred_scene = [torch.stack(positions[mode_p], dim=0) for mode_p in positions.keys()]

        return rel_pred_scene, pred_scene, z_distr_xy, z_distr_x

class VAEEncoder(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(VAEEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_mu = torch.nn.Linear(in_features=self.input_dim, out_features=self.output_dim//2)
        self.fc_var = torch.nn.Linear(in_features=self.input_dim, out_features=self.output_dim//2)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        inputs = torch.reshape(inputs, shape=(-1, self.input_dim))
        z_mu = self.relu(self.fc_mu(inputs))
        z_log_var = 0.01 + self.relu(self.fc_var(inputs))
        return z_mu, z_log_var

class VAEDecoder(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = torch.nn.Linear(in_features=self.input_dim, out_features=self.output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        inputs = torch.reshape(inputs, shape=(-1, self.input_dim))
        return self.relu(self.fc(inputs))

class VAEPredictor(object):
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


    def __call__(self, paths, scene_goal, n_predict=12, modes=1, predict_all=True, obs_length=9, start_length=0, args=None):
        self.model.eval()
        # self.model.train()
        self.model.num_modes = modes
        with torch.no_grad():
            xy = trajnetplusplustools.Reader.paths_to_xy(paths)
            # xy = augmentation.add_noise(xy, thresh=args.thresh, ped=args.ped_type)
            batch_split = [0, xy.shape[1]]

            if args.normalize_scene:
                xy, rotation, center, scene_goal = center_scene(xy, obs_length, goals=scene_goal)
            
            xy = torch.Tensor(xy)  #.to(self.device)
            scene_goal = torch.Tensor(scene_goal) #.to(device)
            batch_split = torch.Tensor(batch_split).long()

            multimodal_outputs = {}
            _, output_scenes_list, _, _ = self.model(xy[start_length:obs_length], scene_goal, batch_split, n_predict=n_predict)
            for num_p, _ in enumerate(output_scenes_list):
                output_scenes = output_scenes_list[num_p]
                output_scenes = output_scenes.numpy()
                if args.normalize_scene:
                    output_scenes = augmentation.inverse_scene(output_scenes, rotation, center)
                output_primary = output_scenes[-n_predict:, 0]
                output_neighs = output_scenes[-n_predict:, 1:]
                if num_p == 0:
                    ## Dictionary of predictions. Each key corresponds to one mode
                    multimodal_outputs[num_p] = [output_primary, output_neighs]
                else:
                    multimodal_outputs[num_p] = [output_primary, []]

        ## Return Dictionary of predictions. Each key corresponds to one mode
        return multimodal_outputs
