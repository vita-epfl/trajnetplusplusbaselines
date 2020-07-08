from collections import defaultdict
import itertools

import numpy as np
import torch
import torch.nn as nn

import trajnetplusplustools

from ..lstm.modules import Hidden2Normal, InputEmbedding

from .. import augmentation
from ..lstm.utils import center_scene

NAN = float('nan')

def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask], mask

def get_noise(shape, noise_type, device):
    ## removed cuda ##
    if noise_type == 'gaussian':
        return torch.randn(*shape, device=device)
    if noise_type == 'uniform':
        return torch.rand(*shape, device=device).sub_(0.5).mul_(2.0)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

class SGAN(torch.nn.Module):
    def __init__(self, generator=None, discriminator=None, add_noise=False, k=1, d_steps=1, g_steps=1):
        super(SGAN, self).__init__()

        ## Generator
        self.generator = generator if generator is not None else LSTMGenerator()
        self.g_steps = g_steps

        self.k = 1
        ## Add Noise for Variety Loss
        if add_noise:
            self.k = k

        ## Discriminator
        self.use_d = False
        self.d_steps = 0
        if discriminator is not None:
            print("Using Discriminator")
            self.discrimator = discriminator
            self.use_d = True
            self.d_steps = d_steps


    def forward(self, observed, goals, prediction_truth=None, n_predict=None, step_type='g', pred_length=12):
        """forward
        observed shape is (seq, n_tracks, observables)
        """

        rel_pred_list = []
        pred_list = []
        for _ in range(self.k):
            # print("k:", k)
            rel_pred_scene, pred_scene = self.generator(observed, goals, prediction_truth, n_predict)
            rel_pred_list.append(rel_pred_scene)
            pred_list.append(pred_scene)

            if step_type == 'd':
                break

        if self.use_d:
            scores_real = self.discrimator(observed, prediction_truth, goals)
            scores_fake = self.discrimator(observed, pred_scene[-pred_length:], goals)
            return rel_pred_list, pred_list, scores_real, scores_fake

        return rel_pred_list, pred_list, None, None

class LSTMGenerator(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128, pool=None, pool_to_input=True, goal_dim=None, goal_flag=False,
                 noise_dim=8, add_noise=False, noise_type='gaussian'):
        super(LSTMGenerator, self).__init__()
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
        input_dim = (self.embedding_dim + self.goal_dim) if self.goal_flag else self.embedding_dim
        self.goal_embedding = InputEmbedding(2, self.goal_dim, scale)

        print("INPUT DIM: ", input_dim)
        ## Pooling
        if self.pool is not None and self.pool_to_input:
            self.encoder = torch.nn.LSTMCell(input_dim + self.pool.out_dim, self.hidden_dim)
            self.decoder = torch.nn.LSTMCell(input_dim + self.pool.out_dim, self.hidden_dim)
        else:
            self.encoder = torch.nn.LSTMCell(input_dim, self.hidden_dim)
            self.decoder = torch.nn.LSTMCell(input_dim, self.hidden_dim)

        # Predict the parameters of a multivariate normal:
        # mu_vel_x, mu_vel_y, sigma_vel_x, sigma_vel_y, rho
        self.hidden2normal = Hidden2Normal(self.hidden_dim)

        ####### GAN Specific #########
        ## Noise
        self.noise_dim = noise_dim
        self.add_noise = add_noise
        self.noise_type = noise_type

        ## MLP Interface between Encoder and Decoder
        ## If required, do in future
        mlp_decoder_context_dims = [
            self.hidden_dim, self.hidden_dim - self.noise_dim
        ]

        self.mlp_decoder_context = make_mlp(
            mlp_decoder_context_dims
        )
        ###############################

    def adding_noise(self, hidden_cell_state):
        hidden_cell_state = (
            torch.stack([h for h in hidden_cell_state[0]], dim=0),
            torch.stack([c for c in hidden_cell_state[1]], dim=0),
        )

        ## hidden_dim --> hidden_dim - noise_dim
        new_hidden_state = self.mlp_decoder_context(hidden_cell_state[0])

        if self.add_noise:
            noise = get_noise((self.noise_dim, ), self.noise_type, device=hidden_cell_state[0].device)
        else:
            ## Add zeroes to inputs (CUDA if necessary)
            noise = torch.zeros(self.noise_dim, device=hidden_cell_state.device)

        z_decoder = noise.repeat(new_hidden_state.size(0), 1)
        new_hidden_state = torch.cat([new_hidden_state, z_decoder], dim=1)

        return (
            list(new_hidden_state),
            list(hidden_cell_state[1]),
        )

    def step(self, lstm, hidden_cell_state, obs1, obs2, goals):
        """Do one step: two inputs to one normal prediction."""
        # mask for pedestrians absent from scene (partial trajectories)
        # consider only the hidden states of pedestrains present in scene
        track_mask = (torch.isnan(obs1[:, 0]) + torch.isnan(obs2[:, 0])) == 0
        obs1, obs2, goals = obs1[track_mask], obs2[track_mask], goals[track_mask]

        ## LSTM-Based Interaction Encoders. Provide track_mask
        if self.pool.__class__.__name__ in {'NN_LSTM', 'TrajectronPooling', 'SAttention', 'SAttention_fast'}:
            self.pool.track_mask = track_mask

        hidden_cell_stacked = [
            torch.stack([h for m, h in zip(track_mask, hidden_cell_state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(track_mask, hidden_cell_state[1]) if m], dim=0),
        ]

        norm_factors = (torch.norm(obs2 - goals, dim=1))
        goal_direction = (obs2 - goals) / norm_factors.unsqueeze(1)
        goal_direction[norm_factors == 0] = torch.tensor([0., 0], device=obs1.device)
        # goal_direction = obs2 - goals

        # input embedding and optional pooling
        if self.pool is None:
            if self.goal_flag:
                input_emb = torch.cat([self.input_embedding(obs2 - obs1), self.goal_embedding(goal_direction)], dim=1)
            else:
                input_emb = self.input_embedding(obs2 - obs1)
        elif self.pool_to_input:
            hidden_states_to_pool = hidden_cell_stacked[0].clone() ## detach()
            pooled = self.pool(hidden_states_to_pool, obs1, obs2)
            # input_emb = self.input_embedding(torch.cat([obs2 - obs1, goal_direction, pooled], dim=1))
            if self.goal_flag:
                input_emb = torch.cat([self.input_embedding(obs2 - obs1), self.goal_embedding(goal_direction), pooled], dim=1)
            else:
                input_emb = torch.cat([self.input_embedding(obs2 - obs1), pooled], dim=1)
        else:
            if self.goal_flag:
                input_emb = torch.cat([self.input_embedding(obs2 - obs1), self.goal_embedding(goal_direction)], dim=1)
            else:
                input_emb = self.input_embedding(obs2 - obs1)
            hidden_states_to_pool = hidden_cell_stacked[0].detach()
            hidden_cell_stacked[0] += self.pool(hidden_states_to_pool, obs1, obs2)


        # step
        hidden_cell_stacked = lstm(input_emb, hidden_cell_stacked)
        normal_masked = self.hidden2normal(hidden_cell_stacked[0])

        # unmask
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

    def tag_step(self, lstm, hidden_cell_state, tag):
        """Update step for all LSTMs with a start tag."""
        hidden_cell_state = (
            torch.stack([h for h in hidden_cell_state[0]], dim=0),
            torch.stack([c for c in hidden_cell_state[1]], dim=0),
        )
        hidden_cell_state = lstm(tag, hidden_cell_state)
        return (
            list(hidden_cell_state[0]),
            list(hidden_cell_state[1]),
        )

    def forward(self, observed, goals, prediction_truth=None, n_predict=None):
        """forward
        observed shape is (seq, n_tracks, observables)
        """
        assert ((prediction_truth is None) + (n_predict is None)) == 1
        if n_predict is not None:
            # -1 because one prediction is done by the encoder already
            prediction_truth = [None for _ in range(n_predict - 1)]

        # initialize: Because of tracks with different lengths and the masked
        # update, the hidden state for every LSTM needs to be a separate object
        # in the backprop graph. Therefore: list of hidden states instead of
        # a single higher rank Tensor.
        n_tracks = observed.size(1)
        hidden_cell_state = (
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(n_tracks)],
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(n_tracks)],
        )

        ## LSTM-Based Interaction Encoders. Initialze Hdden state
        if self.pool.__class__.__name__ in {'NN_LSTM', 'TrajectronPooling', 'SAttention', 'SAttention_fast'}:
            self.pool.reset(num=n_tracks, device=observed.device)

        # list of predictions
        normals = []  # predicted normal parameters for both phases
        positions = []  # true (during obs phase) and predicted positions

        if len(observed) == 2:
            positions = [observed[-1]]

        # encoder
        for obs1, obs2 in zip(observed[:-1], observed[1:]):
            ##LSTM Step
            hidden_cell_state, normal = self.step(self.encoder, hidden_cell_state, obs1, obs2, goals)

            # concat predictions
            normals.append(normal)
            positions.append(obs2 + normal[:, :2])  # no sampling, just mean

        # initialize predictions with last position to form velocity
        prediction_truth = list(itertools.chain.from_iterable(
            (observed[-1:], prediction_truth[:-1])
        ))

########################################################################################################################
        ## ADD NOISE
        hidden_cell_state = self.adding_noise(hidden_cell_state)
########################################################################################################################

        # decoder, predictions
        for obs1, obs2 in zip(prediction_truth[:-1], prediction_truth[1:]):
            if obs1 is None:
                obs1 = positions[-2].detach()  # DETACH!!!
            else:
                obs1[0] = positions[-2][0].detach()  # DETACH!!!
            if obs2 is None:
                obs2 = positions[-1].detach()
            else:
                obs2[0] = positions[-1][0].detach()
            hidden_cell_state, normal = self.step(self.decoder, hidden_cell_state, obs1, obs2, goals)

            # concat predictions
            normals.append(normal)
            positions.append(obs2 + normal[:, :2])  # no sampling, just mean

        ##Pred_scene: Absolute positions -->  19 x n_person x 2
        ##Rel_pred_scene: Next step wrt current step --> 19 x n_person x 5
        rel_pred_scene = torch.stack(normals, dim=0)
        pred_scene = torch.stack(positions, dim=0)

        return rel_pred_scene, pred_scene

class LSTMDiscriminator(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128, pool=None, pool_to_input=True, goal_dim=None, goal_flag=False):
        super(LSTMDiscriminator, self).__init__()
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
        input_dim = (self.embedding_dim + self.goal_dim) if self.goal_flag else self.embedding_dim
        self.goal_embedding = InputEmbedding(2, self.goal_dim, scale)

        print("INPUT DIM: ", input_dim)
        ## Pooling
        if self.pool is not None and self.pool_to_input:
            self.encoder = torch.nn.LSTMCell(input_dim + self.pool.out_dim, self.hidden_dim)
        else:
            self.encoder = torch.nn.LSTMCell(input_dim, self.hidden_dim)

        ## Classifier
        real_classifier_dims = [self.hidden_dim, int(self.hidden_dim / 2), int(self.hidden_dim / 4), 1]
        self.real_classifier = make_mlp(
            real_classifier_dims
            # activation=activation,
            # dropout=dropout
        )

    def step(self, lstm, hidden_cell_state, obs1, obs2, goals):
        """Do one step: two inputs to one normal prediction."""
        # mask for pedestrians absent from scene (partial trajectories)
        # consider only the hidden states of pedestrains present in scene
        track_mask = (torch.isnan(obs1[:, 0]) + torch.isnan(obs2[:, 0])) == 0
        obs1, obs2, goals = obs1[track_mask], obs2[track_mask], goals[track_mask]

        ## LSTM-Based Interaction Encoders. Provide track_mask
        if self.pool.__class__.__name__ in {'NN_LSTM', 'TrajectronPooling', 'SAttention', 'SAttention_fast'}:
            self.pool.track_mask = track_mask

        hidden_cell_stacked = [
            torch.stack([h for m, h in zip(track_mask, hidden_cell_state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(track_mask, hidden_cell_state[1]) if m], dim=0),
        ]

        norm_factors = (torch.norm(obs2 - goals, dim=1))
        goal_direction = (obs2 - goals) / norm_factors.unsqueeze(1)
        goal_direction[norm_factors == 0] = torch.tensor([0., 0], device=obs1.device)
        # goal_direction = obs2 - goals

        # input embedding and optional pooling
        if self.pool is None:
            if self.goal_flag:
                input_emb = torch.cat([self.input_embedding(obs2 - obs1), self.goal_embedding(goal_direction)], dim=1)
            else:
                input_emb = self.input_embedding(obs2 - obs1)
        elif self.pool_to_input:
            hidden_states_to_pool = hidden_cell_stacked[0].clone() ## detach()
            pooled = self.pool(hidden_states_to_pool, obs1, obs2)
            # input_emb = self.input_embedding(torch.cat([obs2 - obs1, goal_direction, pooled], dim=1))
            if self.goal_flag:
                input_emb = torch.cat([self.input_embedding(obs2 - obs1), self.goal_embedding(goal_direction), pooled], dim=1)
            else:
                input_emb = torch.cat([self.input_embedding(obs2 - obs1), pooled], dim=1)
        else:
            if self.goal_flag:
                input_emb = torch.cat([self.input_embedding(obs2 - obs1), self.goal_embedding(goal_direction)], dim=1)
            else:
                input_emb = self.input_embedding(obs2 - obs1)
            hidden_states_to_pool = hidden_cell_stacked[0].detach()
            hidden_cell_stacked[0] += self.pool(hidden_states_to_pool, obs1, obs2)

        # step
        hidden_cell_stacked = lstm(input_emb, hidden_cell_stacked)

        # unmask
        mask_index = [i for i, m in enumerate(track_mask) if m]
        for i, h, c in zip(mask_index,
                           hidden_cell_stacked[0],
                           hidden_cell_stacked[1]):
            hidden_cell_state[0][i] = h
            hidden_cell_state[1][i] = c

        return hidden_cell_state

    def forward(self, observed, prediction, goals):
        """forward
        observed shape is (seq, n_tracks, observables)
        """

        observed = torch.cat([observed, prediction], dim=0)

        # initialize: Because of tracks with different lengths and the masked
        # update, the hidden state for every LSTM needs to be a separate object
        # in the backprop graph. Therefore: list of hidden states instead of
        # a single higher rank Tensor.
        n_tracks = observed.size(1)
        hidden_cell_state = (
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(n_tracks)],
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(n_tracks)],
        )

        ## LSTM-Based Interaction Encoders. Initialze Hdden state
        if self.pool.__class__.__name__ in {'NN_LSTM', 'TrajectronPooling', 'SAttention', 'SAttention_fast'}:
            self.pool.reset(num=n_tracks, device=observed.device)

        if len(observed) == 2:
            positions = [observed[-1]]

        # encoder
        for obs1, obs2 in zip(observed[:-1], observed[1:]):
            ##LSTM Step
            hidden_cell_state = self.step(self.encoder, hidden_cell_state, obs1, obs2, goals)

        hidden_cell_state = (
            torch.stack([h for h in hidden_cell_state[0]], dim=0),
            torch.stack([c for c in hidden_cell_state[1]], dim=0),
        )

        ## Score only the primary pedestrian
        scores = self.real_classifier(hidden_cell_state[0][0])
        return scores

class SGANPredictor(object):
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
        self.model.use_d = False
        modes = 20 #(Trajnet Eval)
        if modes is not None:
            self.model.k = modes

        with torch.no_grad():
            xy = trajnetplusplustools.Reader.paths_to_xy(paths)

            ## Drop Distant
            ## REAL
            xy, mask = drop_distant(xy, r=6.0)
            scene_goal = scene_goal[mask]

            if args.normalize_scene:
                xy, rotation, center, scene_goal = center_scene(xy, obs_length, goals=scene_goal)
            xy = torch.Tensor(xy)  #.to(self.device)
            scene_goal = torch.Tensor(scene_goal) #.to(device)

            multimodal_outputs = {}
            ## model.k outputs
            _, output_scenes_list, _, _ = self.model(xy[:obs_length], scene_goal, n_predict=n_predict)
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
