import itertools

import numpy as np
import torch

import trajnetplusplustools

from .modules import Hidden2Normal, InputEmbedding

from .. import augmentation
from .utils import center_scene

NAN = float('nan')

def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask], mask


class LSTM(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128, pool=None, pool_to_input=True, goal_dim=None, goal_flag=False):
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
        goal_direction[norm_factors == 0] = torch.tensor([0., 0.], device=obs1.device)

        # input embedding and optional goal/pooling
        if self.pool is None:
            if self.goal_flag:
                input_emb = torch.cat([self.input_embedding(obs2 - obs1), self.goal_embedding(goal_direction)], dim=1)
            else:
                input_emb = self.input_embedding(obs2 - obs1)
        elif self.pool_to_input:
            hidden_states_to_pool = hidden_cell_stacked[0].clone() ## detach() ?
            pooled = self.pool(hidden_states_to_pool, obs1, obs2)
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
            (observed[-1:], prediction_truth)
        ))

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

        ##Pred_scene: Absolute positions -->  seq_length x n_person x 2
        ##Rel_pred_scene: Next step wrt current step --> seq_length x n_person x 5
        rel_pred_scene = torch.stack(normals, dim=0)
        pred_scene = torch.stack(positions, dim=0)

        return rel_pred_scene, pred_scene

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


    def __call__(self, paths, scene_goal, n_predict=12, modes=1, predict_all=True, obs_length=9, start_length=0, args=None):
        self.model.eval()
        # self.model.train()
        with torch.no_grad():
            xy = trajnetplusplustools.Reader.paths_to_xy(paths)

            ## Drop Distant (for real data)
            # xy, mask = drop_distant(xy, r=15.0)
            # scene_goal = scene_goal[mask]

            if args.normalize_scene:
                xy, rotation, center, scene_goal = center_scene(xy, obs_length, goals=scene_goal)
            xy = torch.Tensor(xy)  #.to(self.device)
            scene_goal = torch.Tensor(scene_goal) #.to(device)

            multimodal_outputs = {}
            for num_p in range(modes):
                # _, output_scenes = self.model(xy[start_length:obs_length], scene_goal, xy[obs_length:-1].clone())
                _, output_scenes = self.model(xy[start_length:obs_length], scene_goal, n_predict=n_predict)
                output_scenes = output_scenes.numpy()
                if args.normalize_scene:
                    output_scenes = augmentation.inverse_scene(output_scenes, rotation, center)
                output_primary = output_scenes[-n_predict:, 0]
                output_neighs = output_scenes[-n_predict:, 1:]
                ## Dictionary of predictions. Each key corresponds to one mode
                multimodal_outputs[num_p] = [output_primary, output_neighs]

        ## Return Dictionary of predictions. Each key corresponds to one mode
        return multimodal_outputs
