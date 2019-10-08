from collections import defaultdict
import itertools

import numpy as np
import torch
import trajnettools

from .modules import Hidden2Normal, InputEmbedding

NAN = float('nan')

def drop_distant(xy, r=10.0):
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    if not all(any(e == e for e in column) for column in distance_2.T):
        print(distance_2.tolist())
        print(np.nanmin(distance_2, axis=0))
        raise Exception
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask]


class LSTM(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128, pool=None, pool_to_input=True):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.pool = pool
        self.pool_to_input = pool_to_input

        self.input_embedding = InputEmbedding(2, self.embedding_dim, 4.0)
        if self.pool is not None and self.pool_to_input:
            self.input_embedding = InputEmbedding(2 + self.pool.out_dim, self.embedding_dim, 4.0)

        self.encoder = torch.nn.LSTMCell(self.embedding_dim, self.hidden_dim)
        self.decoder = torch.nn.LSTMCell(self.embedding_dim, self.hidden_dim)

        # Predict the parameters of a multivariate normal:
        # mu_vel_x, mu_vel_y, sigma_vel_x, sigma_vel_y, rho
        self.hidden2normal = Hidden2Normal(self.hidden_dim)

    def step(self, lstm, hidden_cell_state, obs1, obs2):
        """Do one step: two inputs to one normal prediction."""
        # mask for pedestrians absent from scene (partial trajectories)
        # consider only the hidden states of pedestrains present in scene
        track_mask = (torch.isnan(obs1[:, 0]) + torch.isnan(obs2[:, 0])) == 0
        obs1, obs2 = obs1[track_mask], obs2[track_mask]
        hidden_cell_stacked = [
            torch.stack([h for m, h in zip(track_mask, hidden_cell_state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(track_mask, hidden_cell_state[1]) if m], dim=0),
        ]

        # input embedding and optional pooling
        if self.pool is None:
            input_emb = self.input_embedding(obs2 - obs1)
        elif self.pool_to_input:
            hidden_states_to_pool = hidden_cell_stacked[0].detach()
            pooled = self.pool(hidden_states_to_pool, obs1, obs2)
            input_emb = self.input_embedding(torch.cat([obs2 - obs1, pooled], dim=1))
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

    def forward(self, observed, prediction_truth=None, n_predict=None):
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

        # list of predictions
        normals = []  # predicted normal parameters for both phases
        positions = []  # true (during obs phase) and predicted positions

        #tag the start of encoding (optional)
        start_enc_tag = self.input_embedding.start_enc(observed[0])
        hidden_cell_state = self.tag_step(self.encoder, hidden_cell_state, start_enc_tag)

        # encoder
        for obs1, obs2 in zip(observed[:-1], observed[1:]):
            ##LSTM Step
            hidden_cell_state, normal = self.step(self.encoder, hidden_cell_state, obs1, obs2)

            # concat predictions
            normals.append(normal)
            positions.append(obs2 + normal[:, :2])  # no sampling, just mean

        # initialize predictions with last position to form velocity
        prediction_truth = list(itertools.chain.from_iterable(
            (observed[-1:], prediction_truth)
        ))

        # decoder, predictions
        start_dec_tag = self.input_embedding.start_dec(observed[0])
        hidden_cell_state = self.tag_step(self.decoder, hidden_cell_state, start_dec_tag)
        for obs1, obs2 in zip(prediction_truth[:-1], prediction_truth[1:]):
            obs1 = positions[-2].detach()  # DETACH!!!
            obs2 = positions[-1].detach()  # DETACH!!!
            hidden_cell_state, normal = self.step(self.decoder, hidden_cell_state, obs1, obs2)

            # concat predictions
            normals.append(normal)
            positions.append(obs2 + normal[:, :2])  # no sampling, just mean
        
        ##Pred_scene: Absolute positions -->  19 x n_person x 2
        ##Rel_pred_scene: Next step wrt current step --> 19 x n_person x 5
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


    def __call__(self, paths, n_predict=12, modes=1):
        self.model.eval()

        observed_path = paths[0]
        ped_id = observed_path[0].pedestrian
        ped_id_ = []
        for j in range(len(paths)):
            ped_id_.append(paths[j][0].pedestrian)
        frame_diff = observed_path[1].frame - observed_path[0].frame
        first_frame = observed_path[8].frame + frame_diff
        with torch.no_grad():
            xy = trajnettools.Reader.paths_to_xy(paths)
            xy = drop_distant(xy, r=10.0)
            xy = torch.Tensor(xy)  #.to(self.device)
            multimodal_outputs = {}
            for np in range(modes):
                _, output_scenes = self.model(xy[:9], n_predict=n_predict)
                outputs = output_scenes[-n_predict:, 0]
                output_scenes = output_scenes[-n_predict:]
                output_primary = [trajnettools.TrackRow(first_frame + i * frame_diff, ped_id, outputs[i, 0],
                                  outputs[i, 1], 0) for i in range(len(outputs))]

                output_all = [[trajnettools.TrackRow(first_frame + i * frame_diff, ped_id_[j], output_scenes[i, j, 0],
                                              output_scenes[i, j, 1], 0) for i in range(len(outputs))] for j in range(1, output_scenes.shape[1])]

                multimodal_outputs[np] = [output_primary, output_all]
        return multimodal_outputs