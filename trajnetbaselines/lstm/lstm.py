from collections import defaultdict
import itertools

import torch
import trajnettools

from .modules import Hidden2Normal, InputEmbedding

NAN = float('nan')


def scene_to_xy(scene, drop_distance=5.0):
    """Return a Torch Tensor representing the scene.

    Drop other paths that are never closer than drop_distance.
    """
    frames = [r.frame for r in scene[0]]
    pedestrians = [path[0].pedestrian for path in scene]

    frame_to_index = {frame: i for i, frame in enumerate(frames)}
    xy = torch.full((len(frames), len(pedestrians), 2), NAN)

    for ped_index, path in enumerate(scene):
        for row in path:
            if row.frame not in frame_to_index:
                continue
            entry = xy[frame_to_index[row.frame]][ped_index]
            entry[0] = row.x
            entry[1] = row.y

    if drop_distance is not None:
        drop_distance_2 = drop_distance**2
        all_distances = xy - xy[:, 0:1]
        all_distances_2 = torch.sum(torch.mul(all_distances, all_distances), dim=2)

        # handle nan entries
        all_distances_2[torch.isnan(all_distances_2)] = drop_distance_2

        mask = torch.min(all_distances_2, dim=0)[0] < drop_distance_2
        xy = xy[:, mask]

    return xy


class LSTM(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128, pool=None):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.input_embedding = InputEmbedding(self.embedding_dim, 4.0)
        self.pool = pool
        self.encoder = torch.nn.LSTMCell(self.embedding_dim, self.hidden_dim)
        self.decoder = torch.nn.LSTMCell(self.embedding_dim, self.hidden_dim)

        # Predict the parameters of a multivariate normal:
        # mu_vel_x, mu_vel_y, sigma_vel_x, sigma_vel_y, rho
        self.hidden2normal = Hidden2Normal(self.hidden_dim)

    def step(self, lstm, hidden_cell_state, obs1, obs2):
        """Do one step: two inputs to one normal prediction."""
        # mask
        track_mask = (torch.isnan(obs1[:, 0]) + torch.isnan(obs2[:, 0])) == 0
        obs1, obs2 = obs1[track_mask], obs2[track_mask]
        hidden_cell_stacked = [
            torch.stack([h for m, h in zip(track_mask, hidden_cell_state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(track_mask, hidden_cell_state[1]) if m], dim=0),
        ]

        # step
        coordinate_emb = self.input_embedding(obs2 - obs1)
        if self.pool is not None:
            hidden_cell_stacked[0] += self.pool(hidden_cell_stacked[0], obs1, obs2)
        hidden_cell_stacked = lstm(coordinate_emb, hidden_cell_stacked)
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
        # without pooling, only look at the primary track
        if self.pool is None:
            observed = observed[:, 0:1]
            if prediction_truth is not None:
                prediction_truth = prediction_truth[:, 0:1]

        assert ((prediction_truth is None) + (n_predict is None)) == 1
        if n_predict is not None:
            prediction_truth = [None for _ in range(n_predict)]

        # initialize: Because of tracks with different lengths and the masked
        # update, the hidden state for every LSTM needs to be a separate object
        # in the backprop graph. Therefore: list of hidden states instead of
        # a single higher rank Tensor.
        n_tracks = observed.size(1)
        hidden_cell_state = (
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(n_tracks)],
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(n_tracks)],
        )

        # encoder
        normals = []  # predicted normal parameters for both phases
        positions = []  # true (during obs phase) and predicted positions
        start_enc_tag = self.input_embedding.start_enc(observed[0])
        hidden_cell_state = self.tag_step(self.encoder, hidden_cell_state, start_enc_tag)
        for obs1, obs2 in zip(observed[:-1], observed[1:]):
            hidden_cell_state, normal = self.step(self.encoder, hidden_cell_state, obs1, obs2)

            # save outputs
            normals.append(normal)
            positions.append(obs2)

        # do not include the last prediction as this will be done with the
        # decoder below
        normals = normals[:-1]
        positions = positions[1:]  # positions are not predicted, but truth

        # initialize predictions with last two positions
        prediction_truth = list(itertools.chain.from_iterable(
            (positions[-2:], prediction_truth)
        ))

        # decoder, predictions
        start_dec_tag = self.input_embedding.start_dec(observed[0])
        hidden_cell_state = self.tag_step(self.decoder, hidden_cell_state, start_dec_tag)
        for obs1, obs2 in zip(prediction_truth[:-1], prediction_truth[1:]):
            if obs1 is None:
                obs1 = positions[-2].detach()  # DETACH!!!
            else:
                obs1[0] = positions[-2][0].detach()  # DETACH!!!
            if obs2 is None:
                obs2 = positions[-1].detach()
            else:
                obs2[0] = positions[-1][0].detach()

            hidden_cell_state, normal = self.step(self.decoder, hidden_cell_state, obs1, obs2)

            # save outputs
            normals.append(normal)
            positions.append(obs2 + normal[:, :2])  # no sampling, just mean

        return torch.stack(normals if self.training else positions, dim=0)[:, 0]


class LSTMPredictor(object):
    def __init__(self, model):
        self.model = model

    def save(self, filename):
        with open(filename, 'wb') as f:
            torch.save(self, f)

        # during dev, good for compatibility across API changes:
        with open(filename.replace('.pkl', '') + '_state_dict.pkl', 'wb') as f:
            torch.save(self.model.state_dict(), f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return torch.load(f)

    def __call__(self, scene, n_predict=12):
        self.model.eval()

        observed_path = scene[0]
        ped_id = observed_path[0].pedestrian
        with torch.no_grad():
            xy = scene_to_xy(scene)
            outputs = self.model(xy[:9], n_predict=n_predict)[-n_predict:]
            # outputs = self.model(xy[:9], xy[9:-1])[-n_predict:]

        return [trajnettools.TrackRow(0, ped_id, x, y) for x, y in outputs]
