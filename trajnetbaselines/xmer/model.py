import itertools
import copy

import numpy as np
import torch

import trajnetplusplustools

from ..lstm.modules import Hidden2Normal, InputEmbedding

from .. import augmentation
from ..lstm.utils import center_scene

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

NAN = float('nan')


def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask], mask

def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class MyNewModel(torch.nn.Module):
    def __init__(self, pool = None, goal_dim = None, d_model= 64, nhead = 4, d_hid = 256, nlayers = 4, dropout = 0.0):
        """"
        d_model : embedding dimension
        nhead : number of heads in nn.MultiheadAttention
        d_hid : dimension of the feedforward network model in nn.TransformerEncoder
        nlayers : number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        dropout : dropout probability
        """

        super(MyNewModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model+pool.out_dim+goal_dim, dropout)
        encoder_layers = TransformerEncoderLayer(d_model+pool.out_dim+goal_dim, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = torch.nn.Linear(2, d_model)
        # self.encoder = InputEmbedding(2, d_model, 4.0)
        self.d_model = d_model
        self.decoder = torch.nn.Linear(d_model+pool.out_dim+goal_dim, 2)

        self.pool = pool

        scale = 4.0
        self.goal_dim = goal_dim
        self.goal_embedding = InputEmbedding(2, goal_dim, scale)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, positions, batch_split, goals):
        velocities = positions[1:]-positions[:-1]
        src = self.encoder(velocities)

        goal_emb = self.goals(positions, goals)

        pooled = self.pooling(positions, batch_split)
        src = torch.cat([src, pooled, goal_emb], dim=2) * math.sqrt(self.d_model+self.pool.out_dim+self.goal_dim)
        src_mask = generate_square_subsequent_mask(velocities.shape[0])
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

    def construct_next_step_input(self, positions, batch_split, goals):
        velocities = positions[1:]-positions[:-1]
        src = self.encoder(velocities)

        goal_emb = self.goals(positions, goals)

        pooled = self.pooling(positions, batch_split)
        src = torch.cat([src, pooled, goal_emb], dim=2) * math.sqrt(self.d_model+self.pool.out_dim+self.goal_dim)
        return src
    
    def forward_step(self, src):
        src_mask = generate_square_subsequent_mask(src.shape[0])
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


    def forward_sequential(self, observations, predictions_gt, batch_split, goals, pred_len=12):
        
        next_step_inp = self.construct_next_step_input(observations, batch_split, goals)
        src = next_step_inp.clone()
        predictions = observations[-1:].clone()
        for pred_ind in range(pred_len):
            curr_pred = self.forward_step(src)
            curr_pred_pos = curr_pred[-1:] + observations[-1:]
            predictions_curr = predictions_gt[pred_ind:pred_ind+1].clone()
            predictions_curr[:,batch_split[:-1],:] = curr_pred_pos[:,batch_split[:-1],:].clone()
            observations = torch.cat([observations, predictions_curr[-1:].detach()], dim=0) # DETACH !!
            next_step_inp = self.construct_next_step_input(observations[-2:], batch_split, goals)
            src = torch.cat([src, next_step_inp], dim=0)
            predictions = torch.cat([predictions, predictions_curr], dim=0)
        rel_predictions = predictions[1:] - predictions[:-1]
        return rel_predictions, predictions

    def get_predictions(self, src, pred_gt, batch_split, pred_len, goals):
        # Using neighbor GT!
        orig = pred_gt.clone()
        orig_src = src.clone()

        curr_pred = self.forward(src, batch_split, goals)
        curr_pred_pos = curr_pred[-1:] + src[-1:]
        predictions = pred_gt[0:1].clone()
        predictions[:,batch_split[:-1],:] = curr_pred_pos[:,batch_split[:-1],:].clone()
        src = torch.cat([src, predictions[-1:]], dim=0)
        for pred_ind in range(pred_len-1):
            curr_pred = self.forward(src, batch_split, goals)
            curr_pred_pos = curr_pred[-1:] + src[-1:]
            predictions_curr = pred_gt[pred_ind+1:pred_ind+2].clone()
            predictions_curr[:,batch_split[:-1],:] = curr_pred_pos[:,batch_split[:-1],:].clone()
            src = torch.cat([src, predictions_curr[-1:]], dim=0)
            predictions = torch.cat([predictions, predictions_curr], dim=0)

        return predictions

        # Using neighbor predictions!
        # curr_pred = self.forward(src, batch_split)
        # predictions = curr_pred[-1:] + src[-1:]
        # src = torch.cat([src, curr_pred[-1:] + src[-1:]], dim=0)
        # for _ in range(pred_len-1):
        #     curr_pred = self.forward(src, batch_split)
        #     predictions = torch.cat([predictions, curr_pred[-1:] + src[-1:]], dim=0)
        #     src = torch.cat([src, curr_pred[-1:] + src[-1:]], dim=0)
        # return predictions

    def goals(self, positions, goals):
        
        goal_emb_full = []
        for obs1, obs2 in zip(positions[:-1], positions[1:]):
            norm_factors = (torch.norm(obs2 - goals, dim=1))
            goal_direction = (obs2 - goals) / norm_factors.unsqueeze(1)
            goal_direction[norm_factors == 0] = torch.tensor([0., 0.], device=obs1.device)
            goal_emb = self.goal_embedding(goal_direction).unsqueeze(0)
            goal_emb_full.append(goal_emb)

        return torch.cat(goal_emb_full)
            
    @staticmethod
    def generate_pooling_inputs(obs2, obs1, track_mask, batch_split):
        # tensor for pooling; filled with nan-mask [bs, max # neighbor, 2]
        max_num_neighbor = (batch_split[1:] - batch_split[:-1]).max()   # number of agents in a scene minus the primary
        batch_size = len(batch_split) - 1
        curr_positions = torch.empty(batch_size, max_num_neighbor, 2).fill_(float('nan')).to(obs1.device) # placeholder
        prev_positions = torch.empty(batch_size, max_num_neighbor, 2).fill_(float('nan')).to(obs1.device) # placeholder
        track_mask_positions = torch.empty(batch_size, max_num_neighbor).fill_(False).bool().to(obs1.device)  # placeholder

        for i in range(batch_size):
            curr_positions[i, :batch_split[i+1]-batch_split[i]] = obs2[batch_split[i]:batch_split[i+1]]
            prev_positions[i, :batch_split[i+1]-batch_split[i]] = obs1[batch_split[i]:batch_split[i+1]]
            track_mask_positions[i, :batch_split[i+1]-batch_split[i]] = True
            # track_mask_positions[i, :batch_split[i+1]-batch_split[i]] = track_mask[batch_split[i]:batch_split[i+1]].bool()

        return curr_positions, prev_positions, track_mask_positions

    def pooling(self, positions, batch_split):
        pooled_output = []
        track_mask = None
        for obs1, obs2 in zip(positions[:-1], positions[1:]):
            curr_positions, prev_positions, track_mask_positions = \
                self.generate_pooling_inputs(obs2, obs1, track_mask, batch_split)
            pool_sample = self.pool(None, prev_positions, curr_positions)
            pooled = pool_sample[track_mask_positions.view(-1)]
            pooled_output.append(pooled)

        return torch.stack(pooled_output)

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MyNewModelPredictor(object):
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
        pass
