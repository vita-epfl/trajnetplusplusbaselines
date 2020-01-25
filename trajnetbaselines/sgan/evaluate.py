import argparse
import os
import torch

from attrdict import AttrDict

import trajnettools
import shutil
import numpy as np 

# from sgan.data.loader import data_loader
from .models import TrajectoryGenerator
from .losses import displacement_error, final_displacement_error
from .utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--data', type=str)
parser.add_argument('--num_samples', default=1, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator

def evaluate(args, generator, num_samples, data):
    datasets = sorted([f for f in os.listdir(data) if not f.startswith('.') and f.endswith('.ndjson')])

    ## Start writing in dataset/test_pred
    for dataset in datasets:
        # Read file from 'test'
        reader = trajnettools.Reader(data + dataset, scene_type='paths')
        scenes = [s for s in reader.scenes()]

        ade_outer, fde_outer = [], []
        total_traj = 0
        with torch.no_grad():
            for scene_id, paths in scenes:
                # print("ID: ", scene_id)
                observed_path = paths[0]
                ped_id = observed_path[0].pedestrian
                frame_diff = observed_path[1].frame - observed_path[0].frame
                first_frame = observed_path[8].frame + frame_diff
                

                xy = trajnettools.Reader.paths_to_xy(paths)
                xy = xy.transpose(1, 0, 2)[~np.isnan(xy.transpose(1, 0, 2)).any(axis=1).any(axis=1)]
                xy = torch.Tensor(xy.transpose(1, 0, 2)).type(torch.float)  #.to(self.device)
                # print(xy.shape)
                # import pdb
                # pdb.set_trace()

                ## Go from xy to Obs
                traj_gt = xy
                obs_traj = traj_gt[:9]
                obs_traj_rel = torch.zeros(obs_traj.shape).type(torch.float)
                obs_traj_rel[1:] = obs_traj[1:] - obs_traj[:-1]

                seq_start_end = torch.LongTensor([[0, obs_traj.shape[1]]])
                
                pred_traj_gt = traj_gt[9:].cuda()

                ade, fde = [], []
                total_traj += pred_traj_gt.size(1)

                ## Multimodal
                for _ in range(num_samples):
                    pred_traj_fake_rel = generator(
                        obs_traj.cuda(), obs_traj_rel.cuda(), seq_start_end.cuda()
                    )
                    pred_traj_fake = relative_to_abs(
                        pred_traj_fake_rel.cuda(), obs_traj[-1].cuda()
                    )
                    ade.append(displacement_error(
                        pred_traj_fake, pred_traj_gt, mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                    ))

                ade_sum = evaluate_helper(ade, seq_start_end)
                fde_sum = evaluate_helper(fde, seq_start_end)

                ade_outer.append(ade_sum)
                fde_outer.append(fde_sum)

        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)

        print("Dataset: ", dataset)
        print("ADE: ", ade)
        print("FDE: ", fde)
    return


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def main(args):
    print("Number of Samples: ", args.num_samples)
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        if 'no_model' in path:
            continue
        print(path)
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        # path = get_dset_path(_args.dataset_name, args.dset_type)
        # _, loader = data_loader(_args, path)
        evaluate(_args, generator, args.num_samples, args.data)
    print('Done')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)