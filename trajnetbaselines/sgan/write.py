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
parser.add_argument('--num_samples', default=20, type=int)
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

def evaluate(args, generator, num_samples, data, model_name):
    datasets = sorted([f for f in os.listdir(data.replace('_pred', '')) if not f.startswith('.') and f.endswith('.ndjson')])

    if not os.path.exists(data + model_name):
        os.makedirs(data + model_name)

    ## Start writing in dataset/test_pred
    for dataset in datasets:
        # Model's name
        name = dataset.replace(data.replace('_pred', '') + 'test/', '')

        # Copy file from test into test/train_pred folder
        shutil.copyfile(data.replace('_pred', '') + name, data + '{}/{}'.format(model_name, name))
        print('processing ' + name)

        # Read file from 'test'
        reader = trajnettools.Reader(data.replace('_pred', '') + dataset, scene_type='paths')
        scenes = [s for s in reader.scenes()]

        print("Model Name: ", model_name)

        # Write the prediction
        with open(data + '{}/{}'.format(model_name, name), "a") as myfile:
            for scene_id, paths in scenes:
                # print("ID: ", scene_id)
                observed_path = paths[0]
                ped_id = observed_path[0].pedestrian
                frame_diff = observed_path[1].frame - observed_path[0].frame
                first_frame = observed_path[8].frame + frame_diff
                with torch.no_grad():
                    xy = trajnettools.Reader.paths_to_xy(paths)
                    # print(xy.shape)
                    xy = xy.transpose(1, 0, 2)[~np.isnan(xy.transpose(1, 0, 2)).any(axis=1).any(axis=1)]
                    # import pdb
                    # pdb.set_trace()

                    # xy = drop_distant(xy)
                    xy = torch.Tensor(xy.transpose(1, 0, 2)).type(torch.float)  #.to(self.device)
                    # print(xy.shape)
                    # import pdb
                    # pdb.set_trace()

                    ## Go from xy to Obs
                    traj_gt = xy
                    obs_traj = traj_gt
                    obs_traj_rel = torch.zeros(obs_traj.shape).type(torch.float)
                    obs_traj_rel[1:] = obs_traj[1:] - obs_traj[:-1]

                    seq_start_end = torch.LongTensor([[0, obs_traj.shape[1]]])
                    ## Multimodal
                    for num_s in range(num_samples):
                        pred_traj_fake_rel = generator(
                            obs_traj.cuda(), obs_traj_rel.cuda(), seq_start_end.cuda()
                        )
                        pred_traj_fake = relative_to_abs(
                            pred_traj_fake_rel.cuda(), obs_traj[-1].cuda()
                        )
                        ## Get OUTPUT Primary
                        outputs = pred_traj_fake[:, 0].cpu().numpy().astype(float)
                        # import pdb
                        # pdb.set_trace()
                        for i in range(len(outputs)):
                            track = trajnettools.TrackRow(first_frame + i * frame_diff, ped_id,
                                                          outputs[i, 0], outputs[i, 1], num_s, scene_id)
                            myfile.write(trajnettools.writers.trajnet(track))
                            myfile.write('\n')
    return

    #                 predictions = []
    #                 for _ in range(num_samples):
    #                     pred_traj_fake_rel = generator(
    #                         obs_traj, obs_traj_rel, seq_start_end
    #                     )
    #                     pred_traj_fake = relative_to_abs(
    #                         pred_traj_fake_rel, obs_traj[-1]
    #                     )
    #                     predictions.append(pred_traj_fake)

    #                 predictions = predictor(paths)
    #                 for m in range(len(predictions)):
    #                     prediction, neigh_predictions = predictions[m]
                        
    #                     ## Write Primary
    #                     for i in range(len(prediction)):
    #                         track = trajnettools.TrackRow(prediction[i].frame, prediction[i].pedestrian,
    #                                                       prediction[i].x.item(), prediction[i].y.item(), m, scene_id)
    #                         myfile.write(trajnettools.writers.trajnet(track))
    #                         myfile.write('\n')

    #         print('')
    # with torch.no_grad():
    #     for batch in loader:
    #         batch = [tensor.cuda() for tensor in batch]
    #         (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
    #          non_linear_ped, loss_mask, seq_start_end) = batch

    #         ade, fde = [], []
    #         total_traj += pred_traj_gt.size(1)

    #         for _ in range(num_samples):
    #             pred_traj_fake_rel = generator(
    #                 obs_traj, obs_traj_rel, seq_start_end
    #             )
    #             pred_traj_fake = relative_to_abs(
    #                 pred_traj_fake_rel, obs_traj[-1]
    #             )

        #         ade.append(displacement_error(
        #             pred_traj_fake, pred_traj_gt, mode='raw'
        #         ))
        #         fde.append(final_displacement_error(
        #             pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
        #         ))

        #     ade_sum = evaluate_helper(ade, seq_start_end)
        #     fde_sum = evaluate_helper(fde, seq_start_end)

        #     ade_outer.append(ade_sum)
        #     fde_outer.append(fde_sum)
        # ade = sum(ade_outer) / (total_traj * args.pred_len)
        # fde = sum(fde_outer) / (total_traj)
        # return ade, fde



# def loader(dataset, batch_size=1):
#     l = len(dataset)
#     for ndx in range(0, l, batch_size):
#         scenes = [torch.Tensor(scene).type(torch.float).permute(1, 0, 2) for (_, scene) in dataset[ndx:min(ndx + batch_size, l)]]
#         ## Obs and Pred
#         traj_gt = torch.cat(scenes, dim=0).permute(1, 0, 2)[:16]
#         obs_traj = traj_gt[:8]
#         pred_traj_gt = traj_gt[8:16]
#         obs_traj_rel = torch.zeros(obs_traj.shape).type(torch.float)
#         obs_traj_rel[1:] = obs_traj[1:] - obs_traj[:-1]
#         pred_traj_gt_rel = torch.zeros(pred_traj_gt.shape).type(torch.float)
#         pred_traj_gt_rel[1:] = pred_traj_gt[1:] - pred_traj_gt[:-1]
        
#         # print(obs_traj.shape)
#         # print(obs_traj_rel.shape)
#         # print(pred_traj_gt.shape)
#         # print(pred_traj_gt_rel.shape)

#         ## Loss Mask
#         loss_mask = torch.ones((traj_gt.shape[1], traj_gt.shape[0])).type(torch.float)
#         # print(loss_mask.shape)

#         ## Non-Linear 
#         non_linear_ped = torch.zeros(traj_gt.shape[1]).type(torch.float)
#         # print(non_linear_ped.shape)
        
#         ## Num Peds
#         num_peds_in_seq = [scene.shape[0] for scene in scenes]
#         cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
#         seq_start_end = np.array([
#             [start, end]
#             for start, end in zip(cum_start_idx, cum_start_idx[1:])
#         ])
#         seq_start_end = torch.LongTensor(seq_start_end)
#         # print(seq_start_end)

#         # import pdb
#         # pdb.set_trace()

#         yield (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
#                non_linear_ped, loss_mask, seq_start_end)

    ## Model names are passed as arguments
    # for model in ['blah/sgan.pkl']:
    #     model_name = model.split('/')[-1].replace('.pkl', '')


        # ## Make a directory in DATA_BLOCK which will contain the model outputs
        # ## If model is already written, you skip writing
        # if not os.path.exists(data):
        #     os.makedirs(data)
        # if not os.path.exists(data + model_name):
        #     os.makedirs(data + model_name)
        # else:
        #     continue

def main(args):
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
        model_name = path.split('/')[-1].replace('_with_model.pt', '')
        print(model_name)
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        # path = get_dset_path(_args.dataset_name, args.dset_type)
        # _, loader = data_loader(_args, path)
        evaluate(_args, generator, args.num_samples, args.data, model_name)
        print('Done')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)