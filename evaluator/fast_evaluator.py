""" A fast evaluator for 'overall score' that does not save the model predictions file """

import os
from collections import OrderedDict
import argparse
import pickle

import numpy as np
import scipy
import torch

import trajnetplusplustools
import trajnetbaselines

## Parallel Compute
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

def process_scene(predictor, model_name, paths, scene_goal, args):
    ## For each scene, get predictions
    predictions = predictor(paths, scene_goal, n_predict=args.pred_length, obs_length=args.obs_length, modes=args.modes, args=args)
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='trajdata',
                        help='directory of data to test')
    parser.add_argument('--output', required=True, nargs='+',
                        help='relative path to saved model')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--disable-write', action='store_true',
                        help='disable writing new files')
    parser.add_argument('--disable-collision', action='store_true',
                        help='disable collision metrics')
    parser.add_argument('--labels', required=False, nargs='+',
                        help='labels of models')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='augment scenes')
    parser.add_argument('--unimodal', action='store_true',
                        help='provide unimodal evaluation')
    parser.add_argument('--topk', action='store_true',
                        help='provide topk evaluation')
    parser.add_argument('--multimodal', action='store_true',
                        help='provide multimodal nll evaluation')
    parser.add_argument('--modes', default=1, type=int,
                        help='number of modes to predict')
    parser.add_argument('--scene_type', default=0, type=int,
                        choices=(0, 1, 2, 3, 4),
                        help='type of scene to evaluate')
    parser.add_argument('--thresh', default=0.0, type=float,
                        help='noise thresh')
    parser.add_argument('--ped_type', default='primary',
                        help='type of ped to add noise to')

    args = parser.parse_args()

    scipy.seterr('ignore')

    ## Path to the data folder name to predict
    args.path = 'DATA_BLOCK/' + args.path + '/'

    ## Test_pred: Folders for saving model predictions
    args.path = args.path + 'test_pred/'

    if (not args.unimodal) and (not args.topk) and (not args.multimodal):
        args.unimodal = True # Compute unimodal metrics by default

    if args.topk:
        args.modes = 3

    if args.multimodal:
        args.modes = 20

    enable_col1 = True
    ## drop pedestrians that appear post observation
    def drop_post_obs(ground_truth, obs_length):
        obs_end_frame = ground_truth[0][obs_length].frame
        ground_truth = [track for track in ground_truth if track[0].frame < obs_end_frame]
        return ground_truth

    ## Writes to Test_pred
    ## Does this overwrite existing predictions? No. ###
    datasets = sorted([f.split('.')[-2] for f in os.listdir(args.path.replace('_pred', '')) if not f.startswith('.') and f.endswith('.ndjson')])

    ## Model names are passed as arguments
    for model in args.output:
        model_name = model.split('/')[-1].replace('.pkl', '')

        # Loading the appropriate model (functionality only for SGAN and LSTM)
        print("Model Name: ", model_name)
        if 'sgan' in model_name:
            predictor = trajnetbaselines.sgan.SGANPredictor.load(model)
            goal_flag = predictor.model.generator.goal_flag
        elif 'vae' in model_name:
            predictor = trajnetbaselines.vae.VAEPredictor.load(model)
            goal_flag = predictor.model.goal_flag
        else:
            predictor = trajnetbaselines.lstm.LSTMPredictor.load(model)
            goal_flag = predictor.model.goal_flag

        # On CPU
        device = torch.device('cpu')
        predictor.model.to(device)

        total_scenes = 0
        average = 0
        final = 0
        gt_col = 0.
        pred_col = 0.
        neigh_scenes = 0
        topk_average = 0
        topk_final = 0
        all_goals = {}
        average_nll = 0

        ## Start writing in dataset/test_pred
        for dataset in datasets:
            # Model's name
            name = dataset.replace(args.path.replace('_pred', '') + 'test/', '')

            # Copy file from test into test/train_pred folder
            print('processing ' + name)
            if 'collision_test' in name:
                continue

            ## Filter for Scene Type
            reader_tag = trajnetplusplustools.Reader(args.path.replace('_pred', '_private') + dataset + '.ndjson', scene_type='tags')
            if args.scene_type != 0:
                filtered_scene_ids = [s_id for s_id, tag, s in reader_tag.scenes() if tag[0] == args.scene_type]
            else:
                filtered_scene_ids = [s_id for s_id, _, _ in reader_tag.scenes()]

            # Read file from 'test'
            reader = trajnetplusplustools.Reader(args.path.replace('_pred', '') + dataset + '.ndjson', scene_type='paths')
            ## Necessary modification of train scene to add filename (for goals)
            scenes = [(dataset, s_id, s) for s_id, s in reader.scenes() if s_id in filtered_scene_ids]

            ## Consider goals
            ## Goal file must be present in 'goal_files/test_private' folder 
            ## Goal file must have the same name as corresponding test file 
            if goal_flag:
                goal_dict = pickle.load(open('goal_files/test_private/' + dataset +'.pkl', "rb"))
                all_goals[dataset] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scenes}

            ## Get Goals
            if goal_flag:
                scene_goals = [np.array(all_goals[filename][scene_id]) for filename, scene_id, _ in scenes]
            else:
                scene_goals = [np.zeros((len(paths), 2)) for _, scene_id, paths in scenes]

            print("Getting Predictions")
            scenes = tqdm(scenes)
            ## Get all predictions in parallel. Faster!
            pred_list = Parallel(n_jobs=12)(delayed(process_scene)(predictor, model_name, paths, scene_goal, args)
                                            for (_, _, paths), scene_goal in zip(scenes, scene_goals))

            ## GT Scenes
            reader_gt = trajnetplusplustools.Reader(args.path.replace('_pred', '_private') + dataset + '.ndjson', scene_type='paths')
            scenes_gt = [s for s_id, s in reader_gt.scenes() if s_id in filtered_scene_ids]
            total_scenes += len(scenes_gt)

            print("Evaluating Predictions")
            scenes = tqdm(scenes)
            for (predictions, (_, scene_id, paths), ground_truth) in zip(pred_list, scenes, scenes_gt):

                ## Extract 1) first_frame, 2) frame_diff 3) ped_ids for writing predictions
                observed_path = paths[0]
                frame_diff = observed_path[1].frame - observed_path[0].frame
                first_frame = observed_path[args.obs_length-1].frame + frame_diff
                ped_id = observed_path[0].pedestrian
                ped_id_ = []
                for j, _ in enumerate(paths[1:]): ## Only need neighbour ids
                    ped_id_.append(paths[j+1][0].pedestrian)
                
                if args.unimodal: ## Unimodal
                    ## ADE / FDE
                    prediction, neigh_predictions = predictions[0]
                    prediction = np.round(prediction, 2)
                    ## make Track Rows
                    # primary
                    prediction = [trajnetplusplustools.TrackRow(first_frame + i * frame_diff, ped_id, prediction[i, 0], prediction[i, 1], 0)
                                  for i in range(len(prediction))]

                    primary_tracks = [t for t in prediction if t.prediction_number == 0]
                    frame_gt = [t.frame for t in ground_truth[0]][args.obs_length:args.obs_length+args.pred_length]
                    frame_pred = [t.frame for t in primary_tracks]

                    ## To verify if same scene
                    if frame_gt != frame_pred:
                        raise Exception('frame numbers are not consistent')

                    average_l2 = trajnetplusplustools.metrics.average_l2(ground_truth[0][args.obs_length:args.obs_length+args.pred_length], primary_tracks, n_predictions=args.pred_length)
                    final_l2 = trajnetplusplustools.metrics.final_l2(ground_truth[0][args.obs_length:args.obs_length+args.pred_length], primary_tracks)

                    # aggregate FDE and ADE
                    average += average_l2
                    final += final_l2

                    ground_truth = drop_post_obs(ground_truth, args.obs_length)
                    ## Collision Metrics
                    for j in range(1, len(ground_truth)):
                        if trajnetplusplustools.metrics.collision(primary_tracks, ground_truth[j], n_predictions=args.pred_length):
                            gt_col += 1
                            break

                    num_gt_neigh = len(ground_truth) - 1
                    num_predicted_neigh = neigh_predictions.shape[1]
                    if num_gt_neigh != num_predicted_neigh:
                        enable_col1 = False
                    # [Col-I] only if neighs in gt = neighs in prediction
                    if enable_col1:
                        neigh_scenes += 1
                        for n in range(neigh_predictions.shape[1]):
                            neigh = neigh_predictions[:, n]
                            neigh = np.round(neigh, 2)
                            neigh_track = [trajnetplusplustools.TrackRow(first_frame + j * frame_diff, n, neigh[j, 0], neigh[j, 1], 0)
                                           for j in range(len(neigh))]
                            if trajnetplusplustools.metrics.collision(primary_tracks, neigh_track, n_predictions=args.pred_length):
                                pred_col += 1
                                break

                primary_tracks_all = [trajnetplusplustools.TrackRow(first_frame + i * frame_diff, ped_id, x, y, m)
                                      for m, (prim, neighs) in predictions.items() for i, (x, y) in enumerate(prim)]

                if args.topk:
                    topk_ade, topk_fde = trajnetplusplustools.metrics.topk(primary_tracks_all, ground_truth[0][args.obs_length:args.obs_length+args.pred_length], n_predictions=args.pred_length)
                    topk_average += topk_ade
                    topk_final += topk_fde

                if args.multimodal:
                    nll_val = trajnetplusplustools.metrics.nll(primary_tracks_all, ground_truth[0], n_predictions=args.pred_length, n_samples=20)
                    average_nll += nll_val

        if args.unimodal:
            ## Average ADE and FDE
            average /= total_scenes
            final /= total_scenes
            gt_col /= (total_scenes * 0.01)
            if not enable_col1:
                pred_col = -1
            else:
                pred_col /= (neigh_scenes * 0.01)

            print('ADE: ', np.round(average, 3))
            print('FDE: ', np.round(final, 3))
            print("Col-I: ", np.round(pred_col, 2))
            print("Col-II: ", np.round(gt_col, 2))

        if args.topk:
            topk_average /= total_scenes
            topk_final /= total_scenes
            print('Topk_ADE: ', topk_average)
            print('Topk_FDE: ', topk_final)

        if args.multimodal:
            average_nll /= total_scenes
            print('Average NLL: ', average_nll)

if __name__ == '__main__':
    main()
