""" Fast evaluator for visualizing the LRP of LSTM forecasting models """

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

def process_scene(predictor, model_name, paths, scene_goal, scene_id, args):
    ## For each scene, get predictions
    predictions = predictor(paths, scene_goal, n_predict=args.pred_length, obs_length=args.obs_length, modes=args.modes, scene_id=scene_id, args=args)
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
    parser.add_argument('--normalize_scene', action='store_true',
                        help='augment scenes')
    parser.add_argument('--modes', default=1, type=int,
                        help='number of modes to predict')
    args = parser.parse_args()

    scipy.seterr('ignore')

    ## Path to the data folder name to predict
    args.path = 'DATA_BLOCK/' + args.path + '/'

    ## Test_pred: Folders for saving model predictions
    args.path = args.path + 'test_pred/'

    ## Writes to Test_pred
    ## Does this overwrite existing predictions? No. ###
    datasets = sorted([f.split('.')[-2] for f in os.listdir(args.path.replace('_pred', '')) if not f.startswith('.') and f.endswith('.ndjson')])

    ## Model names are passed as arguments
    for model in args.output:
        model_name = model.split('/')[-1].replace('.pkl', '')

        # Loading the appropriate model (functionality only for SGAN and LSTM)
        print("Model Name: ", model_name)
        predictor = trajnetbaselines.lstm.LSTMPredictor.load(model)
        goal_flag = predictor.model.goal_flag
        # On CPU
        device = torch.device('cpu')
        predictor.model.to(device)

        all_goals = {}
        ## Start writing in dataset/test_pred
        for dataset in datasets:
            # Model's name
            name = dataset.replace(args.path.replace('_pred', '') + 'test/', '')

            # Read file from 'test'
            reader = trajnetplusplustools.Reader(args.path.replace('_pred', '') + dataset + '.ndjson', scene_type='paths')
            ## Necessary modification of train scene to add filename (for goals)
            scenes = [(dataset, s_id, s) for s_id, s in reader.scenes()]

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

            print("Saving LRP GIFs")
            ## Change Scene IDs to be saved as animations.
            pred_list = Parallel(n_jobs=1)(delayed(process_scene)(predictor, model_name, paths, scene_goal, scene_id, args)
                                            for (_, scene_id, paths), scene_goal in zip(scenes[41:42], scene_goals[41:42]))

if __name__ == '__main__':
    main()