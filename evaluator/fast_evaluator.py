import shutil
import os
import warnings
from collections import OrderedDict
import argparse

import numpy
import trajnettools
import trajnetbaselines
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='trajdata',
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
    args = parser.parse_args()

    ## Path to the data folder name to predict 
    args.data = 'DATA_BLOCK/' + args.data + '/'

    ## Test_pred: Folders for saving model predictions
    args.data = args.data + 'test_pred/'

    ## Writes to Test_pred
    ## Does this overwrite existing predictions? No. ###
    datasets = sorted([f for f in os.listdir(args.data.replace('_pred', '')) if not f.startswith('.') and f.endswith('.ndjson')])

    ## Model names are passed as arguments
    for model in args.output:
        model_name = model.split('/')[-1].replace('.pkl', '')
        # Loading the appropriate model (currently written only for LSTMs)
        print("Model Name: ", model_name)
        predictor = trajnetbaselines.lstm.LSTMPredictor.load(model)
        # On CPU
        device = torch.device('cpu')
        predictor.model.to(device)

        total_scenes = 0
        average = 0
        final = 0
        topk_average = 0
        topk_final = 0

        ## Start writing in dataset/test_pred
        for dataset in datasets:
            # Model's name
            name = dataset.replace(args.data.replace('_pred', '') + 'test/', '')

            # Copy file from test into test/train_pred folder
            print('processing ' + name)
            if 'collision_test' in name:
                continue

            # Read file from 'test'
            reader = trajnettools.Reader(args.data.replace('_pred', '') + dataset, scene_type='paths')
            scenes = [s for _, s in reader.scenes()]

            reader_gt = trajnettools.Reader(args.data.replace('_pred', '_private') + dataset, scene_type='paths')
            scenes_gt = [s for _, s in reader_gt.scenes()]
            scenes_id_gt = [i for i, _ in reader_gt.scenes()]
            total_scenes += len(scenes_gt)

            for i, paths in enumerate(scenes):
                ground_truth = scenes_gt[i]
                predictions = predictor(paths, n_predict=args.pred_length, obs_length=args.obs_length)

                ## Considers only the First MODE
                prediction, neigh_predictions = predictions[0]

                primary_tracks = [t for t in prediction if t.prediction_number == 0]
                # frame_gt = [t.frame for t in ground_truth[0]][-args.pred_length:]
                frame_gt = [t.frame for t in ground_truth[0]][args.obs_length:args.obs_length+args.pred_length]
                frame_pred = [t.frame for t in primary_tracks]

                ## To verify if same scene
                if frame_gt != frame_pred:
                    raise Exception('frame numbers are not consistent')

                average_l2 = trajnettools.metrics.average_l2(ground_truth[0][args.obs_length:args.obs_length+args.pred_length], primary_tracks, n_predictions=args.pred_length)
                final_l2 = trajnettools.metrics.final_l2(ground_truth[0][args.obs_length:args.obs_length+args.pred_length], primary_tracks)
                
                # aggregate FDE and ADE
                average += average_l2
                final += final_l2

                if len(predictions) > 2:
                    # print(predictions)
                    primary_tracks_all = [t for mode in predictions for t in predictions[mode][0]]
                    # import pdb
                    # pdb.set_trace()
                    topk_ade, topk_fde = trajnettools.metrics.topk(primary_tracks_all, ground_truth[0][args.obs_length:args.obs_length+args.pred_length], n_predictions=args.pred_length)
                    topk_average += topk_ade
                    topk_final += topk_fde

        ## Average ADE and FDE
        average /= total_scenes
        final /= total_scenes

        # ##Adding value to dict
        print('ADE: ', average)
        print('FDE: ', final)

        if len(predictions) > 2:
            topk_average /= total_scenes
            topk_final /= total_scenes            

            # ##Adding value to dict    
            print('Topk_ADE: ', topk_average)
            print('Topk_FDE: ', topk_final)

 
if __name__ == '__main__':
    main()

