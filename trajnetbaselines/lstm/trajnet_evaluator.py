import os
import argparse
import pickle

from joblib import Parallel, delayed
import scipy
import torch
from tqdm import tqdm

from evaluator.trajnet_evaluator import trajnet_evaluate
from evaluator.write_utils import load_test_datasets, preprocess_test, write_predictions
from .lstm import LSTMPredictor


def predict_scene(predictor, model_name, paths, scene_goal, args):
    """For each scene, get model predictions"""
    paths = preprocess_test(paths, args.obs_length)
    predictions = predictor(paths, scene_goal, n_predict=args.pred_length, obs_length=args.obs_length, modes=args.modes, args=args)
    return predictions


def load_predictor(model, device='cpu'):
    """Loading the APPROPRIATE model"""
    predictor = LSTMPredictor.load(model)
    predictor.model.to(torch.device(device))
    return predictor


def get_predictions(args):
    """Get model predictions for each test scene and write the predictions in appropriate folders"""
    ## List of .json file inside the args.path (waiting to be predicted by the testing model)
    datasets = sorted([f.split('.')[-2] for f in os.listdir(args.path.replace('_pred', '')) if not f.startswith('.') and f.endswith('.ndjson')])

    ## Extract Model names from arguments and create its own folder in 'test_pred' for storing predictions
    ## WARNING: If Model predictions already exist from previous run, this process SKIPS WRITING
    for model in args.output:
        model_name = model.split('/')[-1].replace('.pkl', '')
        model_name = model_name + '_modes' + str(args.modes)

        ## Check if model predictions already exist
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        if not os.path.exists(args.path + model_name):
            os.makedirs(args.path + model_name)
        else:
            print('Predictions corresponding to {} already exist.'.format(model_name))
            print('Loading the saved predictions')
            continue

        print("Model Name: ", model_name)
        predictor = load_predictor(model)
        goal_flag = predictor.model.goal_flag

        # Iterate over test datasets
        for dataset in datasets:
            # Load dataset
            dataset_name, scenes, scene_goals = load_test_datasets(dataset, goal_flag, args)

            # Get all predictions in parallel. Faster!
            scenes = tqdm(scenes)
            pred_list = Parallel(n_jobs=12)(delayed(predict_scene)(predictor, model_name, paths, scene_goal, args)
                                            for (_, _, paths), scene_goal in zip(scenes, scene_goals))
            
            # Write all predictions
            write_predictions(pred_list, scenes, model_name, dataset_name, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='trajdata',
                        help='directory of data to test')
    parser.add_argument('--output', nargs='+',
                        help='relative path to saved model')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--write_only', action='store_true',
                        help='disable writing new files')
    parser.add_argument('--disable-collision', action='store_true',
                        help='disable collision metrics')
    parser.add_argument('--labels', required=False, nargs='+',
                        help='labels of models')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='augment scenes')
    parser.add_argument('--modes', default=1, type=int,
                        help='number of modes to predict')
    args = parser.parse_args()

    scipy.seterr('ignore')

    args.output = args.output if args.output is not None else []
    args.path = 'DATA_BLOCK/' + args.path + '/test_pred/'

    ## Writes to Test_pred
    ## Does NOT overwrite existing predictions if they already exist ###
    get_predictions(args)
    if args.write_only: # For submission to AICrowd.
        print("Predictions written in test_pred folder")
        exit()

    ## Evaluate using TrajNet++ evaluator
    trajnet_evaluate(args)


if __name__ == '__main__':
    main()
