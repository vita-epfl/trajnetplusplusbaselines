import os
import argparse
import pickle

from joblib import Parallel, delayed
import scipy
import torch
from tqdm import tqdm

from evaluator.trajnet_evaluator import trajnet_evaluate
from evaluator.write_utils import load_test_datasets, preprocess_test, write_predictions


def predict_scene(predictor, model_name, paths, scene_goal, args):
    """For each scene, get model predictions"""
    paths = preprocess_test(paths, args.obs_length)
    if 'sf_opt' in model_name:
        predictions = predictor(paths, sf_params=[0.5, 5.0, 0.3], n_predict=args.pred_length, obs_length=args.obs_length) ## optimal sf_params (no collision constraint) [0.5, 1.0, 0.1],
    elif 'orca_opt' in model_name:
        predictions = predictor(paths, orca_params=[0.4, 1.0, 0.3], n_predict=args.pred_length, obs_length=args.obs_length) ## optimal orca_params (no collision constraint) [0.25, 1.0, 0.3]
    elif  ('sf' in model_name) or ('orca' in model_name) or ('kf' in model_name):
        predictions = predictor(paths, n_predict=args.pred_length, obs_length=args.obs_length)
    elif 'cv' in model_name:
        predictions = predictor(paths, n_predict=args.pred_length, obs_length=args.obs_length)
    else:
        raise NotImplementedError
    return predictions


def load_predictor(model_name):
    """Loading the APPROPRIATE model"""
    if 'kf' in model_name:
        print("Kalman")
        from .kalman import predict as predictor
    elif 'sf' in model_name:
        print("Social Force")
        from .socialforce import predict as predictor
    elif 'orca' in model_name:
        print("ORCA")
        from .orca import predict as predictor
    elif 'cv' in model_name:
        print("CV")
        from .constant_velocity import predict as predictor
    return predictor


def get_predictions(args):
    """Get model predictions for each test scene and write the predictions in appropriate folders"""
    ## List of .json file inside the args.path (waiting to be predicted by the testing model)
    datasets = sorted([f.split('.')[-2] for f in os.listdir(args.path.replace('_pred', '')) if not f.startswith('.') and f.endswith('.ndjson')])

    ## Handcrafted Baselines (if included)
    if args.kf:
        args.output.append('/kf.pkl')
    if args.sf:
        args.output.append('/sf.pkl')
        args.output.append('/sf_opt.pkl')
    if args.orca:
        args.output.append('/orca.pkl')
        args.output.append('/orca_opt.pkl')
    if args.cv:
        args.output.append('/cv.pkl')

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
        predictor = load_predictor(model_name)
        goal_flag = False

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
    parser.add_argument('--sf', action='store_true',
                        help='consider socialforce in evaluation')
    parser.add_argument('--orca', action='store_true',
                        help='consider orca in evaluation')
    parser.add_argument('--kf', action='store_true',
                        help='consider kalman in evaluation')
    parser.add_argument('--cv', action='store_true',
                        help='consider constant velocity in evaluation')
    args = parser.parse_args()

    scipy.seterr('ignore')

    args.output = []
    ## assert length of output models is not None
    if (not args.sf) and (not args.orca) and (not args.kf) and (not args.cv):
        assert 'No handcrafted baseline mentioned'

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
