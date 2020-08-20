import shutil
import os
import pickle

import torch
import numpy as np

import trajnetplusplustools
import trajnetbaselines

def get_goals(paths, goal_dict, filename, scene_id):
    ## get goals
    if len(goal_dict):
        scene_goal = np.array(goal_dict[filename][scene_id])
    else:
        scene_goal = np.array([[0, 0] for path in paths])
    return scene_goal

def main(args=None):
    ## List of .json file inside the args.path (waiting to be predicted by the testing model)
    datasets = sorted([f.split('.')[-2] for f in os.listdir(args.path.replace('_pred', '')) if not f.startswith('.') and f.endswith('.ndjson')])
    all_goals = {}
    seq_length = args.obs_length + args.pred_length

    ## Handcrafted Baselines (if included)
    if args.kf:
        args.output.append('/kf.pkl')
    if args.sf:
        args.output.append('/sf.pkl')
        args.output.append('/sf_opt.pkl')
    if args.orca:
        args.output.append('/orca.pkl')
        args.output.append('/orca_opt.pkl')

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

        ## Start writing predictions in dataset/test_pred
        for dataset in datasets:
            # Model's name
            name = dataset.replace(args.path.replace('_pred', '') + 'test/', '') + '.ndjson'
            print('NAME: ', name)

            # Read Scenes from 'test' folder
            reader = trajnetplusplustools.Reader(args.path.replace('_pred', '') + dataset + '.ndjson', scene_type='paths')
            ## Necessary modification of train scene to add filename (for goals)
            scenes = [(dataset, s_id, s) for s_id, s in reader.scenes()]
            ## Consider goals
            ## Goal file must be present in 'goal_files/test_private' folder 
            ## Goal file must have the same name as corresponding test file 
            if args.goals:
                goal_dict = pickle.load(open('goal_files/test_private/' + dataset +'.pkl', "rb"))
                all_goals[dataset] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scenes}

            # Loading the APPROPRIATE model
            ## Keep Adding Different Model Architectures to this List
            print("Model Name: ", model_name)
            if model_name == 'kf':
                print("Kalman")
                predictor = trajnetbaselines.classical.kalman.predict
            elif model_name in {'sf', 'sf_opt'}:
                print("Social Force")
                predictor = trajnetbaselines.classical.socialforce.predict
            elif model_name in {'orca', 'orca_opt'}:
                print("ORCA")
                predictor = trajnetbaselines.classical.orca.predict
            elif 'sgan' in model_name:
                print("SGAN")
                predictor = trajnetbaselines.sgan.SGANPredictor.load(model)
                device = torch.device('cpu')
                predictor.model.to(device)
            elif 'lstm' in model_name:
                print("LSTM")
                predictor = trajnetbaselines.lstm.LSTMPredictor.load(model)
                device = torch.device('cpu')
                predictor.model.to(device)
            else:
                print("Model Architecture not recognized")
                raise ValueError

            # Get the model prediction and write them in corresponding test_pred file
            # VERY IMPORTANT: Prediction Format
            # The predictor function should output a dictionary. The keys of the dictionary should correspond to the prediction modes.
            # ie. predictions[0] corresponds to the first mode. predictions[m] corresponds to the m^th mode.... Multimodal predictions!
            # Each modal prediction comprises of primary prediction and neighbour (surrrounding) predictions i.e. predictions[m] = [primary_prediction, neigh_predictions]
            # Note: Return [primary_prediction, []] if model does not provide neighbour predictions
            # Shape of primary_prediction: Tensor of Shape (Prediction length, 2)
            # Shape of Neighbour_prediction: Tensor of Shape (Prediction length, n_tracks - 1, 2).
            # (See LSTMPredictor.py for more details)
            with open(args.path + '{}/{}'.format(model_name, name), "a") as myfile:
                for filename, scene_id, paths in scenes:
                    ## Extract 1) first_frame, 2) frame_diff 3) ped_ids for writing predictions
                    observed_path = paths[0]
                    frame_diff = observed_path[1].frame - observed_path[0].frame
                    first_frame = observed_path[args.obs_length-1].frame + frame_diff
                    ped_id = observed_path[0].pedestrian
                    ped_id_ = []
                    for j, _ in enumerate(paths[1:]): ## Only need neighbour ids
                        ped_id_.append(paths[j+1][0].pedestrian)

                    ## For each scene, get predictions
                    if model_name == 'sf_opt':
                        predictions = predictor(paths, sf_params=[0.5, 5.0, 0.3], n_predict=args.pred_length, obs_length=args.obs_length) ## optimal sf_params (no collision constraint) [0.5, 1.0, 0.1],
                    elif model_name == 'orca_opt':
                        predictions = predictor(paths, orca_params=[0.4, 1.0, 0.3], n_predict=args.pred_length, obs_length=args.obs_length) ## optimal orca_params (no collision constraint) [0.25, 1.0, 0.3]
                    elif model_name in {'sf', 'orca', 'kf'}:
                        predictions = predictor(paths, n_predict=args.pred_length, obs_length=args.obs_length, args=args)
                    else:
                        goals = get_goals(paths, all_goals, filename, scene_id) ## Zeros if no goals utilized
                        predictions = predictor(paths, goals, n_predict=args.pred_length, obs_length=args.obs_length, modes=args.modes, args=args)

                    ## Write SceneRow
                    scenerow = trajnetplusplustools.SceneRow(scene_id, ped_id, observed_path[0].frame, 
                                                             observed_path[0].frame + seq_length - 1, 2.5, 0)
                    myfile.write(trajnetplusplustools.writers.trajnet(scenerow))
                    myfile.write('\n')

                    for m in range(len(predictions)):
                        prediction, neigh_predictions = predictions[m]
                        ## Write Primary
                        for i in range(len(prediction)):
                            track = trajnetplusplustools.TrackRow(first_frame + i * frame_diff, ped_id,
                                                                  prediction[i, 0].item(), prediction[i, 1].item(), m, scene_id)
                            myfile.write(trajnetplusplustools.writers.trajnet(track))
                            myfile.write('\n')

                        ## Write Neighbours (if non-empty)
                        if len(neigh_predictions):
                            for n in range(neigh_predictions.shape[1]):
                                neigh = neigh_predictions[:, n]
                                for j in range(len(neigh)):
                                    track = trajnetplusplustools.TrackRow(first_frame + j * frame_diff, ped_id_[n],
                                                                          neigh[j, 0].item(), neigh[j, 1].item(), m, scene_id)
                                    myfile.write(trajnetplusplustools.writers.trajnet(track))
                                    myfile.write('\n')
        print('')

if __name__ == '__main__':
    main()
