""" Writes the Model Predictions of test file in test_pred folder"""

import trajnettools
import trajnetbaselines
import shutil
import os
import argparse
import torch

def main(args, kf=False, sf=False, orca=False):
    ## List of test files (.json) inside the test folder (waiting to be predicted by the prediction model)
    datasets = sorted([f for f in os.listdir(args.data.replace('_pred', '')) if not f.startswith('.') and f.endswith('.ndjson')])

    ## Handcrafted Baselines (if required to compare)
    if kf:
        args.output.append('/kf.pkl')
    if sf:
        args.output.append('/sf.pkl')
        args.output.append('/sf_opt.pkl')
    if orca:
        args.output.append('/orca.pkl')
        args.output.append('/orca_opt.pkl')

    ## Extract Model names from arguments and create its own folder in 'test_pred' for storing predictions
    ## WARNING: If Model predictions already exist from previous run, this process SKIPS WRITING
    for model in args.output:
        model_name = model.split('/')[-1].replace('.pkl', '')

        ## Check if model predictions already exist
        if not os.path.exists(args.data):
            os.makedirs(args.data)
        if not os.path.exists(args.data + model_name):
            os.makedirs(args.data + model_name)
        else:
            continue

        ## Start writing predictions in dataset/test_pred
        for dataset in datasets:
            # Model's name
            name = dataset.replace(args.data.replace('_pred', '') + 'test/', '')

            # Copy observations from test folder into test_pred folder
            shutil.copyfile(args.data.replace('_pred', '') + name, args.data + '{}/{}'.format(model_name, name))
            print('processing ' + name)

            # Read Scenes from 'test' folder
            reader = trajnettools.Reader(args.data.replace('_pred', '') + dataset, scene_type='paths')
            scenes = [s for s in reader.scenes()]

            # Loading the APPROPRIATE model
            ## Keep Adding Different Models to this List
            print("Model Name: ", model_name)
            if model_name == 'kf':
                print("Kalman")
                predictor = trajnetbaselines.classical.kalman.predict
            elif model_name == 'sf' or model_name == 'sf_opt':
                print("Social Force")
                predictor = trajnetbaselines.classical.socialforce.predict
            elif model_name == 'orca' or model_name == 'orca_opt':
                print("ORCA")
                predictor = trajnetbaselines.classical.orca.predict
            elif 'sgan' in model_name:
                print("SGAN")
                predictor = trajnetbaselines.sgan.SGANPredictor.load(model)
                # On CPU
                device = torch.device('cpu')
                predictor.model.to(device)
            else:
                print("LSTM")
                predictor = trajnetbaselines.lstm.LSTMPredictor.load(model)
                # On CPU
                device = torch.device('cpu')
                predictor.model.to(device)

            # Get the model prediction and write them in corresponding test_pred file
            """ 
            VERY IMPORTANT: Prediction Format

            The predictor function should output a dictionary. The keys of the dictionary should correspond to the prediction modes. 
            ie. predictions[0] corresponds to the first mode. predictions[m] corresponds to the m^th mode.... Multimodal predictions!
            Each modal prediction comprises of primary prediction and neighbour (surrrounding) predictions i.e. predictions[m] = [primary_prediction, neigh_predictions]
            Note: Return [primary_prediction, []] if model does not provide neighbour predictions

            Shape of primary_prediction: Tensor of Shape (Prediction length, 2)
            Shape of Neighbour_prediction: Tensor of Shape (Prediction length, n_tracks - 1, 2).
            (See LSTMPredictor.py for more details)
            """
            with open(args.data + '{}/{}'.format(model_name, name), "a") as myfile:
                for scene_id, paths in scenes:

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
                        predictions = predictor(paths, sf_params=[0.5, 1.0, 0.1], n_predict=args.pred_length, obs_length=args.obs_length) ## optimal sf_params
                    elif model_name == 'orca_opt':
                        predictions = predictor(paths, orca_params=[0.25, 1.0, 0.3], n_predict=args.pred_length, obs_length=args.obs_length) ## optimal orca_params
                    else:
                        predictions = predictor(paths, n_predict=args.pred_length, obs_length=args.obs_length)

                    for m in range(len(predictions)):
                        prediction, neigh_predictions = predictions[m]
                        ## Write Primary
                        for i in range(len(prediction)):
                            # print(i)
                            track = trajnettools.TrackRow(first_frame + i * frame_diff, ped_id,
                                                          prediction[i, 0].item(), prediction[i, 1].item(), m, scene_id)
                            myfile.write(trajnettools.writers.trajnet(track))
                            myfile.write('\n')

                        ## Write Neighbours (if non-empty)
                        for n in range(neigh_predictions.shape[1]):
                            neigh = neigh_predictions[:, n]
                            for j in range(len(neigh)):
                                track = trajnettools.TrackRow(first_frame + j * frame_diff, ped_id_[n],
                                                              neigh[j, 0].item(), neigh[j, 1].item(), m, scene_id)
                                myfile.write(trajnettools.writers.trajnet(track))
                                myfile.write('\n')
        print('')

if __name__ == '__main__':
    main()