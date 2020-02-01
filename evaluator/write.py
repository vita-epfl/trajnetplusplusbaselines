import trajnettools
import trajnetbaselines.kalman as kalman
import trajnetbaselines
import shutil
import os
import argparse
import torch

def main(args, kf=False, sf=False, orca=False):
    ## List of .json file inside the args.data (waiting to be predicted by the testing model)
    datasets = sorted([f for f in os.listdir(args.data.replace('_pred', '')) if not f.startswith('.') and f.endswith('.ndjson')])

    ## Handcrafted Baselines 
    if kf:
        args.output.append('/kf.pkl')
    if sf:
        args.output.append('/sf.pkl')
        args.output.append('/sf_opt.pkl')
    if orca:
        args.output.append('/orca.pkl')
        args.output.append('/orca_opt.pkl')

    ## Model names are passed as arguments
    for model in args.output:
        model_name = model.split('/')[-1].replace('.pkl', '')

        ## Make a directory in DATA_BLOCK which will contain the model outputs
        ## If model is already written, you skip writing
        if not os.path.exists(args.data):
            os.makedirs(args.data)
        if not os.path.exists(args.data + model_name):
            os.makedirs(args.data + model_name)
        else:
            continue

        ## Start writing in dataset/test_pred
        for dataset in datasets:
            # Model's name
            name = dataset.replace(args.data.replace('_pred', '') + 'test/', '')

            # Copy file from test into test/train_pred folder
            shutil.copyfile(args.data.replace('_pred', '') + name, args.data + '{}/{}'.format(model_name, name))
            print('processing ' + name)

            # Read file from 'test'
            reader = trajnettools.Reader(args.data.replace('_pred', '') + dataset, scene_type='paths')
            scenes = [s for s in reader.scenes()]

            # Loading the appropriate model
            print("Model Name: ", model_name)
            if model_name == 'kf':
                print("Kalman")
                predictor = trajnetbaselines.kalman.predict
            elif model_name == 'sf' or model_name == 'sf_opt':
                print("Social Force")
                predictor = trajnetbaselines.socialforce.socialforce.predict
            elif model_name == 'orca' or model_name == 'orca_opt':
                print("ORCA")
                predictor = trajnetbaselines.socialforce.orca.predict
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

            # Write the prediction
            with open(args.data + '{}/{}'.format(model_name, name), "a") as myfile:
                for scene_id, paths in scenes:
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
                            track = trajnettools.TrackRow(prediction[i].frame, prediction[i].pedestrian,
                                                          prediction[i].x.item(), prediction[i].y.item(), m, scene_id)
                            myfile.write(trajnettools.writers.trajnet(track))
                            myfile.write('\n')

                        ## Write Neighbours
                        for n in range(len(neigh_predictions)):
                            neigh = neigh_predictions[n]
                            for j in range(len(neigh)):
                                track = trajnettools.TrackRow(neigh[j].frame, neigh[j].pedestrian,
                                                              neigh[j].x.item(), neigh[j].y.item(), m, scene_id)
                                myfile.write(trajnettools.writers.trajnet(track))
                                myfile.write('\n')
        print('')

if __name__ == '__main__':
    main()