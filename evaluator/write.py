import trajnettools
import trajnetbaselines.kalman as kalman
# from trajnetbaselines import socialforce
import trajnetbaselines
import shutil
import os
import argparse
import torch

def main(args, kf=False, sf=False):
    ## List of .json file inside the args.data (waiting to be predicted by the testing model)
    datasets = sorted([f for f in os.listdir(args.data.replace('_pred', '')) if not f.startswith('.') and f.endswith('.ndjson')])

    if kf:
        args.output.append('/kf.pkl')
    if sf:
        args.output.append('/sf.pkl')
    
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

            print("Model Name: ", model_name)
            # Load the model
            if model_name == 'kf':
                predictor = trajnetbaselines.kalman.predict
            elif model_name == 'sf':
                predictor = trajnetbaselines.socialforce.predict
            else:
                predictor = trajnetbaselines.lstm.LSTMPredictor.load(model)
                # On CPU
                device = torch.device('cpu')
                predictor.model.to(device)
            
            # Write the prediction
            with open(args.data + '{}/{}'.format(model_name, name), "a") as myfile:
                for scene_id, paths in scenes:
                    predictions = predictor(paths)
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