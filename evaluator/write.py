import trajnettools
import trajnetbaselines.kalman as kalman
import trajnetbaselines
import shutil
import os
import argparse
import pdb

def main(args):
    ## List of .json file inside the args.data (waiting to be predicted by the testing model)
    datasets = sorted([f for f in os.listdir(args.data.replace('_pred', '')) if not f.startswith('.') and f.endswith('.ndjson')])

    ## Load your model
    #models = [f for f in os.listdir(args.output) if f.endswith('.pkl')]

    ## Model names are passed as arguments
    for model in args.output:
        j = -1
        while model[j] != '/':
            j -= 1
        model_name = model[j+1:]
        print('model name: ', model_name)

        ## Make a directory in DATA_BLOCK which will contain the model outputs
        ## If model is already written, you skip writing
        if not os.path.exists(args.data):
            os.makedirs(args.data)
        if not os.path.exists(args.data + model_name.replace('.pkl', '')):
            os.makedirs(args.data + model_name.replace('.pkl', ''))
        else:
            continue

        ## Start writing in dataset/test_pred
        for dataset in datasets:
            # Model's name
            name = dataset.replace(args.data.replace('_pred', '') + 'test/', '')

            # Copy file from test into test/train_pred folder
            shutil.copyfile(args.data.replace('_pred', '') + name, args.data + '{}/{}'.format(model_name.replace('.pkl', ''), name))
            print('processing ' + name)

            # Read file from 'test'
            reader = trajnettools.Reader(args.data.replace('_pred', '') + dataset, scene_type='paths')
            scenes = [s for s in reader.scenes()]

            # Load the model
            lstm_predictor = trajnetbaselines.lstm.LSTMPredictor.load(model)

            # Write the prediction
            with open(args.data + '{}/{}'.format(model_name.replace('.pkl', ''), name), "a") as myfile:
                for scene_id, paths in scenes:
                    predictions = lstm_predictor(paths)
                    for m in range(len(predictions)):
                        prediction, neigh_predictions = predictions[m]
                        # print("Primary: ", prediction[0].pedestrian)
                        
                        ## Write Primary
                        for i in range(len(prediction)):
                            track = trajnettools.TrackRow(prediction[i].frame, prediction[i].pedestrian,
                                                          prediction[i].x.item(), prediction[i].y.item(), m, scene_id)
                            myfile.write(trajnettools.writers.trajnet(track))
                            myfile.write('\n')

                        ## Write Neighbours
                        for n in range(len(neigh_predictions)):
                            # print("n:", n )
                            neigh = neigh_predictions[n]
                            # print(neigh[0].pedestrian)
                            for j in range(len(neigh)):
                                track = trajnettools.TrackRow(neigh[j].frame, neigh[j].pedestrian,
                                                              neigh[j].x.item(), neigh[j].y.item(), m, scene_id)
                                myfile.write(trajnettools.writers.trajnet(track))
                                myfile.write('\n')
        print('')

if __name__ == '__main__':
    main()