import pickle

import numpy as np
import trajnetplusplustools


def load_test_datasets(dataset, goal_flag, args):
    """Load Test Prediction file with goals (optional)"""
    all_goals = {}
    dataset_name = dataset.replace(args.path.replace('_pred', '') + 'test/', '') + '.ndjson'
    print('Dataset Name: ', dataset_name)

    # Read Scenes from 'test' folder
    reader = trajnetplusplustools.Reader(args.path.replace('_pred', '') + dataset + '.ndjson', scene_type='paths')
    ## Necessary modification of train scene to add filename (for goals)
    scenes = [(dataset, s_id, s) for s_id, s in reader.scenes()]

    ## Consider goals
    ## Goal file must be present in 'goal_files/test_private' folder
    ## Goal file must have the same name as corresponding test file
    if goal_flag:
        print("Loading Goal File: ", 'goal_files/test_private/' + dataset +'.pkl')
        goal_dict = pickle.load(open('goal_files/test_private/' + dataset +'.pkl', "rb"))
        all_goals[dataset] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scenes}
        scene_goals = [np.array(all_goals[filename][scene_id]) for filename, scene_id, _ in scenes]
    else:
        scene_goals = [np.zeros((len(paths), 2)) for _, scene_id, paths in scenes]

    return dataset_name, scenes, scene_goals


def preprocess_test(scene, obs_len):
    """Remove pedestrian trajectories that appear post observation period.
    Can occur when the test set has overlapping scenes."""
    obs_frames = [primary_row.frame for primary_row in scene[0]][:obs_len]
    last_obs_frame = obs_frames[-1]
    scene = [[row for row in ped if row.frame <= last_obs_frame]
                for ped in scene if ped[0].frame <= last_obs_frame]
    return scene


def write_predictions(pred_list, scenes, model_name, dataset_name, args):
    """Write predictions corresponding to the scenes in the respective file"""
    seq_length = args.obs_length + args.pred_length
    with open(args.path + '{}/{}'.format(model_name, dataset_name), "a") as myfile:
        ## Write All Predictions
        for (predictions, (_, scene_id, paths)) in zip(pred_list, scenes):
            ## Extract 1) first_frame, 2) frame_diff 3) ped_ids for writing predictions
            observed_path = paths[0]
            frame_diff = observed_path[1].frame - observed_path[0].frame
            first_frame = observed_path[args.obs_length-1].frame + frame_diff
            ped_id = observed_path[0].pedestrian
            ped_id_ = []
            for j, _ in enumerate(paths[1:]): ## Only need neighbour ids
                ped_id_.append(paths[j+1][0].pedestrian)

            ## Write SceneRow
            scenerow = trajnetplusplustools.SceneRow(scene_id, ped_id, observed_path[0].frame, 
                                                        observed_path[0].frame + (seq_length - 1) * frame_diff, 2.5, 0)
            # scenerow = trajnetplusplustools.SceneRow(scenerow.scene, scenerow.pedestrian, scenerow.start, scenerow.end, 2.5, 0)
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
