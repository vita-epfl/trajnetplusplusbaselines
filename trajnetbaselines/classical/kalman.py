import numpy as np
import pykalman
import trajnetplusplustools


def predict(paths, predict_all=True, n_predict=12, obs_length=9):
    multimodal_outputs = {}
    neighbours_tracks = []

    primary = paths[0]
    start_frame = primary[obs_length-1].frame
    frame_diff = primary[1].frame - primary[0].frame
    first_frame = start_frame + frame_diff

    ## Primary Prediction
    if not predict_all:
        paths = paths[0:1]

    for i, path in enumerate(paths):
        path = paths[i]
        ped_id = path[0].pedestrian
        past_path = [t for t in path if t.frame <= start_frame]
        past_frames = [t.frame for t in path if t.frame <= start_frame]

        ## To consider agent or not consider.
        if start_frame not in past_frames:
            continue
        if len(past_path) < 2:
            continue

        initial_state_mean = [path[0].x, 0, path[0].y, 0]
        transition_matrix = [[1, 1, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 1],
                             [0, 0, 0, 1]]

        observation_matrix = [[1, 0, 0, 0],
                              [0, 0, 1, 0]]

        kf = pykalman.KalmanFilter(transition_matrices=transition_matrix,
                                   observation_matrices=observation_matrix,
                                   transition_covariance=1e-5 * np.eye(4),
                                   observation_covariance=0.05**2 * np.eye(2),
                                   initial_state_mean=initial_state_mean)
        # kf.em([(r.x, r.y) for r in path[:9]], em_vars=['transition_matrices',
        #                                                'observation_matrices'])
        kf.em([(r.x, r.y) for r in past_path])
        observed_states, _ = kf.smooth([(r.x, r.y) for r in past_path])


        # sample predictions (first sample corresponds to last state)
        # average 5 sampled predictions
        predictions = None
        for _ in range(5):
            _, pred = kf.sample(n_predict + 1, initial_state=observed_states[-1])
            if predictions is None:
                predictions = pred
            else:
                predictions += pred
        predictions /= 5.0

        #write
        if i == 0:
            primary_track = predictions[1:]
        else:
            neighbours_tracks.append(np.array(predictions[1:]))

    ## Unimodal Ouput
    neighbours_tracks = []
    if len(np.array(neighbours_tracks)):
        neighbours_tracks = np.array(neighbours_tracks).transpose(1, 0, 2)

    multimodal_outputs[0] = primary_track, neighbours_tracks
    return multimodal_outputs
