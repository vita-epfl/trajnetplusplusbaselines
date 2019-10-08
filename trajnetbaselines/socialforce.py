import numpy as np
from scipy.interpolate import interp1d

import trajnettools

import socialforce

def predict(input_paths, predict_all=False):

    def init_states(input_paths, start_frame):
        initial_state = []
        for i, _ in enumerate(input_paths):
            path = input_paths[i]
            past_path = [t for t in path if t.frame <= start_frame]
            past_frames = [t.frame for t in path if t.frame <= start_frame]
            len_path = len(past_path)

            ## To consider agent or not consider.
            if (start_frame in past_frames) and len_path >= 4:
                curr = past_path[-1]
                prev = past_path[-4]
                [v_x, v_y] = vel_state(prev, curr, 3)
                if np.linalg.norm([v_x, v_y]) < 1e-6:
                    continue
                [d_x, d_y] = dest_state(past_path, len_path-1)
                if np.linalg.norm([curr.x - d_x, curr.y - d_y]) < 1e-6:
                    continue

                initial_state.append([curr.x, curr.y, v_x, v_y, d_x, d_y])
        return np.array(initial_state)

    def vel_state(prev, curr, stride):
        diff = np.array([curr.x - prev.x, curr.y - prev.y])
        theta = np.arctan2(diff[1], diff[0])
        speed = np.linalg.norm(diff) / (stride * 0.4)
        return [speed*np.cos(theta), speed*np.sin(theta)]

    def dest_state(path, stride):
        x = [t.x for t in path]
        y = [t.y for t in path]
        time = list(range(stride+1))
        f = interp1d(x=time, y=[x, y], fill_value='extrapolate')
        return f(time[-1] + 12)

    multimodal_outputs = {}
    primary = input_paths[0]
    neighbours_tracks = []
    frame_diff = primary[1].frame - primary[0].frame
    start_frame = primary[8].frame
    first_frame = primary[8].frame + frame_diff

    # initialize
    initial_state = init_states(input_paths, start_frame)

    # run
    s = socialforce.Simulator(initial_state)
    states = np.stack([s.step().state.copy() for _ in range(12)])
    ## states : 12 x num_ped x 7

    # predictions
    for i in range(states.shape[1]):
        ped_id = input_paths[i][0].pedestrian
        if i == 0:
            primary_track = [trajnettools.TrackRow(first_frame + j * frame_diff, ped_id, x, y)
                             for j, (x, y) in enumerate(states[:, i, 0:2])]
        else:
            neighbours_tracks.append([trajnettools.TrackRow(first_frame + j * frame_diff, ped_id, x, y)
                                      for j, (x, y) in enumerate(states[:, i, 0:2])])

    ## Primary Prediction
    if not predict_all:
        neighbours_tracks = []

    # Unimodal Prediction
    multimodal_outputs[0] = primary_track, neighbours_tracks
    return multimodal_outputs
