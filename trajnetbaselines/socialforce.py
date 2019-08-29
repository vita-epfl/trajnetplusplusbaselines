import numpy as np
import socialforce
from scipy.interpolate import interp1d
import trajnettools

def predict(input_paths):

    def init_states(input_paths):
        initial_state = []
        for i in range(len(input_paths)):
            path = input_paths[i][:9]
            track_prev = path[-2]
            track = path[-1]
            x = track.x 
            y = track.y
            [v_x, v_y] = vel_state(path)
            [d_x, d_y] = dest_state(path)
            initial_state.append([x, y, v_x, v_y, d_x, d_y])
        return np.array(initial_state)

    def vel_state(path):
        prev = path[-2]
        curr = path[-1] 
        diff = np.array([curr.x - prev.x, curr.y - prev.y])
        theta = np.arctan2(diff[1], diff[0])
        speed = np.linalg.norm(diff) / (1 * 0.4)
        return [speed*np.cos(theta), speed*np.sin(theta)]

    def dest_state(path):
        x = [t.x for t in path]
        y = [t.y for t in path]
        time = [i for i in range(9)]
        f = interp1d(x=time[:9], y=[x, y], fill_value='extrapolate')
        return f(20)

    multimodal_outputs = {}
    primary = input_paths[0]
    neighbours_tracks = []
    frame_diff = primary[1].frame - primary[0].frame
    first_frame = primary[-1].frame + frame_diff

    initial_state = init_states(input_paths)

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

    multimodal_outputs[0] = primary_track, neighbours_tracks
    return multimodal_outputs
