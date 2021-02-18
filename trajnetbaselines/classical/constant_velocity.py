import numpy as np
import trajnetplusplustools

def predict(input_paths, predict_all=True, n_predict=12, obs_length=9):
    multimodal_outputs = {}
    pred_length = n_predict

    xy = trajnetplusplustools.Reader.paths_to_xy(input_paths)
    curr_position = xy[-1]
    curr_velocity = xy[-1] - xy[-2]
    output_rel_scenes = np.array([i * curr_velocity for i in range(1, n_predict+1)])
    output_scenes = curr_position + output_rel_scenes

    output_primary = output_scenes[-n_predict:, 0]
    output_neighs = output_scenes[-n_predict:, 1:]

    # Unimodal Prediction
    multimodal_outputs[0] = output_primary, output_neighs

    return multimodal_outputs