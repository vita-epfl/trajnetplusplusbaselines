import numpy as np
from scipy.interpolate import interp1d

import trajnettools

import socialforce

def predict(input_paths, dest_dict=None, dest_type='true', orca_params=None, predict_all=False):

    def init_states(input_paths, start_frame, dest_dict, dest_type):
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

                positions.append((curr.x, curr.y))
                goals.append((d_x, d_y))

                sim.addAgent((curr.x, curr.y))
                speed.append((v_x, v_y))
        
        trajectories = [[positions[i]] for i in range(len(positions))]
        return trajectories, positions, goals, speed

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

    sim = rvo2.PyRVOSimulator(1 / 2.5, 2, 10, 2, 2, 0.4, 1.2)

    # initialize
    trajectories, positions, goals, speed = init_states(input_paths, start_frame, dest_dict, dest_type)
    
    ##Simulate a scene
    while not done and count < 12:
        sim.doStep()
        reaching_goal = []
        for i in range(num_ped):
            if count == 0:
                trajectories[i].pop(0)
            position = sim.getAgentPosition(i)

            ## Append only if Goal not reached
            if not reaching_goal_by_ped[i]:
                trajectories[i].append(position)

            # check if this agent reaches the goal
            if np.linalg.norm(np.array(position) - np.array(goals[i])) < end_range:
                reaching_goal.append(True)
                sim.setAgentPrefVelocity(i, (0, 0))
                reaching_goal_by_ped[i] = True
            else:
                reaching_goal.append(False)
                velocity = np.array((goals[i][0] - position[0], goals[i][1] - position[1]))
                speed = np.linalg.norm(velocity)
                pref_vel = 1 * velocity / speed if speed > 1 else velocity
                sim.setAgentPrefVelocity(i, tuple(pref_vel.tolist()))
        count += 1
        done = all(reaching_goal)


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
