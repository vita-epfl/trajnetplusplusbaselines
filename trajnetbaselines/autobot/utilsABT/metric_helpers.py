import numpy as np
import torch


def min_xde_K(xdes, probs, K):
    best_ks = probs.argsort(axis=1)[:, -K:]
    dummy_rows = np.expand_dims(np.arange(len(xdes)), 1)
    new_xdes = xdes[dummy_rows, best_ks]
    return np.nanmean(np.sort(new_xdes), axis=0)


def yaw_from_predictions(preds, ego_in, agents_in):
    '''
    For collision detection in the interaction dataset. This function computes the final yaw based on the predicted
    delta yaw.
    '''
    device = preds.device
    new_preds = preds.clone()
    last_obs_yaws = torch.cat((ego_in[:, -1, 7].unsqueeze(-1), agents_in[:, -1, :, 7]), dim=-1)
    T = preds.shape[1]
    yaws = torch.zeros((preds.shape[:4])).to(device)
    for t in range(T):
        yaws[:, t] = last_obs_yaws.unsqueeze(0) + preds[:, t, :, :, 2]
    new_preds[:, :, :, :, 2] = yaws

    return new_preds


def interpolate_trajectories(preds):
    '''
    This function is used for the Interaction dataset. Since we downsample the trajectories during training for
     efficiency, we now interpolate the trajectories to bring it back to the original number of timesteps.
     for evaluation on the test server.
    '''
    device = preds.device
    K = preds.shape[0]
    T_in = preds.shape[1]
    B = preds.shape[2]
    N = preds.shape[3]
    out = preds.shape[4]

    new_preds = torch.zeros((K, 2*T_in, B, N, out)).to(device)
    T_in = preds.shape[1]
    preds = preds.permute(0, 2, 3, 1, 4).reshape(-1, T_in, out)
    new_idx = 0
    for t in range(T_in):
        if t == 0:
            new_pred = (preds[:, t, :2] - torch.tensor([[0.0, 0.0]]).to(device)) / 2.0
            new_pred = new_pred.view(K, B, N, 2)
            pred_t = preds[:, t, :2].view(K, B, N, 2)
            new_pred_yaw = (preds[:, t, -1] - torch.tensor([0.0]).to(device)) / 2.0
            new_pred_yaw = new_pred_yaw.view(K, B, N)
            pred_t_yaw = preds[:, t, -1].view(K, B, N)

            new_preds[:, new_idx, :, :, :2] = new_pred
            new_preds[:, new_idx, :, :, 2] = new_pred_yaw
            new_idx += 1
            new_preds[:, new_idx, :, :, :2] = pred_t
            new_preds[:, new_idx, :, :, 2] = pred_t_yaw
            new_idx += 1
        else:
            new_pred = ((preds[:, t, :2] - preds[:, t-1, :2]) / 2.0) + preds[:, t-1, :2]
            new_pred = new_pred.view(K, B, N, 2)
            pred_t = preds[:, t, :2].view(K, B, N, 2)
            new_pred_yaw = ((preds[:, t, -1] - preds[:, t-1, -1]) / 2.0) + preds[:, t-1, -1]
            new_pred_yaw = new_pred_yaw.view(K, B, N)
            pred_t_yaw = preds[:, t, -1].view(K, B, N)

            new_preds[:, new_idx, :, :, :2] = new_pred
            new_preds[:, new_idx, :, :, 2] = new_pred_yaw
            new_idx += 1
            new_preds[:, new_idx, :, :, :2] = pred_t
            new_preds[:, new_idx, :, :, 2] = pred_t_yaw
            new_idx += 1

    return new_preds


def make_2d_rotation_matrix(angle_in_radians: float) -> np.ndarray:
    """
    Makes rotation matrix to rotate point in x-y plane counterclockwise
    by angle_in_radians.
    """
    return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                     [np.sin(angle_in_radians), np.cos(angle_in_radians)]])


def convert_local_coords_to_global(coordinates: np.ndarray, yaw: float) -> np.ndarray:
    """
    Converts global coordinates to coordinates in the frame given by the rotation quaternion and
    centered at the translation vector. The rotation is meant to be a z-axis rotation.
    :param coordinates: x,y locations. array of shape [n_steps, 2].
    :param translation: Tuple of (x, y, z) location that is the center of the new frame.
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
    :return: x,y locations in frame stored in array of share [n_times, 2].
    """
    transform = make_2d_rotation_matrix(angle_in_radians=-yaw)
    if len(coordinates.shape) > 2:
        coord_shape = coordinates.shape
        return np.dot(transform, coordinates.reshape((-1, 2)).T).T.reshape(*coord_shape)
    return np.dot(transform, coordinates.T).T[:, :2]


# the next two functions are taken from the Interaction dataset gihub page
# https://github.com/interaction-dataset/INTERPRET_challenge_multi-agent/blob/main/calculate_collision.py
def return_circle_list(x, y, l, w, yaw):
    """
     This function returns the list of origins of circles for the given vehicle at all
     predicted timestamps and modalities. x, y, and yaw have the same shape (T, Modality).
     l, w are scalars represents the length and width of the vehicle.
     The output has the shape (T, Modality, c, 2) where c is the number of circles
     determined by the length of the given vehicle.
    """
    r = w/np.sqrt(2)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    if l < 4.0:
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c = [c1, c2]
    elif l >= 4.0 and l < 8.0:
        c0 = [x, y]
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c = [c0, c1, c2]
    else:
        c0 = [x, y]
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c3 = [x-(l-w)/2*cos_yaw/2, y-(l-w)/2*sin_yaw/2]
        c4 = [x+(l-w)/2*cos_yaw/2, y+(l-w)/2*sin_yaw/2]
        c = [c0, c1, c2, c3, c4]
    for i in range(len(c)):
        c[i] = np.stack(c[i], axis=-1)
    c = np.stack(c, axis=-2)
    return c


def return_collision_threshold(w1, w2):
    """
     This function returns the threshold for collision.
     If any of two circles' origins' distance between two vehicles is lower than this threshold,
     it is considered as a collision at that timestamp.
     w1, w2 are scalar values which represents the width of vehicle 1 and vehicle 2.
    """
    return (w1 + w2) / np.sqrt(3.8)


def collisions_for_inter_dataset(preds, agent_types, ego_in, agents_in, translations, device="cpu"):
    '''
    1. Rotate and Translate all agents to the same coordinate system.
    2. Get all agent width and length (if they are vehicles only).
    3. Use provided functions to compute the circles.
    4. Check for collisions between vehicles.
    :return
        batch_collisions: collisions per-item in batch.
        new_preds:
    '''
    new_preds = preds.copy()
    last_obs_yaws = np.concatenate((ego_in[:, -1, 7:8], agents_in[:, -1, :, 7]), axis=-1)
    angles_of_rotation = (np.pi / 2) + np.sign(-last_obs_yaws) * np.abs(last_obs_yaws)
    lengths = np.concatenate((ego_in[:, -1, 8:9], agents_in[:, -1, :, 8]), axis=-1)
    widths = np.concatenate((ego_in[:, -1, 9:10], agents_in[:, -1, :, 9]), axis=-1)

    vehicles_only = agent_types[:, :, 0] == 1.0
    K = preds.shape[0]
    N = preds.shape[3]
    B = ego_in.shape[0]
    batch_collisions = np.zeros(B)
    for b in range(B):
        agents_circles = []
        agents_widths = []
        curr_preds = preds[:, :, b]
        for n in range(N):
            if not vehicles_only[b, n]:
                if agent_types[b, n, 1]:
                    if n == 0:
                        diff = ego_in[b, -1, 0:2] - ego_in[b, -5, 0:2]
                        yaw = np.arctan2(diff[1], diff[0])
                    else:
                        diff = agents_in[b, -1, n-1, 0:2] - agents_in[b, -5, n-1, 0:2]
                        yaw = np.arctan2(diff[1], diff[0])
                    angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
                    new_preds[:, :, b, n, :2] = convert_local_coords_to_global(coordinates=curr_preds[:, :, n, :2],
                                                                               yaw=angle_of_rotation) + \
                                                translations[b, n].reshape(1, 1, -1)
                continue
            yaw = angles_of_rotation[b, n]
            new_preds[:, :, b, n, :2] = convert_local_coords_to_global(coordinates=curr_preds[:, :, n, :2], yaw=yaw) + \
                                        translations[b, n].reshape(1, 1, -1)
            preds_for_circle = new_preds[:, :, b, n, :3].transpose(1, 0, 2)
            circles_list = return_circle_list(x=preds_for_circle[:, :, 0],
                                              y=preds_for_circle[:, :, 1],
                                              l=lengths[b, n], w=widths[b, n],
                                              yaw=preds_for_circle[:, :, 2])
            agents_circles.append(circles_list)
            agents_widths.append(widths[b, n])

        # We now have all rotated agents. Now we need to compute the circles using the lengths and widths of all agents
        collisions = np.zeros(K)
        for n in range(len(agents_circles)):
            curr_agent_circles = agents_circles[n]
            for _n in range(len(agents_circles)):
                if n == _n:
                    continue
                other_agent_circles = agents_circles[_n]
                threshold_between_agents = return_collision_threshold(agents_widths[n], agents_widths[_n])

                for curr_circle_idx in range(curr_agent_circles.shape[2]):
                    for other_circle_idx in range(other_agent_circles.shape[2]):
                        dists = np.linalg.norm(curr_agent_circles[:, :, curr_circle_idx] - other_agent_circles[:, :, other_circle_idx], axis=-1)
                        collisions += (dists <= threshold_between_agents).sum(0)

        batch_collisions[b] = (collisions > 0.0).sum() / K

    return batch_collisions, torch.from_numpy(new_preds).to(device), vehicles_only
