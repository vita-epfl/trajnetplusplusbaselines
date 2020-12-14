import math
import random

import numpy
import matplotlib.pyplot as plt
from matplotlib import cm

import trajnetplusplustools
from trajnetplusplustools import show


def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = numpy.sum(numpy.square(xy - xy[:, 0:1]), axis=2)
    mask = numpy.argsort(distance_2)[0]
    # print("DIST")
    # import pdb
    # pdb.set_trace()
    # mask = numpy.nanmin(distance_2, axis=0) < r**2
    return mask

def random_rotation(xy, goals=None):
    theta = random.random() * 2.0 * math.pi
    ct = math.cos(theta)
    st = math.sin(theta)
    r = numpy.array([[ct, st], [-st, ct]])
    if goals is None:
        return numpy.einsum('ptc,ci->pti', xy, r)
    return numpy.einsum('ptc,ci->pti', xy, r), numpy.einsum('tc,ci->ti', goals, r)

def shift(xy, center):
    # theta = random.random() * 2.0 * math.pi
    xy = xy - center[numpy.newaxis, numpy.newaxis, :]
    return xy

def theta_rotation(xy, theta):
    # theta = random.random() * 2.0 * math.pi
    ct = math.cos(theta)
    st = math.sin(theta)

    r = numpy.array([[ct, st], [-st, ct]])
    return numpy.einsum('ptc,ci->pti', xy, r)

def center_scene(xy, obs_length=9, ped_id=0, goals=None):
    if goals is not None:
        goals = goals[numpy.newaxis, :, :]
    ## Center
    center = xy[obs_length-1, ped_id] ## Last Observation
    xy = shift(xy, center)
    if goals is not None:
        goals = shift(goals, center)

    ## Rotate
    last_obs = xy[obs_length-1, ped_id]
    second_last_obs = xy[obs_length-2, ped_id]
    diff = numpy.array([last_obs[0] - second_last_obs[0], last_obs[1] - second_last_obs[1]])
    thet = numpy.arctan2(diff[1], diff[0])
    rotation = -thet + numpy.pi/2
    xy = theta_rotation(xy, rotation)
    if goals is not None:
        goals = theta_rotation(goals, rotation)
        return xy, rotation, center, goals[0]
    return xy, rotation, center

def visualize_scene(scene, goal=None, weights=None, pool_weight=None):
    # print("Primary: ", scene[0, 0], scene[8, 0])
    for t in range(scene.shape[1]):
        path = scene[:, t]
        color = 'r' if t == 0 else 'b'
        if t == 0 and weights is not None:
            # import pdb
            # pdb.set_trace()
            plt.plot(path[:, 0], path[:, 1], c=color)
            plt.scatter(path[:, 0], path[:, 1], c=weights, cmap='Greys', vmin=0.0, vmax=1.5)
        elif t != 0 and pool_weight is not None:
            # import pdb
            # pdb.set_trace()
            plt.plot(path[:, 0], path[:, 1], c=color, alpha=pool_weight[t-1])
            plt.scatter(path[:, 0], path[:, 1], c=color, alpha=pool_weight[t-1], vmin=0.0, vmax=1.5)
        else:
            plt.plot(path[:, 0], path[:, 1], c=color)
            plt.scatter(path[:, 0], path[:, 1], c=color)

    if goal is not None:
        for t in range(goal.shape[0]):
            goal_t = goal[t]
            plt.scatter(goal_t[0], goal_t[1])


    plt.show()
    plt.close()

def visualize_lrp(output_scenes, vel_weights, neigh_weights, TIME_STEPS):
    # print("Weight: ", vel_weights)
    for t in range(8, TIME_STEPS):
        mask = drop_distant(output_scenes[t:t+1], r=3.0)
        # import pdb
        # pdb.set_trace()
        mask = numpy.sort(mask[:len(neigh_weights[t-7])+1])
        curr_output_scenes = output_scenes[:, mask]
        visualize_scene(curr_output_scenes[:t+2], weights=vel_weights[t-7], pool_weight=neigh_weights[t-7])

def visualize_grid(grid):
    sum_grid = numpy.abs(grid.numpy().sum(axis=0))
    ax = plt.gca()
    fig = plt.gcf()
    viridis = cm.get_cmap('viridis', 256)
    psm = ax.pcolormesh(sum_grid, cmap=viridis, rasterized=True)
    fig.colorbar(psm, ax=ax)
    plt.show()
    plt.close()
    print("Showed Grid")


def xy_to_paths(xy_paths):
    return [trajnetplusplustools.TrackRow(i, 0, xy_paths[i, 0].item(), xy_paths[i, 1].item(), 0, 0)
            for i in range(len(xy_paths))]

def viz(groundtruth, prediction, visualize, output_file=None):
    pred_paths = {}

    groundtruth = groundtruth.cpu().numpy().transpose(1, 0, 2)
    prediction = prediction.cpu().numpy().transpose(1, 0, 2)
    gt_paths = [xy_to_paths(path) for path in groundtruth]
    pred = [xy_to_paths(path) for path in prediction]

    pred_paths[0] = pred[0]
    pred_neigh_paths = None
    if visualize:
        pred_neigh_paths = {}
        pred_neigh_paths[0] = pred[1:]

    with show.predicted_paths(gt_paths, pred_paths, pred_neigh_paths, output_file):
        pass
