import math
import random

import numpy
import matplotlib.pyplot as plt

import trajnetplusplustools
from trajnetplusplustools import show

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

def visualize_scene(scene, goal=None):
    for t in range(scene.shape[1]):
        path = scene[:, t]
        plt.plot(path[:, 0], path[:, 1])
    if goal is not None:
        for t in range(goal.shape[0]):
            goal_t = goal[t]
            plt.scatter(goal_t[0], goal_t[1])


    plt.show()
    plt.close()

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
