import math
import random

import numpy
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

from celluloid import Camera

import trajnetplusplustools
from trajnetplusplustools import show

import labellines

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

def visualize_scene(scene, goal=None, weights=None, pool_weight=None, show=True):
    # print("Primary: ", scene[0, 0], scene[8, 0])
    # print("pool weights: ", pool_weight)

    for t in reversed(range(scene.shape[1])):
        path = scene[:, t]
        color = 'b' if t == 0 else 'orange'
        linewidth = 3.0 if t == 0 else 2.0

        plt.plot(path[:-1, 0], path[:-1, 1], c=color, linewidth=linewidth)

        # dict_map = {4: 'N4', 7: 'N3', 8: 'N2', 9: 'N1'}

        if t == 0 and weights is not None:
            # plt.plot(path[:, 0], path[:, 1], c=color)
            pass
            # plt.scatter(path[:-1, 0], path[:-1, 1], c=weights[:-2], cmap='Blues', vmin=0.0, vmax=1.5)
            # plt.plot(path[-2:, 0], path[-2:, 1], c='g')
            # plt.scatter(path[:, 0], path[:, 1], c=color, alpha='Greys', vmin=0.0, vmax=1.5)
        elif t != 0 and pool_weight is not None:
            # if t in {4, 7, 8, 9}:
            # plt.plot(path[:, 0], path[:, 1], label='N{} = {}'.format(t, numpy.round(pool_weight[t-1], 2)))
            # else:
                # plt.plot(path[:, 0], path[:, 1])
            plt.scatter(path[:-1, 0], path[:-1, 1], color=color, alpha=pool_weight[t-1], vmin=0.0, vmax=1.5, s=60.0)
            # plt.plot(path[-2:, 0], path[-2:, 1], c='g')
        else:
            # plt.plot(path[:, 0], path[:, 1], c=color)
            plt.scatter(path[:-1, 0], path[:-1, 1], c=color)
            # plt.plot(path[-2:, 0], path[-2:, 1], c='g')
        plt.arrow(path[-2, 0], path[-2, 1], path[-1, 0] - path[-2, 0], path[-1, 1] - path[-2, 1], width=0.05, color='g')

    ## Crowds Zara Scene 19
    # dict_map = {4: 'N4', 7: 'N3', 8: 'N2', 9: 'N1'}
    # for t in reversed(range(scene.shape[1])):
    #     path = scene[:, t]
    #     if t in {4, 7, 8, 9}:
    #         print(t)
    #         plt.plot(0.5*(path[-3, 0] + path[-4, 0]), 0.5*(path[-3, 1] + path[-4, 1]), linestyle = 'None', marker='${}$'.format(dict_map[t]), markersize=15.0, color='r', label='${0:0.2f}$'.format(pool_weight[t-1]))

    ## Crowds Zara Scene 0
    # dict_map = {1: 'N1', 2: 'N2'}
    # for t in reversed(range(scene.shape[1])):
    #     path = scene[:, t]
    #     if t in {1, 2}:
    #         print(t)
    #         plt.plot(0.5*(path[-3, 0] + path[-4, 0]), 0.5*(path[-3, 1] + path[-4, 1]), linestyle = 'None', marker='${}$'.format(dict_map[t]), markersize=15.0, color='r', label='${0:0.2f}$'.format(pool_weight[t-1]))

    ## Crowds Zara Scene 164
    dict_map = {4: 'N4', 8: 'N1', 9: 'N2', 10: 'N3'}
    for t in reversed(range(scene.shape[1])):
        path = scene[:, t]
        if t in {4, 8, 9, 10}:
            print(t)
            print(pool_weight[t-1])
            plt.plot(0.5*(path[-3, 0] + path[-4, 0]), 0.5*(path[-3, 1] + path[-4, 1]), linestyle = 'None', marker='${}$'.format(dict_map[t]), markersize=15.0, color='r', label='${0:0.2f}$'.format(pool_weight[t-1]))

    if goal is not None:
        for t in range(goal.shape[0]):
            goal_t = goal[t]
            plt.scatter(goal_t[0], goal_t[1])

    plt.gca().set_aspect('equal')
    xmin = numpy.round(2 * numpy.nanmin(scene[:, :, 0])) * 0.5
    xmax = numpy.round(2 * numpy.nanmax(scene[:, :, 0])) * 0.5
    # xcent = 0.5*(xmin + xmax)
    ymin = numpy.round(2 * numpy.nanmin(scene[:, :, 1])) * 0.5
    ymax = numpy.round(2 * numpy.nanmax(scene[:, :, 1])) * 0.5

    ## Crowds Zara Scene 19
    # xmin = -4
    # xmax = 5
    # ymin = -4
    # ymax = 0

    ## Crowds Zara Scene 0
    # xmin = -2
    # xmax = 5
    # ymin = -4
    # ymax = 0

    ycent = 0.5*(ymin + ymax)
    length_plot = 0.5*(max(xmax - xmin, ymax - ymin) + 1)
    plt.xticks(numpy.arange(xmin - 1, xmax + 2), fontsize=10)
    # plt.gca().set_xticklabels(fontsize = 10, va='bottom', ha='left')
    plt.yticks(numpy.arange(ymin - 1, ymax + 2), fontsize=10)
    plt.legend(loc=4)
    if show:
        plt.show()
        plt.close()

def visualize_lrp(output_scenes, vel_weights, neigh_weights, TIME_STEPS):
    for t in range(8, TIME_STEPS):
        ## Neighbour weights arrange
        # mask = drop_distant(output_scenes[t:t+1])
        # mask = numpy.sort(mask[:len(neigh_weights[t-7])+1])
        # p_weights = 0.0 * numpy.ones(output_scenes.shape[1])
        # p_weights[mask[1:]] = neigh_weights[t-7]
        # import pdb
        # pdb.set_trace()
        # visualize_scene(output_scenes[:t+2], weights=vel_weights[t-7], pool_weight=p_weights[1:])
        print("Neigh Weights: ", neigh_weights[t-7])
        visualize_scene(output_scenes[:t+2], weights=vel_weights[t-7], pool_weight=neigh_weights[t-7])

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

def animate_lrp(output_scenes, vel_weights, neigh_weights, TIME_STEPS, scene_id=0, pool_type='directional'):
    fig = plt.figure()
    camera = Camera(fig)
    for t in range(8, TIME_STEPS):
        visualize_scene(output_scenes[:t+2], weights=vel_weights[t-7], pool_weight=neigh_weights[t-7], show=False)
        camera.snap()

    animation = camera.animate()
    animation.save('anims/lrp_crowds_{}_scene{}_check.mp4'.format(pool_type, scene_id), fps=2, writer = 'ffmpeg')
    plt.close()
