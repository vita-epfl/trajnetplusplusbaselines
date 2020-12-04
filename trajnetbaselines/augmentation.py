import math
import random

import numpy
import trajnetplusplustools


def rotate_path(path, theta):
    ct = math.cos(theta)
    st = math.sin(theta)

    return [trajnetplusplustools.TrackRow(r.frame, r.pedestrian, ct * r.x + st * r.y, -st * r.x + ct * r.y)
            for r in path]


def random_rotation_of_paths(paths):
    theta = random.random() * 2.0 * math.pi
    return [rotate_path(path, theta) for path in paths]


def random_rotation(xy):
    theta = random.random() * 2.0 * math.pi
    ct = math.cos(theta)
    st = math.sin(theta)

    r = numpy.array([[ct, st], [-st, ct]])
    return numpy.einsum('ptc,ci->pti', xy, r)


def rotate_path(path, theta):
    ct = math.cos(theta)
    st = math.sin(theta)

    return [trajnetplusplustools.TrackRow(r.frame, r.pedestrian, ct * r.x + st * r.y, -st * r.x + ct * r.y)
            for r in path]


def theta_rotation(xy, theta):
    # theta = random.random() * 2.0 * math.pi
    ct = math.cos(theta)
    st = math.sin(theta)

    r = numpy.array([[ct, st], [-st, ct]])
    return numpy.einsum('ptc,ci->pti', xy, r)

def shift(xy, center):
    # theta = random.random() * 2.0 * math.pi
    xy = xy - center[numpy.newaxis, numpy.newaxis, :]
    return xy

def center_scene(xy, obs_length=9, ped_id=0):
    ## Center
    center = xy[obs_length-1, ped_id] ## Last Observation
    xy = shift(xy, center)
    ## Rotate
    last_obs = xy[obs_length-1, ped_id]
    second_last_obs = xy[obs_length-2, ped_id]
    diff = numpy.array([last_obs[0] - second_last_obs[0], last_obs[1] - second_last_obs[1]])
    thet = numpy.arctan2(diff[1], diff[0])
    rotation = -thet + numpy.pi/2
    xy = theta_rotation(xy, rotation)
    return xy, rotation, center


def inverse_scene(xy, rotation, center):
    xy = theta_rotation(xy, -rotation)
    xy = shift(xy, -center)
    return xy

def drop_unobserved(xy, obs_length=9):
    loc_at_obs = xy[obs_length-1]
    absent_at_obs = numpy.isnan(loc_at_obs).any(axis=1)
    mask = ~absent_at_obs
    return xy[:, mask], mask

def neigh_nan(xy):
    return numpy.isnan(xy).all()

def add_noise(observation, thresh=0.005, obs_length=9, ped='primary'):
    if ped=='primary':
        observation[:obs_length, 0] += numpy.random.uniform(-thresh, thresh, observation[:obs_length, 0].shape)
    elif ped=='neigh':
        observation[:obs_length, 1:] += numpy.random.uniform(-thresh, thresh, observation[:obs_length, 1:].shape)
    else:
        raise ValueError

    return observation