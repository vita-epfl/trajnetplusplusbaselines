import math
import random

import numpy
import trajnettools


def rotate_path(path, theta):
    ct = math.cos(theta)
    st = math.sin(theta)

    return [trajnettools.TrackRow(r.frame, r.pedestrian, ct * r.x + st * r.y, -st * r.x + ct * r.y)
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
