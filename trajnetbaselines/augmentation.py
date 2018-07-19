import math
import random
import trajnettools


def rotate_path(path, theta):
    ct = math.cos(theta)
    st = math.sin(theta)

    return [trajnettools.TrackRow(r.frame, r.pedestrian, ct * r.x + st * r.y, -st * r.x + ct * r.y)
            for r in path]


def random_rotation(paths):
    theta = random.random() * 2.0 * math.pi
    return [rotate_path(path, theta) for path in paths]
