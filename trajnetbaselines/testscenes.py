"""Create hand-crafted test cases."""

import random

import numpy as np
import pysparkling
import trajnettools
from trajnettools.data import SceneRow, TrackRow

from . import socialforce


def linear(scene_id, start_frame):
    linear_person = [TrackRow(start_frame + i * 10, 0, -5.0 + i * 0.5, 0.0)
                     for i in range(21)]

    scene = SceneRow(scene_id, 0, start_frame, start_frame + 200)
    return linear_person + [scene]


def linear_static(scene_id, start_frame, perpendicular_distance=0.2):
    linear_person = [TrackRow(start_frame + i * 10, 0, -5.0 + i * 0.5, perpendicular_distance)
                     for i in range(21)]
    static_person = [TrackRow(start_frame + i * 10, 1, 0.0, 0.0)
                     for i in range(21)]

    scene = SceneRow(scene_id, 0, start_frame, start_frame + 200)
    return static_person + linear_person + [scene]


def linear_random(scene_id, start_frame, perpendicular_distance=0.2, random_radius=0.2):
    linear_person = [TrackRow(start_frame + i * 10, 0, -5.0 + i * 0.5, perpendicular_distance)
                     for i in range(21)]
    static_person = [TrackRow(start_frame + i * 10, 1,
                              random_radius * 2.0 * (random.random() - 0.5),
                              random_radius * 2.0 * (random.random() - 0.5))
                     for i in range(21)]

    scene = SceneRow(scene_id, 0, start_frame, start_frame + 200)
    return static_person + linear_person + [scene]


def opposing(scene_id, start_frame, perpendicular_distance=0.2):
    person0 = [TrackRow(start_frame + i * 10, 0, -5.0 + i * 0.5, 0.0)
               for i in range(21)]
    person1 = [TrackRow(start_frame + i * 10, 1, 10.0 - i * 0.5, perpendicular_distance)
               for i in range(21)]

    scene = SceneRow(scene_id, 0, start_frame, start_frame + 200)
    return person0 + person1 + [scene]


def sf_opposing(scene_id, start_frame, perpendicular_distance=-0.3):
    """This is the same parametrization as in test_social_force.py::test_opposite()."""
    initial_state = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [perpendicular_distance, 10.0, -1.0, 0.0],
    ])
    destinations = np.array([
        [0.0, 10.0],
        [perpendicular_distance, 0.0],
    ])
    s = socialforce.SocialForceSim(initial_state, destinations)
    states = np.stack([s.step().state.copy() for _ in range(21)])

    person0 = [TrackRow(start_frame + i * 10, 0, x, y)
               for i, (x, y) in enumerate(states[:, 0, 0:2])]
    person1 = [TrackRow(start_frame + i * 10, 1, x, y)
               for i, (x, y) in enumerate(states[:, 1, 0:2])]
    scene = SceneRow(scene_id, 0, start_frame, start_frame + 200)
    return person0 + person1 + [scene]


def main():
    sc = pysparkling.Context()
    sc.parallelize(
        linear(0, 0) +
        linear_static(1, 1000, perpendicular_distance=0.2) +
        linear_static(2, 2000, perpendicular_distance=0.5) +
        linear_random(3, 3000, perpendicular_distance=0.2, random_radius=0.2) +
        linear_random(4, 4000, perpendicular_distance=1.0, random_radius=0.5) +
        opposing(5, 5000, perpendicular_distance=0.2) +
        opposing(6, 6000, perpendicular_distance=1.5) +
        sf_opposing(7, 7000, perpendicular_distance=-0.3) +
        sf_opposing(8, 8000, perpendicular_distance=0.3)
    ).map(trajnettools.writers.trajnet).saveAsTextFile('data/testscenes.ndjson')

    (sc
     .parallelize(range(1000))
     .flatMap(lambda i:
              sf_opposing(i, i*1000,
                          perpendicular_distance=((i % 2) - 0.5) * 2.0 * 0.3))
     .map(trajnettools.writers.trajnet)
     .saveAsTextFile('data/socialforce_overtrain.ndjson'))


if __name__ == '__main__':
    main()
