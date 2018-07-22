"""Create hand-crafted test cases."""

import random

import pysparkling
import trajnettools
from trajnettools.data import SceneRow, TrackRow


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


def main():
    sc = pysparkling.Context()
    sc.parallelize(
        linear(0, 0) +
        linear_static(1, 1000, perpendicular_distance=0.2) +
        linear_static(2, 2000, perpendicular_distance=0.5) +
        linear_random(3, 3000, perpendicular_distance=0.2, random_radius=0.2) +
        linear_random(4, 4000, perpendicular_distance=1.0, random_radius=0.5)
    ).map(trajnettools.writers.trajnet).saveAsTextFile('data/testscenes.ndjson')


if __name__ == '__main__':
    main()
