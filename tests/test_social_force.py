import numpy as np
import pytest
import trajnetbaselines
from trajnetbaselines.socialforce import V0, SIGMA_V


def test_rab():
    initial_state = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    destinations = np.array([
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    s = trajnetbaselines.socialforce.SocialForceSim(initial_state, destinations)
    assert s.rab().tolist() == [[
        [0.0, 0.0],
        [-1.0, 0.0],
    ], [
        [1.0, 0.0],
        [0.0, 0.0],
    ]]


def test_fab():
    initial_state = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    destinations = np.array([
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    s = trajnetbaselines.socialforce.SocialForceSim(initial_state, destinations)
    force_at_unit_distance = 0.25  # TODO confirm
    assert s.fab() == pytest.approx(np.array([[
        [0.0, 0.0],
        [-force_at_unit_distance, 0.0],
    ], [
        [force_at_unit_distance, 0.0],
        [0.0, 0.0],
    ]]), abs=0.05)


def test_b_zero_vel():
    initial_state = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    destinations = np.array([
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    s = trajnetbaselines.socialforce.SocialForceSim(initial_state, destinations)
    assert s.b(s.rab()).tolist() == [
        [0.0, 1.0],
        [1.0, 0.0],
    ]


def test_w():
    initial_state = np.array([
        [0.0, 0.0, 0.5, 0.5],
        [10.0, 0.3, -0.5, 0.5],
    ])
    destinations = np.array([
        [10.0, 10.0],
        [0.0, 10.0],
    ])
    s = trajnetbaselines.socialforce.SocialForceSim(initial_state, destinations)
    w = s.w(s.desired_direction(), -s.fab())
    assert w.tolist() == [
        [0, 1],
        [1, 0],
    ]


def test_crossing():
    initial_state = np.array([
        [0.0, 0.0, 0.5, 0.5],
        [10.0, 0.3, -0.5, 0.5],
    ])
    destinations = np.array([
        [10.0, 10.0],
        [0.0, 10.0],
    ])
    s = trajnetbaselines.socialforce.SocialForceSim(initial_state, destinations)
    states = np.stack([s.step().state.copy() for _ in range(50)])

    # visualize
    print('')
    import trajnettools.show
    with trajnettools.show.canvas() as ax:
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        for ped in range(2):
            x = states[:, ped, 0]
            y = states[:, ped, 1]
            ax.plot(x, y, '-o', label='ped {}'.format(ped), markersize=2.5)
        ax.legend()


def test_crossing_narrow():
    initial_state = np.array([
        [0.0, 0.0, 0.5, 0.5],
        [2.0, 0.3, -0.5, 0.5],
    ])
    destinations = np.array([
        [2.0, 10.0],
        [0.0, 10.0],
    ])
    s = trajnetbaselines.socialforce.SocialForceSim(initial_state, destinations)
    states = np.stack([s.step().state.copy() for _ in range(40)])

    # visualize
    print('')
    import trajnettools.show
    with trajnettools.show.canvas() as ax:
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        for ped in range(2):
            x = states[:, ped, 0]
            y = states[:, ped, 1]
            ax.plot(x, y, '-o', label='ped {}'.format(ped), markersize=2.5)
        ax.legend()


def test_opposite():
    initial_state = np.array([
        [0.0, 0.0, 0.5, 0.0],
        [0.3, 10.0, -0.5, 0.0],
    ])
    destinations = np.array([
        [0.0, 10.0],
        [0.3, 0.0],
    ])
    s = trajnetbaselines.socialforce.SocialForceSim(initial_state, destinations)
    states = np.stack([s.step().state.copy() for _ in range(40)])

    # visualize
    print('')
    import trajnettools.show
    with trajnettools.show.canvas() as ax:
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        for ped in range(2):
            x = states[:, ped, 0]
            y = states[:, ped, 1]
            ax.plot(x, y, '-o', label='ped {}'.format(ped), markersize=2.5)
        ax.legend()


def test_2opposite():
    initial_state = np.array([
        [0.0, 0.0, 0.5, 0.0],
        [0.6, 10.0, -0.5, 0.0],
        [2.0, 10.0, -0.5, 0.0],
    ])
    destinations = np.array([
        [0.0, 10.0],
        [0.6, 0.0],
        [2.0, 0.0],
    ])
    s = trajnetbaselines.socialforce.SocialForceSim(initial_state, destinations)
    states = np.stack([s.step().state.copy() for _ in range(40)])

    # visualize
    print('')
    import trajnettools.show
    with trajnettools.show.canvas() as ax:
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        for ped in range(3):
            x = states[:, ped, 0]
            y = states[:, ped, 1]
            ax.plot(x, y, '-o', label='ped {}'.format(ped), markersize=2.5)
        ax.legend()
