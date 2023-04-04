"""Utility functions for plots and animations."""

from contextlib import contextmanager

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as mpl_animation
except ImportError:
    plt = None
    mpl_animation = None


@contextmanager
def track_canvas(image_file=None, show=True, tight_layout=True, **kwargs):
    """Generic matplotlib context."""
    fig, ax = plt.subplots(**kwargs)

    def format_ax(this_ax):
        if isinstance(this_ax, np.ndarray):
            for a in this_ax:
                format_ax(a)
            return

        this_ax.grid(linestyle='dotted')
        this_ax.set_aspect(1.0, 'datalim')
        this_ax.set_axisbelow(True)
        this_ax.set_xlabel('$x_1$ [m]')
        this_ax.set_ylabel('$x_2$ [m]')

    format_ax(ax)

    yield ax

    if tight_layout:
        fig.set_tight_layout(True)
    if image_file:
        fig.savefig(image_file, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


@contextmanager
def canvas(image_file=None, show=True, **kwargs):
    """Generic matplotlib context."""
    fig, ax = plt.subplots(**kwargs)

    yield ax

    fig.set_tight_layout(True)
    if image_file:
        fig.savefig(image_file, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


@contextmanager
def animation(n, movie_file=None, writer=None, **kwargs):
    """Context for animations."""
    fig, ax = plt.subplots(**kwargs)
    fig.set_tight_layout(True)
    ax.grid(linestyle='dotted')
    ax.set_aspect(1.0, 'datalim')
    ax.set_axisbelow(True)

    context = {'fig': fig, 'ax': ax, 'update_function': None}
    yield context

    ani = mpl_animation.FuncAnimation(fig, context['update_function'], range(n))
    if movie_file:
        ani.save(movie_file, writer=writer)
    fig.show()
    plt.close(fig)


# pylint: disable=too-many-locals
def state_animation(ax, all_states, *,
                    delta_t=0.4, movie_file=None, writer=None, show_speed=False):
    pedestrian_actors = []
    text_actors = []

    for ped in range(all_states.shape[1]):
        speed = np.linalg.norm(all_states[0, ped, 2:4])
        radius = 0.2 + speed / 2.0 * 0.3
        p = plt.Circle(all_states[0, ped, 0:2], radius=radius,
                       facecolor='black' if all_states[0, ped, 6] > 0 else 'white',
                       edgecolor='black', zorder=10.0)
        pedestrian_actors.append(p)
        ax.add_patch(p)
        if show_speed:
            t = ax.text(0.1, 0.9 - 0.08 * ped, '', transform=ax.transAxes)
            text_actors.append(t)

    def update(i):
        for ped, p in enumerate(pedestrian_actors):
            # p.set_data(all_states[i:i+5, ped, 0], all_states[i:i+5, ped, 1])
            p.center = all_states[i, ped, 0:2]
            speed = np.linalg.norm(all_states[i, ped, 2:4])
            p.set_radius(0.2 + speed / 2.0 * 0.3)
            if show_speed:
                acc = np.linalg.norm(all_states[i, ped, 4:6])
                text_actors[ped].set_text(
                    f'$v_{ped} = {speed:.2f}m/s, a_{ped} = {acc:.2f}m/s^2$')
        return pedestrian_actors + text_actors

    ani = mpl_animation.FuncAnimation(
        ax.get_figure(), update,
        frames=len(all_states), interval=delta_t * 1000.0, blit=True)
    if movie_file:
        ani.save(movie_file, writer=writer, dpi=200)
    return ani


def states(ax, states, *, labels=None, monochrome=False, **kwargs):  # pylint: disable=redefined-outer-name
    states = np.asarray(states)

    if monochrome:
        kwargs['color'] = 'black'

    for ped in range(states.shape[1]):
        x = states[:, ped, 0]
        y = states[:, ped, 1]
        label = f'ped {ped}'
        if labels:
            label = labels[ped]
        tracks = ax.plot(x, y, '-o', label=label, markersize=2.5, markeredgewidth=0, **kwargs)

        marker_color = tracks[0].get_color()
        marker_alpha = tracks[0].get_alpha()
        marker_zorder = tracks[0].get_zorder()
        marker_edgewidth = 0
        marker_edgecolor = marker_color
        if monochrome and states[0, ped, 2] < 0.0:
            marker_color = 'white'
            marker_edgewidth = 1
        ax.plot(states[0, ped:ped + 1, 0], states[0, ped:ped + 1, 1],
                'o', color=marker_color,
                alpha=marker_alpha,
                markeredgewidth=marker_edgewidth,
                markeredgecolor=marker_edgecolor,
                zorder=marker_zorder,
                label='start' if ped == 0 else None)
        ax.plot(states[0, ped:ped + 1, 6], states[0, ped:ped + 1, 7],
                'x', color=marker_color,
                alpha=marker_alpha,
                markeredgewidth=2,
                zorder=marker_zorder,
                label='goal' if ped == 0 else None)


def space(ax, ped_space_potential):
    for s in ped_space_potential.space:
        s = np.asarray(s)
        ax.plot(s[:, 0], s[:, 1], '-o', color='black', markersize=2.5)


def potential_1d(V, ax1, ax2=None, **kwargs):
    parameters = list(V.parameters())
    dtype = parameters[0].dtype if parameters else torch.float32
    b = torch.linspace(0.0, 3.0, 200, dtype=dtype)

    with torch.no_grad():
        y = V.value_b(b)
        y = y - y[-1]

    ax1.set_xlabel('$b$ [m]')
    ax1.set_ylabel('$V$')
    ax1.plot(b, y, **kwargs)
    ax1.legend()

    if ax2 is not None:
        ax2.set_xlabel(r'$b$ [m]')
        ax2.set_ylabel(r'$\nabla V$')
        delta_b = b[1:] - b[:-1]
        average_b = 0.5 * (b[:-1] + b[1:])
        diff_b = y[1:] - y[:-1]
        ax2.plot(average_b, diff_b / delta_b, **kwargs)
        ax2.legend()


def potential_1d_parametric(V, ax1, ax2=None, label=None, sigma_label=None,
                            linestyle=None, **kwargs):
    potential_1d(V, ax1, ax2, linestyle=linestyle, label=label, **kwargs)
    ax1.axvline(V.sigma, linestyle='dotted', label=sigma_label, **kwargs)
    if ax2 is not None:
        ax2.axvline(V.sigma, linestyle='dotted', label=sigma_label, **kwargs)


def potential_2d(V, ax, nx=600, ny=400, **kwargs):
    # the "pedestrian of interest" is beta and the probe pedestrians are alpha

    parameters = list(V.parameters())
    dtype = parameters[0].dtype if parameters else torch.float32
    x1 = torch.linspace(-1.0, 2.0, nx, dtype=dtype)
    x2 = torch.linspace(-1.0, 1.0, ny, dtype=dtype)
    xx1, xx2 = torch.meshgrid(x1, x2)
    r_ab_probe = torch.stack((xx1, xx2), dim=-1).view(-1, 2).unsqueeze(1)

    speeds_b = torch.full((1,), 1.0, dtype=dtype)
    desired_directions_b = torch.zeros((1, 2), dtype=dtype)
    desired_directions_b[:, 0] = 1.0

    ax.axhline(0.0, ls='dotted', color='gray')
    ax.axvline(0.0, ls='dotted', color='gray')

    with torch.no_grad():
        values = V.value_r_ab(r_ab_probe, speeds_b, desired_directions_b)
        values -= torch.min(values)
        values = values.view((len(x1), len(x2)))

    ax.clabel(
        ax.contour(x1, x2, values.T,
                   levels=np.linspace(0.1, 1.5, 15), vmin=0.1, vmax=1.5, **kwargs),
        inline=1, fontsize=10)
    ax.plot([0.0], [0.0], '-', label=r'$V$', color='seagreen')  # just for legend

    ax.plot([0.0], [0.0], 'o', label='pedestrian', markersize=5.0, color='black')
    ax.arrow(0.0, 0.0, 0.4, 0.0, width=0.01, zorder=10, color='black')
    ax.set_xlabel('$x_1$ [m]')
    ax.set_ylabel('$x_2$ [m]')
    ax.set_aspect('equal')
    ax.legend()


def potential_2d_grad(V, ax, nx=600, ny=400, **kwargs):
    # the "pedestrian of interest" is beta and the probe pedestrians are alpha

    parameters = list(V.parameters())
    dtype = parameters[0].dtype if parameters else torch.float32
    x1 = torch.linspace(-1.0, 2.0, nx, dtype=dtype)
    x2 = torch.linspace(-1.0, 1.0, ny, dtype=dtype)
    xx1, xx2 = torch.meshgrid(x1, x2)
    r_ab_probe = torch.stack((xx1, xx2), dim=-1).view(-1, 2).unsqueeze(1)

    speeds_b = torch.full((1,), 1.0, dtype=dtype)
    desired_directions_b = torch.zeros((1, 2), dtype=dtype)
    desired_directions_b[:, 0] = 1.0

    ax.axhline(0.0, ls='dotted', color='gray')
    ax.axvline(0.0, ls='dotted', color='gray')

    with torch.no_grad():
        grad = V.grad_r_ab_(r_ab_probe, speeds_b, desired_directions_b)
        values = torch.linalg.norm(grad, ord=2, dim=-1)
        # values -= torch.min(values)
        values = values.view((len(x1), len(x2)))

    ax.clabel(
        ax.contour(x1, x2, values.T,
                   levels=np.linspace(0.1, 1.5, 15), vmin=0.1, vmax=1.5, **kwargs),
        inline=1, fontsize=10)
    ax.plot([0.0], [0.0], '-', label=r'$|\nabla V|$', color='seagreen')  # just for legend

    ax.plot([0.0], [0.0], 'o', label='pedestrian', markersize=5.0, color='black')
    ax.arrow(0.0, 0.0, 0.4, 0.0, width=0.01, zorder=10, color='black')
    ax.set_xlabel('$x_1$ [m]')
    ax.set_ylabel('$x_2$ [m]')
    ax.set_aspect('equal')
    ax.legend()


def experience(ax, experiences, *, reference_ped=0, **kwargs):
    ax.axhline(0.0, ls='dotted', color='gray')
    ax.axvline(0.0, ls='dotted', color='gray')

    for t0, t1 in experiences:
        for other_i in range(t0.shape[0]):
            if other_i == reference_ped:
                continue
            diff_t0 = t0[other_i] - t0[reference_ped]
            diff_t1 = t1[other_i] - t1[reference_ped]
            x = [diff_t0[0], diff_t1[0]]
            y = [diff_t0[1], diff_t1[1]]
            ax.plot(x, y, '-o', markersize=2.5, markeredgewidth=0, **kwargs)
