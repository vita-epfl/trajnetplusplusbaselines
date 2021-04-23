"""Utility functions for plots and animations."""

from contextlib import contextmanager

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as mpl_animation
except ImportError:
    plt = None
    mpl_animation = None


@contextmanager
def canvas(image_file=None, **kwargs):
    """Generic matplotlib context."""
    fig, ax = plt.subplots(**kwargs)
    ax.grid(linestyle='dotted')
    ax.set_aspect(1.0, 'datalim')
    ax.set_axisbelow(True)

    yield ax

    fig.set_tight_layout(True)
    if image_file:
        fig.savefig(image_file, dpi=300)
    fig.show()
    plt.close(fig)


@contextmanager
def animation(n, movie_file=None, writer=None, **kwargs):
    """Context for animations."""
    fig, ax = plt.subplots(**kwargs)
    fig.set_tight_layout(True)
    ax.grid(linestyle='dotted')
    ax.set_aspect(1.0, 'datalim')
    ax.set_axisbelow(True)

    context = {'ax': ax, 'update_function': None}
    yield context

    ani = mpl_animation.FuncAnimation(fig, context['update_function'], range(n))
    if movie_file:
        ani.save(movie_file, writer=writer)
    fig.show()
    plt.close(fig)
