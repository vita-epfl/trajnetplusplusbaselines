"""Field of view computation."""

import numpy as np


class FieldOfView(object):
    """Compute field of view prefactors.

    The field of view angle twophi is given in degrees.
    out_of_view_factor is C in the paper.
    """
    def __init__(self, twophi=200.0, out_of_view_factor=0.5):
        self.cosphi = np.cos(twophi / 2.0 / 180.0 * np.pi)
        self.out_of_view_factor = out_of_view_factor

    def __call__(self, e, f):
        """Weighting factor for field of view.

        e is rank 2 and normalized in the last index.
        f is a rank 3 tensor.
        """
        in_sight = np.einsum('aj,abj->ab', e, f) > np.linalg.norm(f, axis=-1) * self.cosphi
        out = self.out_of_view_factor * np.ones_like(in_sight)
        out[in_sight] = 1.0
        np.fill_diagonal(out, 0.0)
        return out
