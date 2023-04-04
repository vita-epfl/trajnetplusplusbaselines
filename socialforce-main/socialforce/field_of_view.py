"""Field of view computation."""

import math
import torch


class FieldOfView:
    """Compute field of view prefactors.

    The field of view angle twophi is given in degrees.
    out_of_view_factor is C in the paper.
    """
    out_of_view_factor = 0.5

    def __init__(self, twophi=200.0):
        self.cosphi = math.cos(twophi / 2.0 / 180.0 * math.pi)

    def __call__(self, e, f):
        """Weighting factor for field of view.

        e is rank 2 and normalized in the last index.
        f is a rank 3 tensor.
        """
        cosphi_l = torch.einsum('aj,abj->ab', (e, f))
        in_sight = cosphi_l > torch.linalg.norm(f, ord=2, dim=-1) * self.cosphi

        out = torch.full_like(cosphi_l, self.out_of_view_factor)
        out[in_sight] = 1.0
        torch.diagonal(out)[:] = 0.0
        return out
