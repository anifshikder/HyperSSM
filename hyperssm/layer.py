"""One HyperSSM layer: K channels over one number system, computed by a parallel scan.

For each channel j, with input u_k at step k:

    x_k = a(u_k) x_{k-1} + B(u_k) u_k,      a(u_k) = exp(-alpha) * exp(v(u_k)),      x_0 = 1

* v(u_k) is a generator, linear in the input (and, in a stack, in the features of the layer
  below at the previous step).
* alpha >= 0 is an optional learned decay (off by default: the "tracking regime").
* B(u_k) u_k is an optional input term (off by default).

With complex channels this is a diagonal SSM. With quaternion channels the input chooses a
rotation axis as well as an angle, so the memory keeps the order of its inputs.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .algebras import get_algebra
from .scan import prefix_scan, affine_scan


class HyperLayer(nn.Module):
    def __init__(self, d_in, algebra="H", channels=1, below_dim=0, decay=False, input_term=False,
                 init_scale=1.0):
        super().__init__()
        self.alg = get_algebra(algebra)
        self.K, self.d_in, self.below_dim = channels, d_in, below_dim
        self.generator = nn.Linear(d_in * (1 + below_dim), channels * self.alg.gen_dim)
        with torch.no_grad():                                   # initial generators
            if self.alg.name == "C":
                self.generator.weight.uniform_(0, 2 * math.pi)  # angles cover the circle
            elif self.alg.name == "R":
                self.generator.weight.normal_(0, 1.5)
            else:
                self.generator.weight.normal_(0, init_scale)
            self.generator.bias.zero_()
        self.decay = nn.Parameter(torch.full((channels,), -3.0)) if decay else None
        self.input_term = nn.Linear(d_in, channels * self.alg.state_dim) if input_term else None

    @property
    def tracking(self):
        return self.decay is None and self.input_term is None

    def forward(self, u, mask, below=None, hard=False):
        """u: (B, T, d_in) inputs; mask: (B, T) True on real steps;
        below: (B, T, below_dim) features of the layer below at the SAME step (shifted here).
        Returns states (B, T, K, state_dim)."""
        B, T, _ = u.shape
        alg = self.alg
        if self.below_dim:
            prev = torch.cat([below[:, :1].new_zeros(B, 1, self.below_dim), below[:, :-1]], 1)
            prev[:, 0] = self._identity_features(below)
            z = torch.cat([torch.ones_like(prev[..., :1]), prev], -1)          # [1, chi_{k-1}]
            inp = (u.unsqueeze(-1) * z.unsqueeze(-2)).flatten(2)              # u (x) [1, chi]
        else:
            inp = u
        g = self.generator(inp).view(B, T, self.K, alg.gen_dim)
        a = alg.transition(g, hard)
        ident = alg.identity((B, T, self.K), u.device, a.dtype)
        live = mask[..., None, None]
        a = torch.where(live, a, ident)                                        # padding = identity
        if self.tracking:
            return prefix_scan(a, alg.mul)
        if self.decay is not None:
            a = torch.where(live, a * torch.exp(-F.softplus(self.decay))[None, None, :, None], ident)
        b = torch.zeros_like(a)
        if self.input_term is not None:
            b = self.input_term(u).view(B, T, self.K, alg.state_dim) * live
        b = torch.cat([alg.mul(a[:, :1], ident[:, :1]) + b[:, :1], b[:, 1:]], 1)  # start from x_0 = 1
        return affine_scan(a, b, alg.mul)

    def _identity_features(self, below):
        return self._below_identity.to(below.device, below.dtype).expand(below.shape[0], -1)

    def set_below_identity(self, feats):
        """Features of the lower layer's initial state (used at the first step)."""
        self.register_buffer("_below_identity", feats.detach().clone(), persistent=False)
