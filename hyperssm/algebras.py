"""Number systems for HyperSSM channels.

Every linear recurrent memory splits into blocks over the real, complex and quaternion
numbers, so these three are the building blocks. Each algebra defines:

  gen_dim      numbers in a generator v (what the input controls)
  state_dim    real numbers stored per channel
  transition   v -> a = exp(v), a bounded, invertible element (a rotation)
  mul          the product of the algebra (associative, not necessarily commutative)
  features     what a readout (or the layer above) may read from a state

States are stored as real tensors of shape (..., state_dim); complex numbers as (re, im),
quaternions as (w, x, y, z).
"""
import math
import torch


class Algebra:
    name = ""
    gen_dim = 0
    state_dim = 0

    def identity(self, shape, device=None, dtype=torch.float32):
        e = torch.zeros(*shape, self.state_dim, device=device, dtype=dtype)
        e[..., 0] = 1
        return e

    def transition(self, g, hard=False):
        raise NotImplementedError

    def mul(self, a, b):
        raise NotImplementedError

    def feature_dim(self):
        raise NotImplementedError

    def features(self, x):
        raise NotImplementedError

    def __repr__(self):
        return self.name


class Real(Algebra):
    """One real number with a learned sign. The only bounded real transitions are +1 and -1,
    so the transition is exactly sign(g) in the forward pass, with the gradient of tanh(g)
    (a straight-through estimator). Tracks parity (the group of order 2)."""
    name, gen_dim, state_dim = "R", 1, 1

    def transition(self, g, hard=False):
        soft = torch.tanh(g)
        exact = torch.where(g >= 0, torch.ones_like(g), -torch.ones_like(g))
        return soft + (exact - soft).detach()

    def mul(self, a, b):
        return a * b

    def feature_dim(self):
        return 1

    def features(self, x):
        return x


class Complex(Algebra):
    """One complex number: a clock hand. The transition turns it by an angle g.
    Tracks counting modulo n (cyclic groups) and every abelian group with enough channels.
    `lattice = n` snaps angles to multiples of 2*pi/n (see hyperssm.snap)."""
    name, gen_dim, state_dim = "C", 1, 2

    def __init__(self, lattice=None):
        self.lattice = lattice

    def angle(self, g, hard=False):
        if hard and self.lattice:
            step = 2 * math.pi / self.lattice
            return torch.round(g / step) * step
        return g

    def transition(self, g, hard=False):
        th = self.angle(g, hard)
        return torch.cat([torch.cos(th), torch.sin(th)], -1)

    def mul(self, a, b):
        ar, ai = a[..., :1], a[..., 1:]
        br, bi = b[..., :1], b[..., 1:]
        return torch.cat([ar * br - ai * bi, ar * bi + ai * br], -1)

    def feature_dim(self):
        return 2

    def features(self, x):
        return x


class Quaternion(Algebra):
    """One unit quaternion: a 3D orientation. The transition rotates it about the axis v/|v|
    by the angle 2|v|; turns about different axes do not commute, so order is kept.

    sign_invariant=True reads the 10 entries of q q^T, which ignore the sign of q
    (q and -q are the same rotation). This readout tracks the rotation groups of the
    Platonic solids, including the icosahedral group A5, and the Bloch vector of a qubit.
    sign_invariant=False reads q itself as well (needed for the quaternion group Q8)."""
    name, gen_dim, state_dim = "H", 3, 4
    _iu = torch.triu_indices(4, 4)

    def __init__(self, sign_invariant=True):
        self.sign_invariant = sign_invariant

    def transition(self, g, hard=False):
        n = torch.sqrt((g * g).sum(-1, keepdim=True) + 1e-12)
        return torch.cat([torch.cos(n), torch.sin(n) / n * g], -1)

    def mul(self, a, b):
        aw, ax, ay, az = a.unbind(-1)
        bw, bx, by, bz = b.unbind(-1)
        return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                            aw * bx + ax * bw + ay * bz - az * by,
                            aw * by - ax * bz + ay * bw + az * bx,
                            aw * bz + ax * by - ay * bx + az * bw], -1)

    def feature_dim(self):
        return 10 if self.sign_invariant else 14

    def features(self, x):
        i, j = self._iu.to(x.device)
        qq = (x.unsqueeze(-1) * x.unsqueeze(-2))[..., i, j]
        return qq if self.sign_invariant else torch.cat([x, qq], -1)


ALGEBRAS = {"R": Real, "C": Complex, "H": Quaternion}


def get_algebra(spec):
    """'R', 'C', 'H' or an Algebra instance."""
    if isinstance(spec, Algebra):
        return spec
    return ALGEBRAS[spec]()
