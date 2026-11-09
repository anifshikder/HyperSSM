"""Parallel prefix scans: O(T) work, O(log T) depth.

Both scans only need the multiplication to be associative, not commutative, which is
why a channel may use any associative number system (real, complex, quaternion, matrix).
"""
import torch


def prefix_scan(a, mul):
    """Inclusive product scan along dim 1: out[:, t] = a[:, t] * ... * a[:, 0].

    a:   (B, T, ...) elements of an algebra
    mul: associative product, mul(x, y) = x * y (x applied after y)
    """
    T, d = a.shape[1], 1
    while d < T:
        a = torch.cat([a[:, :d], mul(a[:, d:], a[:, :-d])], 1)
        d *= 2
    return a


def affine_scan(a, b, mul):
    """States of x_t = a_t x_{t-1} + b_t with x_{-1} = 0, as a scan over affine maps.

    Steps compose as (a2, b2) o (a1, b1) = (a2 a1, a2 b1 + b2), which is associative.
    To start from a state x_0 instead of 0, fold it into b_0: b_0 <- a_0 x_0 + b_0.
    """
    T, d = a.shape[1], 1
    while d < T:
        a, b = (torch.cat([a[:, :d], mul(a[:, d:], a[:, :-d])], 1),
                torch.cat([b[:, :d], mul(a[:, d:], b[:, :-d]) + b[:, d:]], 1))
        d *= 2
    return b
