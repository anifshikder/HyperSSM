"""The two-layer S5 construction of the paper, set by hand: a sign channel beneath a quaternion
channel (five numbers of state), exact on S5 with 32 generators at length 1000.

S5 = A5 x| C2. Write each element as g = t n with t in {e, (0 1)} and n in A5. Reading s sends
t to the representative t' of the coset of s t, and n to (t'^-1 s t) n: a left multiplication in
A5 chosen by s and by the sign below. A5 is the rotation group of the icosahedron, so the
quaternion channel multiplies by lifts of these rotations into the binary icosahedral group.
"""
import itertools
import math

import numpy as np
import torch

from hyperssm import HyperSSM, Quaternion
from hyperssm.tasks import _perm_group, word_problem

PHI = (1 + 5 ** 0.5) / 2


def binary_icosahedral():
    """The 120 unit quaternions of 2I."""
    qs = set()
    for i in range(4):
        for s in (1, -1):
            v = [0.0] * 4; v[i] = s; qs.add(tuple(v))
    for signs in itertools.product((0.5, -0.5), repeat=4):
        qs.add(signs)
    base = (PHI / 2, 0.5, 1 / (2 * PHI), 0.0)
    even = [p for p in itertools.permutations(range(4))
            if sum(p[i] > p[j] for i in range(4) for j in range(i + 1, 4)) % 2 == 0]
    for p in even:
        for signs in itertools.product((1, -1), repeat=3):
            v = [0.0] * 4
            vals = [base[0] * signs[0], base[1] * signs[1], base[2] * signs[2], 0.0]
            for k in range(4):
                v[p[k]] = vals[k]
            qs.add(tuple(v))
    Q = np.array(sorted(qs))
    assert len(Q) == 120
    return Q


def qmul(a, b):
    return Quaternion().mul(torch.tensor(a), torch.tensor(b)).numpy()


def rot_key(q):
    q = q if (q[0] > 1e-9 or (abs(q[0]) < 1e-9 and tuple(np.sign(q[1:]).tolist()) >= (0, 0, 0))) else -q
    return tuple(np.round(q, 6))


def a5_into_2i():
    """A homomorphism A5 -> 2I/{+-1}: lifts (up to sign) of every element, as quaternions."""
    elems, idx, mul = _perm_group("A5")
    a, b = idx[(1, 2, 0, 3, 4)], idx[(1, 2, 3, 4, 0)]
    Q = binary_icosahedral()
    order = lambda q: next(k for k in range(1, 11) if abs(_qpow(q, k)[0]) > 1 - 1e-9)
    cands3 = [q for q in Q if order(q) == 3]
    cands5 = [q for q in Q if order(q) == 5]
    for x in cands3:
        for y in cands5:
            img = {idx[tuple(range(5))]: np.array([1.0, 0, 0, 0])}
            frontier, ok = [idx[tuple(range(5))]], True
            while frontier and ok:
                g = frontier.pop()
                for gen, q in ((a, x), (b, y)):
                    h = mul[gen, g]
                    qh = qmul(q, img[g])
                    if h in img:
                        if rot_key(img[h]) != rot_key(qh):
                            ok = False; break
                    else:
                        img[h] = qh; frontier.append(h)
            if ok and len(img) == 60:
                return img
    raise RuntimeError("no embedding found")


def _qpow(q, k):
    r = np.array([1.0, 0, 0, 0])
    for _ in range(k):
        r = qmul(q, r)
    return r


def qlog(q):
    """v with exp(v) = q."""
    q = q / np.linalg.norm(q)
    th = math.acos(max(-1.0, min(1.0, q[0])))
    n = np.linalg.norm(q[1:])
    return np.zeros(3) if n < 1e-12 else th * q[1:] / n


def test_handset_s5_two_layers_exact():
    gen, vocab, classes = word_problem("S5", 32)
    elems, idx, mul = _perm_group("S5")
    sign = lambda p: (-1) ** sum(p[i] > p[j] for i in range(5) for j in range(i + 1, 5))
    inv = {g: next(h for h in range(120) if mul[g, h] == idx[tuple(range(5))]) for g in range(120)}
    e, tau = idx[tuple(range(5))], idx[(1, 0, 2, 3, 4)]
    a5_elems, a5_idx, _ = _perm_group("A5")
    lift = a5_into_2i()
    gens = list(np.random.default_rng(12345 + 32).choice(
        [i for i in range(120) if i not in (idx[(1, 0, 2, 3, 4)], idx[(1, 2, 3, 4, 0)]) and i != e], 30, replace=False))
    gens = [idx[(1, 0, 2, 3, 4)], idx[(1, 2, 3, 4, 0)]] + gens

    model = HyperSSM(classes, layers=["R", "H"], vocab_size=vocab, readout="stack")
    with torch.no_grad():
        sgn, top = model.layers
        sgn.generator.weight.zero_(); sgn.generator.bias.zero_()
        top.generator.weight.zero_(); top.generator.bias.zero_()
        sgn.generator.weight[0, 0] = 1.0
        for tok, g in enumerate(gens, start=1):
            sgn.generator.weight[0, tok] = float(sign(elems[g]))
            v = {}
            for chi, t in ((1.0, e), (-1.0, tau)):
                st = mul[g, t]
                t2 = e if sign(elems[st]) == 1 else tau
                m = mul[inv[t2], st]                                   # t'^-1 s t, in A5
                v[chi] = qlog(lift[a5_idx[elems[m]]])
            W0, W1 = (v[1.0] + v[-1.0]) / 2, (v[1.0] - v[-1.0]) / 2   # v = W0 + chi W1
            top.generator.weight[:, tok * 2 + 0] = torch.tensor(W0, dtype=torch.float32)
            top.generator.weight[:, tok * 2 + 1] = torch.tensor(W1, dtype=torch.float32)
        # readout: logit_g = (q . p_g)^2 (1 + s s_g) / 2, linear in qq^T (x) [1, s]
        iu = torch.triu_indices(4, 4)
        model.head.weight.zero_(); model.head.bias.zero_()
        for g in range(120):
            t = e if sign(elems[g]) == 1 else tau
            n = mul[inv[t], g]
            p = torch.tensor(lift[a5_idx[elems[n]]], dtype=torch.float32)
            w = (p[:, None] * p[None, :] * (2 - torch.eye(4)))[iu[0], iu[1]]
            s_g = float(sign(elems[g]))
            model.head.weight[g] = torch.stack([w / 2, w * s_g / 2], -1).flatten()
    x, y, L = gen(128, 900, 1000, np.random.default_rng(0))
    pred = HyperSSM.last(model(torch.as_tensor(x)), torch.as_tensor(L)).argmax(-1).numpy()
    assert (pred == y).mean() == 1.0
