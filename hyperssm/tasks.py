"""State-tracking tasks (the protocol of Terzic et al., NeurIPS 2025, and Walker et al., 2025).

Each generator returns (tokens, targets, lengths) as numpy arrays:
  tokens  (B, L) int, 0 = padding, symbols start at 1
  targets (B,)   int, the answer after the whole sequence
  lengths (B,)   int
"""
import itertools
import numpy as np


def _lengths(B, lmin, lmax, rng):
    return rng.integers(lmin, lmax + 1, B)


def _pad(x, L):
    x[np.arange(x.shape[1])[None] >= L[:, None]] = 0
    return x


def parity(B, lmin, lmax, rng):
    """Is the number of 'b' symbols even? Group: Z2. Vocabulary 3, classes 3 (labels 1, 2)."""
    L = _lengths(B, lmin, lmax, rng)
    x = _pad(rng.integers(1, 3, (B, lmax)), L)
    return x, np.where(((x == 2).sum(1) % 2) == 0, 1, 2), L


def cycle_navigation(B, lmin, lmax, rng):
    """Walk on a cycle of 5 positions with steps stay/+1/-1. Group: Z5. Vocabulary 9, classes 9."""
    L = _lengths(B, lmin, lmax, rng)
    x = _pad(rng.integers(1, 4, (B, lmax)), L)
    return x, 4 + ((x == 2).sum(1) - (x == 3).sum(1)) % 5, L


def even_pairs(B, lmin, lmax, rng):
    """Do the first and last symbols agree? Read as Z2 on (previous, current) pairs.
    Vocabulary 3, classes 3. Use HyperSSM(..., pairs=True)."""
    L = _lengths(B, lmin, lmax, rng)
    x = _pad(rng.integers(1, 3, (B, lmax)), L)
    return x, np.where(x[:, 0] == x[np.arange(B), L - 1], 1, 2), L


def _perm_group(name):
    def inv(p):
        return sum(p[i] > p[j] for i in range(5) for j in range(i + 1, 5))
    elems = [p for p in itertools.permutations(range(5)) if name == "S5" or inv(p) % 2 == 0]
    idx = {p: i for i, p in enumerate(elems)}
    mul = np.array([[idx[tuple(g[h[i]] for i in range(5))] for h in elems] for g in elems])
    return elems, idx, mul


_STD = {"A5": [(1, 2, 0, 3, 4), (1, 2, 3, 4, 0)],     # (0 1 2), (0 1 2 3 4)
        "S5": [(1, 0, 2, 3, 4), (1, 2, 3, 4, 0)]}     # (0 1),   (0 1 2 3 4)


def word_problem(group="A5", n_generators=2, seed=12345):
    """Word problem of A5 (60 elements) or S5 (120): compose a sequence of generators.
    Two standard generators plus n_generators - 2 random group elements.
    Returns (generator function, vocabulary size, number of classes)."""
    elems, idx, mul = _perm_group(group)
    e = idx[tuple(range(5))]
    gens = [idx[p] for p in _STD[group]]
    rest = [i for i in range(len(elems)) if i not in gens and i != e]
    gens += list(np.random.default_rng(seed + n_generators).choice(rest, n_generators - 2, replace=False))
    gens = np.array(gens)

    def gen(B, lmin, lmax, rng):
        L = _lengths(B, lmin, lmax, rng)
        tok = _pad(rng.integers(1, len(gens) + 1, (B, lmax)), L)
        cur = np.full(B, e)
        for t in range(lmax):
            cur = np.where(t < L, mul[gens[np.maximum(tok[:, t] - 1, 0)], cur], cur)  # g_t o (previous)
        return tok, cur, L

    return gen, len(gens) + 1, len(elems)


TASKS = {
    "parity": (parity, 3, 3),
    "cycle_navigation": (cycle_navigation, 9, 9),
    "even_pairs": (even_pairs, 3, 3),
}


def get_task(name):
    """'parity', 'cycle_navigation', 'even_pairs', or 'A5_n' / 'S5_n' (n generators)."""
    if name in TASKS:
        return TASKS[name]
    group, n = name.split("_")
    return word_problem(group, int(n))
