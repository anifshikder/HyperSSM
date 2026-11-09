"""HyperSSM: a stack of HyperLayers with a linear readout.

    model = HyperSSM(vocab_size=3, n_classes=3, layers=["R"])               # parity
    model = HyperSSM(vocab_size=3, n_classes=60, layers=["H"])              # A5
    model = HyperSSM(vocab_size=33, n_classes=120, layers=["R", "H"],
                     readout="stack")                                       # S5
    model = HyperSSM(d_in=3, n_classes=4, layers=["H"])                     # continuous inputs

Inputs are either integer tokens (B, T) with 0 = padding (when vocab_size is given) or
float features (B, T, d_in) with an optional mask. The output has one prediction per step.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .algebras import get_algebra
from .layer import HyperLayer


def _spec(s):
    return dict(algebra=s) if isinstance(s, str) else dict(s)


class HyperSSM(nn.Module):
    def __init__(self, n_classes, layers=("H",), vocab_size=None, d_in=None, pairs=False,
                 readout="top", init_scale=1.0):
        """
        n_classes:  output size of the linear readout
        layers:     list of algebra names ('R', 'C', 'H') or dicts with keys
                    algebra, channels, decay, input_term
        vocab_size: number of token ids (0 is padding); omit for float inputs
        d_in:       input feature size for float inputs
        pairs:      read (previous, current) token pairs instead of single tokens
        readout:    'top' (features of the top layer) or 'stack' (top features times
                    [1, features of the layer below], for a stack such as R -> H for S5)
        """
        super().__init__()
        assert (vocab_size is None) != (d_in is None), "give vocab_size (tokens) or d_in (features)"
        self.vocab_size, self.pairs, self.readout = vocab_size, pairs, readout
        d = (vocab_size ** 2 if pairs else vocab_size) if vocab_size else d_in
        self.layers, below_dim = nn.ModuleList(), 0
        for s in layers:
            s = _spec(s)
            alg = get_algebra(s.pop("algebra"))
            layer = HyperLayer(d, alg, below_dim=below_dim, init_scale=init_scale, **s)
            if below_dim:
                prev = self.layers[-1]
                layer.set_below_identity(self._feats(prev, prev.alg.identity((1, prev.K)))[0])
            self.layers.append(layer)
            below_dim = layer.K * alg.feature_dim()
        top = self.layers[-1]
        f = top.K * top.alg.feature_dim()
        if readout == "stack":
            assert len(self.layers) > 1
            low = self.layers[-2]
            f *= 1 + low.K * low.alg.feature_dim()
        self.head = nn.Linear(f, n_classes)
        self.hard = False                       # True: complex channels snap to their lattice

    @staticmethod
    def _feats(layer, x):
        return layer.alg.features(x).flatten(-2)

    def encode(self, x, mask=None):
        if self.vocab_size:
            mask = x > 0 if mask is None else mask
            tok = x
            if self.pairs:
                prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)
                tok = prev * self.vocab_size + x
            u = F.one_hot(tok, self.vocab_size ** 2 if self.pairs else self.vocab_size).float()
        else:
            u = x
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if mask is None else mask
        return u, mask

    def states(self, x, mask=None):
        """States of every layer: list of (B, T, K, state_dim)."""
        u, mask = self.encode(x, mask)
        out, below = [], None
        for layer in self.layers:
            s = layer(u, mask, below, hard=self.hard)
            out.append(s)
            below = self._feats(layer, s)
        return out

    def forward(self, x, mask=None):
        st = self.states(x, mask)
        f = self._feats(self.layers[-1], st[-1])
        if self.readout == "stack":
            low = self._feats(self.layers[-2], st[-2])
            low = torch.cat([torch.ones_like(low[..., :1]), low], -1)
            f = (f.unsqueeze(-1) * low.unsqueeze(-2)).flatten(-2)
        return self.head(f)

    @staticmethod
    def last(outputs, lengths):
        """Pick the output at the last real step of each sequence: (B, T, C) -> (B, C)."""
        return outputs[torch.arange(outputs.shape[0], device=outputs.device), lengths - 1]

    def numbers_of_state(self):
        return sum(l.K * l.alg.state_dim for l in self.layers)
