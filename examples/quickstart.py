"""Quickstart: train HyperSSM on three state-tracking tasks and read the learned rule off the weights.

    python examples/quickstart.py          # about a minute on a laptop CPU

Each task gets the number system the theory prescribes:
  parity            -> one real channel (+1 / -1)
  cycle navigation  -> one complex channel (a clock hand); its lattice n is discovered
  A5 (icosahedron)  -> one quaternion channel (a 3D orientation), sign-invariant readout
"""
import math

import torch

from hyperssm import train_word_problem

torch.manual_seed(0)

for task, kw in [("parity", dict(layers=["R"], steps=1500)),
                 ("cycle_navigation", dict(layers=["C"], steps=5000)),
                 ("A5_2", dict(layers=["H"], steps=4000))]:
    model, report = train_word_problem(task, log=None, **kw)
    print(f"{task:17s} accuracy on lengths 40-256: {report['val']:.4f}   "
          f"numbers of state: {report['numbers_of_state']}   discovered lattice: {report['lattices'] or '-'}")

# Read the A5 model: each symbol's transition is a rotation by 2*arccos|a0|.
# The two generators of A5 used here have orders 3 and 5, so the learned rotations should be
# 120 degrees and 72 or 144 degrees, about symmetry axes of one icosahedron.
layer = model.layers[0]
with torch.no_grad():
    for token, name in [(1, "(0 1 2)"), (2, "(0 1 2 3 4)")]:
        u = torch.zeros(1, 1, layer.d_in); u[0, 0, token] = 1
        a = layer.alg.transition(layer.generator(u).view(3))
        angle = math.degrees(2 * math.acos(min(1.0, abs(a[0].item()))))
        print(f"learned transition for {name:12s}: rotation by {angle:6.2f} degrees")
