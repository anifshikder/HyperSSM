"""Task-free snapping of complex channels.

A channel that tracks a finite group has transitions in a finite group, and the finite
groups of unit complex numbers are the n-th roots of unity. After training, each complex
layer is snapped to the SMALLEST n whose roots of unity give the best validation accuracy;
n is read off the trained model, never given.
"""
from .algebras import Complex


def discover_lattices(model, evaluate, max_n=64, target=0.9995):
    """evaluate(model) -> validation accuracy. Sets model.layers[i].alg.lattice for every complex
    layer and returns {layer index: n}. Leaves model.hard = True."""
    found = {}
    for i, layer in enumerate(model.layers):
        if not isinstance(layer.alg, Complex):
            continue
        best_n, best = None, -1.0
        for n in range(1, max_n + 1):
            layer.alg.lattice = n
            model.hard = True
            acc = evaluate(model)
            if acc > best + 1e-9:
                best_n, best = n, acc
            if best >= target:
                break
        layer.alg.lattice = best_n
        found[i] = best_n
    model.hard = True
    return found
