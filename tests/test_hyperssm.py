"""Run with:  pytest -q"""
import math

import numpy as np
import torch

from hyperssm import Complex, HyperSSM, Quaternion, Real, affine_scan, prefix_scan, get_task


def loop_product(a, mul):
    out, cur = [], a[:, 0]
    out.append(cur)
    for t in range(1, a.shape[1]):
        cur = mul(a[:, t], cur)
        out.append(cur)
    return torch.stack(out, 1)


def test_prefix_scan_matches_loop():
    torch.manual_seed(0)
    for alg in (Complex(), Quaternion()):
        a = alg.transition(torch.randn(2, 37, alg.gen_dim, dtype=torch.float64))
        assert torch.allclose(prefix_scan(a, alg.mul), loop_product(a, alg.mul), atol=1e-10)


def test_affine_scan_matches_loop():
    torch.manual_seed(0)
    alg = Quaternion()
    a = 0.9 * alg.transition(torch.randn(2, 29, 3, dtype=torch.float64))
    b = torch.randn(2, 29, 4, dtype=torch.float64)
    x, out = torch.zeros(2, 4, dtype=torch.float64), []
    for t in range(29):
        x = alg.mul(a[:, t], x) + b[:, t]
        out.append(x)
    assert torch.allclose(affine_scan(a, b, alg.mul), torch.stack(out, 1), atol=1e-10)


def test_quaternion_scan_matches_su2_matrices():
    """i, j, k <-> -i sigma_x, -i sigma_y, -i sigma_z."""
    torch.manual_seed(0)
    g = torch.randn(64, 3, dtype=torch.float64)
    sig = [torch.tensor([[0, 1], [1, 0]], dtype=torch.cdouble),
           torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cdouble),
           torch.tensor([[1, 0], [0, -1]], dtype=torch.cdouble)]
    M = torch.eye(2, dtype=torch.cdouble)
    for v in g:
        M = torch.linalg.matrix_exp(-1j * sum(v[k] * sig[k] for k in range(3))) @ M
    H = Quaternion()
    w, x, y, z = prefix_scan(H.transition(g[None]), H.mul)[0, -1]
    Mq = torch.tensor([[w - 1j * z, -1j * x - y], [-1j * x + y, w + 1j * z]], dtype=torch.cdouble)
    assert (Mq - M).abs().max() < 1e-10


def test_order_matters_only_for_quaternions():
    C, H = Complex(), Quaternion()
    a, b = C.transition(torch.tensor([0.3])), C.transition(torch.tensor([1.1]))
    assert torch.allclose(C.mul(a, b), C.mul(b, a))
    p, q = H.transition(torch.tensor([0.8, 0, 0])), H.transition(torch.tensor([0, 0, 0.8]))
    assert not torch.allclose(H.mul(p, q), H.mul(q, p))


def test_sign_invariant_readout_ignores_sign():
    H = Quaternion(sign_invariant=True)
    q = H.transition(torch.randn(5, 3))
    assert torch.allclose(H.features(q), H.features(-q))


def test_real_channel_is_exactly_plus_minus_one():
    g = torch.randn(1000, requires_grad=True)
    a = Real().transition(g)
    assert set(a.detach().unique().tolist()) <= {-1.0, 1.0}
    a.sum().backward()
    assert g.grad.abs().sum() > 0                      # straight-through gradient


def test_handset_parity_is_exact_at_every_length():
    """One real channel: symbol 'b' (token 2) flips the sign. Exact at length 1000."""
    gen, vocab, classes = get_task("parity")
    model = HyperSSM(classes, layers=["R"], vocab_size=vocab)
    with torch.no_grad():
        model.layers[0].generator.weight[:] = torch.tensor([[1.0, 1.0, -1.0]])
        model.layers[0].generator.bias.zero_()
        model.head.weight[:] = torch.tensor([[0.0], [5.0], [-5.0]])
        model.head.bias[:] = torch.tensor([-10.0, 0.0, 0.0])
    x, y, L = gen(256, 500, 1000, np.random.default_rng(0))
    pred = HyperSSM.last(model(torch.as_tensor(x)), torch.as_tensor(L)).argmax(-1).numpy()
    assert (pred == y).all()


def test_handset_cycle_navigation_with_lattice():
    """One complex channel turning by +-2pi/5, snapped to the 5th roots of unity."""
    gen, vocab, classes = get_task("cycle_navigation")
    model = HyperSSM(classes, layers=["C"], vocab_size=vocab)
    step = 2 * math.pi / 5
    with torch.no_grad():
        model.layers[0].generator.weight[:] = torch.tensor([[0, 0, step + 0.01, -step - 0.02] + [0] * 5])
        model.layers[0].generator.bias.zero_()
        model.head.weight.zero_(); model.head.bias.fill_(-10.0)
        for j in range(5):
            model.head.weight[4 + j] = 10 * torch.tensor([math.cos(j * step), math.sin(j * step)])
            model.head.bias[4 + j] = 0.0
    model.layers[0].alg.lattice = 5
    model.hard = True
    x, y, L = gen(256, 500, 1000, np.random.default_rng(0))
    pred = HyperSSM.last(model(torch.as_tensor(x)), torch.as_tensor(L)).argmax(-1).numpy()
    assert (pred == y).all()


def test_stack_shapes_and_state_count():
    model = HyperSSM(120, layers=["R", "H"], vocab_size=33, readout="stack")
    x = torch.randint(1, 33, (4, 50))
    assert model(x).shape == (4, 50, 120)
    assert model.numbers_of_state() == 5


def test_float_inputs():
    model = HyperSSM(4, layers=[dict(algebra="H", channels=2)], d_in=3)
    assert model(torch.randn(2, 10, 3)).shape == (2, 10, 4)
