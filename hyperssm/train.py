"""A small training loop for the state-tracking tasks (final-token loss, as in the benchmarks).

    from hyperssm import HyperSSM, train_word_problem
    model, report = train_word_problem("A5_2", layers=["H"])
"""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from .model import HyperSSM
from .snap import discover_lattices
from .tasks import get_task


def batches(gen, n, lmin, lmax, seed, bs=256):
    rng = np.random.default_rng(seed)
    xs, ys, Ls = [], [], []
    for i in range(0, n, bs):
        x, y, L = gen(min(bs, n - i), lmin, lmax, rng)
        xs.append(np.pad(x, ((0, 0), (0, lmax - x.shape[1])))); ys.append(y); Ls.append(L)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(Ls)


@torch.no_grad()
def accuracy(model, data, device="cpu", bs=128):
    model.eval()
    xs, ys, Ls = data
    correct = 0
    for i in range(0, len(xs), bs):
        x = torch.as_tensor(xs[i:i + bs], device=device)
        L = torch.as_tensor(Ls[i:i + bs], device=device)
        pred = HyperSSM.last(model(x), L).argmax(-1).cpu().numpy()
        correct += (pred == ys[i:i + bs]).sum()
    model.train()
    return correct / len(xs)


def train_word_problem(task, layers=("H",), pairs=False, readout="top", steps=5000, lr=1e-2, batch=256,
                       train_len=(3, 40), val_len=(40, 256), n_val=512, restarts=8, seed=0,
                       eval_every=250, device="cpu", log=print):
    """Train a HyperSSM on a state-tracking task with the final-token loss.

    Curriculum: lengths 3-5 for the first 30% of training, then a ramp to 40.
    Restarts: an attempt below 90% halfway through is restarted from a new initialization.
    Complex channels are snapped to their discovered lattice at the end.
    Returns (model, report)."""
    gen, vocab, classes = get_task(task)
    val = batches(gen, n_val, *val_len, seed=10_000 + seed)
    best_model, best = None, -1.0
    for attempt in range(1 + restarts):
        torch.manual_seed(1000 * seed + attempt)
        rng = np.random.default_rng(seed + 31 * attempt)
        model = HyperSSM(classes, layers=layers, vocab_size=vocab, pairs=pairs, readout=readout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        warm = max(1, steps // 10)                              # 10% warmup, then cosine decay
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else
            (1e-5 + 0.5 * (lr - 1e-5) * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))) / lr)
        a_best, a_state = -1.0, None
        for s in range(1, steps + 1):
            frac = s / steps
            lmax = 3 + int(2 * frac / 0.3) if frac < 0.3 else min(train_len[1], 5 + int((frac - 0.3) / 0.4 * 35))
            x, y, L = (torch.as_tensor(a, device=device) for a in gen(batch, train_len[0], lmax, rng))
            loss = F.cross_entropy(HyperSSM.last(model(x), L), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            if s % eval_every == 0 or s == steps:
                acc = _snapped_accuracy(model, val, device)
                if acc > a_best:
                    a_best, a_state = acc, copy.deepcopy(model.state_dict())
                if log:
                    log(f"[{task}] attempt {attempt} step {s:5d} loss {loss.item():.3f} "
                        f"val {acc:.4f} (best {a_best:.4f})")
                if a_best >= 0.9995 or (s >= steps // 2 and a_best < 0.9 and attempt < restarts):
                    break
        if a_best > best:
            model.load_state_dict(a_state)
            best, best_model = a_best, model
        if best >= 0.9995:
            break
    model = best_model
    lattices = discover_lattices(model, lambda m: accuracy(m, val, device))
    report = dict(task=task, val=float(accuracy(model, val, device)), lattices=lattices,
                  attempts=attempt + 1, numbers_of_state=model.numbers_of_state())
    return model, report


def _snapped_accuracy(model, val, device):
    """Accuracy with complex channels snapped to their best lattice (training stays continuous)."""
    found = discover_lattices(model, lambda m: accuracy(m, val, device))
    acc = accuracy(model, val, device)
    for i in found:
        model.layers[i].alg.lattice = None
    model.hard = False
    return acc
