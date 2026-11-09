# HyperSSM

**State-space models whose memory uses the right number system for the task.**

Companion code for *Symmetry in Time: A Representation Theory of Linear Recurrent Memory*
(Anif N. Shikder, Fields Lab for Network Computation, Fields Institute).

---

## The idea in one minute

A state-space model (SSM) such as S4, LRU or Mamba stores its past in a bank of numbers and
updates each one by multiplying it by another number chosen by the input. Real and complex
numbers commute (`ab = ba`), so the same steps in a different order end in the same state:
the memory cannot tell *a quarter turn forward, then left* from *left, then forward*.

The paper shows that a linear recurrent memory is a *representation* of the sequence it reads,
and that every such memory splits into independent blocks over three number systems:

| number system | numbers per channel | what one channel can remember | example task |
|---|---|---|---|
| real ℝ (±1) | 1 | a single bit: parity | is the number of `b`s even? |
| complex ℂ | 2 | counting modulo n (a clock hand) | position on a cycle |
| quaternion ℍ | 4 | 3D rotations, including the 60 rotations of an icosahedron (A5) | orientation of a body, a qubit |
| ℝ beneath ℍ (two layers) | 5 | all 120 permutations of five objects (S5) | shuffling five cards |

HyperSSM lets each channel choose its number system, keeps the parallel scan of an SSM
(linear time, log-depth training), and reads its state with a small linear readout.

## Install

```bash
git clone https://github.com/<your-username>/hyperssm.git
cd hyperssm
pip install -e .            # needs PyTorch >= 2.0 and NumPy
pytest -q                   # optional: 11 tests, a few seconds
```

## Quickstart

```python
from hyperssm import train_word_problem

# A5: the 60 rotations of an icosahedron, from 2 generators. One quaternion channel (4 numbers).
model, report = train_word_problem("A5_2", layers=["H"], steps=4000)
print(report)   # {'task': 'A5_2', 'val': 1.0, ..., 'numbers_of_state': 4}
```

Trained on sequences of length 3–40 and evaluated on lengths 40–256; it trains in seconds on a CPU.

```bash
python examples/quickstart.py          # parity, cycle navigation and A5, then reads the learned rotations
python examples/continuous_inputs.py   # learns to integrate a miscalibrated gyroscope (a few minutes)
```

## Using the model

```python
import torch
from hyperssm import HyperSSM

# Token inputs (0 = padding). One prediction per step; take the last real step.
model = HyperSSM(n_classes=60, layers=["H"], vocab_size=3)
tokens = torch.randint(1, 3, (8, 100))
logits = model(tokens)                                  # (8, 100, 60)
last = HyperSSM.last(logits, torch.full((8,), 100))     # (8, 60)

# Float inputs (e.g. sensor readings).
model = HyperSSM(n_classes=4, layers=["H"], d_in=3)
states = model.states(torch.randn(8, 100, 3))           # list of (8, 100, channels, 4) per layer

# A stack: a sign channel beneath a quaternion channel, read together (the S5 model).
model = HyperSSM(n_classes=120, layers=["R", "H"], vocab_size=33, readout="stack")

# More channels, decay, or an input term (x_k = a(u_k) x_{k-1} + B(u_k) u_k):
model = HyperSSM(n_classes=10, vocab_size=20,
                 layers=[dict(algebra="C", channels=8, decay=True, input_term=True),
                         dict(algebra="H", channels=2)])
```

| argument | meaning |
|---|---|
| `layers` | list of `"R"`, `"C"`, `"H"`, or dicts `{algebra, channels, decay, input_term}` |
| `vocab_size` / `d_in` | token inputs (one-hot inside) or float inputs |
| `pairs=True` | read (previous, current) token pairs, like a width-2 convolution |
| `readout` | `"top"`: features of the top layer; `"stack"`: top features × [1, features of the layer below] |
| `model.hard = True` | snap complex channels to their lattice (set by `discover_lattices`) |

What the readout sees from each number system (`algebra.features`):

* ℝ: the sign.  ℂ: the point on the circle.
* ℍ: the 10 entries of `q qᵀ` (default, `sign_invariant=True`), which ignore the sign of `q`.
  `q` and `−q` are the same rotation, so this is the right readout for rotations, for A5, and for
  the Bloch vector of a qubit. Use `Quaternion(sign_invariant=False)` when the sign matters (the
  quaternion group Q8).

## How it works

```
hyperssm/
  algebras.py   Real, Complex, Quaternion: transition exp(v), product, readout features
  scan.py       parallel prefix scans (only associativity is needed, not commutativity)
  layer.py      HyperLayer: x_k = exp(-alpha) exp(v(u_k)) x_{k-1} + B(u_k) u_k
  model.py      HyperSSM: a stack of layers + linear readout
  snap.py       task-free snapping of complex channels to the smallest lattice n
  tasks.py      parity, cycle navigation, even pairs, A5 / S5 word problems
  train.py      a small training loop with the benchmark protocol
```

Each channel's update is `x_k = a(u_k) x_{k-1}` with `a = exp(v)`, where the generator `v` is
linear in the input (and, in a stack, in the readout features of the layer below at the previous
step). For a complex channel `v` is an angle; for a quaternion channel it is an axis times an angle.
Because the product is associative, all states come from one parallel scan.

**Task-free training.** Nothing about a task's group is given to the model:

* real channels are exactly ±1 in the forward pass (straight-through gradient);
* complex channels train continuously and are then snapped to the *smallest* n whose n-th roots of
  unity give the best validation accuracy (`discover_lattices`);
* quaternion channels stay continuous.

## Reproducing the paper

The `notebooks/` folder holds the exact code behind every number in the paper. Each runs on
Google Colab (GPU recommended for 3 and 4) and downloads its own data.

| notebook | paper | data |
|---|---|---|
| `1_state_tracking_benchmarks.ipynb` | Tables I–II: regular languages, A5, S5 | generated |
| `2_gyroscope_orientation.ipynb` | Fig. 4(a): orientation from a real gyroscope | [BROAD](https://github.com/dlaidig/broad) (downloaded) |
| `3_qubit_gates.ipynb` | Fig. 4(b), Fig. S1: qubit circuits, imperfect gates | generated |
| `4_protein_backbone.ipynb` | Fig. 4(c), Table S1: protein backbones | [PDB](https://www.rcsb.org) (downloaded) |

The `hyperssm/` package is a clean, reusable version of the same model. The notebooks keep
their own copies of the code so that each one runs on its own.

**Training notes.**
* Products of rotations have local minima, so some runs need a restart: `train_word_problem`
  restarts an attempt that is still far from exact halfway through.
* A5 trains in seconds and cycle navigation in under a minute on a CPU. S5 is much harder to
  *find* by gradient descent, although the model can represent it exactly: the test
  `tests/test_s5_construction.py` sets the two-layer S5 model by hand and checks it is exact at
  length 1000. Training S5 needs long attempts (20,000 steps each, several restarts; a GPU helps).
  Use `notebooks/1_state_tracking_benchmarks.ipynb` for the exact protocol behind the paper.

## Citation

```bibtex
@article{shikder2026symmetry,
  title  = {Symmetry in Time: A Representation Theory of Linear Recurrent Memory},
  author = {Shikder, Anif N.},
  year   = {2026}
}
```

## License

MIT (see `LICENSE`).
