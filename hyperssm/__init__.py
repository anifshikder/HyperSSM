"""HyperSSM: state-space models whose channels use real, complex or quaternion numbers.

Companion code for "Symmetry in Time: A Representation Theory of Linear Recurrent Memory".
"""
from .algebras import Algebra, Real, Complex, Quaternion, get_algebra
from .scan import prefix_scan, affine_scan
from .layer import HyperLayer
from .model import HyperSSM
from .snap import discover_lattices
from .tasks import get_task, word_problem
from .train import train_word_problem, accuracy, batches

__version__ = "0.1.0"
