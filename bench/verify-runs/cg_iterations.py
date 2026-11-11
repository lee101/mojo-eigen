"""Check the quoted CG iteration count on the benchmark's own inputs.

bench.py builds the 120x120 Poisson system and a seed-7 rhs; record what each
side does so the text matches the run.
"""
import sys

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

sys.path.insert(0, "python")
import mojo_eigen as me  # noqa: E402


def poisson(side):
    one = np.ones(side)
    line = sparse.diags([-one, 4 * one, -one], [-1, 0, 1], shape=(side, side))
    identity = sparse.eye(side)
    coupling = sparse.diags([-one, -one], [-1, 1], shape=(side, side))
    return sparse.csr_matrix(sparse.kron(identity, line) + sparse.kron(coupling, identity))


rng = np.random.default_rng(7)
# replay bench.py's draws so the CG rhs is the one the benchmark uses
rng.normal(size=(250_000, 3, 3)); rng.normal(size=(250_000, 3, 3))
rng.normal(size=(150_000, 4, 4)); rng.normal(size=(150_000, 4))
rng.normal(size=(4, 4)); rng.normal(size=4)
rng.normal(size=(4, 4))
rng.normal(size=(3, 3))
system = poisson(120)
b = rng.normal(size=system.shape[0])

solver = spla.cg(system, b, rtol=1e-8, atol=0, maxiter=500)
print("scipy result type:", type(solver).__name__)
res = me.cg(system.data, system.indices, system.indptr, b, tol=1e-8, max_iter=500)
print(f"mojo iterations={res.iterations} relative_error={res.relative_error:.3e} info={res.info}")
x_scipy, info = solver[0], solver[1]
print(f"scipy info={info} relres={np.linalg.norm(system @ x_scipy - b) / np.linalg.norm(b):.3e}")
print(f"max|x_mojo - x_scipy| = {np.abs(res.x - x_scipy).max():.3e}")
