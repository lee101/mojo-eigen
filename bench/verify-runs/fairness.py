"""Audit that both sides of each benchmark row do the work the row claims.

Prints the reference-side cost model and checks that the Mojo side actually
consumes distinct matrices rather than repeating one factorization.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "python")
import eigenpy  # noqa: E402
import mojo_eigen as me  # noqa: E402
from scipy import sparse  # noqa: E402

rng = np.random.default_rng(11)


def poisson(side):
    one = np.ones(side)
    line = sparse.diags([-one, 4 * one, -one], [-1, 0, 1], shape=(side, side))
    identity = sparse.eye(side)
    coupling = sparse.diags([-one, -one], [-1, 1], shape=(side, side))
    return sparse.csr_matrix(sparse.kron(identity, line) + sparse.kron(coupling, identity))


def best(fn, r=3):
    b = float("inf")
    for _ in range(r):
        t = time.perf_counter(); fn(); b = min(b, time.perf_counter() - t)
    return b * 1e3


print("### LDLT: 20k distinct factorizations vs one factor reused 20k times")
raw = rng.normal(size=(20000, 4, 4))
distinct = np.swapaxes(raw, -2, -1) @ raw + np.eye(4)
rhs = rng.normal(size=(20000, 4))
one = distinct[:1]
one_rhs = rhs[:1]

def ref_constructor_per_item():
    out = np.empty((20000, 4))
    for i in range(20000):
        out[i] = eigenpy.LDLT(distinct[i]).solve(rhs[i])
    return out

def ref_repeat_factored_rhs():
    # bench.py: one factorization, but a new right-hand side per iteration
    f = eigenpy.LDLT(distinct[0])
    return [f.solve(rhs[i]) for i in range(20000)]

def ref_factor_once():
    f = eigenpy.LDLT(distinct[0])
    return [f.solve(rhs[0]) for _ in range(20000)]

print(f"mojo, 20k distinct:                    {best(lambda: me.solve(distinct, rhs, 'ldlt')):7.3f} ms")
print(f"eigenpy, 20k construct+solve:          {best(ref_constructor_per_item):7.3f} ms")
print(f"eigenpy (bench.py), 20k rhs re-solves: {best(ref_repeat_factored_rhs):7.3f} ms")
print(f"eigenpy, one factored rhs reused:      {best(ref_factor_once):7.3f} ms")
print("mojo solution of distinct items correct:",
      np.allclose(distinct @ me.solve(distinct, rhs, "ldlt")[..., None], rhs[..., None], rtol=2e-11, atol=2e-11))

print()
print("### eigh: bench.py times 20000 constructions, not 20000 eigensolves")
sym = rng.normal(size=(4, 4)); sym = (sym + sym.T) * 0.5
batch = np.repeat(sym[None, ...], 20000, axis=0)

def ref_solve(every):
    for i in range(0, 20000, every):
        s = eigenpy.SelfAdjointEigenSolver(sym)
        if every == 1:
            s.eigenvalues()

print(f"mojo, 20k solves:                           {best(lambda: me.eigh(batch)):7.3f} ms")
print(f"eigenpy, 20k constructions (as benchmarked): {best(lambda: ref_solve(20000)):7.3f} ms")
print(f"eigenpy, alternating construct+solve (20k):  {best(lambda: [eigenpy.SelfAdjointEigenSolver(sym).eigenvalues() for _ in range(20000)]):7.3f} ms")

print()
print("### SVD: bench.py times 20000 constructions without asking for U/S/Vh")
dense = rng.normal(size=(3, 3)); db = np.repeat(dense[None, ...], 20000, axis=0)
print(f"mojo, 20k full SVDs:                         {best(lambda: me.svd(db)):7.3f} ms")
print(f"eigenpy, 20k construct only (as benchmarked): {best(lambda: [eigenpy.HhJacobiSVD(dense, 20) for _ in range(20000)]):7.3f} ms")
print(f"eigenpy, 20k construct + singularValues():    {best(lambda: [eigenpy.HhJacobiSVD(dense, 20).singularValues() for _ in range(20000)]):7.3f} ms")

print()
print("### CG: same CSR system, same tolerance")
A = poisson(120)
b = rng.normal(size=A.shape[0])
r = me.cg(A.data, A.indices, A.indptr, b, tol=1e-8, max_iter=500)
from scipy.sparse import linalg as spla  # noqa: E402
x, info = spla.cg(A, b, rtol=1e-8, atol=0, maxiter=500)
print(f"mojo iterations={r.iterations} relres={r.relative_error:.3e} info={r.info}")
print(f"scipy converged={info == 0} relres={np.linalg.norm(A @ x - b) / np.linalg.norm(b):.3e}")
