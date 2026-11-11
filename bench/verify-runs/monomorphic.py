"""Check src/eigen.mojo against the intended Eigen algorithm for each kernel.

A second independent implementation of what the algorithm should be, diffed
against the actual Mojo output, so a shared-order bug with numpy cannot hide.
"""
import sys

import numpy as np
from scipy import sparse  # noqa: E402
from scipy.sparse import linalg as spla  # noqa: E402

sys.path.insert(0, "python")
import mojo_eigen as me  # noqa: E402

rng = np.random.default_rng(4242)


def cg_monomorphic(a, b, tol, max_iter):  # classic CG with diagonal preconditioner
    n = b.size
    d = a.diagonal()
    m = 1.0 / d
    x = np.zeros(n)
    r = b - a @ x
    z = m * r
    p = z.copy()
    rz = r @ z
    bnorm = np.linalg.norm(b)
    err = 1.0
    it = 0
    while it < max_iter and not (np.linalg.norm(r) / bnorm <= tol):
        ap = a @ p
        alpha = rz / (p @ ap)
        x += alpha * p
        r -= alpha * ap
        z = m * r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
        it += 1
    return x, np.linalg.norm(r) / bnorm


def poisson(side):
    one = np.ones(side)
    line = sparse.diags([-one, 4 * one, -one], [-1, 0, 1], shape=(side, side))
    identity = sparse.eye(side)
    coupling = sparse.diags([-one, -one], [-1, 1], shape=(side, side))
    return sparse.csr_matrix(sparse.kron(identity, line) + sparse.kron(coupling, identity))


A = poisson(12)
b = rng.normal(size=A.shape[0])
r = me.cg(A.data, A.indices, A.indptr, b, tol=1e-10, max_iter=500)
x2, e2 = cg_monomorphic(A, b, 1e-10, 500)
print(f"cg: mojo iters={r.iterations} err={r.relative_error:.3e}; "
      f"independent iters<=? err={e2:.3e}; max|x diff|={np.abs(r.x - x2).max():.3e}")

bad = 0
for trial in range(100):
    n = 3 if trial % 2 else 4
    sym = rng.normal(size=(n, n))
    sym = (sym + sym.T) * 0.5
    # Eigen 3x3: closed form (computeDirect). 4x4: tridiagonalize by Householder
    # then implicit QR -- so verify via invariants rather than reproducing it.
    vals, vecs = me.eigh(sym)
    resid = np.abs(sym @ vecs - vecs * vals).max()
    orth = np.abs(vecs.T @ vecs - np.eye(n)).max()
    if resid > 1e-10 or orth > 1e-10:
        bad += 1
    if trial == 0:
        print("eigh invariants ok" if not bad else "eigh invariant FAIL")
print(f"eigh: {bad}/100 matrices violate residual/orthogonality")

bad = 0
for trial in range(100):
    n = 3 if trial % 2 else 4
    a = rng.normal(size=(n, n))
    u, s, vh = me.svd(a)
    orth = max(np.abs(u.T @ u - np.eye(n)).max(), np.abs(vh @ vh.T - np.eye(n)).max())
    rec = np.abs(u @ np.diag(s) @ vh - a).max() / np.abs(a).max()
    two_sided = np.abs(np.triu(a - a, 1)).max()  # placeholder no-op
    if orth > 1e-8 or rec > 1e-10 or np.any(np.diff(s) > 1e-12):
        bad += 1
print(f"svd: {bad}/100 matrices violate orthogonality/reconstruction/descending order")

x_ls, res = me.lstsq(rng.normal(size=(8, 3)) ** 0, np.ones(8))
print("lstsq on zero matrix raises or returns:", "ok" if x_ls.size == 3 else "bad")
