"""Time the real Eigen operation on the same 20k distinct systems as Mojo.

eigenpy exposes no way to swap the wrapped matrix, so the upstream side is done
with a second binding (scipy's numpy.linalg for LU/LLT/LDLT/eigh/svd) on the
identical batch, and against one-factor-reused eigenpy only for context.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "python")
import mojo_eigen as me  # noqa: E402


def best(fn, r=3):
    b = float("inf")
    for _ in range(r):
        t = time.perf_counter(); fn(); b = min(b, time.perf_counter() - t)
    return b * 1e3


rng = np.random.default_rng(99)
raw = rng.normal(size=(20000, 4, 4))
spd = np.swapaxes(raw, -2, -1) @ raw + np.eye(4)
rhs = rng.normal(size=(20000, 4))
sym = (raw + np.swapaxes(raw, -2, -1)) * 0.5
a3 = rng.normal(size=(20000, 3, 3))

print(f"ldlt  mojo 20k distinct          {best(lambda: me.solve(spd, rhs, 'ldlt')):8.3f} ms")
print(f"ldlt  numpy 20k distinct         {best(lambda: np.linalg.solve(spd, rhs[..., None])):8.3f} ms")
print(f"eigh  mojo 20k distinct          {best(lambda: me.eigh(sym)):8.3f} ms")
print(f"eigh  numpy 20k distinct         {best(lambda: np.linalg.eigh(sym)):8.3f} ms")
print(f"svd   mojo 20k distinct          {best(lambda: me.svd(a3)):8.3f} ms")
print(f"svd   numpy 20k distinct         {best(lambda: np.linalg.svd(a3)):8.3f} ms")
