"""Small additional checks: batch-vs-single consistency and determinism."""

import sys

import numpy as np

sys.path.insert(0, "python")
import mojo_eigen as me  # noqa: E402

rng = np.random.default_rng(77)
raw = rng.normal(size=(600, 4, 4))
sym = (raw + np.swapaxes(raw, -2, -1)) * 0.5
spd = np.swapaxes(raw, -2, -1) @ raw + np.eye(4)
rhs = rng.normal(size=(600, 4))
a3 = rng.normal(size=(600, 3, 3))

vals_b, vecs_b = me.eigh(sym)
svd_b = me.svd(a3)
ldlt_b = me.solve(spd, rhs, "ldlt")
worst = {"eigh": 0.0, "svd": 0.0, "ldlt": 0.0}
for i in range(600):
    v, w = me.eigh(sym[i])
    worst["eigh"] = max(worst["eigh"], np.abs(v - vals_b[i]).max(),
                        np.abs(np.abs(w) - np.abs(vecs_b[i])).max())
    u, s, vh = me.svd(a3[i])
    worst["svd"] = max(worst["svd"], np.abs(s - svd_b[1][i]).max(),
                       np.abs(u @ np.diag(s) @ vh - a3[i]).max())
    x = me.solve(spd[i], rhs[i], "ldlt")
    worst["ldlt"] = max(worst["ldlt"], np.abs(x - ldlt_b[i]).max())
print("serially-batched vs single-call max difference:", {k: f"{v:.2e}" for k, v in worst.items()})

repeat = [me.matmul(raw, raw) for _ in range(3)]
print("matmul bitwise deterministic across calls:",
      all(np.array_equal(repeat[0], r) for r in repeat[1:]))
