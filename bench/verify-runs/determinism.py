"""Does the same large batch produce the same numbers on every call?

The batched kernels are split across worker chunks, so a race in the result or
scratch buffers would show up here as run-to-run differences.
"""
import sys

import numpy as np

sys.path.insert(0, "python")
import mojo_eigen as me  # noqa: E402

rng = np.random.default_rng(31)
raw = rng.normal(size=(20000, 4, 4))
spd = np.swapaxes(raw, -2, -1) @ raw + np.eye(4)
rhs = rng.normal(size=(20000, 4))
sym = (raw + np.swapaxes(raw, -2, -1)) * 0.5
a3 = rng.normal(size=(20000, 3, 3))

cases = {
    "eigh": lambda: me.eigh(sym),
    "svd": lambda: me.svd(a3),
    "ldlt": lambda: me.solve(spd, rhs, "ldlt"),
}
for name, call in cases.items():
    first = call()
    runs = [call(), call(), call()]
    identical = all(all(np.array_equal(a, b) for a, b in zip(first, r)) for r in runs)
    print(f"{name:5s} 20000-item batch bitwise-identical across 4 calls: {identical}")
