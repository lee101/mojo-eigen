"""Probe how close the current build gets to the test-suite tolerances.

For each kernel, compare against upstream eigenpy over many random inputs and
report the worst relative error as a multiple of the tolerance the tests use.
"""
import sys

import numpy as np

sys.path.insert(0, "python")
import eigenpy  # noqa: E402
import mojo_eigen as me  # noqa: E402

TOL = {np.float64: 2e-11, np.float32: 3e-4}


def rel(a, b, dtype):
    scale = max(np.abs(b).max(), 1.0)
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / scale / TOL[dtype])


def main():
    rng = np.random.default_rng(2026)
    worst = {}
    for dtype in (np.float64, np.float32):
        tol = TOL[dtype]
        for n in (3, 4):
            for _ in range(200):
                raw = rng.normal(size=(n, n)).astype(dtype)
                spd = raw.T @ raw + np.eye(n, dtype=dtype) * dtype(0.25)
                b = rng.normal(size=n).astype(dtype)
                for method, ref in (("lu", eigenpy.PartialPivLU), ("llt", eigenpy.LLT), ("ldlt", eigenpy.LDLT)):
                    e = rel(me.solve(spd, b, method=method), ref(spd).solve(b), dtype)
                    key = f"{np.dtype(dtype).name} solve {method} {n}x{n}"
                    worst[key] = max(worst.get(key, 0), e)
                sym = (raw + raw.T) * dtype(0.5)
                ref_eig = np.asarray(eigenpy.SelfAdjointEigenSolver(sym).eigenvalues(), dtype=dtype)
                key = f"{np.dtype(dtype).name} eigh {n}x{n}"
                worst[key] = max(worst.get(key, 0), rel(me.eigh(sym)[0], ref_eig, dtype))
                sv = np.asarray(eigenpy.HhJacobiSVD(raw, 20).singularValues(), dtype=dtype)
                key = f"{np.dtype(dtype).name} svd {n}x{n} (x3 tol)"
                worst[key] = max(worst.get(key, 0), rel(me.svd(raw)[1], sv, dtype) / 3)
    print("worst |mojo - eigenpy| as a multiple of the test tolerance")
    print("(values well below 1 are inside tolerance; the value shown is already")
    print(" divided by any multiplier the test applies, e.g. SVD x3)")
    for k in sorted(worst):
        print(f"{k:34s} {worst[k]:.3f}")


if __name__ == "__main__":
    main()
