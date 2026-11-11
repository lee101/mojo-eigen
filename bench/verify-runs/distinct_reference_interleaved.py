"""Comparable-work reference: 20k distinct factorizations on both sides.

Every call runs in one process and the two sides are interleaved across pairs
of rounds, so both see the same machine state. (Timing a side alone in a fresh
process flatters whichever side gets the higher turbo clock.)
"""
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, "python")
import mojo_eigen as me  # noqa: E402


def best(fn, r=3):
    b = float("inf")
    for _ in range(r):
        t = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t)
    return b * 1e3


def main():
    for attempt in range(3):
        rng = np.random.default_rng(99 + attempt)
        raw = rng.normal(size=(20000, 4, 4))
        spd = np.swapaxes(raw, -2, -1) @ raw + np.eye(4)
        rhs = rng.normal(size=(20000, 4))
        sym = (raw + np.swapaxes(raw, -2, -1)) * 0.5
        a3 = rng.normal(size=(20000, 3, 3))
        calls = {
            "ldlt": (lambda: me.solve(spd, rhs, "ldlt"), lambda: np.linalg.solve(spd, rhs[..., None])),
            "eigh": (lambda: me.eigh(sym), lambda: np.linalg.eigh(sym)),
            "svd": (lambda: me.svd(a3), lambda: np.linalg.svd(a3)),
        }
        for name, (ours, ref) in calls.items():
            ours(); ref()
            ours_samples, ref_samples = [], []
            for _ in range(3):
                ours_samples.append(best(ours))
                ref_samples.append(best(ref))
            print(f"attempt {attempt} {name:5s}: mojo med {statistics.median(ours_samples):7.2f} ms"
                  f" ({min(ours_samples):.2f}-{max(ours_samples):.2f}), "
                  f"numpy med {statistics.median(ref_samples):8.2f} ms "
                  f"({min(ref_samples):.2f}-{max(ref_samples):.2f}), "
                  f"numpy/mojo {statistics.median(ref_samples) / statistics.median(ours_samples):6.2f}x",
                  flush=True)


if __name__ == "__main__":
    main()
