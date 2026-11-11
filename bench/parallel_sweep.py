"""Batch-size and worker-count sweep for the batched kernels.

Development tool behind the threshold and worker cap recorded in the README.
It times the public Python entry points (which is what callers actually pay:
ctypes crossing plus the Mojo kernel) at a range of batch sizes, and for the
thread-count rows it pins the fan-out width through
``mojo_eigen.linalg._MAX_BATCH_WORKERS`` so the same build is measured at 1, 2,
4, ... workers.

    pixi run python bench/parallel_sweep.py [--repeat N] [--kernels ...]

Numbers are best-of-N wall times per call. Run it on an idle machine; the host
is shared, so treat differences under ~5% as noise.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojo_eigen as me  # noqa: E402
from mojo_eigen import linalg  # noqa: E402


BATCHES = [1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 20000]
WORKERS = [1, 2, 3, 4, 6, 8]


def best_of(function, repeat):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def make_cases(batch, rng, dtype):
    raw = rng.normal(size=(batch, 4, 4)).astype(dtype)
    spd = np.swapaxes(raw, -2, -1) @ raw + np.eye(4, dtype=dtype) * dtype(0.25)
    rhs = rng.normal(size=(batch, 4)).astype(dtype)
    symmetric = ((raw + np.swapaxes(raw, -2, -1)) * dtype(0.5)).astype(dtype)
    return {
        "ldlt4": lambda: me.solve(spd, rhs, method="ldlt"),
        "eigh4": lambda: me.eigh(symmetric),
        "svd4": lambda: me.svd(raw),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--kernels", default="ldlt4,eigh4,svd4")
    parser.add_argument("--dtype", default="float64")
    args = parser.parse_args()
    kernels = args.kernels.split(",")
    dtype = np.dtype(args.dtype).type
    rng = np.random.default_rng(11)

    print(f"repeat={args.repeat} dtype={dtype}")
    for kernel in kernels:
        print(f"\n## {kernel}: batch scaling at the current settings")
        print("| batch | ms |")
        print("| ---: | ---: |")
        for batch in BATCHES:
            call = make_cases(batch, rng, dtype)[kernel]
            call()
            print(f"| {batch} | {best_of(call, args.repeat) * 1e3:.3f} |")

        print(f"\n## {kernel}: worker sweep at fixed batches")
        header = "| batch | " + " | ".join(f"{w}w" for w in WORKERS) + " |"
        print(header)
        print("| ---: | " + " | ".join("---:" for _ in WORKERS) + " |")
        for batch in [512, 20000]:
            call = make_cases(batch, rng, dtype)[kernel]
            times = []
            for workers in WORKERS:
                linalg._MAX_BATCH_WORKERS = workers
                call()
                times.append(best_of(call, args.repeat) * 1e3)
            linalg._MAX_BATCH_WORKERS = 8
            print(f"| {batch} | " + " | ".join(f"{t:.3f}" for t in times) + " |")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
