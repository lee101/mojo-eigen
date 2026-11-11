"""Side-by-side A/B timing of kernel variants in one process.

Every variant is compiled into its own shared library and loaded through
ctypes, then the variants are timed interleaved on the same buffers, so a run
compares them on the same machine and load. This is a development tool; the
numbers that go into the README come from `pixi run bench`.

    KERNEL=matmul python bench/microbench.py base=old.mojo new=src/eigen.mojo

Needs `MODULAR_HOME` set (`pixi run` does that).
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The parity tolerance the test suite uses for f64.
TOLERANCE = 2e-11

CASES = {
    "matmul": {"3x3 250k": (250_000, 3), "4x4 250k": (250_000, 4)},
    "matvec": {"3x3 150k": (150_000, 3), "4x4 150k": (150_000, 4)},
    "svd": {"3x3 20k": (20_000, 3), "4x4 20k": (20_000, 4)},
}


def build(source, output):
    subprocess.run(
        [
            "mojo", "build", "--emit", "shared-lib", os.path.abspath(source),
            "-o", output, "-Xlinker", "-lm",
        ],
        check=True,
        cwd=ROOT,
    )
    return ctypes.CDLL(output)


def ptr(array):
    return array.ctypes.data


class Case:
    """One kernel, one size: the buffers, the call arguments, and a checker."""

    def __init__(self, kernel, batch, n):
        rng = np.random.default_rng(1 + n)
        self.kernel = kernel
        self.batch = batch
        self.n = n
        # Every buffer stays referenced by self.buffers for the whole run:
        # a freed array would leave the recorded address dangling.
        self.buffers = {"a": rng.normal(size=(batch, n, n))}
        b = self.buffers
        if kernel == "matmul":
            b["b"] = rng.normal(size=(batch, n, n))
            b["dst"] = np.empty((batch, n, n))
        elif kernel == "matvec":
            b["x"] = rng.normal(size=(batch, n))
            b["dst"] = np.empty((batch, n))
        elif kernel == "svd":
            b["values"] = np.empty((batch, n))
            b["u"] = np.empty((batch, n, n))
            b["v"] = np.empty((batch, n, n))
            b["work"] = np.empty((batch, n, n))
            b["rotations"] = np.empty((batch, 4))
            b["statuses"] = np.zeros(batch, dtype=np.int64)
        else:
            raise SystemExit(f"unknown KERNEL {kernel}")

    def call(self, symbol):
        b = self.buffers
        if self.kernel == "matmul":
            return symbol(ptr(b["a"]), ptr(b["b"]), ptr(b["dst"]),
                          self.n, self.batch)
        if self.kernel == "matvec":
            return symbol(ptr(b["a"]), ptr(b["x"]), ptr(b["dst"]),
                          self.n, self.batch)
        # Batched entry point: it consumes the work/rotation/status scratch.
        return symbol(
            ptr(b["a"]), ptr(b["values"]), ptr(b["u"]), ptr(b["v"]),
            ptr(b["work"]), ptr(b["rotations"]), ptr(b["statuses"]),
            self.n, 0, self.batch,
        )

    def check(self):
        b = self.buffers
        close = lambda x, y: np.allclose(  # noqa: E731
            x, y, rtol=TOLERANCE, atol=TOLERANCE
        )
        if self.kernel == "matmul":
            return close(b["dst"], b["a"] @ b["b"])
        if self.kernel == "matvec":
            return close(b["dst"], (b["a"] @ b["x"][..., None])[..., 0])
        if not b["statuses"].all():
            return False
        reconstructed = b["u"] @ (b["values"][..., None] * np.swapaxes(b["v"], -1, -2))
        if os.environ.get("SHOW_ERROR"):
            print("max abs err", np.abs(reconstructed - b["a"]).max(),
                  "max |a|", np.abs(b["a"]).max())
        return close(reconstructed, b["a"])

    def reset(self):
        for name, buffer in self.buffers.items():
            if name != "a":
                buffer[...] = 0.0
        if self.kernel == "svd":
            self.buffers["statuses"][...] = 0


def load(library, kernel):
    if kernel == "svd":
        name, count = "me_svd_batch_f64", 10
    else:
        name, count = f"me_{kernel}_f64", 5
    symbol = getattr(library, name)
    symbol.restype = None
    symbol.argtypes = [ctypes.c_int64] * count
    return symbol


def main():
    kernel = os.environ.get("KERNEL", "matmul")
    sources = sys.argv[1:]
    libraries = {}
    for spec in sources:
        label, source = spec.split("=", 1)
        libraries[label] = build(source, os.path.join("/tmp", f"ab-{label}.so"))
    labels = list(libraries)
    results = {}
    for name, (batch, n) in CASES[kernel].items():
        case = Case(kernel, batch, n)
        symbols = {
            label: load(library, kernel)
            for label, library in libraries.items()
        }
        samples = {label: [] for label in labels}
        for label in labels:
            case.reset()
            case.call(symbols[label])
            if not case.check():
                raise SystemExit(f"{name} {label}: result mismatch")
        for _ in range(9):
            for label in labels:
                start = time.perf_counter()
                case.call(symbols[label])
                samples[label].append(time.perf_counter() - start)
        for label in labels:
            results[(label, name)] = min(samples[label])
            print(f"{name:12s} {label:12s} {results[(label, name)] * 1e3:9.3f} ms")
        print()
    for name in CASES[kernel]:
        for label in labels[1:]:
            speedup = results[(labels[0], name)] / results[(label, name)]
            print(f"{name:12s} {label} vs {labels[0]}: {speedup:.3f}x")


if __name__ == "__main__":
    main()
