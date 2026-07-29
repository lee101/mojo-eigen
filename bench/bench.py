"""Benchmarks against Eigen (eigenpy), NumPy, and SciPy."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time

import eigenpy
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojo_eigen as me  # noqa: E402


def timeit(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def poisson(side):
    one = np.ones(side)
    line = sparse.diags([-one, 4 * one, -one], [-1, 0, 1], shape=(side, side))
    identity = sparse.eye(side)
    coupling = sparse.diags([-one, -one], [-1, 1], shape=(side, side))
    return sparse.csr_matrix(sparse.kron(identity, line) + sparse.kron(coupling, identity))


def cpu_name():
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def gpu_free_mib():
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=2, check=True,
        )
        return min(
            int(line.strip()) for line in result.stdout.splitlines() if line.strip()
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def main():
    rng = np.random.default_rng(7)
    cases = []
    gpu_skip = None

    a3 = rng.normal(size=(250_000, 3, 3))
    b3 = rng.normal(size=(250_000, 3, 3))
    cases.append(("batched 3x3 matmul (250k)", lambda: me.matmul(a3, b3), lambda: a3 @ b3))

    a4 = rng.normal(size=(150_000, 4, 4))
    x4 = rng.normal(size=(150_000, 4))
    cases.append(
        (
            "batched 4x4 matvec (150k)",
            lambda: me.matvec(a4, x4),
            lambda: (a4 @ x4[..., None])[..., 0],
        )
    )

    spd = rng.normal(size=(4, 4))
    spd = spd.T @ spd + np.eye(4)
    rhs = rng.normal(size=4)
    spd_batch = np.repeat(spd[None, ...], 20_000, axis=0)
    rhs_batch = np.repeat(rhs[None, ...], 20_000, axis=0)
    eigen_ldlt = eigenpy.LDLT(spd)
    cases.append(
        (
            "batched 4x4 LDLT solve (20k)",
            lambda: me.solve(spd_batch, rhs_batch, "ldlt"),
            lambda: [eigen_ldlt.solve(rhs) for _ in range(20_000)],
        )
    )

    symmetric = (rng.normal(size=(4, 4)))
    symmetric = (symmetric + symmetric.T) * 0.5
    symmetric_batch = np.repeat(symmetric[None, ...], 20_000, axis=0)
    cases.append(
        (
            "batched symmetric 4x4 eigh (20k)",
            lambda: me.eigh(symmetric_batch),
            lambda: [eigenpy.SelfAdjointEigenSolver(symmetric) for _ in range(20_000)],
        )
    )

    dense = rng.normal(size=(3, 3))
    dense_batch = np.repeat(dense[None, ...], 20_000, axis=0)
    cases.append(
        (
            "batched Jacobi 3x3 SVD (20k)",
            lambda: me.svd(dense_batch),
            lambda: [eigenpy.HhJacobiSVD(dense, 20) for _ in range(20_000)],
        )
    )
    if os.environ.get("MOJO_EIGEN_BENCH_GPU") == "1":
        free_mib = gpu_free_mib()
        if free_mib is not None and free_mib >= 4000:
            cases.append(
                (
                    "batched symmetric 4x4 eigh GPU vs CPU (20k)",
                    lambda: me.eigh(symmetric_batch, device="gpu"),
                    lambda: me.eigh(symmetric_batch),
                )
            )
            cases.append(
                (
                    "batched Jacobi 3x3 SVD GPU vs CPU (20k)",
                    lambda: me.svd(dense_batch, device="gpu"),
                    lambda: me.svd(dense_batch),
                )
            )
        else:
            available = "unavailable" if free_mib is None else f"{free_mib} MiB free"
            gpu_skip = f"GPU benchmarks skipped: {available}; 4000 MiB required."

    system = poisson(120)
    sparse_rhs = rng.normal(size=system.shape[0])
    cases.append(
        (
            "CSR CG, 120x120 Poisson",
            lambda: me.cg(
                system.data, system.indices, system.indptr, sparse_rhs,
                tol=1e-8, max_iter=500,
            ),
            lambda: spla.cg(system, sparse_rhs, rtol=1e-8, atol=0, maxiter=500),
        )
    )

    print(f"Machine: {cpu_name()}; {platform.platform()}")
    if gpu_skip:
        print(gpu_skip)
    print()
    print("| case | mojo-eigen | reference | reference / Mojo |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, reference in cases:
        ours()
        reference()
        a = timeit(ours, repeat=3)
        b = timeit(reference, repeat=3)
        marker = "faster" if a < b else "slower"
        print(
            f"| {name} | {a * 1e3:.2f} ms | {b * 1e3:.2f} ms | "
            f"{b / a:.2f}x {marker} |"
        )


if __name__ == "__main__":
    main()
