# mojo-eigen

`mojo-eigen` is a focused Mojo port of selected
[Eigen](https://gitlab.com/libeigen/eigen) algorithms used by VCGLib's dense
geometry and sparse smoothing/parametrization code. VCGLib bundles Eigen under
`eigenlib/Eigen`. This package provides NumPy-in/NumPy-out Python functions
backed by one compiled Mojo shared library.

This is a derived work of Eigen at upstream revision `d53ac33`, distributed
under the [Mozilla Public License 2.0](LICENSE). The kernels were ported from
the real upstream headers, and each non-obvious kernel names its Eigen source
file and function in the Mojo source.

## Scope

The implemented subset is:

| area | operations |
| --- | --- |
| fixed dense | 3×3/4×4 matrix-matrix and matrix-vector products (including batches), inverse |
| direct solvers | partial-pivot LU, LLT, diagonally pivoted LDLT, including batched LDLT |
| least squares | unpivoted Householder QR for `m >= n` |
| eigensolver | batched self-adjoint 3×3 direct solve and 4×4 Householder tridiagonalization with implicit shifted QR |
| SVD | batched two-sided Jacobi SVD for square 3×3/4×4 matrices |
| sparse | diagonally preconditioned conjugate gradient over full CSR matrices |
| precision | concrete float32 and float64 kernels and C ABI exports; mixed or other dtypes are rejected |

This is deliberately not a general Eigen replacement. It does not implement
dynamic dense expressions, rectangular SVD, complex scalars, maps/strides,
sparse factorizations such as SimplicialLDLT, geometry classes, or Eigen's
compile-time expression system. The Python API uses row-major C-contiguous
buffers at the FFI boundary (non-contiguous inputs are copied). It is not ABI
compatible with Eigen. The self-adjoint eigensolver uses the lower triangle,
matching Eigen.

## Install

```bash
git clone https://github.com/lee101/mojo-eigen.git
cd mojo-eigen
pixi install
pixi run build
pixi run test
```

`pixi install` supplies the pinned Mojo nightly, Python, NumPy, SciPy, and
eigenpy. Linux x86-64 is the currently supported build platform. The test suite
uses eigenpy, a binding of the upstream C++ Eigen library, for dense parity.
eigenpy does not expose Eigen's ConjugateGradient class, so sparse tests compare
against SciPy on the same CSR systems and assert relative residual and
convergence invariants.

## Usage

```python
import numpy as np
import mojo_eigen as eigen

a = np.array([
    [4.0, 1.0, 0.0],
    [1.0, 3.0, 1.0],
    [0.0, 1.0, 2.0],
])
b = np.array([1.0, 2.0, 3.0])

x = eigen.solve(a, b, method="ldlt")
values, vectors = eigen.eigh(a)
u, singular_values, vh = eigen.svd(a)

assert np.allclose(a @ x, b)
assert np.allclose(a @ vectors, vectors * values)
assert np.allclose(u @ np.diag(singular_values) @ vh, a)
```

`solve(..., method="ldlt")`, `eigh`, and `svd` also accept C-contiguous
`(batch, n, n)` inputs; batched LDLT right-hand sides have shape `(batch, n)`.
Large batches are split across physical CPU cores, while batches smaller than
512 stay serial to avoid thread-launch overhead.

Float64 4×4 `eigh` and 3×3/4×4 SVD have an explicit optional GPU route:

```python
values, vectors = eigen.eigh(matrix_batch, device="gpu")
u, singular_values, vh = eigen.svd(matrix_batch, device="gpu")
```

CPU remains the default. The GPU route is attempted only with at least
4000 MiB of reported free device memory and a planned allocation below 2 GB;
an unavailable device, unsupported dtype/shape, or runtime failure falls back
to CPU. LDLT is CPU-only.

For an existing SciPy CSR matrix:

```python
result = eigen.cg(
    matrix.data, matrix.indices, matrix.indptr, b,
    tol=1e-10, max_iter=500,
)
x, info = result
```

`info == 0` means the Eigen-style relative residual tolerance was reached.
`result.iterations` and `result.relative_error` expose the solver diagnostics.

## Performance

Measured on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
6.8.0-147-generic x86-64, on 2026-11-11. Each mojo-eigen column is the median
of six sequential `pixi run bench` processes (that benchmark keeps the best of
three calls per process). Reference columns use NumPy for products,
eigenpy/Eigen for decompositions, and SciPy for CG.

That shared host ran at a load average of 65-90 during measurement; the Mojo
times above 2 ms spread by +-15% to +25% across the six runs. Reference medians
come from the same runs, which alternate with the Mojo side; the min-max range
shows the residual host noise, which is large (up to 3x for CG). For LDLT,
eigh and SVD the benchmark's reference column does less work than mojo-eigen,
so its ratio is a lower bound on the upstream side; the fourth column re-times
upstream on the same 20,000 factorizations.

| case | mojo-eigen (measured) | benchmark's reference | benchmark's ratio | same 20k inputs: numpy | upstream/Mojo |
| --- | ---: | ---: | ---: | ---: | ---: |
| batched 3×3 matmul (250k) | 7.02 ms | NumPy 87-102 ms | 12.85× | 87-102 ms | 12.85× |
| batched 4×4 matvec (150k) | 2.50 ms | NumPy 14.2-16.1 ms | 6.16× | 14.2-16.1 ms | 6.16× |
| batched 4×4 LDLT solve (20k) | 2.75 ms | eigenpy 17.9-24.0 ms, one factorization | 6.99× | numpy 11.8-12.5 ms | 6.00-6.75× |
| batched symmetric 4×4 eigh (20k) | 4.57 ms | eigenpy 59.5-80.4 ms, construction only | 15.07× | numpy 67.8-69.7 ms | 12.35-13.39× |
| batched Jacobi 3×3 SVD (20k) | 10.13 ms | eigenpy 57.2-74.3 ms, construction only | 6.79× | numpy 77.0-79.1 ms | 7.20-8.42× |
| CSR CG, 120×120 Poisson | 69.68 ms | SciPy 7.4-23.1 s | 144.16× | SciPy, same system | ~1× |

Ratios are computed from medians; reference ranges are the min and max of the
six runs. Every row is slower than upstream per operation (all ratios are
upstream-faster), so none is dropped.

What the benchmark actually times, and where it is generous to mojo-eigen:

- matmul and matvec feed identical arrays to both sides, and both sides do the
  same arithmetic. These two rows are like-for-like.
- LDLT: mojo-eigen factorizes 20,000 different matrices. The eigenpy column
  factorizes one matrix and re-solves a different right-hand side 20,000 times.
- eigh and SVD: mojo-eigen runs 20,000 decompositions. The eigenpy column only
  constructs 20,000 solvers and never requests eigenvalues or singular values;
  the calls are lazy.
- The fourth column re-times upstream on the same 20,000 inputs, interleaved
  with mojo-eigen in one process (eigenpy cannot swap its wrapped matrix, so
  that side is numpy). Timing a side alone in a fresh process changes the CPU
  clock: numpy's LDLT measured 22.2 ms alone versus 11.8-12.5 ms interleaved.
- CG: both sides solve the same 14,400-unknown CSR system to `rtol=1e-8`; the
  benchmark's own inputs take 363 iterations here and the two solutions agree to
  8.2e-14. For CG the 144× is a genuine mojo-faster figure (both sides converge
  the same system in the same iteration count); what it does not show is a
  performance gap, because the SciPy time swings by 3x with host load. Time the
  two back to back and they are within noise of each other.

`bench/verify.md` records the runs behind these numbers.

GPU rows are not in the table because `pixi run bench` does not run them
unless `MOJO_EIGEN_BENCH_GPU=1` is set, and on this host's pinned
max/CUDA stack the in-process GPU benchmark crashes: a second device kernel in
one process faults in `cuMemcpyDtoHAsync` inside `libcuda.so.1`. Run one kernel
per process they are exact (eigh max error 1.6e-15, SVD 6.0e-15) and on their
own side they reproduce; the GPU numbers do not (the GPU route is
latency-bound and tracks host load, while the CPU route gets cheaper when the
machine is quiet):

| case | mojo-eigen GPU | Mojo CPU | GPU speedup |
| --- | ---: | ---: | ---: |
| batched symmetric 4×4 eigh (20k) | 1.12-5.01 ms | 4.95-6.28 ms | 1.25-4.43× |
| batched Jacobi 3×3 SVD (20k) | 1.22-1.57 ms | 7.91-7.99 ms | 5.09-6.47× |

The decomposition batches remove repeated Python scratch allocation and FFI
crossings, then parallelize independent matrices above the threshold. Timings
are machine- and load-dependent; run `pixi run bench` for local results.

matmul and matvec now combine each output element's products as a balanced
tree instead of one running sum, so results can differ from the previous
revision in the last bits. The test tolerances are unchanged. The speedup the
source comments claim for that change (~1.6x for 3×3, ~1.3x for 4×4) did not
reproduce when the two revisions' libraries were timed against each other on
this host: matmul ratios 1.01 and 0.96, matvec 0.84 and 1.13, all within
run-to-run noise.

## How it works

All numerical code lives in one Mojo compilation unit. Templates over scalar
type became internal `DType` parameters, while the ABI exposes concrete
`f32`/`f64` symbols because exported Mojo functions cannot be parametric.

Python owns every CPU input, output, and scratch buffer. C-contiguous NumPy
arrays cross ctypes as integer addresses without an input copy; other layouts
are copied to temporary contiguous arrays whose lifetimes cover the synchronous
call. The concrete export reconstructs an
`UnsafePointer[..., AnyOrigin[mut=True]]`. CPU Mojo code performs no heap
allocation, so there are no cross-language ownership or deallocation rules.
Dense arrays are flat row-major buffers; validated CSR inputs are converted to
int64 `indices` and `indptr` before entering Mojo. Optional GPU calls
stage these buffers only for the duration of the call and release device
buffers on return.

The tests include random parity, zero/repeated/rank-deficient matrices,
non-positive and singular factorization failures, ignored upper triangles,
empty systems, zero sparse right-hand sides, initial guesses, invalid CSR, and
forced non-convergence.
