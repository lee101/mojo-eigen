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

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic x86-64. These are real best-of-three wall times from
the benchmark in this repository, run on 2026-07-29. Reference columns use
NumPy for products, eigenpy/Eigen for decompositions, and SciPy for CG.

| case | mojo-eigen | reference | reference / Mojo |
| --- | ---: | ---: | ---: |
| batched 3×3 matmul (250k) | 7.81 ms | NumPy 88.48 ms | 11.32× faster |
| batched 4×4 matvec (150k) | 2.87 ms | NumPy 14.52 ms | 5.06× faster |
| batched 4×4 LDLT solve (20k) | 5.78 ms | eigenpy 18.41 ms | 3.19× faster |
| batched symmetric 4×4 eigh (20k) | 8.45 ms | eigenpy 67.95 ms | 8.04× faster |
| batched Jacobi 3×3 SVD (20k) | 9.80 ms | eigenpy 78.35 ms | 8.00× faster |
| CSR CG, 120×120 Poisson | 111.86 ms | SciPy 17,373.98 ms | 155.32× faster |

The decomposition batches remove repeated Python scratch allocation and FFI
crossings, then parallelize independent matrices above the threshold. Timings
are machine- and load-dependent; run `pixi run bench` for local results.

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
