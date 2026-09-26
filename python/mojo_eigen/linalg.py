"""Small dense linear algebra and sparse conjugate gradient."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import operator
import subprocess

import numpy as np

from ._lib import address, lib


_GPU_HEADROOM = None

_BATCH_PARALLEL_MIN = 512
_ITEMS_PER_WORKER = 256
_MAX_BATCH_WORKERS = 8


def _batch_chunks(batch):
    """Split an independent-item batch across worker threads.

    Each item is a self-contained 3x3 or 4x4 dense factorization running out
    of L1, so the batch is compute-bound and threads scale it. ctypes drops
    the GIL for the foreign call, so this is real fan-out.
    """
    if batch < _BATCH_PARALLEL_MIN:
        return 1
    per_worker = -(-batch // _ITEMS_PER_WORKER)
    return max(1, min(_MAX_BATCH_WORKERS, per_worker))


def _run_batch(call, batch):
    chunks = _batch_chunks(batch)
    if chunks == 1:
        call(0, batch)
        return
    bounds = [i * batch // chunks for i in range(chunks + 1)]
    with ThreadPoolExecutor(max_workers=chunks) as pool:
        list(pool.map(lambda c: call(bounds[c], bounds[c + 1]), range(chunks)))


def _array(value, *, ndim=None):
    original = np.asarray(value)
    if original.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("arrays must have dtype float32 or float64")
    array = np.ascontiguousarray(original)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"expected a {ndim}-dimensional array")
    return array


def _suffix(array):
    return "f32" if array.dtype == np.float32 else "f64"


def _check_small_square(array):
    if array.shape[-2:] not in ((3, 3), (4, 4)):
        raise ValueError("matrix dimensions must be 3x3 or 4x4")


def _same_dtype(*arrays):
    if any(array.dtype != arrays[0].dtype for array in arrays[1:]):
        raise TypeError("all floating-point arrays must have the same dtype")


def _int64_array(value, name):
    original = np.asarray(value)
    if original.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")
    if original.size:
        if np.any(original < 0) or np.any(original > np.iinfo(np.int64).max):
            raise ValueError(f"{name} values must fit in non-negative int64")
    return np.ascontiguousarray(original, dtype=np.int64)


def _gpu_has_headroom():
    global _GPU_HEADROOM
    if _GPU_HEADROOM is None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi", "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=2, check=True,
            )
            _GPU_HEADROOM = min(
                int(line.strip()) for line in result.stdout.splitlines() if line.strip()
            ) >= 4000
        except (OSError, subprocess.SubprocessError, ValueError):
            _GPU_HEADROOM = False
    return _GPU_HEADROOM


def _within_gpu_limit(elements):
    return elements * 8 <= 2_000_000_000


def matmul(a, b):
    a = _array(a)
    b = _array(b)
    if a.shape != b.shape or a.ndim not in (2, 3):
        raise ValueError("a and b must have identical (3,3), (4,4), or batched shapes")
    _check_small_square(a)
    _same_dtype(a, b)
    result = np.empty_like(a)
    n = a.shape[-1]
    batch = 1 if a.ndim == 2 else a.shape[0]
    if batch == 0:
        return result
    getattr(lib(), f"me_matmul_{_suffix(a)}")(
        address(a), address(b), address(result), n, batch
    )
    return result


def matvec(a, x):
    a = _array(a)
    x = _array(x)
    _check_small_square(a)
    n = a.shape[-1]
    if a.ndim == 2:
        if x.shape != (n,):
            raise ValueError("vector shape does not match matrix")
        batch = 1
    elif a.ndim == 3:
        if x.shape != (a.shape[0], n):
            raise ValueError("batched vector shape does not match matrices")
        batch = a.shape[0]
    else:
        raise ValueError("a must be a matrix or batch of matrices")
    _same_dtype(a, x)
    result = np.empty_like(x)
    if batch == 0:
        return result
    getattr(lib(), f"me_matvec_{_suffix(a)}")(
        address(a), address(x), address(result), n, batch
    )
    return result


def solve(a, b, method="lu"):
    a = _array(a)
    b = _array(b)
    if a.ndim not in (2, 3) or b.ndim != a.ndim - 1:
        raise ValueError("solve requires a square matrix and matching vector")
    if a.shape[-2] != a.shape[-1] or b.shape != a.shape[:-1]:
        raise ValueError("solve requires a square matrix and matching vector")
    _same_dtype(a, b)
    n = a.shape[-1]
    if n == 0:
        if a.ndim == 3 and method != "ldlt":
            raise ValueError("batched solve currently supports method='ldlt'")
        if method not in ("lu", "llt", "ldlt"):
            raise ValueError("method must be 'lu', 'llt', or 'ldlt'")
        return np.empty_like(b)
    x = np.empty_like(b)
    work = np.empty_like(a)
    suffix = _suffix(a)
    if a.ndim == 3:
        if method != "ldlt":
            raise ValueError("batched solve currently supports method='ldlt'")
        batch = a.shape[0]
        if batch == 0:
            return x
        temporary = np.empty_like(b)
        pivots = np.empty_like(b, dtype=np.int64)
        statuses = np.empty(batch, dtype=np.int64)
        batch_ldlt = getattr(lib(), f"me_ldlt_batch_{suffix}")
        _run_batch(
            lambda lo, hi: batch_ldlt(
                address(a), address(b), address(x), address(work),
                address(temporary), address(pivots), address(statuses), n,
                lo, hi,
            ),
            batch,
        )
        if not np.all(statuses):
            raise np.linalg.LinAlgError("LDLT factorization failed")
        return x
    if method == "lu":
        pivots = np.empty(n, dtype=np.int64)
        ok = getattr(lib(), f"me_lu_{suffix}")(
            address(a), address(b), address(x), address(work), address(pivots), n
        )
    elif method == "llt":
        ok = getattr(lib(), f"me_llt_{suffix}")(
            address(a), address(b), address(x), address(work), n
        )
    elif method == "ldlt":
        temporary = np.empty(n, dtype=a.dtype)
        pivots = np.empty(n, dtype=np.int64)
        ok = getattr(lib(), f"me_ldlt_{suffix}")(
            address(a), address(b), address(x), address(work),
            address(temporary), address(pivots), n,
        )
    else:
        raise ValueError("method must be 'lu', 'llt', or 'ldlt'")
    if not ok:
        raise np.linalg.LinAlgError(f"{method.upper()} factorization failed")
    return x


def inv(a, method="lu"):
    a = _array(a, ndim=2)
    if a.shape not in ((3, 3), (4, 4)):
        raise ValueError("inverse requires a 3x3 or 4x4 matrix")
    result = np.empty_like(a)
    identity = np.eye(a.shape[0], dtype=a.dtype)
    for column in range(a.shape[0]):
        result[:, column] = solve(a, identity[:, column], method=method)
    return result


def lstsq(a, b):
    a = _array(a, ndim=2)
    b = _array(b, ndim=1)
    m, n = a.shape
    if m < n or b.shape != (m,):
        raise ValueError("Householder QR requires m >= n and a matching vector")
    _same_dtype(a, b)
    if n == 0:
        residuals = (
            np.array([np.sum(b * b)], dtype=a.dtype)
            if m > 0
            else np.empty(0, dtype=a.dtype)
        )
        return np.empty(0, dtype=a.dtype), residuals
    x = np.empty(n, dtype=a.dtype)
    work = np.empty_like(a)
    transformed = np.empty(m, dtype=a.dtype)
    ok = getattr(lib(), f"me_qr_{_suffix(a)}")(
        address(a), address(b), address(x), address(work),
        address(transformed), m, n,
    )
    if not ok:
        raise np.linalg.LinAlgError("Householder QR found an exact zero pivot")
    residuals = np.array([np.sum((a @ x - b) ** 2)], dtype=a.dtype) if m > n else np.empty(0, dtype=a.dtype)
    return x, residuals


def eigh(a, *, device="cpu"):
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    a = _array(a)
    if a.ndim not in (2, 3):
        raise ValueError("eigh requires a matrix or batch of matrices")
    _check_small_square(a)
    n = a.shape[-1]
    values = np.empty(a.shape[:-1], dtype=a.dtype)
    vectors = np.empty_like(a)
    workspace_size = 28 if n == 4 else 18
    batch = 1 if a.ndim == 2 else a.shape[0]
    if batch == 0:
        return values, vectors
    gpu_elements = batch * (2 * n * n + n + workspace_size + 1)
    if (
        device == "gpu" and n == 4 and batch and a.dtype == np.float64
        and _within_gpu_limit(gpu_elements) and _gpu_has_headroom()
    ):
        statuses = np.empty(batch, dtype=np.int64)
        used_gpu = lib().me_eigh_batch_gpu_f64(
            address(a), address(values), address(vectors), address(statuses), n, batch
        )
        if used_gpu:
            if not np.all(statuses):
                raise np.linalg.LinAlgError(
                    "self-adjoint eigensolver did not converge"
                )
            return values, vectors
    work_shape = (workspace_size,) if a.ndim == 2 else (a.shape[0], workspace_size)
    work = np.empty(work_shape, dtype=a.dtype)
    if a.ndim == 3:
        statuses = np.empty(batch, dtype=np.int64)
        batch_eigh = getattr(lib(), f"me_eigh_batch_{_suffix(a)}")
        _run_batch(
            lambda lo, hi: batch_eigh(
                address(a), address(values), address(vectors), address(work),
                address(statuses), n, lo, hi,
            ),
            batch,
        )
        if not np.all(statuses):
            raise np.linalg.LinAlgError("self-adjoint eigensolver did not converge")
        return values, vectors
    ok = getattr(lib(), f"me_eigh_{_suffix(a)}")(
        address(a), address(values), address(vectors), address(work), n
    )
    if not ok:
        raise np.linalg.LinAlgError("self-adjoint eigensolver did not converge")
    return values, vectors


def svd(a, *, device="cpu"):
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    a = _array(a)
    if a.ndim not in (2, 3):
        raise ValueError("svd requires a matrix or batch of matrices")
    _check_small_square(a)
    n = a.shape[-1]
    values = np.empty(a.shape[:-1], dtype=a.dtype)
    u = np.empty_like(a)
    v = np.empty_like(a)
    batch = 1 if a.ndim == 2 else a.shape[0]
    if batch == 0:
        return u, values, np.swapaxes(v, -2, -1)
    gpu_elements = batch * (4 * n * n + n + 5)
    if (
        device == "gpu" and batch and a.dtype == np.float64
        and _within_gpu_limit(gpu_elements) and _gpu_has_headroom()
    ):
        statuses = np.empty(batch, dtype=np.int64)
        used_gpu = lib().me_svd_batch_gpu_f64(
            address(a), address(values), address(u), address(v),
            address(statuses), n, batch,
        )
        if used_gpu:
            if not np.all(statuses):
                raise np.linalg.LinAlgError("Jacobi SVD did not converge")
            return u, values, np.swapaxes(v, -2, -1)
    work = np.empty_like(a)
    rotations_shape = (4,) if a.ndim == 2 else (a.shape[0], 4)
    rotations = np.empty(rotations_shape, dtype=a.dtype)
    if a.ndim == 3:
        batch = a.shape[0]
        statuses = np.empty(batch, dtype=np.int64)
        batch_svd = getattr(lib(), f"me_svd_batch_{_suffix(a)}")
        _run_batch(
            lambda lo, hi: batch_svd(
                address(a), address(values), address(u), address(v),
                address(work), address(rotations), address(statuses), n,
                lo, hi,
            ),
            batch,
        )
        if not np.all(statuses):
            raise np.linalg.LinAlgError("Jacobi SVD did not converge")
        return u, values, np.swapaxes(v, -2, -1)
    ok = getattr(lib(), f"me_svd_{_suffix(a)}")(
        address(a), address(values), address(u), address(v),
        address(work), address(rotations), n,
    )
    if not ok:
        raise np.linalg.LinAlgError("Jacobi SVD did not converge")
    return u, values, v.T


@dataclass(frozen=True)
class CGResult:
    x: np.ndarray
    info: int
    iterations: int
    relative_error: float

    def __iter__(self):
        yield self.x
        yield self.info


def cg(values, indices, indptr, b, *, x0=None, tol=1e-10, max_iter=None):
    values = _array(values, ndim=1)
    indices = _int64_array(indices, "indices")
    indptr = _int64_array(indptr, "indptr")
    b = _array(b, ndim=1)
    _same_dtype(values, b)
    n = b.size
    if (
        indptr.ndim != 1
        or indptr.shape != (n + 1,)
        or indptr[0] != 0
        or indptr[-1] != values.size
        or np.any(indptr < 0)
        or np.any(indptr[1:] < indptr[:-1])
    ):
        raise ValueError("invalid CSR indptr")
    if indices.shape != values.shape or np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("invalid CSR column indices")
    if x0 is None:
        x = np.zeros(n, dtype=values.dtype)
    else:
        initial = _array(x0, ndim=1)
        _same_dtype(values, initial)
        x = initial.copy()
    if x.shape != (n,):
        raise ValueError("x0 shape does not match b")
    max_iter = n if max_iter is None else operator.index(max_iter)
    if max_iter > np.iinfo(np.int64).max:
        raise ValueError("max_iter must fit in int64")
    if max_iter < 0 or not math.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and non-negative; max_iter must be non-negative")
    if n == 0:
        return CGResult(x, 0, 0, 0.0)
    work = np.empty(5 * n, dtype=values.dtype)
    error = np.empty(1, dtype=values.dtype)
    iterations = getattr(lib(), f"me_cg_{_suffix(values)}")(
        address(values), address(indices), address(indptr), address(b),
        address(x), address(work), address(error), n, max_iter, tol,
    )
    relative_error = float(error[0])
    info = 0 if relative_error <= tol else max(1, iterations)
    return CGResult(x, info, int(iterations), relative_error)
