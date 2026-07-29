"""Parity against eigenpy, which binds Eigen's actual C++ implementation."""

from __future__ import annotations

import numpy as np
import pytest

import eigenpy
import mojo_eigen as me


DTYPES = [np.float64, np.float32]


def tolerance(dtype):
    return 2e-11 if dtype == np.float64 else 3e-4


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [3, 4])
def test_fixed_matmul_and_matvec(dtype, n):
    rng = np.random.default_rng(10 + n)
    a = rng.normal(size=(32, n, n)).astype(dtype)
    b = rng.normal(size=(32, n, n)).astype(dtype)
    x = rng.normal(size=(32, n)).astype(dtype)
    assert np.allclose(me.matmul(a, b), a @ b, rtol=tolerance(dtype), atol=tolerance(dtype))
    assert np.allclose(me.matvec(a, x), (a @ x[..., None])[..., 0], rtol=tolerance(dtype), atol=tolerance(dtype))
    assert me.matmul(a[0], b[0]).dtype == dtype
    assert me.matvec(a[0], x[0]).dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("method,reference", [
    ("lu", eigenpy.PartialPivLU),
    ("llt", eigenpy.LLT),
    ("ldlt", eigenpy.LDLT),
])
def test_dense_solve_against_eigenpy(dtype, n, method, reference):
    rng = np.random.default_rng(100 + n)
    for _ in range(20):
        raw = rng.normal(size=(n, n)).astype(dtype)
        a = raw.T @ raw + np.eye(n, dtype=dtype) * dtype(0.25)
        b = rng.normal(size=n).astype(dtype)
        actual = me.solve(a, b, method=method)
        expected = np.asarray(reference(a).solve(b), dtype=dtype)
        assert np.allclose(actual, expected, rtol=tolerance(dtype), atol=tolerance(dtype))
        assert np.allclose(a @ actual, b, rtol=tolerance(dtype), atol=tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_ldlt_diagonal_pivoting(dtype):
    a = np.diag(np.array([1e-8, 8.0, 0.25, 2.0], dtype=dtype))
    a[0, 1] = a[1, 0] = dtype(1e-5)
    b = np.array([1.0, -2.0, 0.5, 3.0], dtype=dtype)
    actual = me.solve(a, b, method="ldlt")
    expected = np.asarray(eigenpy.LDLT(a).solve(b), dtype=dtype)
    assert np.allclose(actual, expected, rtol=tolerance(dtype), atol=tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(3, 3), (8, 3), (12, 4)])
def test_householder_qr_against_eigenpy(dtype, shape):
    rng = np.random.default_rng(sum(shape))
    a = rng.normal(size=shape).astype(dtype)
    b = rng.normal(size=shape[0]).astype(dtype)
    actual, residuals = me.lstsq(a, b)
    expected = np.asarray(eigenpy.HouseholderQR(a).solve(b), dtype=dtype)
    assert np.allclose(actual, expected, rtol=tolerance(dtype), atol=tolerance(dtype))
    assert np.linalg.norm(a.T @ (a @ actual - b)) <= tolerance(dtype) * 10
    if shape[0] > shape[1]:
        assert residuals[0] == pytest.approx(float(np.sum((a @ actual - b) ** 2)))
    else:
        assert residuals.size == 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [3, 4])
def test_selfadjoint_eigensolver_against_eigenpy(dtype, n):
    rng = np.random.default_rng(200 + n)
    cases = [np.zeros((n, n), dtype=dtype), np.eye(n, dtype=dtype) * dtype(7)]
    for _ in range(30):
        raw = rng.normal(size=(n, n)).astype(dtype)
        cases.append((raw + raw.T) * dtype(0.5))
    repeated = np.diag(np.array(([1, 1, 3] if n == 3 else [1, 1, 3, 3]), dtype=dtype))
    cases.append(repeated)
    for a in cases:
        solver = eigenpy.SelfAdjointEigenSolver(a)
        expected = np.asarray(solver.eigenvalues(), dtype=dtype)
        values, vectors = me.eigh(a)
        assert np.allclose(values, expected, rtol=tolerance(dtype), atol=tolerance(dtype))
        assert np.allclose(a @ vectors, vectors * values, rtol=tolerance(dtype), atol=tolerance(dtype))
        assert np.allclose(vectors.T @ vectors, np.eye(n), rtol=tolerance(dtype), atol=tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [3, 4])
def test_selfadjoint_uses_lower_triangle_like_eigen(dtype, n):
    rng = np.random.default_rng(300 + n)
    lower = np.tril(rng.normal(size=(n, n))).astype(dtype)
    noisy_upper = lower.copy()
    noisy_upper[np.triu_indices(n, 1)] = dtype(1e6)
    symmetric = lower + np.tril(lower, -1).T
    values, vectors = me.eigh(noisy_upper)
    expected = np.asarray(eigenpy.SelfAdjointEigenSolver(noisy_upper).eigenvalues(), dtype=dtype)
    assert np.allclose(values, expected, rtol=tolerance(dtype), atol=tolerance(dtype))
    assert np.allclose(symmetric @ vectors, vectors * values, rtol=tolerance(dtype), atol=tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [3, 4])
def test_jacobi_svd_against_eigenpy(dtype, n):
    rng = np.random.default_rng(400 + n)
    cases = [
        np.zeros((n, n), dtype=dtype),
        np.eye(n, dtype=dtype),
        np.diag(np.arange(n, 0, -1, dtype=dtype)),
    ]
    rank_deficient = rng.normal(size=(n, n)).astype(dtype)
    rank_deficient[-1] = rank_deficient[0] + rank_deficient[1]
    cases.append(rank_deficient)
    cases.extend(rng.normal(size=(n, n)).astype(dtype) for _ in range(30))
    for a in cases:
        reference = eigenpy.HhJacobiSVD(a, 20)
        u, values, vh = me.svd(a)
        expected = np.asarray(reference.singularValues(), dtype=dtype)
        assert np.allclose(values, expected, rtol=tolerance(dtype) * 3, atol=tolerance(dtype))
        assert np.allclose(u @ np.diag(values) @ vh, a, rtol=tolerance(dtype) * 3, atol=tolerance(dtype))
        assert np.allclose(u.T @ u, np.eye(n), rtol=tolerance(dtype) * 3, atol=tolerance(dtype))
        assert np.allclose(vh @ vh.T, np.eye(n), rtol=tolerance(dtype) * 3, atol=tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("batch", [7, 512])
def test_batched_decompositions_across_parallel_threshold(dtype, n, batch):
    rng = np.random.default_rng(5000 + batch + n)
    raw = rng.normal(size=(batch, n, n)).astype(dtype)
    spd = np.swapaxes(raw, -2, -1) @ raw + np.eye(n, dtype=dtype) * dtype(0.25)
    rhs = rng.normal(size=(batch, n)).astype(dtype)

    solution = me.solve(spd, rhs, method="ldlt")
    assert np.allclose(spd @ solution[..., None], rhs[..., None],
                       rtol=tolerance(dtype), atol=tolerance(dtype))

    symmetric = (raw + np.swapaxes(raw, -2, -1)) * dtype(0.5)
    values, vectors = me.eigh(symmetric)
    assert np.allclose(symmetric @ vectors, vectors * values[..., None, :],
                       rtol=tolerance(dtype), atol=tolerance(dtype))
    assert np.allclose(np.swapaxes(vectors, -2, -1) @ vectors, np.eye(n),
                       rtol=tolerance(dtype), atol=tolerance(dtype))

    u, singular_values, vh = me.svd(raw)
    reconstructed = (u * singular_values[..., None, :]) @ vh
    assert np.allclose(reconstructed, raw, rtol=tolerance(dtype) * 3,
                       atol=tolerance(dtype))
    assert np.shares_memory(vh, vh.base)


def test_batched_ldlt_reports_any_failed_factorization():
    matrices = np.repeat(np.eye(3)[None, ...], 7, axis=0)
    matrices[4, 1, 1] = 0
    rhs = np.ones((7, 3))
    with pytest.raises(np.linalg.LinAlgError):
        me.solve(matrices, rhs, method="ldlt")


def test_optional_gpu_decompositions_match_cpu_or_fall_back():
    rng = np.random.default_rng(9000)
    raw = rng.normal(size=(37, 4, 4))
    symmetric = (raw + np.swapaxes(raw, -2, -1)) * 0.5

    cpu_values, _ = me.eigh(symmetric)
    gpu_values, gpu_vectors = me.eigh(symmetric, device="gpu")
    assert np.allclose(gpu_values, cpu_values, rtol=tolerance(np.float64),
                       atol=tolerance(np.float64))
    assert np.allclose(symmetric @ gpu_vectors,
                       gpu_vectors * gpu_values[..., None, :],
                       rtol=tolerance(np.float64), atol=tolerance(np.float64))

    _, cpu_singular_values, _ = me.svd(raw)
    gpu_u, gpu_singular_values, gpu_vh = me.svd(raw, device="gpu")
    assert np.allclose(gpu_singular_values, cpu_singular_values,
                       rtol=tolerance(np.float64) * 3,
                       atol=tolerance(np.float64))
    assert np.allclose(
        (gpu_u * gpu_singular_values[..., None, :]) @ gpu_vh,
        raw, rtol=tolerance(np.float64) * 3, atol=tolerance(np.float64),
    )


@pytest.mark.parametrize("function", [me.eigh, me.svd])
def test_decomposition_device_validation(function):
    with pytest.raises(ValueError):
        function(np.eye(3), device="cuda")


def test_factorization_failures_and_empty_system():
    singular = np.diag([1.0, 0.0, 2.0])
    with pytest.raises(np.linalg.LinAlgError):
        me.solve(singular, np.ones(3), method="lu")
    with pytest.raises(np.linalg.LinAlgError):
        me.solve(singular, np.ones(3), method="ldlt")
    with pytest.raises(np.linalg.LinAlgError):
        me.solve(np.diag([1.0, -1.0, 2.0]), np.ones(3), method="llt")
    with pytest.raises(np.linalg.LinAlgError):
        me.lstsq(np.zeros((2, 2)), np.ones(2))
    assert me.solve(np.empty((0, 0)), np.empty(0), method="lu").size == 0
    assert me.lstsq(np.empty((0, 0)), np.empty(0))[0].size == 0


def test_shape_errors():
    with pytest.raises(ValueError):
        me.matmul(np.eye(2), np.eye(2))
    with pytest.raises(ValueError):
        me.matvec(np.eye(3), np.ones(4))
    with pytest.raises(ValueError):
        me.solve(np.ones((3, 2)), np.ones(3))
    with pytest.raises(ValueError):
        me.lstsq(np.ones((2, 3)), np.ones(2))


@pytest.mark.parametrize(
    "function,args",
    [
        (me.matmul, (np.eye(3, dtype=np.float32), np.eye(3))),
        (me.matvec, (np.eye(3, dtype=np.float32), np.ones(3))),
        (me.solve, (np.eye(3, dtype=np.float32), np.ones(3))),
        (me.lstsq, (np.eye(3, dtype=np.float32), np.ones(3))),
    ],
)
def test_mixed_dtypes_are_rejected_instead_of_narrowed(function, args):
    with pytest.raises(TypeError, match="same dtype"):
        function(*args)


@pytest.mark.parametrize("dtype", [np.int64, np.float16, np.complex128])
def test_unsupported_dtypes_are_rejected(dtype):
    with pytest.raises(TypeError, match="float32 or float64"):
        me.eigh(np.eye(3, dtype=dtype))


def test_empty_batches_do_not_enter_ffi():
    empty3 = np.empty((0, 3, 3))
    assert me.matmul(empty3, empty3).shape == empty3.shape
    assert me.matvec(empty3, np.empty((0, 3))).shape == (0, 3)
    assert me.solve(empty3, np.empty((0, 3)), method="ldlt").shape == (0, 3)
    assert me.eigh(empty3)[0].shape == (0, 3)
    assert me.svd(empty3)[1].shape == (0, 3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [3, 4])
def test_fixed_inverse(dtype, n):
    rng = np.random.default_rng(600 + n)
    a = rng.normal(size=(n, n)).astype(dtype) + np.eye(n, dtype=dtype)
    actual = me.inv(a)
    expected = eigenpy.PartialPivLU(a).inverse()
    assert np.allclose(actual, expected, rtol=tolerance(dtype), atol=tolerance(dtype))
    assert np.allclose(a @ actual, np.eye(n), rtol=tolerance(dtype), atol=tolerance(dtype))
