"""Sparse CG parity and convergence invariants."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as spla

import mojo_eigen as me


def poisson_2d(side, dtype=np.float64):
    one = np.ones(side, dtype=dtype)
    line = sparse.diags([-one, 4 * one, -one], [-1, 0, 1], shape=(side, side))
    identity = sparse.eye(side, dtype=dtype)
    coupling = sparse.diags([-one, -one], [-1, 1], shape=(side, side))
    return sparse.csr_matrix(sparse.kron(identity, line) + sparse.kron(coupling, identity))


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-10), (np.float32, 2e-5)])
def test_cg_against_scipy(dtype, tol):
    a = poisson_2d(12, dtype)
    rng = np.random.default_rng(50)
    b = rng.normal(size=a.shape[0]).astype(dtype)
    result = me.cg(a.data, a.indices, a.indptr, b, tol=tol, max_iter=500)
    expected, info = spla.cg(a, b, rtol=tol, atol=0, maxiter=500)
    assert info == 0
    assert result.info == 0
    assert result.iterations > 0
    assert result.relative_error <= tol
    assert np.linalg.norm(a @ result.x - b) / np.linalg.norm(b) <= tol * 1.2
    assert np.allclose(result.x, expected, rtol=tol * 10, atol=tol * 10)


def test_cg_initial_guess_zero_rhs_and_nonconvergence():
    a = poisson_2d(6)
    b = np.arange(a.shape[0], dtype=np.float64)
    exact = spla.spsolve(a, b)
    converged = me.cg(a.data, a.indices, a.indptr, b, x0=exact, tol=1e-12)
    assert converged.info == 0 and converged.iterations == 0
    zero = me.cg(a.data, a.indices, a.indptr, np.zeros_like(b), x0=np.ones_like(b))
    assert zero.info == 0
    assert np.array_equal(zero.x, np.zeros_like(b))
    stopped = me.cg(a.data, a.indices, a.indptr, b, tol=1e-15, max_iter=1)
    assert stopped.info != 0
    assert np.linalg.norm(a @ stopped.x - b) < np.linalg.norm(b)


def test_cg_empty_and_invalid_csr():
    result = me.cg(
        np.empty(0), np.empty(0, dtype=np.int64), np.array([0]), np.empty(0)
    )
    assert result.info == 0 and result.x.size == 0
    with pytest.raises(ValueError):
        me.cg(np.ones(2), np.array([0, 3]), np.array([0, 1, 2]), np.ones(2))
    with pytest.raises(ValueError):
        me.cg(np.ones(1), np.array([0]), np.array([1, 1]), np.ones(1))
    with pytest.raises(ValueError):
        me.cg(
            np.ones(2), np.array([0, 1]), np.array([0, 2, 1, 2]), np.ones(3)
        )


def test_cg_rejects_unsafe_or_narrowing_inputs():
    with pytest.raises(TypeError, match="same dtype"):
        me.cg(
            np.ones(1, dtype=np.float32),
            np.array([0]),
            np.array([0, 1]),
            np.ones(1, dtype=np.float64),
        )
    with pytest.raises(TypeError, match="same dtype"):
        me.cg(
            np.ones(1),
            np.array([0]),
            np.array([0, 1]),
            np.ones(1),
            x0=np.ones(1, dtype=np.float32),
        )
    with pytest.raises(ValueError, match="finite"):
        me.cg(
            np.ones(1), np.array([0]), np.array([0, 1]), np.ones(1), tol=np.nan
        )
    with pytest.raises(TypeError, match="integers"):
        me.cg(np.ones(1), np.array([0.0]), np.array([0, 1]), np.ones(1))
    with pytest.raises(TypeError):
        me.cg(
            np.ones(1), np.array([0]), np.array([0, 1]), np.ones(1), max_iter=1.5
        )
