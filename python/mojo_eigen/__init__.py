"""Eigen's VCGLib-focused linear algebra subset, ported to Mojo."""

from .linalg import CGResult, cg, eigh, inv, lstsq, matmul, matvec, solve, svd

__all__ = ["CGResult", "cg", "eigh", "inv", "lstsq", "matmul", "matvec", "solve", "svd"]
