"""ctypes loader for the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_EIGEN_LIB") or os.path.join(
    ROOT, "dist", "libmojo-eigen.so"
)

I = ctypes.c_int64
F32 = ctypes.c_float
F64 = ctypes.c_double

_SIGNATURES = {
    "me_matmul_f64": ([I] * 5, None),
    "me_matmul_f32": ([I] * 5, None),
    "me_matvec_f64": ([I] * 5, None),
    "me_matvec_f32": ([I] * 5, None),
    "me_lu_f64": ([I] * 6, I),
    "me_lu_f32": ([I] * 6, I),
    "me_llt_f64": ([I] * 5, I),
    "me_llt_f32": ([I] * 5, I),
    "me_ldlt_f64": ([I] * 7, I),
    "me_ldlt_f32": ([I] * 7, I),
    "me_ldlt_batch_f64": ([I] * 10, None),
    "me_ldlt_batch_f32": ([I] * 10, None),
    "me_qr_f64": ([I] * 7, I),
    "me_qr_f32": ([I] * 7, I),
    "me_eigh_f64": ([I] * 5, I),
    "me_eigh_f32": ([I] * 5, I),
    "me_eigh_batch_f64": ([I] * 8, None),
    "me_eigh_batch_f32": ([I] * 8, None),
    "me_eigh_batch_gpu_f64": ([I] * 6, I),
    "me_svd_f64": ([I] * 7, I),
    "me_svd_f32": ([I] * 7, I),
    "me_svd_batch_f64": ([I] * 10, None),
    "me_svd_batch_f32": ([I] * 10, None),
    "me_svd_batch_gpu_f64": ([I] * 7, I),
    "me_cg_f64": ([I] * 9 + [F64], I),
    "me_cg_f32": ([I] * 9 + [F32], I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    sources = [
        os.path.join(ROOT, "src", "eigen.mojo"),
        os.path.join(ROOT, "build", "build.sh"),
    ]
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= max(map(os.path.getmtime, sources))
    ):
        return LIB
    pixi = shutil.which("pixi")
    command = (
        [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"]
        if pixi
        else ["bash", os.path.join(ROOT, "build", "build.sh")]
    )
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=1800
    )
    if result.returncode or not os.path.exists(LIB):
        raise BuildError((result.stderr or result.stdout).strip()[:8000])
    return LIB


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_loaded, name)
            function.argtypes = argtypes
            function.restype = restype
    return _loaded


def address(array) -> int:
    return int(array.ctypes.data)
