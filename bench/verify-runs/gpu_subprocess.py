"""Run the benchmark GPU cases in a fresh subprocess (one kernel per process).

The pinned max/CUDA stack segfaults inside cuMemcpyDtoHAsync when the process
runs two device kernels, so each measurement gets its own interpreter.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORKER = r'''
import sys, time, json, numpy as np
sys.path.insert(0, sys.argv[1])
import mojo_eigen as me
kind, n, batch, mode = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
rng = np.random.default_rng(7)
if kind == "eigh":
    raw = rng.normal(size=(n, n)); m = ((raw + raw.T) * 0.5).astype(np.float64)
    data = np.repeat(m[None, ...], batch, axis=0)
    call = (lambda: me.eigh(data, device="gpu")) if mode == "gpu" else (lambda: me.eigh(data))
else:
    raw = rng.normal(size=(n, n)).astype(np.float64)
    data = np.repeat(raw[None, ...], batch, axis=0)
    call = (lambda: me.svd(data, device="gpu")) if mode == "gpu" else (lambda: me.svd(data))
call()
best = float("inf")
for _ in range(3):
    t = time.perf_counter(); out = call(); best = min(best, time.perf_counter() - t)
res = {"ms": best * 1e3}
if kind == "eigh":
    values, vectors = out
    ref = np.linalg.eigvalsh(data)
    res["err"] = float(np.abs(np.sort(values, axis=-1) - ref).max())
    res["resid"] = float(np.abs(np.matmul(data, vectors) - vectors * values[..., None, :]).max())
    res["same_all"] = bool(np.allclose(values, values[0]))
else:
    u, s, vh = out
    res["err"] = float(np.abs(np.matmul(u * s[..., None, :], vh) - data).max())
    res["resid"] = float(np.abs(np.matmul(np.swapaxes(u, -2, -1), u) - np.eye(n)).max())
    res["same_all"] = bool(np.allclose(s, s[0]))
print("RESULT " + json.dumps(res))
'''


def main():
    out = {}
    env = dict(os.environ)
    for kind, n, batch, label in [("eigh", 4, 20000, "eigh4"), ("svd", 3, 20000, "svd3")]:
        for mode in ("gpu", "cpu"):
            try:
                r = subprocess.run(
                    [sys.executable, "-c", WORKER, os.path.join(ROOT, "python"),
                     kind, str(n), str(batch), mode],
                    capture_output=True, text=True, timeout=1800, env=env)
            except subprocess.TimeoutExpired:
                out[f"{label}_{mode}"] = {"timeout": True}
                print(f"{label} {mode}: TIMEOUT", flush=True)
                continue
            line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
            rec = json.loads(line[-1][7:]) if line else {
                "error": (r.stderr or r.stdout)[-800:], "code": r.returncode}
            rec["exit"] = r.returncode
            out[f"{label}_{mode}"] = rec
            print(f"{label:6s} {mode:3s} exit={r.returncode} {rec}", flush=True)
    with open(os.path.join(ROOT, "bench", "verify-runs", "gpu_subprocess.json"), "w") as f:
        json.dump(out, f, indent=1)
    for label in ("eigh4", "svd3"):
        g, c = out[f"{label}_gpu"], out[f"{label}_cpu"]
        if "ms" in g and "ms" in c:
            print(f"{label} GPU/CPU speedup: {c['ms'] / g['ms']:.2f}x "
                  f"({g['ms']:.2f} ms vs {c['ms']:.2f} ms)")


if __name__ == "__main__":
    main()
