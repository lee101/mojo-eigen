"""Like-for-like timing of two builds (HEAD vs the parent commit) on the same host."""
import os
import subprocess
import sys

import numpy as np

WORKER = r'''
import os, sys, time, numpy as np
sys.path.insert(0, sys.argv[1])
os.environ["MOJO_EIGEN_LIB"] = sys.argv[2]
import mojo_eigen as me
rng = np.random.default_rng(7)
a3 = rng.normal(size=(250_000, 3, 3)); b3 = rng.normal(size=(250_000, 3, 3))
a4 = rng.normal(size=(150_000, 4, 4)); x4 = rng.normal(size=(150_000, 4))
def best(fn):
    b = float("inf")
    for _ in range(5):
        t = time.perf_counter(); fn(); b = min(b, time.perf_counter() - t)
    return b * 1e3
me.matmul(a3, b3); me.matvec(a4, x4)
mm = sorted(best(lambda: me.matmul(a3, b3)) for _ in range(9))
mv = sorted(best(lambda: me.matvec(a4, x4)) for _ in range(9))
print("RESULT", np.min(mm), np.median(mm), np.min(mv), np.median(mv))
'''


def run(lib, root):
    r = subprocess.run([sys.executable, "-c", WORKER, os.path.join(root, "python"), lib],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode:
        raise SystemExit(r.stderr[-2000:])
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")][0]
    return [float(v) for v in line.split()[1:]]


if __name__ == "__main__":
    root, head_lib, parent_lib = sys.argv[1], sys.argv[2], sys.argv[3]
    for attempt in range(2):
        h = run(head_lib, root)
        p = run(parent_lib, root)
        print(f"attempt {attempt} HEAD   matmul best {h[0]:.2f} med {h[1]:.2f} ms"
              f" | matvec best {h[2]:.2f} med {h[3]:.2f} ms", flush=True)
        print(f"attempt {attempt} PARENT matmul best {p[0]:.2f} med {p[1]:.2f} ms"
              f" | matvec best {p[2]:.2f} med {p[3]:.2f} ms", flush=True)
        print(f"attempt {attempt} HEAD/PARENT matmul {h[0] / p[0]:.3f}"
              f"  matvec {h[2] / p[2]:.3f} (best-of-best)", flush=True)
