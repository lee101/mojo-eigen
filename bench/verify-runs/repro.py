"""Verify harness: run the exact bench.py cases with more samples and a
shuffled order, then compare medians against the README figures."""
from __future__ import annotations
import importlib.util, json, os, platform, random, statistics, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("bench_cases", os.path.join(ROOT, "bench", "bench.py"))
bench = importlib.util.module_from_spec(spec)
sys.argv = ["bench.py"]
spec.loader.exec_module(bench)  # defines helpers; main() is not called on import

me = bench.me

def best(function, repeat=3):
    b = float("inf")
    for _ in range(repeat):
        t = time.perf_counter(); function(); b = min(b, time.perf_counter() - t)
    return b

def main():
    rounds = int(os.environ.get("ROUNDS", "9"))
    out = {}
    rng = np.random.default_rng(7)
    a3 = rng.normal(size=(250_000, 3, 3)); b3 = rng.normal(size=(250_000, 3, 3))
    cases = [("matmul3", lambda: me.matmul(a3, b3), lambda: a3 @ b3)]
    a4 = rng.normal(size=(150_000, 4, 4)); x4 = rng.normal(size=(150_000, 4))
    cases.append(("matvec4", lambda: me.matvec(a4, x4), lambda: (a4 @ x4[..., None])[..., 0]))
    spd = rng.normal(size=(4, 4)); spd = spd.T @ spd + np.eye(4); rhs = rng.normal(size=4)
    spd_batch = np.repeat(spd[None, ...], 20_000, axis=0); rhs_batch = np.repeat(rhs[None, ...], 20_000, axis=0)
    eigen_ldlt = bench.eigenpy.LDLT(spd)
    cases.append(("ldlt", lambda: me.solve(spd_batch, rhs_batch, "ldlt"),
                  lambda: [eigen_ldlt.solve(rhs) for _ in range(20_000)]))
    sym = rng.normal(size=(4, 4)); sym = (sym + sym.T) * 0.5
    sym_batch = np.repeat(sym[None, ...], 20_000, axis=0)
    cases.append(("eigh4", lambda: me.eigh(sym_batch),
                  lambda: [bench.eigenpy.SelfAdjointEigenSolver(sym) for _ in range(20_000)]))
    dense = rng.normal(size=(3, 3)); dense_batch = np.repeat(dense[None, ...], 20_000, axis=0)
    cases.append(("svd3", lambda: me.svd(dense_batch),
                  lambda: [bench.eigenpy.HhJacobiSVD(dense, 20) for _ in range(20_000)]))
    system = bench.poisson(120); srhs = rng.normal(size=system.shape[0])
    cases.append(("cg", lambda: me.cg(system.data, system.indices, system.indptr, srhs, tol=1e-8, max_iter=500),
                  lambda: bench.spla.cg(system, srhs, rtol=1e-8, atol=0, maxiter=500)))
    if os.environ.get("MOJO_EIGEN_BENCH_GPU") == "1":
        cases.append(("eigh4_gpu", lambda: me.eigh(sym_batch, device="gpu"), lambda: me.eigh(sym_batch)))
        cases.append(("svd3_gpu", lambda: me.svd(dense_batch, device="gpu"), lambda: me.svd(dense_batch)))

    for name, ours, ref in cases:
        ours(); ref()
    order = list(range(len(cases)))
    samples = {c[0]: {"ours": [], "ref": []} for c in cases}
    random.Random(1234).shuffle(order)
    for r in range(rounds):
        random.Random(1000 + r).shuffle(order)
        for i in order:
            name, ours, ref = cases[i]
            sides = [("ours", ours), ("ref", ref)]
            random.Random(2000 + r).shuffle(sides)
            for side, fn in sides:
                duration = best(fn, 3)
                samples[name][side].append(duration * 1e3)
    for name in samples:
        o = sorted(samples[name]["ours"]); q = sorted(samples[name]["ref"])
        out[name] = {
            "ours_median": statistics.median(o), "ours_min": o[0], "ours_max": o[-1],
            "ref_median": statistics.median(q), "ref_min": q[0], "ref_max": q[-1],
            "ratio_median": statistics.median(q) / statistics.median(o),
            "samples": samples[name],
        }
    print(f"machine: {bench.cpu_name()}; {platform.platform()}; rounds={rounds}")
    print("| case | mojo median ms | mojo min-max | ref median ms | ref min-max | ref/mojo |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, r in out.items():
        print(f"| {name} | {r['ours_median']:.2f} | {r['ours_min']:.2f}-{r['ours_max']:.2f} | "
              f"{r['ref_median']:.2f} | {r['ref_min']:.2f}-{r['ref_max']:.2f} | {r['ratio_median']:.2f}x |")
    path = os.environ.get("OUT", os.path.join(ROOT, "bench", "verify-runs", "repro.json"))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

if __name__ == "__main__":
    main()
