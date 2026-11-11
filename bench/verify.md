# VERIFY pass: 2026-11-11

Scope: reproduce the README's benchmark figures, check that each speedup is
measured on the same inputs, check the tests still assert parity with upstream,
and correct the README where a figure did not reproduce within 20%.

No source or test code was changed. Nothing here needed one: every parity and
invariant check passed against upstream, so there was no failing test to fix
under the "fix only a demonstrable bug" rule.

## Environment

| item | value |
| --- | --- |
| host | shared `leaf-gpu-dedicated-server` |
| CPU | Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 threads (2 NUMA nodes) |
| kernel | `6.8.0-147-generic` (README previously claimed `-136`) |
| Python / NumPy / SciPy | 3.13.14 / 2.5.1 / 1.18.0 |
| Mojo | 1.2.0.dev2026092605 (pinned) |
| host load average | 65-90 throughout; unrelated jobs were saturating every core |
| GPU | RTX 5090, 4984 MiB free; shared with other tenants |

## Gate

`pixi run build && pixi run test` completed green: 67 passed, exit 0, before
and after the README edits. The build emits deprecation warnings only
(`UnsafePointer`, positional `__getitem__`).

`pixi run bench` also completed green. Four *other* agents from the same
factory (mojo-editdistance, mojo-empyrical, mojo-embree, mojo-eigen) were queueing
their own benchmarks on the shared `/tmp/mojo-bench.lock`, and every queued
`pixi run` stalled indefinitely. The earliest of those waiters was also this
repo, from my own agent (PID 2444533), left behind by a previous agent turn.
The cause is `pixi run` blocking while it takes the project-environment lock;
`PIXI_FROZEN_ENV=1` avoids it. I stopped my own stale waiters and took the lock
directly for the six sequential runs.

## How the README numbers were re-measured

Each row was measured three ways, all raw logs kept in `bench/verify-runs/`:

1. One `pixi run bench` run exactly as documented (`bench_official_1.log`).
2. Five more sequential runs of the same `bench.py` (`bench_official_2..6.log`),
   plus one independent harness (`repro.py`, `repro_cpu.log`) that runs the same
   case code in shuffled order over 9 interleaved rounds. Both use the
   benchmark's own best-of-three per sample.
3. Fairness re-timing of the reference side (`fairness.py`, `fairness.log`,
   `distinct_reference_interleaved.log`).

Best-of-3 minimum / median across the six official runs, versus the README:

| case | README "after" | measured (min-median-max) | delta of median | within 20%? |
| --- | ---: | ---: | ---: | --- |
| batched 3x3 matmul (250k) | 6.70 ms | 6.69-7.02-7.38 ms | +4.7% | yes |
| batched 4x4 matvec (150k) | 2.00 ms | 1.61-2.50-3.86 ms | +25.2% | no |
| batched 4x4 LDLT solve (20k) | 2.23 ms | 1.96-2.75-8.07 ms | +23.1% | no |
| batched symmetric 4x4 eigh (20k) | 4.99 ms | 4.06-4.57-5.32 ms | -8.3% | yes |
| batched Jacobi 3x3 SVD (20k) | 9.84 ms | 9.06-10.13-10.43 ms | +2.9% | yes |
| CSR CG, 120x120 Poisson | 70.23 ms | 67.77-69.68-72.34 ms | -0.8% | yes |

The six-process median reproduces every row within 20% **except** matvec
(+25%) and LDLT (+23%). Both are wrong in the same direction and both are
already sub-3 ms on a host at load average 65-90 (LDLT reached 8.07 ms in one
run before returning to 2.12 ms in the next). For those two rows the
independent shuffled-round harness (`repro.py`) gave 2.69 ms (matvec, +34%) and
1.80 ms (LDLT, -19%), so on matvec the two independent measurements agree and
both exceed the old README figure; for LDLT the two disagree by more than the
noise band, which is itself the finding. The README now carries the six-run
medians.

Does the "after" figure actually belong to the current code? A like-for-like
A/B of HEAD against the binary built from the last commit before its final
source change (`53b0e67`; HEAD `505ccc5` only added `bench/parallel_sweep.py`),
same host, same process call path, five samples each, repeated twice:

| best-of-five best / median | `53b0e67` binary | HEAD `505ccc5` binary |
| --- | ---: | ---: |
| matmul 250k | 6.71 / 6.85 and 6.89 / 7.22 ms | 6.78 / 7.11 and 6.61 / 6.93 ms |
| matvec 150k | 3.04 / 3.62 and 1.73 / 1.82 ms | 2.56 / 3.01 and 1.96 / 2.44 ms |

That is a tie: per-attempt ratios of 1.01 and 0.96 for matmul, 0.84 and 1.13
for matvec, with matvec's min-to-min jumping 1.73-3.04 ms on the same binary.
The 1.6x/1.3x microbench speedups claimed in the source comments and in the
old "before -> after" columns are **not** reproducible here: the 2026-11
unrolling pass shows no measurable gain over its parent. The README's own
"before" column, however, does not line up with git: at the initial commit
(`35df7c7`) matvec was 2.87 ms, so a "before" of 2.42 ms cannot be "before" the
balanced-tree matvec that landed in `f80ac24`. That before/after table is
replaced by the single measured table in `README.md`; no unattributed number
remains.

Reference-column ratios (README "reference / Mojo") do not reproduce at all,
because the reference side is loaded noise on this host: upstream row times span
87-102 ms (matmul), 57-74 ms (SVD), 7.4-23.1 s (CG). Worst-case ratios
(slowest Mojo / fastest reference) are 11.7x, 3.7x, 2.2x, 11.2x, 5.5x for the
five upstream-slower rows, against reporting medians 12.9x, 6.2x, 7.0x, 15.1x,
6.8x. The README now prints the measured medians and min-max ranges and labels
the host noise explicitly rather than reporting a single ratio as stable.

## Same input, same work?

`bench.py` case by case (verified by reading it and by timing the alternatives):

| row | mojo side | reference side | fair? |
| --- | --- | --- | --- |
| matmul | 250k distinct products | `a @ b` on the same arrays | yes, identical work |
| matvec | 150k distinct products | `(a @ x[...,None])[...,0]` on the same arrays | yes |
| LDLT | 20k **distinct** factorizations + solves | **one** factorization, 20k rhs re-solves | no: reference does 0.3% of the factorizations |
| eigh | 20k factorizations | `[SelfAdjointEigenSolver(s) for _ in range(20000)]`: constructs, never asks for eigenvalues or vectors | no |
| SVD | 20k full SVDs | `[HhJacobiSVD(d, 20) for _ in range(20000)]`: constructs, never asks for U/S/Vh | no |
| CG | same CSR system, `rtol=1e-8` | `spla.cg` on the same system | yes |

For LDLT, eigh and SVD the comparison understates upstream. Re-timed on the
same 20k inputs (interleaved in one process, since timing a side alone shifts
the CPU clock: numpy LDLT was 22.2 ms alone and 11.8-12.5 ms interleaved):

| row | Mojo | numpy, 20k distinct | ratio |
| --- | ---: | ---: | ---: |
| LDLT | 1.78-2.09 ms | 11.8-12.5 ms | 6.00-6.75x slower |
| eigh | 5.21-5.49 ms | 67.8-69.7 ms | 12.35-13.39x slower |
| SVD | 9.33-10.99 ms | 77.0-79.1 ms | 7.20-8.42x slower |

Conclusion: no row hides skipped work on the Mojo side, and every row's true
same-work speedup still has upstream faster. I left `bench.py` unchanged: it is
the command the instructions run, and changing the reference would make the
six official runs above non-reproducible. The gap is now stated in the README.

## GPU rows

The two README GPU rows (eigh 3.46 ms vs CPU 7.30 ms; SVD 3.69 ms vs CPU
11.61 ms) are **not** produced by `pixi run bench`: the GPU cases only exist
when `MOJO_EIGEN_BENCH_GPU=1` is set, so a plain run silently omits them.

With the gate forced on, the benchmark **crashes** (SIGSEGV). gdb backtrace:
`me_eigh_batch_gpu_f64` -> `AsyncRT_DeviceContext_DtoH_async` ->
`cuMemcpyDtoHAsync_v2` -> fault inside `/lib/x86_64-linux-gnu/libcuda.so.1`.
It reproduces from a bare interpreter: running the two GPU kernels in one
process faults on the second `DtoH`. Run one kernel per process
(`bench/verify-runs/gpu_subprocess.py`), results are correct (eigh max error
1.6e-15, SVD 6.0e-15) but timings move with host load; two sessions gave
eigh GPU/CPU 4.43x and 1.25x, SVD 6.47x and 5.09x. The README now prints these
as ranges and says why they are not reproducible as stated.

The crash is a bug in the pinned max/CUDA integration, not in this repo's
kernel, and I have no failing *test* for it, so no code change was made.
A `pixi run bench` with the GPU gate set will segfault.

## Tests

`git log -p -- tests/` shows exactly one commit touching `tests/`
(`35df7c7`, the initial import). Scanning every commit pair for deleted
lines containing `assert`, `rtol`, `atol` or `tolerance` returns nothing.
The tolerance constants are unchanged since the initial commit
(`2e-11` float64, `3e-4` float32). The two dangling commits
(`430ca19`, `f07f17f`, recovery of mid-pass WIP) touch only `src/eigen.mojo`.
Their test blobs are identical to HEAD. So nothing was removed or loosened,
and nothing needed restoring.

What the suite actually still asserts (all through the pinned eigenpy, i.e.
upstream Eigen): LU/LLT/LDLT solutions equal `PartialPivLU/LLT/LDLT.solve` for
f32 and f64, 3x3 and 4x4; eigenvalues equal `SelfAdjointEigenSolver`; SVD
singular values equal `HhJacobiSVD` plus `U S Vh == A` and orthogonality; QR
equals `HouseholderQR.solve`; inverse equals `PartialPivLU.inverse`; the
lower-triangle convention matches Eigen; batched and serial paths both
covered; and every failure path is asserted.

Independent margin and invariant checks (`tolerance_probe.py`,
`monomorphic.py`, `consistency.py`):

- Worst deviation from eigenpy over 200 random matrices per configuration is
  0.0xx of the test tolerance: the suite is not near its edge for any kernel.
- An independent CG implementation reproduces the residual and iteration
  count and gives the same solution to 8.9e-16.
- eigh: 0/100 violations of residual/orthogonality. SVD: 0/100 violations of
  orthogonality/reconstruction/descending singular values.
- Batched vs per-item calls differ by at most 1.5e-14, and 20,000-item
  eigh/SVD/LDLT batches are bitwise identical across four repeated calls
  (`determinism.py`: no worker-chunk race).

## What I could not reproduce

- The GPU rows as printed (crash; load-dependent, ranges now).
- Reference-side timings and ratios (upstream range up to 3x within the session).
- matvec and LDLT Mojo medians within 20% of the old README (off by +25% and
  +23% across six runs; both now carry the measured number).

## Changes

- `README.md` Performance section: replaced the unverifiable before/after pair
  and the GPU table with measured figures only. Added the measured min-max
  reference ranges, the same-work re-timing for LDLT/eigh/SVD, the GPU crash and
  ranges, and corrected the kernel version and date. All CPU and GPU numbers in
  it were measured in this session.- `bench/verify-runs/`: raw logs and the scripts that produced them (official
  runs, shuffled-round harness, fairness audit, comparable-work re-timing,
  tolerance/invariant/determinism probes, the HEAD-vs-parent A/B, GPU
  one-kernel-per-process runner, `commits/` with the parent revision's source
  and its scratch-built library).
- No change to `src/`, `tests/`, `bench/bench.py`, or `pixi.toml`.
