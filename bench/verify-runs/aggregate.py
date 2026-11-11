"""Aggregate the sequential official bench runs and compare them to the README."""

import glob
import json
import os
import re
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
readme_rows = {
    "batched 3x3 matmul (250k)": (6.70, 13.73),
    "batched 4x4 matvec (150k)": (2.00, 9.00),
    "batched 4x4 LDLT solve (20k)": (2.23, 7.00),
    "batched symmetric 4x4 eigh (20k)": (4.99, 13.04),
    "batched Jacobi 3x3 SVD (20k)": (9.84, 5.12),
    "CSR CG, 120x120 Poisson": (70.23, 84.25),
}
rows = {}
for path in sorted(glob.glob(os.path.join(ROOT, "bench", "verify-runs", "bench_official_*.log"))):
    for line in open(path):
        m = re.match(r"\| (batched .*?|CSR CG.*?) \| ([\d.]+) ms \| ([\d.]+) ms \| ([\d.]+)x", line)
        if m:
            rows.setdefault(m.group(1), []).append(
                (float(m.group(2)), float(m.group(3)), float(m.group(4))))

print("| case | README after | official medians (6 runs) | README ref/mojo | measured ref/mojo | ratio to README |")
print("| --- | ---: | ---: | ---: | ---: | ---: |")
for case, samples in rows.items():
    ours = sorted(s[0] for s in samples)
    ref = sorted(s[1] for s in samples)
    ratio = sorted(s[2] for s in samples)
    med = statistics.median(ours)
    rmed = statistics.median(ratio)
    readme_after, readme_ratio = readme_rows[case]
    pct = (med - readme_after) / readme_after * 100
    print(f"| {case} | {readme_after:.2f} ms | {med:.2f} ms ({ours[0]:.2f}-{ours[-1]:.2f}) "
          f"| {readme_ratio:.2f}x | {rmed:.2f}x | {pct:+.1f}% |")
    print(f"    ref medians: {['%.2f' % r for r in ref]}")
with open(os.path.join(ROOT, "bench", "verify-runs", "official_aggregate.json"), "w") as f:
    json.dump(rows, f, indent=1)
