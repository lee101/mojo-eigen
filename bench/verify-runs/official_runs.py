"""Run the official benchmark command several times, sequentially, and log it.

Each run is the same command pixi.toml's `bench` task invokes
(`flock /tmp/mojo-bench.lock python bench/bench.py`); these runs are already
sequential, so only the inner python command is started here.
"""
import subprocess
import time

for i in range(2, 7):
    log = f"bench/verify-runs/bench_official_{i}.log"
    t = time.time()
    with open(log, "w") as f:
        rc = subprocess.call([".pixi/envs/default/bin/python", "-u", "bench/bench.py"],
                             stdout=f, stderr=subprocess.STDOUT)
    print(f"run {i}: exit={rc} {time.time() - t:.0f}s", flush=True)
