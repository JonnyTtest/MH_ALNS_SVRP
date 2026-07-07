#!/usr/bin/env python3
"""Benchmark skillvrp.py over all instances: run with a time limit, validate
with checker.py, and report profit vs. a singleton-feasibility upper bound.

Usage: python3 benchmark.py [timeout] [solver]
"""
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, "..", "data", "dataset")
CHECKER = os.path.join(BASE, "..", "data", "checker.py")
TIMEOUT = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SOLVER = sys.argv[2] if len(sys.argv) > 2 else "skillvrp.py"
SEEDS = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1]
SOLDIR = os.path.join(BASE, "bench_sols")

sys.path.insert(0, BASE)
from instance_reader import read_instance  # noqa: E402


def upper_bound(inst):
    """Sum of profits over customers servable as a singleton tour."""
    return sum(inst.profit[c] for c in inst.customers if inst.solo_feasible_vehicles[c])


def main():
    os.makedirs(SOLDIR, exist_ok=True)
    files = sorted((f for f in os.listdir(DATASET)
                    if re.search(r"n(\d+)", f) and f.endswith(".txt")),
                   key=lambda f: int(re.search(r"n(\d+)", f).group(1)))
    seed_hdr = " ".join(f"{'s'+str(s):>7}" for s in SEEDS)
    print(f"{'instance':42s} {'n':>5} {'B':>3} {seed_hdr} {'mean':>7} {'%UB':>6} "
          f"{'check':>6}", flush=True)
    rows = []
    for fname in files:
        ipath = os.path.join(DATASET, fname)
        inst = read_instance(ipath)
        ub = upper_bound(inst)
        objs = []
        all_ok = True
        for seed in SEEDS:
            spath = os.path.join(SOLDIR, fname.replace(".txt", f"_s{seed}.sol"))
            with open(spath, "w") as out:
                subprocess.run([sys.executable, os.path.join(BASE, SOLVER),
                                ipath, str(TIMEOUT), "--seed", str(seed)],
                               stdout=out, timeout=TIMEOUT + 15)
            chk = subprocess.run([sys.executable, CHECKER, ipath, spath],
                                 capture_output=True, text=True)
            if chk.returncode != 0:
                all_ok = False
                print("   checker: " + (chk.stdout + chk.stderr).strip()[:300], flush=True)
            with open(spath) as f:
                m = re.search(r"###OBJECTIVE:\s*(\d+)", f.read())
            objs.append(int(m.group(1)) if m else -1)

        mean = sum(objs) / len(objs)
        pct = 100.0 * mean / ub if ub else 100.0
        ok = "OK" if all_ok else "FAIL"
        obj_cols = " ".join(f"{o:>7}" for o in objs)
        print(f"{fname:42s} {inst.num_customers:>5} {inst.num_vehicles:>3} "
              f"{obj_cols} {mean:>7.0f} {pct:>5.1f}% {ok:>6}", flush=True)
        rows.append((fname, mean, ub, pct, ok))

    fails = [r for r in rows if r[4] != "OK"]
    avg = sum(r[3] for r in rows) / len(rows)
    print(f"\nAverage %UB (mean over {len(SEEDS)} seeds): {avg:.1f}%   infeasible: {len(fails)}")


if __name__ == "__main__":
    main()
