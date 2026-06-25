#!/usr/bin/env python3
"""Benchmark skillvrp+.py over all instances: run with a time limit, validate
with checker.py, and report profit vs. a singleton-feasibility upper bound."""
import importlib.util
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, "dataset")
TIMEOUT = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SOLVER = sys.argv[2] if len(sys.argv) > 2 else "skillvrp+.py"
SOLDIR = os.path.join(BASE, "bench_sols_" + re.sub(r"\W", "", SOLVER[:-3]))

spec = importlib.util.spec_from_file_location("svrp", os.path.join(BASE, "skillvrp+.py"))
svrp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svrp)


def upper_bound(inst):
    """Sum of profits over customers servable as a singleton tour by >=1 courier."""
    ub = 0
    for c in range(1, inst.n_nodes):
        for b in range(inst.n_vehicles):
            if inst.req_skills[c] <= inst.v_skills[b] and \
                    svrp.compute_timing(inst, b, [c])[0]:
                ub += inst.profit[c]
                break
    return ub


def main():
    os.makedirs(SOLDIR, exist_ok=True)
    files = sorted((f for f in os.listdir(DATASET)
                    if re.search(r"n(\d+)", f) and f.endswith(".txt")),
                   key=lambda f: int(re.search(r"n(\d+)", f).group(1)))
    print(f"{'instance':42s} {'n':>5} {'B':>3} {'obj':>7} {'UB':>7} {'%UB':>6} "
          f"{'time':>6} {'check':>6}")
    rows = []
    for fname in files:
        ipath = os.path.join(DATASET, fname)
        spath = os.path.join(SOLDIR, fname.replace(".txt", ".sol"))
        t0 = time.time()
        with open(spath, "w") as out:
            subprocess.run([sys.executable, os.path.join(BASE, SOLVER),
                            ipath, str(TIMEOUT)], stdout=out,
                           timeout=TIMEOUT + 15)
        wall = time.time() - t0

        chk = subprocess.run([sys.executable, os.path.join(BASE, "checker.py"),
                              ipath, spath], capture_output=True, text=True)
        ok = "OK" if chk.returncode == 0 else "FAIL"

        with open(spath) as f:
            m = re.search(r"###OBJECTIVE:\s*(\d+)", f.read())
        obj = int(m.group(1)) if m else -1

        inst = svrp.parse_instance(ipath)
        ub = upper_bound(inst)
        pct = 100.0 * obj / ub if ub else 100.0
        print(f"{fname:42s} {inst.n_customers:>5} {inst.n_vehicles:>3} "
              f"{obj:>7} {ub:>7} {pct:>5.1f}% {wall:>5.1f}s {ok:>6}", flush=True)
        if ok != "OK":
            print("   checker: " + (chk.stdout + chk.stderr).strip()[:300], flush=True)
        rows.append((fname, obj, ub, pct, ok))

    fails = [r for r in rows if r[4] != "OK"]
    avg = sum(r[3] for r in rows) / len(rows)
    print(f"\nAverage %UB: {avg:.1f}%   infeasible: {len(fails)}")


if __name__ == "__main__":
    main()
