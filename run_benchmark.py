#!/usr/bin/env python3
"""Benchmark: skillvrp.py ueber alle Instanzen laufen lassen und mit
checker.py verifizieren.

Verwendung:
    python3 run_benchmark.py [timeout_sekunden]   # Default: 30

Schreibt pro Instanz eine Zeile nach bench_<timeout>s.txt (Profit +
Checker-Status) und am Ende die Summe aller Profits. Die Loesungsdateien
landen in bench_sols/.
"""

import re
import subprocess
import sys
import time
from pathlib import Path

TIMEOUT = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
DATASET_DIR = Path("../data/dataset")
SOLVER = Path("skillvrp.py")
CHECKER = Path("checker.py")
SOL_DIR = Path("bench_sols")
OUT_PATH = Path(f"bench_{TIMEOUT:g}s.txt")

SOL_DIR.mkdir(exist_ok=True)


def n_customers(path: Path) -> int:
    m = re.search(r"_n(\d+)_", path.name)
    return int(m.group(1)) if m else 0


instances = sorted(DATASET_DIR.glob("skillvrp_*.txt"), key=n_customers)
if not instances:
    sys.exit(f"Keine Instanzen in {DATASET_DIR} gefunden")

lines = []
total = 0
all_ok = True
t_start = time.perf_counter()

for idx, inst in enumerate(instances, start=1):
    sol_path = SOL_DIR / f"{inst.stem}.sol"

    t0 = time.perf_counter()
    with open(sol_path, "w") as f:
        subprocess.run(
            [sys.executable, str(SOLVER), str(inst), str(TIMEOUT)],
            stdout=f,
        )
    elapsed = time.perf_counter() - t0

    check = subprocess.run(
        [sys.executable, str(CHECKER), str(inst), str(sol_path)],
        capture_output=True, text=True,
    )
    check_out = (check.stdout or check.stderr).strip()
    ok = check.returncode == 0

    m = re.search(r"profit = (\d+)", check_out)
    profit = int(m.group(1)) if m else 0
    total += profit
    all_ok &= ok

    line = f"{inst.name:42s} profit={profit:>7d}  {elapsed:5.1f}s  | {check_out}"
    lines.append(line)
    print(f"[{idx:2d}/{len(instances)}] {line}", flush=True)

total_min = (time.perf_counter() - t_start) / 60.0
lines.append("")
lines.append(f"SUMME: {total}  |  alle checker-ok: {all_ok}  |  Gesamtzeit: {total_min:.1f} min")

OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\n{lines[-1]}")
print(f"Geschrieben: {OUT_PATH}")
