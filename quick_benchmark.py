"""
30-Sekunden-Benchmark mit dem aktuellen ALNS + Grid-Search-Parametern.
Alle 20 Instanzen, validiert mit checker.py.

Verwendung:
    python3 quick_benchmark.py

Laufzeit: 20 Instanzen x 30s = ca. 10 Minuten.
Ergebnis: Tabelle mit Profit pro Instanz + Gesamtprofit.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from alns import run_alns, SAParams, ALNSParams, PenaltyParams
from instance_reader import read_instance
from initial_solution import greedy_initial_solution
from make_operators_fixed import make_operators

DATASET_DIR = Path("../data_extracted/dataset")
CHECKER    = Path("checker.py")
SOL_DIR    = Path("quick_benchmark_solutions")
SOL_DIR.mkdir(parents=True, exist_ok=True)
TOP3_PATH  = Path("gridsearch_top3_setups.json")

RUNTIME = 30.0
SEED    = 1

# Bestes Grid-Search-Setup laden (Fallback: vernuenftige Defaults)
if TOP3_PATH.exists():
    with open(TOP3_PATH) as f:
        setup = json.load(f)[0]
    # JSON liefert Listen statt Tupel — alns.py erwartet tuple fuer fraction-Ranges
    for key in ("base_fraction", "related_fraction"):
        if isinstance(setup.get(key), list):
            setup[key] = tuple(setup[key])
    print(f"Setup: id={setup['setup_id']}  "
          f"temp_factor={setup['temp_factor']}  "
          f"profile={setup['destroy_profile']}")
else:
    setup = {}
    print("Kein gridsearch_top3_setups.json gefunden — nutze Defaults")

SA = SAParams(
    initial_temperature=100.0,
    min_temperature=0.001,
    cooling_rate=0.9995,
    reheat_factor=0.75,
)
PARAMS = ALNSParams(
    random_seed=SEED,
    segment_length=int(setup.get("segment_length", 100)),
    no_accept_limit=int(setup.get("no_accept_limit", 700)),
    reaction_factor=float(setup.get("reaction_factor", 0.20)),
    min_operator_weight=float(setup.get("min_operator_weight", 0.05)),
    score_global_best=25.0, score_better_current=10.0,
    score_accepted=3.0, score_rejected=0.0,
    time_cost_alpha=0.5, time_scale_seconds=0.01,
    verbose=False, polish_interval_seconds=0.0,
)
PENALTIES = PenaltyParams(
    time_window_penalty=float(setup.get("time_window_penalty", 20.0)),
    shift_penalty=float(setup.get("shift_penalty", 20.0)),
    skill_penalty=float(setup.get("skill_penalty", 10000.0)),
)

inst_files = sorted(
    DATASET_DIR.glob("skillvrp_*.txt"),
    key=lambda p: int(re.search(r"n(\d+)", p.name).group(1)),
)

print(f"\n{'Instanz':<40} {'n':>5}  {'Greedy':>7}  {'ALNS':>7}  {'Verbes.':>7}  {'Check':>6}")
print("-" * 80)

total_profit = 0
rows = []

for inst_path in inst_files:
    inst  = read_instance(str(inst_path))
    start = greedy_initial_solution(inst, strategy="multi_start", verbose=False)

    d_ops, r_ops = make_operators(inst, setup if setup else None)

    result = run_alns(
        runtime=RUNTIME, inst=inst, start_solution=start,
        sa=SA, params=PARAMS, penalties=PENALTIES,
        destroy_operators=d_ops, repair_operators=r_ops,
    )

    profit   = result.best_evaluation.profit
    feasible = result.best_evaluation.feasible
    improv   = 100.0 * (profit - start.objective) / max(start.objective, 1)

    # Loesung fuer Checker speichern
    sol_path = SOL_DIR / inst_path.name.replace(".txt", ".sol")
    result.best_solution.write_for_checker(str(sol_path))

    chk = subprocess.run(
        [sys.executable, str(CHECKER), str(inst_path), str(sol_path)],
        capture_output=True, text=True,
    )
    check_ok = "OK" if chk.returncode == 0 else "FAIL"

    total_profit += profit
    rows.append((inst_path.name, inst.num_customers, start.objective, profit, improv, check_ok))

    print(f"{inst_path.name:<40} {inst.num_customers:>5}  "
          f"{start.objective:>7}  {profit:>7}  {improv:>+6.1f}%  {check_ok:>6}",
          flush=True)

# Zusammenfassung
fails = [r for r in rows if r[5] != "OK"]
avg_improv = sum(r[4] for r in rows) / len(rows)

print("-" * 80)
print(f"\nGesamtprofit (alle {len(rows)} Instanzen): {total_profit:,}")
print(f"Andere Gruppe (Vergleich):                310,000")
print(f"Differenz:                               +{total_profit - 310000:,}")
print(f"Durchschn. Verbesserung ggue. Greedy:     {avg_improv:.1f}%")
print(f"Infeasible:                               {len(fails)}")
