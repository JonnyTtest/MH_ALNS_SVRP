"""
Ablationsstudie fuer das ALNS-Framework.

Fuer jeden Operator wird einmal ein Lauf ohne diesen Operator durchgefuehrt
und mit dem Baseline-Lauf (alle Operatoren) verglichen. Die Differenz zeigt
wie viel jeder einzelne Operator zum Gesamtergebnis beitraegt.

Ergebnisse werden als CSV gespeichert:
  ablation_results/ablation_results.csv
  ablation_results/ablation_summary.csv

Verwendung:
  python3 ablation_study.py

Laufzeit: ca. 60-90 Minuten (abhaengig von RUNTIME_PER_RUN und SEEDS).
"""

from __future__ import annotations

import csv
import os
import random
import time
from pathlib import Path

from instance_reader import read_instance
from initial_solution import greedy_initial_solution
from alns import (
    run_alns, SAParams, ALNSParams, PenaltyParams,
    RandomRemoval, WorstDensityRemoval, SkillScarcityRemoval,
    RelatedRemoval, WorstDetourRemoval, WorstDetourRemovalV2,
    ShawRelatedRemoval, RouteRemoval, TimeWindowSegmentRemoval,
    SequenceRemoval, ClusterRemoval, LargestSavingRemoval,
    TemporalShawRemoval, HistoryBasedRemoval,
    GreedyBestInsertionRepair, Regret2InsertionRepair,
    SequentialCheapestInsertionRepair, NoisyGreedyInsertionRepair,
    ScarceSkillFirstRepair, LastRemovedFirstInsertedRepair,
    RegretKInsertionRepair, RandomPositionInsertionRepair,
    ShawInsertionRepair,
)

# -----------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------

DATASET_DIR = Path("dataset")
OUT_DIR = Path("ablation_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reprasentative Instanzen: klein, mittel, gross
INSTANCES = [
    "skillvrp_n100_v2_s5_k1.0_3.txt",
    "skillvrp_n450_v11_s3_k0.0_11.txt",
    "skillvrp_n1000_v25_s8_k2.0_20.txt",
]

RUNTIME_PER_RUN = 90.0   # Sekunden pro Lauf
SEEDS = [0, 1]            # Zwei Seeds fuer Robustheit

# Beste Parameter aus der Grid Search
SA = SAParams(
    initial_temperature=100.0 * 1.5,   # temp_factor=1.5 war Rank 1
    min_temperature=0.001,
    cooling_rate=0.9995,
    reheat_factor=0.75,
)
PARAMS = ALNSParams(
    random_seed=0,          # wird pro Lauf ueberschrieben
    segment_length=150,
    no_accept_limit=700,
    reaction_factor=0.20,
    min_operator_weight=0.01,   # niedrig damit schwache Operatoren sichtbar bleiben
    score_global_best=25.0,
    score_better_current=10.0,
    score_accepted=3.0,
    score_rejected=0.0,
    time_cost_alpha=0.5,
    time_scale_seconds=0.01,
    verbose=False,
    polish_interval_seconds=0.0,
)
PENALTIES = PenaltyParams(
    time_window_penalty=20.0,
    shift_penalty=20.0,
    skill_penalty=10000.0,
)


# -----------------------------------------------------------------------
# Operator-Factory: alle Operatoren korrekt instanziiert
# Basiert auf make_operators_fixed.py mit korrigierten Signaturen
# -----------------------------------------------------------------------

def make_all_operators(inst):
    """Erstellt das vollstaendige, korrekte Operator-Set (keine Bugs)."""
    n = inst.num_customers
    max_remove = max(5, int(n * 0.20))
    eul = 80    # extra_unserved_limit

    destroy = [
        RandomRemoval(fraction=0.10, min_remove=1, max_remove=max_remove, initial_weight=1.0),
        WorstDensityRemoval(fraction=0.10, min_remove=1, max_remove=max_remove, noise=0.05, initial_weight=1.0),
        SkillScarcityRemoval(fraction=0.10, min_remove=1, max_remove=max_remove, noise=0.05, initial_weight=1.0),
        RelatedRemoval(fraction=0.15, min_remove=1, max_remove=max_remove, bias=4.0, initial_weight=1.0),
        WorstDetourRemoval(fraction=0.10, min_remove=1, max_remove=max_remove, bias=4.0, initial_weight=1.0),
        WorstDetourRemovalV2(fraction=0.10, min_remove=1, max_remove=max_remove, noise=0.05, selection_bias=3.0, initial_weight=1.0),
        ShawRelatedRemoval(fraction=0.15, min_remove=1, max_remove=max_remove, p_determinism=4.0, w_distance=0.5, w_time=0.25, w_skill=0.25, initial_weight=1.0),
        RouteRemoval(max_routes=2, selection_bias=3.0, initial_weight=1.0),
        TimeWindowSegmentRemoval(window_fraction=0.15, max_remove=max_remove, initial_weight=1.0),
        SequenceRemoval(fraction=0.10, min_remove=1, max_remove=max_remove, initial_weight=1.0),
        ClusterRemoval(fraction=0.10, min_remove=1, max_remove=max_remove, initial_weight=1.0),
        LargestSavingRemoval(fraction=0.10, min_remove=1, max_remove=max_remove, selection_bias=3.0, initial_weight=1.0),
        TemporalShawRemoval(fraction=0.15, min_remove=1, max_remove=max_remove, p_determinism=4.0, w_distance=0.5, w_time=0.25, w_skill=0.25, initial_weight=1.0),
        HistoryBasedRemoval(fraction=0.10, min_remove=1, max_remove=max_remove, selection_bias=3.0, initial_weight=1.0),
    ]

    repair = [
        GreedyBestInsertionRepair(extra_unserved_limit=eul, max_insertions=None, min_delta_score=0.0, initial_weight=1.0),
        Regret2InsertionRepair(extra_unserved_limit=eul, max_insertions=None, min_delta_score=0.0, initial_weight=1.0),
        SequentialCheapestInsertionRepair(extra_unserved_limit=eul, order="profit", initial_weight=1.0),
        NoisyGreedyInsertionRepair(extra_unserved_limit=eul, max_insertions=None, min_delta_score=0.0, noise=0.10, initial_weight=1.0),
        ScarceSkillFirstRepair(extra_unserved_limit=eul, min_delta_score=0.0, initial_weight=1.0),
        LastRemovedFirstInsertedRepair(extra_unserved_limit=eul, min_delta_score=0.0, initial_weight=1.0),
        RegretKInsertionRepair(extra_unserved_limit=eul, max_insertions=None, min_delta_score=0.0, k=3, initial_weight=1.0),
        RandomPositionInsertionRepair(extra_unserved_limit=eul, initial_weight=1.0),
        ShawInsertionRepair(extra_unserved_limit=eul, min_delta_score=0.0, p_determinism=4.0, w_distance=0.5, w_time=0.25, w_skill=0.25, initial_weight=1.0),
    ]

    return destroy, repair


def run_single(inst, destroy_ops, repair_ops, seed, runtime):
    """Fuehrt einen einzelnen ALNS-Lauf durch und gibt den besten Profit zurueck."""
    from dataclasses import replace
    params = replace(PARAMS, random_seed=seed)

    start = greedy_initial_solution(inst, strategy="multi_start", verbose=False)

    result = run_alns(
        runtime=runtime,
        inst=inst,
        start_solution=start,
        sa=SA,
        params=params,
        penalties=PENALTIES,
        destroy_operators=destroy_ops,
        repair_operators=repair_ops,
    )

    if result.best_evaluation.feasible:
        return result.best_evaluation.profit
    return 0


# -----------------------------------------------------------------------
# Hauptschleife
# -----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ABLATIONSSTUDIE")
    print(f"Instanzen: {len(INSTANCES)}  |  Seeds: {SEEDS}  |  Runtime: {RUNTIME_PER_RUN}s/Lauf")
    print("=" * 70)

    # Instanzen einlesen
    instances = []
    for name in INSTANCES:
        path = str(DATASET_DIR / name)
        inst = read_instance(path)
        instances.append((name, inst))
        print(f"Geladen: {name}  ({inst.num_customers} Kunden)")

    rows = []

    # ------------------------------------------------------------------ #
    # Phase 1: Baseline (alle Operatoren)
    # ------------------------------------------------------------------ #
    print("\n--- BASELINE (alle Operatoren) ---")
    baselines = {}   # (instance_name, seed) -> profit

    for inst_name, inst in instances:
        for seed in SEEDS:
            d_ops, r_ops = make_all_operators(inst)
            profit = run_single(inst, d_ops, r_ops, seed, RUNTIME_PER_RUN)
            baselines[(inst_name, seed)] = profit
            print(f"  {inst_name[:30]:30s}  seed={seed}  profit={profit:6d}")
            rows.append({
                "config": "baseline",
                "removed_type": "-",
                "removed_operator": "-",
                "instance": inst_name,
                "seed": seed,
                "profit": profit,
                "vs_baseline_pct": 0.0,
            })

    # ------------------------------------------------------------------ #
    # Phase 2: Je einen Operator entfernen
    # ------------------------------------------------------------------ #
    # Wir testen alle Destroy- und Repair-Operatoren einzeln
    ref_destroy, ref_repair = make_all_operators(instances[0][1])
    destroy_names = [op.name for op in ref_destroy]
    repair_names  = [op.name for op in ref_repair]

    total_configs = len(destroy_names) + len(repair_names)
    done = 0

    # Destroy-Ablation
    for skip_idx, skip_name in enumerate(destroy_names):
        done += 1
        print(f"\n--- DESTROY ablation [{done}/{total_configs}]: ohne '{skip_name}' ---")

        for inst_name, inst in instances:
            d_ops, r_ops = make_all_operators(inst)
            # Diesen einen Operator rausnehmen
            d_ops_reduced = [op for op in d_ops if op.name != skip_name]

            for seed in SEEDS:
                profit = run_single(inst, d_ops_reduced, r_ops, seed, RUNTIME_PER_RUN)
                baseline = baselines[(inst_name, seed)]
                delta = 100.0 * (profit - baseline) / max(baseline, 1)
                print(f"  {inst_name[:30]:30s}  seed={seed}  profit={profit:6d}  vs_baseline={delta:+.1f}%")
                rows.append({
                    "config": f"no_{skip_name}",
                    "removed_type": "destroy",
                    "removed_operator": skip_name,
                    "instance": inst_name,
                    "seed": seed,
                    "profit": profit,
                    "vs_baseline_pct": round(delta, 2),
                })

    # Repair-Ablation
    for skip_idx, skip_name in enumerate(repair_names):
        done += 1
        print(f"\n--- REPAIR ablation [{done}/{total_configs}]: ohne '{skip_name}' ---")

        for inst_name, inst in instances:
            d_ops, r_ops = make_all_operators(inst)
            r_ops_reduced = [op for op in r_ops if op.name != skip_name]

            for seed in SEEDS:
                profit = run_single(inst, d_ops, r_ops_reduced, seed, RUNTIME_PER_RUN)
                baseline = baselines[(inst_name, seed)]
                delta = 100.0 * (profit - baseline) / max(baseline, 1)
                print(f"  {inst_name[:30]:30s}  seed={seed}  profit={profit:6d}  vs_baseline={delta:+.1f}%")
                rows.append({
                    "config": f"no_{skip_name}",
                    "removed_type": "repair",
                    "removed_operator": skip_name,
                    "instance": inst_name,
                    "seed": seed,
                    "profit": profit,
                    "vs_baseline_pct": round(delta, 2),
                })

    # ------------------------------------------------------------------ #
    # Ergebnisse speichern
    # ------------------------------------------------------------------ #
    results_path = OUT_DIR / "ablation_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDetails gespeichert: {results_path}")

    # Summary: Durchschnittliche Auswirkung je Operator
    summary = {}
    for row in rows:
        op = row["removed_operator"]
        if op == "-":
            continue
        if op not in summary:
            summary[op] = {"type": row["removed_type"], "deltas": []}
        summary[op]["deltas"].append(row["vs_baseline_pct"])

    summary_rows = []
    for op_name, data in summary.items():
        avg_delta = sum(data["deltas"]) / len(data["deltas"])
        min_delta = min(data["deltas"])
        summary_rows.append({
            "operator": op_name,
            "type": data["type"],
            "avg_impact_pct": round(avg_delta, 2),
            "worst_impact_pct": round(min_delta, 2),
        })
    summary_rows.sort(key=lambda x: x["avg_impact_pct"])

    summary_path = OUT_DIR / "ablation_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    # Ausgabe Zusammenfassung
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG — Einfluss je Operator (negativer Wert = schadet mehr)")
    print("=" * 70)
    print(f"{'Operator':<35} {'Typ':<8} {'Ø Auswirkung':>12} {'Schlimmster Fall':>16}")
    print("-" * 70)
    for r in summary_rows:
        print(f"{r['operator']:<35} {r['type']:<8} {r['avg_impact_pct']:>+11.1f}%  {r['worst_impact_pct']:>+14.1f}%")

    print(f"\nSummary gespeichert: {summary_path}")


if __name__ == "__main__":
    main()
