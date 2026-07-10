"""Benchmark-Instrumentierung fuer alns.py.

alns.py bleibt komplett unveraendert: dieses Modul importiert alle Originale
und stellt `run_alns_tracked` bereit -- eine 1:1-Kopie der Hauptschleife aus
`alns.run_alns`, erweitert um Tracking-Hooks. Jede Abweichung vom Original
ist mit `# TRACKING` markiert; alles andere ist identisch uebernommen.

Getrackt wird:
  * der Gewichtsverlauf jedes Operators nach jedem Segment-Update
    (inkl. Segment 0 = Initialgewichte und dem finalen Update nach der
    Schleife), zusammen mit der Nutzungshaeufigkeit im Segment
  * jede neue globale Bestloesung mit Iteration, verstrichener Zeit in
    Sekunden und Zielfunktionswert (Profit)

Export als zwei CSVs (operator_weights.csv, best_solution.csv), aus denen
sich die Report-Plots direkt erzeugen lassen.

Typische Verwendung (im Notebook / run_test):

    from alns_tracking import run_alns_tracked, make_tracking_params

    params = make_tracking_params(params, min_operator_weight=1e-3,
                                  segment_length=200)

    result, tracker = run_alns_tracked(
        runtime=60.0, inst=inst, start_solution=start,
        sa=sa, params=params, penalties=penalties,
        destroy_operators=destroys, repair_operators=repairs,
        csv_dir="benchmark_results", csv_prefix="instanz01_",
    )

`result` ist das gewohnte ALNSResult; die CSVs liegen danach unter
benchmark_results/instanz01_operator_weights.csv bzw. ..._best_solution.csv.
"""

from __future__ import annotations

import csv
import os
import random
import time
from dataclasses import dataclass, field, replace

from solution import Solution
from alns import (
    ALNSParams,
    ALNSResult,
    DestroyOperator,
    PenaltyParams,
    RepairOperator,
    SAParams,
    accept_sa,
    copy_solution_with_state,
    evaluation_from_solution,
    initialize_solution_state,
    operator_summary,
    polish_and_fill,
    random_start_solution,
    select_operator,
    update_operator_weights,
)


def make_tracking_params(
    params: ALNSParams,
    min_operator_weight: float = 1e-3,
    segment_length: int | None = None,
) -> ALNSParams:
    """Kopie der ALNSParams mit benchmark-freundlichen Werten.

    min_operator_weight sehr niedrig, damit sich schwache Operatoren im
    Gewichtsverlauf sichtbar von den starken absetzen; segment_length optional
    hoeher, damit die Segment-Mittelwerte weniger verrauscht sind. Das
    Original-Objekt bleibt unveraendert (dataclasses.replace).
    """
    if segment_length is None:
        segment_length = params.segment_length

    return replace(
        params,
        min_operator_weight=min_operator_weight,
        segment_length=segment_length,
    )


@dataclass
class ALNSTracker:
    """Sammelt Gewichtsverlaeufe und Bestloesungs-Historie eines Laufs."""

    # rows: {segment, operator_type, operator_name, weight, segment_uses}
    weight_history: list[dict] = field(default_factory=list)
    # rows: {iteration, elapsed_seconds, objective_value}
    best_history: list[dict] = field(default_factory=list)

    def record_weights(
        self,
        segment: int,
        destroy_operators: list[DestroyOperator],
        repair_operators: list[RepairOperator],
        uses: dict[str, int] | None = None,
    ) -> None:
        for op_type, operators in (
            ("destroy", destroy_operators),
            ("repair", repair_operators),
        ):
            for op in operators:
                self.weight_history.append(
                    {
                        "segment": segment,
                        "operator_type": op_type,
                        "operator_name": op.name,
                        "weight": op.weight,
                        "segment_uses": uses.get(op.name, 0) if uses else 0,
                    }
                )

    def record_best(
        self,
        iteration: int,
        elapsed_seconds: float,
        objective_value: float,
    ) -> None:
        self.best_history.append(
            {
                "iteration": iteration,
                "elapsed_seconds": elapsed_seconds,
                "objective_value": objective_value,
            }
        )

    def export_csv(self, out_dir: str = ".", prefix: str = "") -> tuple[str, str]:
        """Schreibt operator_weights.csv und best_solution.csv nach out_dir.

        Gibt die beiden Dateipfade zurueck. `prefix` erlaubt getrennte
        Dateien pro Instanz/Seed (z. B. prefix="c101_seed3_").
        """
        os.makedirs(out_dir, exist_ok=True)

        weights_path = os.path.join(out_dir, f"{prefix}operator_weights.csv")
        best_path = os.path.join(out_dir, f"{prefix}best_solution.csv")

        with open(weights_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "segment",
                    "operator_type",
                    "operator_name",
                    "weight",
                    "segment_uses",
                ],
            )
            writer.writeheader()
            writer.writerows(self.weight_history)

        with open(best_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["iteration", "elapsed_seconds", "objective_value"],
            )
            writer.writeheader()
            writer.writerows(self.best_history)

        return weights_path, best_path


def _segment_uses_snapshot(
    destroy_operators: list[DestroyOperator],
    repair_operators: list[RepairOperator],
) -> dict[str, int]:
    """Nutzungszaehler VOR update_operator_weights sichern (das Update setzt
    segment_uses zurueck)."""
    uses: dict[str, int] = {}
    for op in destroy_operators:
        uses[op.name] = op.segment_uses
    for op in repair_operators:
        uses[op.name] = op.segment_uses
    return uses


def run_alns_tracked(
    runtime: float,
    inst,
    start_solution: Solution,
    sa: SAParams,
    params: ALNSParams,
    penalties: PenaltyParams,
    destroy_operators: list[DestroyOperator],
    repair_operators: list[RepairOperator],
    tracker: ALNSTracker | None = None,
    csv_dir: str | None = None,
    csv_prefix: str = "",
) -> tuple[ALNSResult, ALNSTracker]:
    """Identisch zu alns.run_alns, plus Tracking (Zeilen mit # TRACKING).

    Gibt (ALNSResult, ALNSTracker) zurueck. Ist csv_dir gesetzt, werden die
    CSVs am Ende automatisch dorthin exportiert.
    """
    if tracker is None:  # TRACKING
        tracker = ALNSTracker()  # TRACKING

    rng = random.Random(params.random_seed)

    current_solution = start_solution.copy()
    current_eval = initialize_solution_state(inst, current_solution, penalties)

    best_solution = copy_solution_with_state(current_solution)
    best_eval = current_eval

    # Time-based cooling: temperature decays exponentially from the current
    # curve start T0 to min_temperature over the remaining wall-clock time.
    # This makes the schedule independent of the iteration throughput, which
    # varies by two orders of magnitude between the smallest and largest
    # instances.
    temperature = sa.initial_temperature
    curve_t0 = sa.initial_temperature
    curve_start = time.perf_counter()

    iterations = 0
    restarts = 0
    no_accept_counter = 0
    last_polish = 0.0
    start_time = time.perf_counter()
    deadline = start_time + runtime

    segment_index = 0  # TRACKING
    tracker.record_weights(0, destroy_operators, repair_operators)  # TRACKING: Initialgewichte
    tracker.record_best(0, 0.0, best_eval.profit)  # TRACKING: Startloesung

    while time.perf_counter() < deadline:
        iterations += 1

        destroy_op = select_operator(destroy_operators, rng)

        t0 = time.perf_counter()
        destroy_result = destroy_op.apply(
            inst=inst,
            solution=current_solution,
            rng=rng,
            penalties=penalties,
        )
        destroy_time = time.perf_counter() - t0

        compatible_repairs = [
            repair_op
            for repair_op in repair_operators
            if repair_op.compatible_with(destroy_result)
        ]

        repair_op = select_operator(compatible_repairs, rng)

        t0 = time.perf_counter()
        candidate_solution = repair_op.apply(
            inst=inst,
            destroy_result=destroy_result,
            rng=rng,
            penalties=penalties,
            deadline=deadline,
        )
        repair_time = time.perf_counter() - t0

        candidate_eval = evaluation_from_solution(inst, candidate_solution)

        accepted = accept_sa(
            current_score=current_eval.score,
            candidate_score=candidate_eval.score,
            temperature=temperature,
            rng=rng,
        )

        new_global_best = (
            candidate_eval.feasible
            and candidate_eval.profit > best_eval.profit
        )

        if new_global_best:
            operator_score = params.score_global_best
        elif accepted and candidate_eval.score > current_eval.score:
            operator_score = params.score_better_current
        elif accepted:
            operator_score = params.score_accepted
        else:
            operator_score = params.score_rejected

        if new_global_best:
            best_solution = copy_solution_with_state(candidate_solution)
            best_eval = candidate_eval

            tracker.record_best(  # TRACKING
                iterations, time.perf_counter() - start_time, best_eval.profit
            )

            now = time.perf_counter()
            if (
                params.polish_interval_seconds > 0
                and now - last_polish >= params.polish_interval_seconds
            ):
                last_polish = now
                polished = polish_and_fill(
                    inst, best_solution, penalties, rng, deadline=deadline
                )
                polished_eval = evaluation_from_solution(inst, polished)

                if (
                    polished_eval.feasible
                    and polished_eval.profit >= best_eval.profit
                ):
                    if polished_eval.profit > best_eval.profit:  # TRACKING
                        tracker.record_best(  # TRACKING
                            iterations,
                            time.perf_counter() - start_time,
                            polished_eval.profit,
                        )
                    best_solution = polished
                    best_eval = polished_eval
                    # continue the search from the compressed solution
                    candidate_solution = copy_solution_with_state(polished)
                    candidate_eval = polished_eval

        if accepted:
            current_solution = candidate_solution
            current_eval = candidate_eval
            no_accept_counter = 0
        else:
            no_accept_counter += 1

        destroy_op.observe(operator_score, destroy_time)
        repair_op.observe(operator_score, repair_time)

        if iterations % params.segment_length == 0:
            uses = _segment_uses_snapshot(destroy_operators, repair_operators)  # TRACKING

            update_operator_weights(
                destroy_operators=destroy_operators,
                repair_operators=repair_operators,
                params=params,
            )

            segment_index += 1  # TRACKING
            tracker.record_weights(  # TRACKING
                segment_index, destroy_operators, repair_operators, uses
            )

            if params.verbose:
                print(
                    f"iter={iterations:>7} | "
                    f"best={best_eval.profit:>8} | "
                    f"current_score={current_eval.score:>10.2f} | "
                    f"current_feasible={current_eval.feasible} | "
                    f"T={temperature:>8.4f}"
                )

        now = time.perf_counter()
        span = max(1e-9, deadline - curve_start)
        frac = min(1.0, (now - curve_start) / span)
        temperature = max(
            sa.min_temperature,
            curve_t0 * (sa.min_temperature / curve_t0) ** frac,
        )

        if no_accept_counter >= params.no_accept_limit:
            current_solution = random_start_solution(inst, rng, penalties, deadline=deadline)
            current_eval = evaluation_from_solution(inst, current_solution)

            curve_t0 = max(temperature, sa.initial_temperature * sa.reheat_factor)
            curve_start = time.perf_counter()
            temperature = curve_t0

            no_accept_counter = 0
            restarts += 1

            if params.verbose:
                print(
                    f"random_restart={restarts} at iter={iterations} | "
                    f"new current profit={current_eval.profit} | "
                    f"best profit={best_eval.profit} | "
                    f"T={temperature:.4f}"
                )

    uses = _segment_uses_snapshot(destroy_operators, repair_operators)  # TRACKING

    update_operator_weights(
        destroy_operators=destroy_operators,
        repair_operators=repair_operators,
        params=params,
    )

    segment_index += 1  # TRACKING: finales Update nach der Schleife
    tracker.record_weights(segment_index, destroy_operators, repair_operators, uses)  # TRACKING

    result = ALNSResult(
        best_solution=best_solution,
        best_evaluation=best_eval,
        current_solution=current_solution,
        current_evaluation=current_eval,
        iterations=iterations,
        runtime_seconds=time.perf_counter() - start_time,
        restarts=restarts,
        destroy_summary=operator_summary(destroy_operators),
        repair_summary=operator_summary(repair_operators),
    )

    if csv_dir is not None:  # TRACKING
        tracker.export_csv(csv_dir, csv_prefix)  # TRACKING

    return result, tracker
