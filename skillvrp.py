#!/usr/bin/env python3
"""skillvrp.py -- entry point for the Prize-Collecting Skill-VRP metaheuristic.

Usage:
    python3 skillvrp.py <path to instance> <timeout (s)>

Pipeline:
  1. Parse the instance (instance_reader.Instance).
  2. Multi-start greedy insertion -> initial feasible solution.
  3. ALNS (destroy/repair with adaptive operator weights, simulated-annealing
     acceptance, random restarts) until the time budget is exhausted.
  4. Print the best feasible solution in the checker format to stdout.

The program always prints a feasible solution: the greedy start is feasible by
construction, the ALNS only replaces the incumbent best with strictly better
*feasible* candidates, and if anything at all goes wrong we fall back to the
empty plan (profit 0), which is always feasible.
"""

from __future__ import annotations

import argparse
import sys
import time

from instance_reader import read_instance
from initial_solution import greedy_initial_solution
from solution import Solution
from alns import (
    run_alns,
    SAParams,
    ALNSParams,
    PenaltyParams,
    RandomRemoval,
    RelatedRemoval,
    WorstDensityRemoval,
    WorstDetourRemoval,
    SkillScarcityRemoval,
    SequentialCheapestInsertionRepair,
    GreedyBestInsertionRepair,
    Regret2InsertionRepair,
)


def initial_temperature(inst) -> float:
    """T0 scaled to the magnitude of a single move: destroy/repair deltas are
    sums of a few customer profits (10..100), independent of instance size.
    T0 = 1.5 x mean profit accepts a typical single-customer loss with
    probability ~0.5 at the start of the (time-based) cooling schedule."""
    profits = [inst.profit[c] for c in inst.customers]
    mean_profit = sum(profits) / max(1, len(profits))
    return max(10.0, 1.5 * mean_profit)


def build_operators(inst):
    """Destroy sizes are part of the operator identity: each geometric
    operator exists in a small (polish) and a large (restructure) variant, so
    the adaptive weights can learn the right destroy-size mix per instance."""
    # Noise amplitude for the noised insertion, scaled to the instance's
    # distance magnitude (Ropke & Pisinger 2006 use a fraction of max dist).
    max_distance = max(map(max, inst.distance))
    noise_amp = 0.1 * max_distance
    small = (0.03, 0.12)
    large = (0.12, 0.28)

    destroy_operators = []

    for label, fraction, max_remove in (("small", small, 40), ("large", large, 70)):
        variants = [
            RandomRemoval(
                fraction=fraction,
                min_remove=1,
                max_remove=max_remove,
                initial_weight=1.0,
            ),
            RelatedRemoval(
                fraction=fraction,
                min_remove=1,
                max_remove=max_remove,
                bias=3.0,
                initial_weight=1.0,
            ),
            WorstDetourRemoval(
                fraction=fraction,
                min_remove=1,
                max_remove=max_remove,
                bias=3.0,
                initial_weight=1.0,
            ),
        ]
        for op in variants:
            op.name += "_" + label
        destroy_operators.extend(variants)

    destroy_operators.extend([
        WorstDensityRemoval(
            fraction=(0.05, 0.18),
            min_remove=1,
            max_remove=50,
            noise=0.05,
            initial_weight=1.0,
        ),
        SkillScarcityRemoval(
            fraction=(0.05, 0.18),
            min_remove=1,
            max_remove=50,
            noise=0.05,
            initial_weight=1.0,
        ),
    ])

    repair_operators = [
        GreedyBestInsertionRepair(
            extra_unserved_limit=100,
            max_insertions=None,
            min_delta_score=0.0,
            initial_weight=1.0,
        ),
        Regret2InsertionRepair(
            extra_unserved_limit=100,
            max_insertions=None,
            min_delta_score=0.0,
            initial_weight=1.0,
        ),
        SequentialCheapestInsertionRepair(
            extra_unserved_limit=100,
            order="profit",
            initial_weight=1.0,
        ),
        SequentialCheapestInsertionRepair(
            extra_unserved_limit=100,
            order="random",
            initial_weight=1.0,
        ),
        SequentialCheapestInsertionRepair(
            extra_unserved_limit=100,
            order="profit",
            initial_weight=1.0,
            noise_amp=noise_amp,
        ),
    ]

    return destroy_operators, repair_operators


def print_empty_solution(num_vehicles: int) -> None:
    print("###RESULT: Feasible")
    print("###OBJECTIVE: 0")
    for b in range(num_vehicles):
        print(f"###VEHICLE {b + 1}: 0 0")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prize-Collecting Skill-VRP metaheuristic (ALNS)")
    parser.add_argument("instance", help="path to the instance file")
    parser.add_argument("timeout", type=float, help="time limit in seconds")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--verbose", action="store_true", help="log ALNS progress to stderr")
    args = parser.parse_args()

    t_start = time.perf_counter()
    # Reserve a slice of the budget for parsing, the greedy start and output,
    # so the program never overruns the limit.
    reserve = 0.3 + min(1.0, 0.02 * args.timeout)

    inst = read_instance(args.instance)

    deadline = t_start + max(0.05, args.timeout - reserve)

    try:
        start_solution = greedy_initial_solution(
            inst, strategy="multi_start", deadline=deadline
        )
    except Exception:
        # Extremely defensive: an empty plan is always feasible.
        print_empty_solution(inst.num_vehicles)
        return

    best_solution: Solution = start_solution

    remaining = args.timeout - reserve - (time.perf_counter() - t_start)

    if remaining > 0:
        sa = SAParams(
            initial_temperature=initial_temperature(inst),
            min_temperature=0.4,
            cooling_rate=0.9995,  # unused: run_alns cools on wall-clock time
            reheat_factor=0.75,
        )

        penalties = PenaltyParams(
            time_window_penalty=20.0,
            shift_penalty=20.0,
            skill_penalty=10000.0,
        )

        params = ALNSParams(
            random_seed=args.seed,
            segment_length=100,
            no_accept_limit=500,
            reaction_factor=0.20,
            min_operator_weight=0.05,
            score_global_best=25.0,
            score_better_current=10.0,
            score_accepted=3.0,
            score_rejected=0.0,
            time_cost_alpha=0.5,
            time_scale_seconds=0.01,
            verbose=False,
            return_to_best_limit=2000,
            polish_interval_seconds=0.5,
        )

        destroy_operators, repair_operators = build_operators(inst)

        try:
            result = run_alns(
                runtime=remaining,
                inst=inst,
                start_solution=start_solution,
                sa=sa,
                params=params,
                penalties=penalties,
                destroy_operators=destroy_operators,
                repair_operators=repair_operators,
            )

            if (
                result.best_evaluation.feasible
                and result.best_evaluation.profit >= best_solution.objective
            ):
                best_solution = result.best_solution

            if args.verbose:
                print(
                    f"[skillvrp] iterations={result.iterations} "
                    f"restarts={result.restarts} "
                    f"best={result.best_evaluation.profit} "
                    f"feasible={result.best_evaluation.feasible}",
                    file=sys.stderr,
                )
        except Exception as exc:  # never crash: keep the greedy solution
            print(f"[skillvrp] ALNS aborted: {exc!r}", file=sys.stderr)

    best_solution.print_for_checker()


if __name__ == "__main__":
    main()
