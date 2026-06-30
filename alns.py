from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import random
import time

from solution import Solution
from initial_solution import find_best_insertion, insert_customer as greedy_insert_customer



# Parameterclasses


@dataclass(slots=True)
class SAParams:
    initial_temperature: float
    min_temperature: float
    cooling_rate: float
    reheat_factor: float


@dataclass(slots=True)
class PenaltyParams:
    time_window_penalty: float
    shift_penalty: float
    skill_penalty: float


@dataclass(slots=True)
class ALNSParams:
    random_seed: int
    segment_length: int
    no_accept_limit: int

    reaction_factor: float
    min_operator_weight: float

    score_global_best: float
    score_better_current: float
    score_accepted: float
    score_rejected: float

    time_cost_alpha: float
    time_scale_seconds: float

    verbose: bool



# Cached evaluation

@dataclass(slots=True)
class RouteCache:
    arrival: list[int]
    service_start: list[int]
    departure: list[int]
    prefix_penalty: list[float]
    end_time: int
    penalty: float


@dataclass(slots=True)
class SolutionEvaluation:
    feasible: bool
    profit: int
    penalty: float
    score: float
    served_customers: int


def build_route_cache(inst, vehicle: int, route: list[int], penalties: PenaltyParams) -> RouteCache:
    arrival_list = []
    service_start_list = []
    departure_list = []
    prefix_penalty = [0.0]
    time_now = inst.vehicle_start[vehicle]
    previous = 0
    penalty = 0.0
    vehicle_skills = inst.vehicle_skills[vehicle]
    distance = inst.distance


    for customer in route:
        if not inst.required_skills[customer].issubset(vehicle_skills):
            penalty += penalties.skill_penalty

        arrival = time_now + distance[previous][customer]

        if arrival > inst.due[customer]:
            penalty += penalties.time_window_penalty * (arrival - inst.due[customer])

        service_start = max(arrival, inst.ready[customer])
        departure = service_start + inst.service[customer]

        arrival_list.append(arrival)
        service_start_list.append(service_start)
        departure_list.append(departure)
        prefix_penalty.append(penalty)

        time_now = departure
        previous = customer

    end_time = time_now + distance[previous][0]

    if end_time > inst.vehicle_end[vehicle]:
        penalty += penalties.shift_penalty * (end_time - inst.vehicle_end[vehicle])


    return RouteCache(
        arrival=arrival_list,
        service_start=service_start_list,
        departure=departure_list,
        prefix_penalty=prefix_penalty,
        end_time=end_time,
        penalty=penalty,
    )


def initialize_solution_state(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
) -> SolutionEvaluation:

    route_cache = []
    total_penalty = 0.0

    for vehicle, route in enumerate(solution.routes):
        cache = build_route_cache(inst, vehicle, route, penalties)
        route_cache.append(cache)
        total_penalty += cache.penalty

    solution.route_cache = route_cache
    solution.total_penalty = total_penalty
    solution.score = solution.objective - total_penalty
    solution.feasible = total_penalty == 0.0

    return evaluation_from_solution(inst, solution)


def evaluation_from_solution(inst, solution: Solution) -> SolutionEvaluation:
    return SolutionEvaluation(
        feasible=solution.feasible,
        profit=solution.objective,
        penalty=solution.total_penalty,
        score=solution.score,
        served_customers=inst.num_customers - len(solution.unserved),
    )


def copy_solution_with_state(solution: Solution) -> Solution:
    new_solution = solution.copy()
    if hasattr(solution, "route_cache"):
        new_solution.route_cache = list(solution.route_cache)
        new_solution.total_penalty = solution.total_penalty
        new_solution.score = solution.score
        new_solution.feasible = solution.feasible

    return new_solution


def update_route_cache(
    inst,
    solution: Solution,
    vehicle: int,
    penalties: PenaltyParams,
) -> None:
    old_penalty = solution.route_cache[vehicle].penalty
    new_cache = build_route_cache(inst, vehicle, solution.routes[vehicle], penalties)
    solution.route_cache[vehicle] = new_cache
    solution.total_penalty += new_cache.penalty - old_penalty
    solution.score = solution.objective - solution.total_penalty
    solution.feasible = solution.total_penalty == 0.0


def full_evaluate_solution(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
) -> SolutionEvaluation:

    penalty = 0.0
    seen = set()

    for vehicle, route in enumerate(solution.routes):
        cache = build_route_cache(inst, vehicle, route, penalties)
        penalty += cache.penalty

        for customer in route:
            if customer < 1 or customer > inst.num_customers:
                penalty += penalties.skill_penalty
            elif customer in seen:
                penalty += penalties.skill_penalty
            else:
                seen.add(customer)

    score = solution.objective - penalty


    return SolutionEvaluation(
        feasible=penalty == 0.0,
        profit=solution.objective,
        penalty=penalty,
        score=score,
        served_customers=len(seen),
    )



# Solution operations

def remove_customer(solution: Solution, inst, customer: int) -> tuple[int, int]:
    vehicle = solution.customer_to_vehicle[customer]
    position = solution.routes[vehicle].index(customer)
    solution.routes[vehicle].pop(position)
    solution.unserved.add(customer)
    del solution.customer_to_vehicle[customer]
    solution.objective -= inst.profit[customer]
    return vehicle, position


def apply_insertion(
    solution: Solution,
    inst,
    penalties: PenaltyParams,
    move: "InsertionMove",
) -> None:
    solution.routes[move.vehicle].insert(move.position, move.customer)
    solution.unserved.remove(move.customer)
    solution.customer_to_vehicle[move.customer] = move.vehicle
    solution.objective += inst.profit[move.customer]
    update_route_cache(inst, solution, move.vehicle, penalties)



def served_customers(solution: Solution) -> list[int]:
    return [c for route in solution.routes for c in route]


# Random restart solution ()

def random_start_solution(
    inst,
    rng: random.Random,
    penalties: PenaltyParams,
) -> Solution:
    solution = Solution.empty(inst)

    customers = [
        c for c in inst.customers
        if inst.solo_feasible_vehicles[c]
    ]

    rng.shuffle(customers)

    for customer in customers:
        insertion = find_best_insertion(inst, solution.routes, customer)

        if insertion is not None:
            greedy_insert_customer(solution, inst, insertion)

    initialize_solution_state(inst, solution, penalties)
    return solution



# Destroy / repair data

@dataclass(slots=True)
class DestroyResult:
    partial_solution: Solution
    removed_customers: list[int]
    origin_vehicle: dict[int, int]
    origin_position: dict[int, int]
    affected_vehicles: set[int]
    tags: set[str]


@dataclass(slots=True)
class InsertionMove:
    customer: int
    vehicle: int
    position: int
    delta_score: float
    travel_delta: int
    new_route_penalty: float



# Operator base classes


class ALNSOperator:
    def __init__(self, name: str, initial_weight: float):
        self.name = name
        self.weight = initial_weight
        self.total_uses = 0
        self.total_time = 0.0
        self.segment_uses = 0
        self.segment_score = 0.0
        self.segment_time = 0.0


    def observe(self, score: float, elapsed: float) -> None:
        self.total_uses += 1
        self.total_time += elapsed
        self.segment_uses += 1
        self.segment_score += score
        self.segment_time += elapsed

    def update_weight(self, params: ALNSParams) -> None:
        if self.segment_uses == 0:
            return

        avg_score = self.segment_score / self.segment_uses
        avg_time = self.segment_time / self.segment_uses

        time_factor = 1.0

        if params.time_cost_alpha > 0.0:
            time_factor += (
                avg_time / max(1e-12, params.time_scale_seconds)
            ) ** params.time_cost_alpha

        utility = avg_score / time_factor

        self.weight = (
            (1.0 - params.reaction_factor) * self.weight
            + params.reaction_factor * utility
        )

        self.weight = max(params.min_operator_weight, self.weight)
        self.segment_uses = 0
        self.segment_score = 0.0
        self.segment_time = 0.0


class DestroyOperator(ALNSOperator):
    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        raise NotImplementedError


class RepairOperator(ALNSOperator):
    requires: set[str]

    def compatible_with(self, destroy_result: DestroyResult) -> bool:
        return self.requires.issubset(destroy_result.tags)

    def apply(
        self,
        inst,
        destroy_result: DestroyResult,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> Solution:
        raise NotImplementedError



# Destroy helpers


def number_to_remove(n_served: int, fraction: float, min_remove: int, max_remove: int) -> int:
    q = round(fraction * n_served)
    q = max(min_remove, q)
    q = min(max_remove, q)
    q = min(n_served, q)
    return q


def build_destroy_result(
    inst,
    solution: Solution,
    removed: list[int],
    tag: str,
    penalties: PenaltyParams,
) -> DestroyResult:
    new_solution = copy_solution_with_state(solution)
    origin_vehicle = {}
    origin_position = {}
    affected_vehicles = set()

    for customer in removed:
        vehicle, position = remove_customer(new_solution, inst, customer)
        origin_vehicle[customer] = vehicle
        origin_position[customer] = position
        affected_vehicles.add(vehicle)

    for vehicle in affected_vehicles:
        update_route_cache(inst, new_solution, vehicle, penalties)

    return DestroyResult(
        partial_solution=new_solution,
        removed_customers=removed,
        origin_vehicle=origin_vehicle,
        origin_position=origin_position,
        affected_vehicles=affected_vehicles,
        tags={"customer_pool", "origin_info", "affected_routes", tag},
    )



###########################################

# Destroy operators

class RandomRemoval(DestroyOperator):
    def __init__(
        self,
        fraction: float,
        min_remove: int,
        max_remove: int,
        initial_weight: float,
    ):
        super().__init__("random_removal", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove

    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        served = served_customers(solution)

        # Fallback: should not normally trigger, an empty solution has nothing to destroy.
        if not served:
            return build_destroy_result(inst, solution, [], "random_removal", penalties)

        q = number_to_remove(
            n_served=len(served),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
        )

        removed = rng.sample(served, q)

        return build_destroy_result(inst, solution, removed, "random_removal", penalties)






class WorstDensityRemoval(DestroyOperator):
    def __init__(
        self,
        fraction: float,
        min_remove: int,
        max_remove: int,
        noise: float,
        initial_weight: float,
    ):
        super().__init__("worst_density_removal", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove
        self.noise = noise

    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        served = served_customers(solution)

        if not served:
            return build_destroy_result(inst, solution, [], "worst_density_removal", penalties)

        q = number_to_remove(
            n_served=len(served),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
        )

        ranked = sorted(
            served,
            key=lambda c: inst.profit_density_lb[c] + self.noise * rng.random(),
        )

        removed = ranked[:q]

        return build_destroy_result(inst, solution, removed, "worst_density_removal", penalties)


class SkillScarcityRemoval(DestroyOperator):
    """Removes served customers that are occupying a courier whose skill set
    is in high demand among the currently-unserved customers, weighted by
    how little profit they themselves contribute.

    RandomRemoval and WorstDensityRemoval only ever reason about travel
    distance/profit density; neither looks at skills at all, even though
    skill compatibility is the defining constraint of this problem variant.
    The idea here: if a courier with a rare skill set is currently tied up
    serving a low-profit customer while several unserved customers would
    also need exactly that skill set, that courier's time is a contested
    resource. Removing the low-profit occupant gives the repair step a
    chance to reassign that capacity to a more profitable customer who
    needs the same scarce skill -- a swap that distance-based operators
    cannot "see" because it has nothing to do with route geometry.
    """

    def __init__(
        self,
        fraction: float,
        min_remove: int,
        max_remove: int,
        noise: float,
        initial_weight: float,
    ):
        super().__init__("skill_scarcity_removal", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove
        self.noise = noise

    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        served = served_customers(solution)

        if not served:
            return build_destroy_result(inst, solution, [], "skill_scarcity_removal", penalties)

        q = number_to_remove(
            n_served=len(served),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
        )

        # demand_per_vehicle[b] = how many currently-unserved customers could
        # courier b serve, skill-wise. High demand -> b's time is contested.
        demand_per_vehicle = [0] * inst.num_vehicles
        for c in solution.unserved:
            for vehicle in inst.skill_feasible_vehicles[c]:
                demand_per_vehicle[vehicle] += 1

        def scarcity_score(c: int) -> float:
            vehicle = solution.customer_to_vehicle[c]
            demand = demand_per_vehicle[vehicle]
            return inst.profit[c] / (1.0 + demand)

        # Low score first: low profit on a high-demand courier is removed first.
        ranked = sorted(
            served,
            key=lambda c: scarcity_score(c) + self.noise * rng.random(),
        )

        removed = ranked[:q]

        return build_destroy_result(inst, solution, removed, "skill_scarcity_removal", penalties)



# Segment-cache insertion evaluation


def insertion_penalty_from_cache(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
    customer: int,
    vehicle: int,
    position: int,
) -> tuple[float, int]:

    route = solution.routes[vehicle]
    cache = solution.route_cache[vehicle]
    distance = inst.distance
    vehicle_skills = inst.vehicle_skills[vehicle]

    if position == 0:
        previous = 0
        time_now = inst.vehicle_start[vehicle]
        penalty = 0.0
    else:
        previous = route[position - 1]
        time_now = cache.departure[position - 1]
        penalty = cache.prefix_penalty[position]

    if not inst.required_skills[customer].issubset(vehicle_skills):
        penalty += penalties.skill_penalty

    arrival = time_now + distance[previous][customer]

    if arrival > inst.due[customer]:
        penalty += penalties.time_window_penalty * (arrival - inst.due[customer])

    time_now = max(arrival, inst.ready[customer]) + inst.service[customer]
    previous = customer

    for suffix_customer in route[position:]:
        if not inst.required_skills[suffix_customer].issubset(vehicle_skills):
            penalty += penalties.skill_penalty

        arrival = time_now + distance[previous][suffix_customer]

        if arrival > inst.due[suffix_customer]:
            penalty += penalties.time_window_penalty * (arrival - inst.due[suffix_customer])

        time_now = max(arrival, inst.ready[suffix_customer]) + inst.service[suffix_customer]
        previous = suffix_customer

    end_time = time_now + distance[previous][0]

    if end_time > inst.vehicle_end[vehicle]:
        penalty += penalties.shift_penalty * (end_time - inst.vehicle_end[vehicle])

    prev_node = 0 if position == 0 else route[position - 1]
    next_node = 0 if position == len(route) else route[position]

    travel_delta = (
        distance[prev_node][customer]
        + distance[customer][next_node]
        - distance[prev_node][next_node]
    )

    return penalty, travel_delta


def insertion_move_for_position(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
    customer: int,
    vehicle: int,
    position: int,
) -> InsertionMove:
    new_route_penalty, travel_delta = insertion_penalty_from_cache(
        inst=inst,
        solution=solution,
        penalties=penalties,
        customer=customer,
        vehicle=vehicle,
        position=position,
    )

    old_route_penalty = solution.route_cache[vehicle].penalty
    additional_penalty = new_route_penalty - old_route_penalty

    delta_score = inst.profit[customer] - additional_penalty

    return InsertionMove(
        customer=customer,
        vehicle=vehicle,
        position=position,
        delta_score=delta_score,
        travel_delta=travel_delta,
        new_route_penalty=new_route_penalty,
    )


def insertion_sort_key(move: InsertionMove) -> tuple[float, int, float]:
    return (
        -move.delta_score,
        move.travel_delta,
        move.new_route_penalty,
    )


def add_top_move(top_moves: list[InsertionMove], move: InsertionMove, k: int) -> None:

    top_moves.append(move)
    top_moves.sort(key=insertion_sort_key)

    if len(top_moves) > k:
        top_moves.pop()


def top_insertions_for_customer(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
    customer: int,
    k: int,
) -> list[InsertionMove]:
    if customer not in solution.unserved:
        return []

    top_moves: list[InsertionMove] = []

    for vehicle in inst.solo_feasible_vehicles[customer]:
        route = solution.routes[vehicle]

        for position in range(len(route) + 1):
            move = insertion_move_for_position(
                inst=inst,
                solution=solution,
                penalties=penalties,
                customer=customer,
                vehicle=vehicle,
                position=position,
            )
            add_top_move(top_moves, move, k)

    return top_moves


def best_insertion_for_customer(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
    customer: int,
) -> InsertionMove | None:
    moves = top_insertions_for_customer(
        inst=inst,
        solution=solution,
        penalties=penalties,
        customer=customer,
        k=1,
    )
    return moves[0] if moves else None


def build_repair_candidates(
    inst,
    solution: Solution,
    removed_customers: list[int],
    extra_unserved_limit: int,
) -> list[int]:
    candidates = []
    seen = set()
    for customer in removed_customers:
        if customer in solution.unserved:
            candidates.append(customer)
            seen.add(customer)

    extra = (
        c for c in solution.unserved
        if c not in seen and inst.solo_feasible_vehicles[c]
    )
    if extra_unserved_limit > 0:
        extra_top = heapq.nsmallest(
            extra_unserved_limit,
            extra,
            key=lambda c: (
                -inst.profit_density_lb[c],
                -inst.profit[c],
                len(inst.solo_feasible_vehicles[c]),
            ),
        )
        candidates.extend(extra_top)
    return candidates


###########################################

# Repair operators


class GreedyBestInsertionRepair(RepairOperator):
    # Repair operators declare which destroy-result tags they need to run;
    # "customer_pool" means "a list of customers available for insertion".
    requires = {"customer_pool"}

    def __init__(
        self,
        extra_unserved_limit: int,
        max_insertions: int | None,
        min_delta_score: float,
        initial_weight: float,
    ):
        super().__init__("greedy_best_insertion", initial_weight)
        self.extra_unserved_limit = extra_unserved_limit
        self.max_insertions = max_insertions
        self.min_delta_score = min_delta_score

    def apply(
        self,
        inst,
        destroy_result: DestroyResult,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> Solution:
        solution = destroy_result.partial_solution

        # apply_insertion() keeps the route-cache in sync after each single
        # insertion; fine here since repair only inserts a bounded candidate
        # list, but would need a batched cache update for larger changes.
        candidates = build_repair_candidates(
            inst=inst,
            solution=solution,
            removed_customers=destroy_result.removed_customers,
            extra_unserved_limit=self.extra_unserved_limit,
        )

        inserted = 0

        while self.max_insertions is None or inserted < self.max_insertions:
            best_move = None

            for customer in candidates:
                move = best_insertion_for_customer(
                    inst=inst,
                    solution=solution,
                    penalties=penalties,
                    customer=customer,
                )

                if move is None:
                    continue

                if move.delta_score <= self.min_delta_score:
                    continue

                if best_move is None:
                    best_move = move
                elif insertion_sort_key(move) < insertion_sort_key(best_move):
                    best_move = move

            if best_move is None:
                break

            apply_insertion(solution, inst, penalties, best_move)
            inserted += 1

        return solution


class Regret2InsertionRepair(RepairOperator):
    requires = {"customer_pool"}

    def __init__(
        self,
        extra_unserved_limit: int,
        max_insertions: int | None,
        min_delta_score: float,
        initial_weight: float,
    ):
        super().__init__("regret2_insertion", initial_weight)
        self.extra_unserved_limit = extra_unserved_limit
        self.max_insertions = max_insertions
        self.min_delta_score = min_delta_score

    def apply(
        self,
        inst,
        destroy_result: DestroyResult,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> Solution:
        solution = destroy_result.partial_solution

        candidates = build_repair_candidates(
            inst=inst,
            solution=solution,
            removed_customers=destroy_result.removed_customers,
            extra_unserved_limit=self.extra_unserved_limit,
        )

        inserted = 0

        while self.max_insertions is None or inserted < self.max_insertions:
            selected_move = None
            selected_regret = None

            for customer in candidates:
                moves = top_insertions_for_customer(
                    inst=inst,
                    solution=solution,
                    penalties=penalties,
                    customer=customer,
                    k=2,
                )

                if not moves:
                    continue

                best = moves[0]

                if best.delta_score <= self.min_delta_score:
                    continue

                if len(moves) == 1:
                    regret = 1_000_000.0 + best.delta_score
                else:
                    regret = best.delta_score - moves[1].delta_score

                if selected_move is None:
                    selected_move = best
                    selected_regret = regret
                elif regret > selected_regret:
                    selected_move = best
                    selected_regret = regret
                elif regret == selected_regret and insertion_sort_key(best) < insertion_sort_key(selected_move):
                    selected_move = best
                    selected_regret = regret

            if selected_move is None:
                break

            apply_insertion(solution, inst, penalties, selected_move)
            inserted += 1

        return solution



# Operator selection and weight update


def select_operator(operators: list[ALNSOperator], rng: random.Random) -> ALNSOperator:
    total_weight = sum(op.weight for op in operators)
    draw = rng.random() * total_weight
    cumulative = 0.0

    for op in operators:
        cumulative += op.weight

        if cumulative >= draw:
            return op

    return operators[-1]


def update_operator_weights(
    destroy_operators: list[DestroyOperator],
    repair_operators: list[RepairOperator],
    params: ALNSParams,
) -> None:
    for op in destroy_operators:
        op.update_weight(params)

    for op in repair_operators:
        op.update_weight(params)


def operator_summary(operators: list[ALNSOperator]) -> list[dict]:
    rows = []

    for op in operators:
        avg_time = op.total_time / op.total_uses if op.total_uses else 0.0

        rows.append(
            {
                "name": op.name,
                "weight": op.weight,
                "uses": op.total_uses,
                "avg_time": avg_time,
                "total_time": op.total_time,
            }
        )

    return rows



# Simulated Annealing

def accept_sa(
    current_score: float,
    candidate_score: float,
    temperature: float,
    rng: random.Random,
) -> bool:
    if candidate_score >= current_score:
        return True

    probability = math.exp((candidate_score - current_score) / temperature)

    return rng.random() < probability


def cool_down(temperature: float, sa: SAParams) -> float:
    return max(
        sa.min_temperature,
        temperature * sa.cooling_rate,
    )


def reheat(temperature: float, sa: SAParams) -> float:
    return max(
        temperature,
        sa.initial_temperature * sa.reheat_factor,
    )



# Result


@dataclass(slots=True)
class ALNSResult:
    best_solution: Solution
    best_evaluation: SolutionEvaluation

    current_solution: Solution
    current_evaluation: SolutionEvaluation

    iterations: int
    runtime_seconds: float
    restarts: int

    destroy_summary: list[dict]
    repair_summary: list[dict]




##################################

# Main ALNS


def run_alns(
    runtime: float,
    inst,
    start_solution: Solution,
    sa: SAParams,
    params: ALNSParams,
    penalties: PenaltyParams,
    destroy_operators: list[DestroyOperator],
    repair_operators: list[RepairOperator],
) -> ALNSResult:
    rng = random.Random(params.random_seed)

    current_solution = start_solution.copy()
    current_eval = initialize_solution_state(inst, current_solution, penalties)


    best_solution = copy_solution_with_state(current_solution)
    best_eval = current_eval


    temperature = sa.initial_temperature


    iterations = 0
    restarts = 0
    no_accept_counter = 0
    start_time = time.perf_counter()

    while time.perf_counter() - start_time < runtime:
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

        if accepted:
            current_solution = candidate_solution
            current_eval = candidate_eval
            no_accept_counter = 0
        else:
            no_accept_counter += 1

        destroy_op.observe(operator_score, destroy_time)
        repair_op.observe(operator_score, repair_time)

        if iterations % params.segment_length == 0:
            update_operator_weights(
                destroy_operators=destroy_operators,
                repair_operators=repair_operators,
                params=params,
            )

            if params.verbose:
                print(
                    f"iter={iterations:>7} | "
                    f"best={best_eval.profit:>8} | "
                    f"current_score={current_eval.score:>10.2f} | "
                    f"current_feasible={current_eval.feasible} | "
                    f"T={temperature:>8.4f}"
                )

        temperature = cool_down(temperature, sa)

        if no_accept_counter >= params.no_accept_limit:
            current_solution = random_start_solution(inst, rng, penalties)
            current_eval = evaluation_from_solution(inst, current_solution)

            temperature = reheat(temperature, sa)

            no_accept_counter = 0
            restarts += 1

            if params.verbose:
                print(
                    f"restart={restarts} at iter={iterations} | "
                    f"new current profit={current_eval.profit} | "
                    f"best profit={best_eval.profit} | "
                    f"T={temperature:.4f}"
                )

    update_operator_weights(
        destroy_operators=destroy_operators,
        repair_operators=repair_operators,
        params=params,
    )

    return ALNSResult(
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
