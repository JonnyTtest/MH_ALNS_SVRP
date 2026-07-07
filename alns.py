from __future__ import annotations

import sys

if sys.version_info >= (3, 10):
    from dataclasses import dataclass as _dc
    import functools
    dataclass = functools.partial(_dc, slots=True)
else:
    from dataclasses import dataclass
import heapq
import math
import random
import time

from solution import Solution
from initial_solution import find_best_insertion, insert_customer as greedy_insert_customer



# Parameterclasses


@dataclass
class SAParams:
    initial_temperature: float
    min_temperature: float
    cooling_rate: float
    reheat_factor: float


@dataclass
class PenaltyParams:
    time_window_penalty: float
    shift_penalty: float
    skill_penalty: float


@dataclass
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

    # Intensification: reset the current solution to the global best after
    # this many iterations without a new global best (0 = disabled).
    return_to_best_limit: int = 0

    # Local-search polish: on a new global best (at most once per this many
    # seconds), Or-opt every route and refill freed capacity (0 = disabled).
    polish_interval_seconds: float = 0.0



# Cached evaluation

@dataclass
class RouteCache:
    arrival: list[int]
    service_start: list[int]
    departure: list[int]
    prefix_penalty: list[float]
    end_time: int
    penalty: float
    # Savelsbergh forward time slack per position; only computed (non-empty)
    # when the route is feasible (penalty == 0). slack[i] = max delay of the
    # service start at position i that keeps positions i..end and the return
    # to the depot feasible. Enables O(1) insertion feasibility checks.
    slack: list[int]


@dataclass
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

    slack: list[int] = []

    if penalty == 0.0 and route:
        m = len(route)
        slack = [0] * m
        slack_end = inst.vehicle_end[vehicle] - end_time
        slack[m - 1] = min(inst.due[route[m - 1]] - service_start_list[m - 1], slack_end)

        for i in range(m - 2, -1, -1):
            wait_next = service_start_list[i + 1] - arrival_list[i + 1]
            slack[i] = min(
                inst.due[route[i]] - service_start_list[i],
                wait_next + slack[i + 1],
            )

    return RouteCache(
        arrival=arrival_list,
        service_start=service_start_list,
        departure=departure_list,
        prefix_penalty=prefix_penalty,
        end_time=end_time,
        penalty=penalty,
        slack=slack,
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
    deadline: float | None = None,
) -> Solution:
    solution = Solution.empty(inst)

    customers = [
        c for c in inst.customers
        if inst.solo_feasible_vehicles[c]
    ]

    rng.shuffle(customers)

    for index, customer in enumerate(customers):
        if (
            deadline is not None
            and (index & 31) == 0
            and time.perf_counter() > deadline
        ):
            break

        insertion = find_best_insertion(inst, solution.routes, customer)

        if insertion is not None:
            greedy_insert_customer(solution, inst, insertion)

    initialize_solution_state(inst, solution, penalties)
    return solution



# Destroy / repair data

@dataclass
class DestroyResult:
    partial_solution: Solution
    removed_customers: list[int]
    origin_vehicle: dict[int, int]
    origin_position: dict[int, int]
    affected_vehicles: set[int]
    tags: set[str]


@dataclass
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
        deadline: float | None = None,
    ) -> Solution:
        raise NotImplementedError



# Destroy helpers


def number_to_remove(
    n_served: int,
    fraction,
    min_remove: int,
    max_remove: int,
    rng: random.Random | None = None,
) -> int:
    """`fraction` is either a fixed float or a (lo, hi) range sampled per call,
    which diversifies the destroy size (small repairs polish, large ones
    restructure)."""
    if isinstance(fraction, tuple):
        lo, hi = fraction
        f = rng.uniform(lo, hi) if rng is not None else (lo + hi) / 2.0
    else:
        f = fraction

    q = round(f * n_served)
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
            rng=rng,
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
            rng=rng,
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
            rng=rng,
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



class RelatedRemoval(DestroyOperator):
    """Shaw-style related removal: picks a random served seed customer and
    removes it together with its nearest served neighbors (biased sampling).
    Spatially correlated removals give the repair step a real chance to
    re-route a whole neighborhood instead of scattered singletons."""

    def __init__(self, fraction, min_remove, max_remove, bias, initial_weight):
        super().__init__("related_removal", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove
        self.bias = bias  # >1: strongly prefer the closest neighbors

    def apply(self, inst, solution, rng, penalties):
        served = served_customers(solution)

        if not served:
            return build_destroy_result(inst, solution, [], "related_removal", penalties)

        q = number_to_remove(
            n_served=len(served),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
            rng=rng,
        )

        seed = rng.choice(served)
        served_set = set(served)
        removed = [seed]
        served_set.discard(seed)

        # nearest_customers[seed] is precomputed by increasing distance.
        neighbors = [c for c in inst.nearest_customers[seed] if c in served_set]

        while len(removed) < q and neighbors:
            index = int((rng.random() ** self.bias) * len(neighbors))
            removed.append(neighbors.pop(index))

        return build_destroy_result(inst, solution, removed, "related_removal", penalties)


class WorstDetourRemoval(DestroyOperator):
    """Solution-dependent worst removal: removes customers with the poorest
    profit per travel detour in their *current* route position (biased
    sampling, Ropke & Pisinger 2006). Complements the static density-based
    ranking, which never changes between iterations."""

    def __init__(self, fraction, min_remove, max_remove, bias, initial_weight):
        super().__init__("worst_detour_removal", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove
        self.bias = bias

    def apply(self, inst, solution, rng, penalties):
        distance = inst.distance
        scored = []

        for route in solution.routes:
            last = len(route) - 1
            for i, customer in enumerate(route):
                prev_node = route[i - 1] if i > 0 else 0
                next_node = route[i + 1] if i < last else 0
                detour = (
                    distance[prev_node][customer]
                    + distance[customer][next_node]
                    - distance[prev_node][next_node]
                )
                # rounded distances can violate the triangle inequality
                ratio = inst.profit[customer] / (max(detour, 0) + 1.0)
                scored.append((ratio, customer))

        if not scored:
            return build_destroy_result(inst, solution, [], "worst_detour_removal", penalties)

        q = number_to_remove(
            n_served=len(scored),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
            rng=rng,
        )

        scored.sort()  # ascending ratio: worst first
        pool = [customer for _, customer in scored]
        removed = []

        while len(removed) < q and pool:
            index = int((rng.random() ** self.bias) * len(pool))
            removed.append(pool.pop(index))

        return build_destroy_result(inst, solution, removed, "worst_detour_removal", penalties)


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
    m = len(route)

    if position == 0:
        previous = 0
        time_now = inst.vehicle_start[vehicle]
        penalty = 0.0
    else:
        previous = route[position - 1]
        time_now = cache.departure[position - 1]
        penalty = cache.prefix_penalty[position]

    next_node = 0 if position == m else route[position]
    travel_delta = (
        distance[previous][customer]
        + distance[customer][next_node]
        - distance[previous][next_node]
    )

    if not inst.required_skills[customer].issubset(vehicle_skills):
        penalty += penalties.skill_penalty

    arrival = time_now + distance[previous][customer]

    if arrival > inst.due[customer]:
        penalty += penalties.time_window_penalty * (arrival - inst.due[customer])

    time_now = max(arrival, inst.ready[customer]) + inst.service[customer]
    previous = customer

    departures = cache.departure
    prefix_penalty = cache.prefix_penalty

    for j in range(position, m):
        if j > position:
            # From here on the predecessor node equals the cached one, so once
            # the timeline re-synchronizes the cached suffix applies verbatim.
            cached_departure = departures[j - 1]

            if time_now == cached_departure:
                return penalty + (cache.penalty - prefix_penalty[j]), travel_delta

            if time_now < cached_departure and cache.penalty == prefix_penalty[j]:
                # Earlier timeline + penalty-free cached suffix: arrivals only
                # get earlier, so the suffix stays penalty-free.
                return penalty, travel_delta

        suffix_customer = route[j]

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

def build_repair_candidates(
    inst,
    solution: Solution,
    removed_customers: list[int],
    extra_unserved_limit: int,
    rng: random.Random | None = None,
) -> list[int]:
    candidates = []
    seen = set()
    for customer in removed_customers:
        if customer in solution.unserved:
            candidates.append(customer)
            seen.add(customer)

    extra = [
        c for c in solution.unserved
        if c not in seen and inst.solo_feasible_vehicles[c]
    ]
    if extra_unserved_limit > 0 and extra:
        # Half exploitation (best static profit density), half exploration
        # (uniform sample) -- a purely static top list would offer the same
        # extra candidates in every iteration and starve the rest.
        top_limit = extra_unserved_limit // 2 if rng is not None else extra_unserved_limit
        extra_top = heapq.nsmallest(
            top_limit,
            extra,
            key=lambda c: (
                -inst.profit_density_lb[c],
                -inst.profit[c],
                len(inst.solo_feasible_vehicles[c]),
            ),
        )
        candidates.extend(extra_top)

        if rng is not None:
            chosen = set(extra_top)
            rest = [c for c in extra if c not in chosen]
            sample_size = min(extra_unserved_limit - len(extra_top), len(rest))
            if sample_size > 0:
                candidates.extend(rng.sample(rest, sample_size))
    return candidates


def vehicle_top_moves(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
    customer: int,
    vehicle: int,
    k: int,
    noise_amp: float = 0.0,
    rng: random.Random | None = None,
) -> list[InsertionMove]:
    """Top-k insertion moves for `customer` restricted to a single vehicle.

    Tight inlined variant of insertion_move_for_position: one pass over all
    positions with the early-exit suffix evaluation, manual top-k tracking
    (k <= 2 in practice) and a pruning bound that stops scanning positions
    once the unavoidable time-window violation alone wipes out the profit
    (departures are non-decreasing along the route).
    """
    route = solution.routes[vehicle]
    cache = solution.route_cache[vehicle]
    distance = inst.distance
    dist_customer = distance[customer]
    departures = cache.departure
    prefix_penalty = cache.prefix_penalty
    old_penalty = cache.penalty

    ready = inst.ready
    due = inst.due
    service = inst.service
    required = inst.required_skills

    ready_c = ready[customer]
    due_c = due[customer]
    service_c = service[customer]
    profit_c = inst.profit[customer]

    tw_pen = penalties.time_window_penalty
    shift_pen = penalties.shift_penalty
    skill_pen = penalties.skill_penalty

    vehicle_skills = inst.vehicle_skills[vehicle]
    v_start = inst.vehicle_start[vehicle]
    v_end = inst.vehicle_end[vehicle]
    m = len(route)

    base_penalty = skill_pen if not required[customer].issubset(vehicle_skills) else 0.0

    if cache.penalty == 0.0 and base_penalty == 0.0:
        # Fast path for feasible routes (the common case): O(1) feasibility
        # per position via the cached forward time slack; only feasible
        # insertions are generated (delta_score = profit, penalty stays 0).
        service_starts = cache.service_start
        slack = cache.slack
        best: list[tuple[int, int]] = []  # (travel_delta, position)

        for position in range(m + 1):
            if position == 0:
                previous = 0
                time_prev = v_start
            else:
                time_prev = departures[position - 1]

                if time_prev > due_c:
                    break  # departures are non-decreasing: no later fit

                previous = route[position - 1]

            d_prev_customer = distance[previous][customer]
            arrival = time_prev + d_prev_customer

            if arrival > due_c:
                continue

            depart_c = (arrival if arrival >= ready_c else ready_c) + service_c

            if position == m:
                next_node = 0

                if depart_c + dist_customer[0] > v_end:
                    continue
            else:
                next_node = route[position]
                arrival_next = depart_c + dist_customer[next_node]
                ready_n = ready[next_node]
                begin_next = arrival_next if arrival_next >= ready_n else ready_n
                delay = begin_next - service_starts[position]

                if delay > 0 and delay > slack[position]:
                    continue

            travel_delta = (
                d_prev_customer
                + dist_customer[next_node]
                - distance[previous][next_node]
            )

            if noise_amp > 0.0 and rng is not None:
                # Noised insertion (Ropke & Pisinger 2006): perturb the
                # ranking so the deterministic cheapest position is not
                # always chosen. Only the ranking is noised; the executed
                # move itself stays feasible.
                travel_delta += rng.uniform(-noise_amp, noise_amp)

            if len(best) < k:
                best.append((travel_delta, position))
                best.sort()
            elif travel_delta < best[-1][0]:
                best[-1] = (travel_delta, position)
                best.sort()

        return [
            InsertionMove(
                customer=customer,
                vehicle=vehicle,
                position=position,
                delta_score=profit_c,
                travel_delta=travel_delta,
                new_route_penalty=0.0,
            )
            for travel_delta, position in best
        ]

    # Positions after this departure time can only yield delta_score <= 0
    # (arrival - due_c alone costs more than the profit); such moves are
    # discarded by the repair operators anyway.
    prune_departure = due_c + profit_c / tw_pen

    results: list[tuple[tuple, int, float, int]] = []  # (sort_key, position, penalty, travel_delta)

    for position in range(m + 1):
        if position == 0:
            previous = 0
            time_now = v_start
            penalty = base_penalty
        else:
            time_now = departures[position - 1]
            if time_now > prune_departure:
                break
            previous = route[position - 1]
            penalty = prefix_penalty[position] + base_penalty

        next_node = 0 if position == m else route[position]
        d_prev_customer = distance[previous][customer]
        travel_delta = (
            d_prev_customer
            + dist_customer[next_node]
            - distance[previous][next_node]
        )

        arrival = time_now + d_prev_customer
        if arrival > due_c:
            penalty += tw_pen * (arrival - due_c)

        time_now = (arrival if arrival >= ready_c else ready_c) + service_c
        previous = customer

        j = position
        while j < m:
            if j > position:
                cached_departure = departures[j - 1]

                if time_now == cached_departure:
                    penalty += old_penalty - prefix_penalty[j]
                    break

                if time_now < cached_departure and old_penalty == prefix_penalty[j]:
                    break

            suffix_customer = route[j]

            if not required[suffix_customer].issubset(vehicle_skills):
                penalty += skill_pen

            arrival = time_now + distance[previous][suffix_customer]
            due_s = due[suffix_customer]

            if arrival > due_s:
                penalty += tw_pen * (arrival - due_s)

            ready_s = ready[suffix_customer]
            time_now = (arrival if arrival >= ready_s else ready_s) + service[suffix_customer]
            previous = suffix_customer
            j += 1
        else:
            end_time = time_now + distance[previous][0]

            if end_time > v_end:
                penalty += shift_pen * (end_time - v_end)

        delta_score = profit_c - (penalty - old_penalty)
        sort_key = (-delta_score, travel_delta, penalty)

        if len(results) < k:
            results.append((sort_key, position, penalty, travel_delta))
            results.sort()
        elif sort_key < results[-1][0]:
            results[-1] = (sort_key, position, penalty, travel_delta)
            results.sort()

    return [
        InsertionMove(
            customer=customer,
            vehicle=vehicle,
            position=position,
            delta_score=-sort_key[0],
            travel_delta=travel_delta,
            new_route_penalty=penalty,
        )
        for sort_key, position, penalty, travel_delta in results
    ]


class MoveCache:
    """Per-(customer, vehicle) top-k insertion-move cache.

    Inserting a customer only modifies one route, so cached moves on all other
    vehicles stay exact. After each insertion only the (candidate, vehicle)
    entries of the modified vehicle are recomputed (Ropke & Pisinger 2006).
    """

    __slots__ = ("inst", "solution", "penalties", "k", "candidates", "by_customer")

    def __init__(
        self,
        inst,
        solution: Solution,
        penalties: PenaltyParams,
        candidates: list[int],
        k: int,
        deadline: float | None = None,
    ):
        self.inst = inst
        self.solution = solution
        self.penalties = penalties
        self.k = k
        self.candidates: list[int] = []
        self.by_customer: dict[int, dict[int, list[InsertionMove]]] = {}

        for index, customer in enumerate(candidates):
            if (
                deadline is not None
                and (index & 7) == 0
                and time.perf_counter() > deadline
            ):
                break

            self.by_customer[customer] = {
                vehicle: vehicle_top_moves(inst, solution, penalties, customer, vehicle, k)
                for vehicle in inst.solo_feasible_vehicles[customer]
            }
            self.candidates.append(customer)

    def top_moves(self, customer: int) -> list[InsertionMove]:
        best_move = None
        best_key = None
        second_move = None
        second_key = None
        want_two = self.k > 1

        for moves in self.by_customer[customer].values():
            for move in moves:
                key = (-move.delta_score, move.travel_delta, move.new_route_penalty)

                if best_key is None or key < best_key:
                    second_move, second_key = best_move, best_key
                    best_move, best_key = move, key
                elif want_two and (second_key is None or key < second_key):
                    second_move, second_key = move, key

        if best_move is None:
            return []

        if not want_two or second_move is None:
            return [best_move]

        return [best_move, second_move]

    def remove_candidate(self, customer: int) -> None:
        self.candidates.remove(customer)
        del self.by_customer[customer]

    def invalidate_vehicle(self, vehicle: int) -> None:
        for customer in self.candidates:
            per_vehicle = self.by_customer[customer]

            if vehicle in per_vehicle:
                per_vehicle[vehicle] = vehicle_top_moves(
                    self.inst, self.solution, self.penalties, customer, vehicle, self.k
                )


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
        deadline: float | None = None,
    ) -> Solution:
        solution = destroy_result.partial_solution

        candidates = build_repair_candidates(
            inst=inst,
            solution=solution,
            removed_customers=destroy_result.removed_customers,
            extra_unserved_limit=self.extra_unserved_limit,
            rng=rng,
        )

        move_cache = MoveCache(
            inst=inst,
            solution=solution,
            penalties=penalties,
            candidates=candidates,
            k=1,
            deadline=deadline,
        )

        inserted = 0

        while self.max_insertions is None or inserted < self.max_insertions:
            if deadline is not None and time.perf_counter() > deadline:
                break

            best_move = None

            for customer in move_cache.candidates:
                moves = move_cache.top_moves(customer)

                if not moves:
                    continue

                move = moves[0]

                if move.delta_score <= self.min_delta_score:
                    continue

                if best_move is None:
                    best_move = move
                elif insertion_sort_key(move) < insertion_sort_key(best_move):
                    best_move = move

            if best_move is None:
                break

            apply_insertion(solution, inst, penalties, best_move)
            move_cache.remove_candidate(best_move.customer)
            move_cache.invalidate_vehicle(best_move.vehicle)
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
        deadline: float | None = None,
    ) -> Solution:
        solution = destroy_result.partial_solution

        candidates = build_repair_candidates(
            inst=inst,
            solution=solution,
            removed_customers=destroy_result.removed_customers,
            extra_unserved_limit=self.extra_unserved_limit,
            rng=rng,
        )

        move_cache = MoveCache(
            inst=inst,
            solution=solution,
            penalties=penalties,
            candidates=candidates,
            k=2,
            deadline=deadline,
        )

        inserted = 0

        while self.max_insertions is None or inserted < self.max_insertions:
            if deadline is not None and time.perf_counter() > deadline:
                break

            selected_move = None
            selected_regret = None

            for customer in move_cache.candidates:
                moves = move_cache.top_moves(customer)

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
            move_cache.remove_candidate(selected_move.customer)
            move_cache.invalidate_vehicle(selected_move.vehicle)
            inserted += 1

        return solution



class SequentialCheapestInsertionRepair(RepairOperator):
    """Single-pass repair (Kovacs et al. 2012 style): candidates are visited
    once in a fixed order (by profit, or shuffled) and each is immediately
    inserted at its cheapest feasible position. Roughly an order of magnitude
    cheaper per call than the best-first repairs, trading insertion-order
    optimality for iteration throughput -- the adaptive weights decide when
    that trade is worth it.
    """

    requires = {"customer_pool"}

    def __init__(
        self,
        extra_unserved_limit: int,
        order: str,
        initial_weight: float,
        noise_amp: float = 0.0,
    ):
        name = f"sequential_insertion_{order}"
        if noise_amp > 0.0:
            name += "_noise"
        super().__init__(name, initial_weight)
        self.extra_unserved_limit = extra_unserved_limit
        self.order = order
        self.noise_amp = noise_amp

    def apply(
        self,
        inst,
        destroy_result: DestroyResult,
        rng: random.Random,
        penalties: PenaltyParams,
        deadline: float | None = None,
    ) -> Solution:
        solution = destroy_result.partial_solution

        candidates = build_repair_candidates(
            inst=inst,
            solution=solution,
            removed_customers=destroy_result.removed_customers,
            extra_unserved_limit=self.extra_unserved_limit,
            rng=rng,
        )

        if self.order == "profit":
            candidates.sort(
                key=lambda c: (-inst.profit[c], inst.due[c] - inst.ready[c])
            )
        else:
            rng.shuffle(candidates)

        for index, customer in enumerate(candidates):
            if (
                deadline is not None
                and (index & 15) == 0
                and time.perf_counter() > deadline
            ):
                break

            best_move = None
            best_key = None

            for vehicle in inst.solo_feasible_vehicles[customer]:
                moves = vehicle_top_moves(
                    inst, solution, penalties, customer, vehicle, 1,
                    noise_amp=self.noise_amp, rng=rng,
                )

                if not moves:
                    continue

                move = moves[0]

                if move.delta_score <= 0.0:
                    continue

                key = (-move.delta_score, move.travel_delta, move.new_route_penalty)

                if best_key is None or key < best_key:
                    best_move, best_key = move, key

            if best_move is not None:
                apply_insertion(solution, inst, penalties, best_move)

        return solution



# Local-search polish (Or-opt) and capacity refill


def polish_route(inst, vehicle: int, route: list[int], deadline: float | None = None):
    """Or-opt within one route: relocate segments of length 1..3 to a position
    that strictly reduces travel time, keeping the route feasible. First
    improvement, repeated until no move improves. Profit is unchanged; the
    point is to free shift capacity for additional insertions.

    Returns (route, improved_flag)."""
    distance = inst.distance

    if len(route) < 3:
        return route, False

    improved_any = False
    improved = True

    while improved:
        improved = False

        if deadline is not None and time.perf_counter() > deadline:
            break

        m = len(route)

        for seg_len in (1, 2, 3):
            if improved:
                break

            for i in range(m - seg_len + 1):
                a = route[i - 1] if i > 0 else 0
                b = route[i]
                c = route[i + seg_len - 1]
                d = route[i + seg_len] if i + seg_len < m else 0

                remove_gain = distance[a][b] + distance[c][d] - distance[a][d]

                if remove_gain <= 0:
                    continue

                rest = route[:i] + route[i + seg_len:]
                segment = route[i:i + seg_len]
                n_rest = len(rest)

                for j in range(n_rest + 1):
                    if j == i:
                        continue  # original position

                    p = rest[j - 1] if j > 0 else 0
                    q = rest[j] if j < n_rest else 0
                    add_cost = distance[p][b] + distance[c][q] - distance[p][q]

                    if add_cost >= remove_gain:
                        continue

                    candidate = rest[:j] + segment + rest[j:]

                    if inst.route_is_feasible(vehicle, candidate):
                        route = candidate
                        improved = True
                        improved_any = True
                        break

                if improved:
                    break

    return route, improved_any


def fill_solution(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
    rng: random.Random,
    limit: int,
    deadline: float | None = None,
) -> None:
    """Sequential pass inserting unserved candidates into free capacity
    (used after polishing, which shortens routes but does not add profit)."""
    candidates = build_repair_candidates(
        inst=inst,
        solution=solution,
        removed_customers=[],
        extra_unserved_limit=limit,
        rng=rng,
    )
    candidates.sort(key=lambda c: (-inst.profit[c], inst.due[c] - inst.ready[c]))

    for index, customer in enumerate(candidates):
        if (
            deadline is not None
            and (index & 15) == 0
            and time.perf_counter() > deadline
        ):
            break

        best_move = None
        best_key = None

        for vehicle in inst.solo_feasible_vehicles[customer]:
            moves = vehicle_top_moves(inst, solution, penalties, customer, vehicle, 1)

            if not moves:
                continue

            move = moves[0]

            if move.delta_score <= 0.0:
                continue

            key = (-move.delta_score, move.travel_delta, move.new_route_penalty)

            if best_key is None or key < best_key:
                best_move, best_key = move, key

        if best_move is not None:
            apply_insertion(solution, inst, penalties, best_move)


def polish_and_fill(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
    rng: random.Random,
    deadline: float | None = None,
    fill_limit: int = 100,
) -> Solution:
    """Or-opt every route of a (feasible) solution, then refill freed
    capacity with unserved customers. Returns a new solution whose profit is
    >= the input's."""
    polished = copy_solution_with_state(solution)
    changed = False

    for vehicle in range(inst.num_vehicles):
        if deadline is not None and time.perf_counter() > deadline:
            break

        new_route, improved = polish_route(
            inst, vehicle, polished.routes[vehicle], deadline
        )

        if improved:
            polished.routes[vehicle] = new_route
            changed = True

    if changed:
        initialize_solution_state(inst, polished, penalties)

    fill_solution(inst, polished, penalties, rng, fill_limit, deadline)
    return polished



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
# Result


@dataclass
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
    since_best = 0
    last_polish = 0.0
    start_time = time.perf_counter()
    deadline = start_time + runtime

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
            since_best = 0

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
                    best_solution = polished
                    best_eval = polished_eval
                    # continue the search from the compressed solution
                    candidate_solution = copy_solution_with_state(polished)
                    candidate_eval = polished_eval
        else:
            since_best += 1

        if (
            params.return_to_best_limit > 0
            and since_best >= params.return_to_best_limit
        ):
            current_solution = copy_solution_with_state(best_solution)
            current_eval = best_eval
            since_best = 0

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
